import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule


class PLModel(LightningModule):
    def __init__(self, hparams, normalization=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.criterion = VAELoss(
            beta=hparams['beta'],
            epoch_resets=hparams['epoch_resets'],
            num_resets=hparams['num_resets']           
        )

        if hparams['model'] == 'mlp':
            self.model = VAE(latent_dim=hparams['latent_dim'], input_dim=hparams['freq_dim'])
        
        elif hparams['model'] == 'cnn':
            self.model = VAECNN(
                latent_dim = hparams['latent_dim'],
                freq_dim = hparams['freq_dim'],
                in_channels = hparams["n_channels"],
                normalization = hparams["normalization"],
                hidden_channels = hparams["hidden_channels"]
                )

        if normalization is not None:
            self.norm_params = normalization
            #self.model.save_norm(mu=normalization['mu'], std=normalization['std'])

    def training_step(self, train_batch, batch_idx):
        x,_,_ = train_batch
        mu, logvar, x_rec = self.model(x)
        nll, kl = self.criterion(x_rec, mu, logvar, x) # kl is multiplied by beta
        
        # log
        kl_loss = kl/self.criterion.current_beta
        loss = nll + kl
        rec_loss = nll
        self.log('train_kl_loss', kl_loss)
        self.log('train_rec_loss', rec_loss)
        self.log('train_loss', loss)
        self.log('beta', self.criterion.current_beta)

        return {'rec_loss': rec_loss, 'kl_loss': kl_loss, 'loss': loss}

    def training_epoch_end(self, outputs):
        self.criterion.update_beta()
        return

    def validation_step(self, val_batch, batch_idx):
        x,_,_ = val_batch
        mu, logvar, x_rec = self.model(x)
        nll, kl = self.criterion(x_rec, mu, logvar, x)
        
        # log
        kl_loss = kl/self.criterion.current_beta
        loss = nll + kl
        rec_loss = nll
        self.log('val_kl_loss', kl_loss)
        self.log('val_rec_loss', rec_loss)
        self.log('val_loss', loss)

        return {'rec_loss': rec_loss, 'kl_loss': kl_loss, 'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['learning_rate'])
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1)
        return [optimizer], [scheduler]

class VAELoss(torch.nn.Module):
    '''VAE loss with annealing.
    '''
    def __init__(self, beta:float, epoch_resets:int, num_resets:int):
        super().__init__()
        self.beta = beta      # final beta value
        self.num_resets = num_resets # no. of KL resets
        self.current_beta = 0 # beta running value
        self.current_time = 0 # tracking the number of resets
        self.mse = torch.nn.MSELoss(reduction='sum')
        self.epoch_resets = epoch_resets  # epochs to achieve final beta
        self.beta_rate = self.beta/self.epoch_resets

    def update_beta(self):
        '''Updates beta value or keep constant.
        '''
        self.current_beta += self.beta_rate
        if self.current_beta > self.beta:
            if self.current_time >= self.num_resets:
                self.current_beta = self.beta
            elif self. current_time < self.num_resets:
                self.current_beta = 0 # reset
                self.current_time += 1

    def forward(self, x_pred, mu, logvar, x):
        kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        nll = self.mse(torch.flatten(x_pred,1), torch.flatten(x,1))
        return nll, self.current_beta*kl


class DownBlock(torch.nn.Module):
    '''Basic downsampling block'''
    def __init__(self, in_channels:int, kernel_size:int, residual:bool=True, batch_norm:bool=True):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2)
        self.conv2 = torch.nn.Conv1d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2)
        self.bn1 = torch.nn.BatchNorm1d(in_channels)
        self.bn2 = torch.nn.BatchNorm1d(in_channels)
        self.residual = residual
        self.down = torch.nn.MaxPool1d(kernel_size=2)
    
    def forward(self, x) -> torch.Tensor:
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = torch.nn.functional.leaky_relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        if self.residual:
            x1 += x
        x1 = torch.nn.functional.leaky_relu(x1)
        x1 = self.down(x1)
        return x1

class UpBlock(torch.nn.Module):
    '''Basic upsampling block'''
    def __init__(self, in_channels:int, kernel_size:int, residual:bool=True, batch_norm:bool=True):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2)
        self.conv2 = torch.nn.Conv1d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2)
        self.bn1 = torch.nn.BatchNorm1d(in_channels)
        self.bn2 = torch.nn.BatchNorm1d(in_channels)
        self.residual = residual
        self.upconv = torch.nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = torch.nn.functional.leaky_relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        if self.residual:
            x1 += x
        x1 = torch.nn.functional.leaky_relu(x1)
        x1 = self.upconv(x1)
        return x1

class ConvEncoder(torch.nn.Module):
    '''Encoder module'''
    def __init__(self, input_dim:int, latent_dim:int, in_channels:int, hidden_channels:int, kernel_size:int):
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2), #1024
            DownBlock(in_channels=hidden_channels, kernel_size=kernel_size), #512
            DownBlock(in_channels=hidden_channels, kernel_size=kernel_size), #256
            DownBlock(in_channels=hidden_channels, kernel_size=kernel_size), #128
            DownBlock(in_channels=hidden_channels, kernel_size=kernel_size), #64
            DownBlock(in_channels=hidden_channels, kernel_size=kernel_size), #32
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(hidden_channels*input_dim//32, latent_dim*2),
        )
        
    def forward(self,x):
        return self.layers(x)

class ConvDecoder(torch.nn.Module):
    '''Decoder module'''
    def __init__(self, out_dim:int, latent_dim:int, out_channels:int, hidden_channels:int, kernel_size:int):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_channels*out_dim//32),
            torch.nn.Unflatten(1, (hidden_channels,out_dim//32)), # 64,16
            UpBlock(in_channels=hidden_channels, kernel_size=kernel_size), #32
            UpBlock(in_channels=hidden_channels, kernel_size=kernel_size), #64
            UpBlock(in_channels=hidden_channels, kernel_size=kernel_size), #128
            UpBlock(in_channels=hidden_channels, kernel_size=kernel_size), #256
            UpBlock(in_channels=hidden_channels, kernel_size=kernel_size), #1024
            torch.nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        )
        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class VAE(torch.nn.Module):
    '''VAE model with linear layers and ReLU activation.
    '''
    def __init__(self, latent_dim:int, input_dim:int):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.normalization = None

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, int(3*input_dim/4)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(3*input_dim/4), int(2*input_dim/4)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(2*input_dim/4), int(2*input_dim/4)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(2*input_dim/4), 2*latent_dim),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, int(2*input_dim/4)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(2*input_dim/4), int(2*input_dim/4)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(2*input_dim/4), int(3*input_dim/4)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(3*input_dim/4), input_dim),
        )

    def save_norm(self, mu, std):
        ''' Saves the z-score normalization used during training.
        '''
        self.mu = mu
        self.std = std
        self.normalization = 'z-score'

    def encode(self, x):
        x = x[:,0,:]
        x = self.encoder(x)
        mu, logvar = x[:,0:self.latent_dim], x[:,self.latent_dim:]
        return mu, logvar
    
    def decode(self, x):
        x = self.decoder(x)
        x = x.unsqueeze(1)
        x = torch.sigmoid(x)
        return x

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return (eps*std) + mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_rec = self.decode(z)
        return mu, logvar, x_rec        

    def normalize(self, x):
        if self.normalization is not None:
            return (x - self.mu) / self.std
        else:
            return x

    def unnormalize(self, x_norm):
        if self.normalization is not None:
            return (x_norm * self.std) + self.mu
        else:
            return x_norm

class OLD_VAECNN(torch.nn.Module):
    def __init__(self, latent_dim:int, input_dim:int):
        '''VAE model with 1D convolutional layers and ReLU activation.
        Data input must have dimensions (n_batch, in_channels, freq_dim).
        Args:
            latent_dim: size of latent vector
            input_dim: in_channels * freq_dim
        Fixed parameters:
            kernel_size = 3
            in_channels = 8
            hidden_channels = 64
        '''
        super(VAECNN, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        kernel_size = 3
        hidden_channels = 64
        self.in_channels = 8
        self.freq_dim = self.input_dim // self.in_channels
        self.encoder = ConvEncoder(self.freq_dim, latent_dim, self.in_channels, hidden_channels, kernel_size)
        self.decoder = ConvDecoder(self.freq_dim, latent_dim, self.in_channels, hidden_channels, kernel_size)

    def reshape_input(self, x):
        '''Reshapes the input tensor x (n_batch, input_dim) to (n_batch, in_channels, freq_dim)'''
        n_batch = x.shape[0]
        out = x.clone().reshape(n_batch, self.in_channels, self.freq_dim)
        for c in range(self.in_channels):
            out[:,c,:] = x[:,c*self.freq_dim : (c+1)*self.freq_dim]
        return out

    def reshape_output(self, x):
        '''Reshapes the input tensor x (n_batch, in_channels, freq_dim) to (n_batch, input_dim)'''
        return x.reshape(x.shape[0], -1) 

    def save_norm(self, mu, std):
        ''' Saves the z-score normalization used during training.
        '''
        self.mu = mu
        self.std = std
        self.normalization = 'z-score'

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = x[:,0:self.latent_dim], x[:,self.latent_dim:]
        return mu, logvar
    
    def decode(self, x):
        x = self.decoder(x)
        return x

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return (eps*std) + mu

    def forward(self, x):
        x = self.reshape_input(x)
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_rec = self.decode(z)
        x_rec = self.reshape_output(x_rec)
        return mu, logvar, x_rec


class VAECNN(torch.nn.Module):
    def __init__(self, latent_dim:int, in_channels:int, freq_dim:int, normalization:str, hidden_channels:int):
        '''VAE model with 1D convolutional layers and ReLU activation.
        Data input must have dimensions (n_batch, in_channels, freq_dim).
        Args:
            latent_dim: size of latent vector
            freq_dim: size of each PSD
            in_channels: number of input channels
            normalization: "min-max" or "z-score", defines the output layer
        Fixed parameters:
            kernel_size = 3
        '''
        super(VAECNN, self).__init__()
        self.latent_dim = latent_dim
        self.freq_dim = freq_dim
        self.in_channels = in_channels
        self.normalization = normalization
        
        kernel_size = 3
        self.encoder = ConvEncoder(self.freq_dim, latent_dim, self.in_channels, hidden_channels, kernel_size)
        self.decoder = ConvDecoder(self.freq_dim, latent_dim, self.in_channels, hidden_channels, kernel_size)

    def save_norm(self, mu, std):
        ''' Saves the z-score normalization used during training.
        '''
        self.mu = mu
        self.std = std
        self.normalization = 'z-score'

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = x[:,0:self.latent_dim], x[:,self.latent_dim:]
        return mu, logvar
    
    def decode(self, x):
        x = self.decoder(x)
        return x

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return (eps*std) + mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_rec = self.decode(z)
        
        if self.normalization == "min-max":
            x_rec = torch.sigmoid(x_rec)

        return mu, logvar, x_rec
