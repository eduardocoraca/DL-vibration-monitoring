import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from typing import Tuple, List
import matplotlib.pyplot as plt

class GAN_PL(LightningModule):
    def __init__(self, hparams:dict, normalization=None) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.weight_decay = hparams["weight_decay"]
        self.encoder = Encoder(
            input_dim = hparams["input_dim"]//8,
            latent_dim = hparams["latent_dim"],
            in_channels = 8,      # fixed
            hidden_channels = 64, # fixed
            kernel_size = 3       # fixed
        )
        
        self.generator = Generator(
            out_dim = hparams["input_dim"]//8,
            latent_dim = hparams["latent_dim"],
            out_channels = 8,      # fixed
            hidden_channels = 64,  # fixed
            kernel_size = 3        # fixed
        )

        self.discriminator = Discriminator(
            input_dim = hparams["input_dim"]//8,
            latent_dim = hparams["latent_dim"],
            in_channels = 8,      # fixed
            hidden_channels = 16, # fixed
            kernel_size = 3       # fixed            
        )

        if normalization is not None:
            self.save_norm(mu=normalization['mu'], std=normalization['std'])

        self.loss = torch.nn.BCELoss()
        self.latent_dim = hparams["latent_dim"]
        self.learning_rate = hparams['learning_rate']

    def save_norm(self, mu, std):
        ''' Saves the z-score normalization used during training.
        '''
        self.mu = mu
        self.std = std
        self.normalization = 'z-score'

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        optimizer_d = torch.optim.SGD(
            params = self.discriminator.parameters(),
            lr = self.learning_rate,
            weight_decay = self.weight_decay
        )

        optimizer_ge = torch.optim.SGD(
            params = list(self.generator.parameters()) + list(self.encoder.parameters()),
            lr = self.learning_rate,
        )

        return [optimizer_d, optimizer_ge]

    def training_step(self, batch, _, optimizer_idx): 
        x_valid,_,_ = batch
        
        y_valid = torch.ones(x_valid.size(0), 1).to(x_valid.device)
        y_fake = torch.zeros(x_valid.size(0), 1).to(x_valid.device)

        z_valid = self.encoder(x_valid)
        z_fake = torch.randn((x_valid.shape[0], self.latent_dim)).to(x_valid.device)
        
        x_fake = self.generator(z_fake)
        
        p_valid = self.discriminator(x_valid, z_valid)
        p_fake = self.discriminator(x_fake, z_fake)

        if optimizer_idx == 0:  # Discriminator
            loss_dis = self.loss(p_valid, y_valid) + self.loss(p_fake, y_fake)
            loss_gen = torch.ones_like(loss_dis)*torch.nan # for logging both losses
            loss = loss_dis.clone()

        else:  # Generator
            loss_gen = self.loss(p_fake, y_valid) + self.loss(p_valid, y_fake)
            loss_dis = torch.ones_like(loss_gen)*torch.nan
            loss = loss_gen.clone()

        return {"loss": loss, "loss_gen": loss_gen.detach(), "loss_dis": loss_dis.detach()}

    # [ [{},{}], [{},{}], ...]
    def training_epoch_end(self, outputs):
        avg_loss_dis = torch.mean(torch.stack([output[0]['loss_dis'] for output in outputs]))
        avg_loss_gen = torch.mean(torch.stack([output[1]['loss_gen'] for output in outputs]))

        self.log("train_dis_loss", avg_loss_dis)
        self.log("train_gen_loss", avg_loss_gen)

    def validation_step(self, batch, _):
        x,_,_ = batch
        with torch.no_grad():
            z = self.encoder(x)
            prob_real = self.discriminator(x=x,z=z)
        return {'val_pred': prob_real}

    # def plot_last(self, x:torch.Tensor):
    #     x = x[-1].detach().cpu().numpy()
    #     num_channels = x.shape[0]
    #     fig = plt.figure(figsize=(14,8))
    #     for c in range(num_channels):
    #         ax = fig.add_subplot(num_channels, 1, c + 1)
    #         ax.plot(x[c,:], color='k')
    #     #run[f'figs_gan/{self.epoch_count}'].log(fig)
    #     plt.close('all')

    # def validation_epoch_end(self, outputs):
    #     avg_proba_dis = torch.mean(torch.vstack([output['val_pred'] for output in outputs]))
    #     run['val_pred'].log(avg_proba_dis)
        
    #     if self.epoch_count % 10 == 0:
    #         x,_,_ = next(iter(val_dataloader))
    #         with torch.no_grad():
    #             z = self.model.sample_z(x.shape[0]).to(self.dev)
    #             x_gen = self.model.generator(z)
    #         self.plot_last(x_gen)
    #     return


class DownBlock(torch.nn.Module):
    '''Basic downsampling block'''
    def __init__(self, in_channels:int, kernel_size:int, residual:bool=True, batch_norm:bool=True):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size-2)
        self.conv2 = torch.nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size-2)
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
        self.conv1 = torch.nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size-2)
        self.conv2 = torch.nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size-2)
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

class Encoder(torch.nn.Module):
    '''Encoder module'''
    def __init__(self, input_dim:int, latent_dim:int, in_channels:int, hidden_channels:int, kernel_size:int):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.input_dim = input_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=kernel_size-2), #1024
            DownBlock(in_channels=hidden_channels, kernel_size=kernel_size), #512
            DownBlock(in_channels=hidden_channels, kernel_size=kernel_size), #256
            DownBlock(in_channels=hidden_channels, kernel_size=kernel_size), #128
            DownBlock(in_channels=hidden_channels, kernel_size=kernel_size), #64
            DownBlock(in_channels=hidden_channels, kernel_size=kernel_size), #32
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(hidden_channels*input_dim//32, latent_dim),
        )

    def reshape_input(self, x):
        '''Reshapes the input tensor x (n_batch, input_dim) to (n_batch, in_channels, freq_dim)'''
        n_batch = x.shape[0]
        out = x.clone().reshape(n_batch, self.in_channels, self.input_dim)
        for c in range(self.in_channels):
            out[:,c,:] = x[:,c*self.input_dim : (c+1)*self.input_dim]
        return out

    def reshape_output(self, x):
        '''Reshapes the input tensor x (n_batch, in_channels, freq_dim) to (n_batch, input_dim)'''
        return x.reshape(x.shape[0], -1) 

    def forward(self,x):
        x = self.reshape_input(x)
        x = self.layers(x)
        return x

class Generator(torch.nn.Module):
    '''Generator module'''
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
            torch.nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, padding=kernel_size-2),
        )

    def reshape_input(self, x):
        '''Reshapes the input tensor x (n_batch, input_dim) to (n_batch, in_channels, freq_dim)'''
        n_batch = x.shape[0]
        out = x.clone().reshape(n_batch, self.in_channels, self.input_dim)
        for c in range(self.in_channels):
            out[:,c,:] = x[:,c*self.input_dim : (c+1)*self.input_dim]
        return out

    def reshape_output(self, x):
        '''Reshapes the input tensor x (n_batch, in_channels, freq_dim) to (n_batch, input_dim)'''
        return x.reshape(x.shape[0], -1) 

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = self.reshape_output(x)
        return x

class ChannelPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(axis=-1)

class Discriminator(torch.nn.Module):
    ''' The discriminator returns the probability of the tuple (x,z) belonging to the true distribution.
    '''
    def __init__(self, latent_dim:int, input_dim:int, hidden_channels:int, in_channels:int, kernel_size:int):
        super().__init__()
        self.in_channels = in_channels
        self.input_dim = input_dim
        self.feat_x = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=kernel_size-2), #1024
            DownBlock(in_channels=hidden_channels, kernel_size=kernel_size), #512
            DownBlock(in_channels=hidden_channels, kernel_size=kernel_size), #256
            DownBlock(in_channels=hidden_channels, kernel_size=kernel_size), #128
            torch.nn.Conv1d(in_channels=hidden_channels, out_channels=128, kernel_size=kernel_size, padding=kernel_size-2), #1024
            ChannelPooling(), 
        ) # output size: (bs, 128)
        
        self.feat_z = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128,128)
        )

        self.fc = torch.nn.Linear(256,1)

    def reshape_input(self, x):
        '''Reshapes the input tensor x (n_batch, input_dim) to (n_batch, in_channels, freq_dim)'''
        n_batch = x.shape[0]
        out = x.clone().reshape(n_batch, self.in_channels, self.input_dim)
        for c in range(self.in_channels):
            out[:,c,:] = x[:,c*self.input_dim : (c+1)*self.input_dim]
        return out

    def reshape_output(self, x):
        '''Reshapes the input tensor x (n_batch, in_channels, freq_dim) to (n_batch, input_dim)'''
        return x.reshape(x.shape[0], -1) 

    def forward(self,x,z) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.reshape_input(x)
        feat_x = self.feat_x(x)
        feat_z = self.feat_z(z)
        feat = torch.concat((feat_x, feat_z),dim=1)
        logits = self.fc(feat)
        proba = torch.sigmoid(logits)
        return proba


class GAN(torch.nn.Module):
    def __init__(self, in_dim:int, latent_dim:int, hidden_dim:int, residual:bool, batch_norm:bool):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(
            in_dim = in_dim,
            latent_dim = latent_dim,
            hidden_dim = hidden_dim,
            residual = residual,
            batch_norm = batch_norm
        )

        self.generator = Decoder(
            out_dim = in_dim,
            latent_dim = latent_dim,
            hidden_dim = hidden_dim,
            residual = residual,
            batch_norm = batch_norm
        )

        self.discriminator = Discriminator(
            in_dim = in_dim,
            hidden_dim = 16,
            residual = residual,
            batch_norm = batch_norm
        )

    def sample_z(self, batch_size) -> torch.Tensor:
        z = torch.randn(batch_size, self.latent_dim)
        return z


    def __init__(self, in_dim:int, latent_dim:int, hidden_dim:int, residual:bool, batch_norm:bool):
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            DownBlock(in_dim=hidden_dim, residual=residual, batch_norm=batch_norm),
            DownBlock(in_dim=hidden_dim//2, residual=residual, batch_norm=batch_norm),
            DownBlock(in_dim=hidden_dim//4, residual=residual, batch_norm=batch_norm),
            DownBlock(in_dim=hidden_dim//8, residual=residual, batch_norm=batch_norm),
            torch.nn.Linear(hidden_dim//16, latent_dim),
        )
        
    def forward(self,x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.flatten(x, start_dim=1)
        x = x.to(device)
        return self.layers(x)