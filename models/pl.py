import pytorch_lightning as pl
from models.vae import VAE, VAELoss, VAERec
import torch

#### OLD

class PLModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = VAE(
			input_dim = hparams['input_dim'],
	    	latent_dim = hparams['latent_dim'],
		)
  
        self.criterion = VAELoss(
            beta=hparams['beta'],
            epochs_reset=hparams['epochs_reset'],
            num_resets=hparams['num_resets']
        )

    def training_step(self, train_batch, batch_idx):
        x,_,_ = train_batch
        mu, logvar, x_rec = self.model(x)
        nll, kl = self.criterion(x_rec, mu, logvar, x, 'train')
        return {'rec_loss': nll, 'kl_loss': kl/self.criterion.current_beta, 'loss': nll + kl}

    def training_epoch_end(self, outputs):
        avg_rec_loss = torch.stack([output['rec_loss'] for output in outputs]).mean()
        avg_kl_loss = torch.stack([output['kl_loss'] for output in outputs]).mean()
        avg_vae_loss = torch.stack([output['loss'] for output in outputs]).mean()

        self.log('train_rec_loss', avg_rec_loss.item())
        self.log('train_kl_loss', avg_kl_loss.item())
        self.log('train_vae_loss', avg_vae_loss.item())
        self.log('beta', self.criterion.current_beta)
        self.criterion.update_beta()
        return

    def validation_step(self, val_batch, batch_idx):
        x,_,_ = val_batch
        mu, logvar, x_rec = self.model(x)
        nll, kl = self.criterion(x_rec, mu ,logvar, x, 'val')
        return {'rec_loss': nll, 'kl_loss': kl/self.criterion.current_beta, 'loss': nll + kl}

    def validation_epoch_end(self, outputs):
        avg_rec_loss = torch.stack([output['rec_loss'] for output in outputs]).mean()
        avg_kl_loss = torch.stack([output['kl_loss'] for output in outputs]).mean()
        avg_vae_loss = torch.stack([output['loss'] for output in outputs]).mean()

        self.log('val_rec_loss', avg_rec_loss.item())
        self.log('val_kl_loss', avg_kl_loss.item())
        self.log('val_vae_loss', avg_vae_loss.item())
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['learning_rate'])
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1)
        return [optimizer], [scheduler]


class PLModelRec(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.latent_dim = hparams['latent_dim']
        self.save_hyperparameters(hparams)
        self.model = VAERec(
			input_dim = hparams['input_dim'],
	    	latent_dim = hparams['latent_dim'],
		)
  
        self.criterion = VAELoss(
            beta=hparams['beta'],
            epochs_reset=hparams['epochs_reset'],
            num_resets=hparams['num_resets']
        )

    def training_step(self, train_batch, batch_idx):
        x,_,_ = train_batch
        mu, logvar, x_rec = self.model(x)
        nll, kl = self.criterion(x_rec, mu, logvar, x, 'train')
        return {'rec_loss': nll, 'kl_loss': kl/self.criterion.current_beta, 'loss': nll + kl}

    def training_epoch_end(self, outputs):
        self.model.set_z0()
        avg_rec_loss = torch.stack([output['rec_loss'] for output in outputs]).mean()
        avg_kl_loss = torch.stack([output['kl_loss'] for output in outputs]).mean()
        avg_vae_loss = torch.stack([output['loss'] for output in outputs]).mean()

        self.log('train_rec_loss', avg_rec_loss.item())
        self.log('train_kl_loss', avg_kl_loss.item())
        self.log('train_vae_loss', avg_vae_loss.item())
        self.log('beta', self.criterion.current_beta)
        self.criterion.update_beta()
        return

    def validation_step(self, val_batch, batch_idx):
        x,_,_ = val_batch
        mu, logvar, x_rec = self.model(x)
        nll, kl = self.criterion(x_rec, mu ,logvar, x, 'val')
        return {'rec_loss': nll, 'kl_loss': kl/self.criterion.current_beta, 'loss': nll + kl}

    def validation_epoch_end(self, outputs):
        self.model.set_z0()
        avg_rec_loss = torch.stack([output['rec_loss'] for output in outputs]).mean()
        avg_kl_loss = torch.stack([output['kl_loss'] for output in outputs]).mean()
        avg_vae_loss = torch.stack([output['loss'] for output in outputs]).mean()

        self.log('val_rec_loss', avg_rec_loss.item())
        self.log('val_kl_loss', avg_kl_loss.item())
        self.log('val_vae_loss', avg_vae_loss.item())
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['learning_rate'])
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1)
        return [optimizer], [scheduler]


#