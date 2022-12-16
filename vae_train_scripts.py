from lib import *
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import torchvision
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import gc
import wandb
from models.vae import VAE, VAELoss, PLModel, VAECNN
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def train_normal(config=None):
    ''' Executes the training and evaluation procedure for the damage models. 
    Vibration data from each sensor is used to train a single CNN-based VAE, which is then used
    as a data compression tool. The average reconstruction error of each sensor is used a health indicator.
    Dataset splits:
        - data_train_norm: used for training the VAE
        - data_val_norm: used for modelling the reconstruction error
        - data_test_norm: used for evaluating the anomaly detector (negative examples)
        - data_dmg_all: used for evaluating the anomaly detector (positive examples)
    Models and results are logged with Wandb.
    '''
    with wandb.init(config=config):
        config = wandb.config

        hparams = {
            'batch_size': config.batch_size,
            'beta': config.beta,
            'epochs':config.epochs,
            'epoch_resets': config.epoch_resets,
            'latent_dim': config.latent_dim,
            'learning_rate': config.learning_rate,
            'model': config.model,
            'normalization': config.normalization,
            'num_resets': config.num_resets,
            'channels': config.channels,
        }

        path_data = "data/"

        data_all = load_preprocessed_psd(
            filename=path_data + "data_4_psd",
            channels=hparams['channels']
        )

        # splitting in normal/damage
        data_split = split_by_meta(
            data=data_all,
            path_to_meta="data/ds4_metadata.csv"
        )

        data_train_norm = data_split["normal_train"]
        data_val_norm = data_split["normal_val"]
        data_test_norm = data_split["normal_test"]

        data_dmg_1 = data_split["dmg_1"]
        data_dmg_2 = data_split["dmg_2"]

        data_dmg_all = merge_data((data_dmg_1, data_dmg_2))
        data_all = merge_data(
            (
                data_train_norm, data_val_norm, data_test_norm, data_dmg_1, data_dmg_2
            )
        )

        gc.collect()

        hparams["freq_dim"] = data_train_norm["features"].shape[-1]
        hparams["n_channels"] = data_train_norm["features"].shape[-2]

        ### dataset and dataloader
        if hparams['normalization'] == "min-max":
            transform = torchvision.transforms.Lambda(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
            normalization = None
        
        elif hparams["normalization"] == "z-score":
            mu_train = data_train_norm["features"].mean(axis=0)
            std_train = data_train_norm["features"].mean(axis=0)
            transform = torchvision.transforms.Lambda(
                lambda x: (x - mu_train) / std_train
            )
            normalization = {'mu':mu_train, 'std':std_train}
        
        elif hparams["normalization"] == "db-z-score":
            mu_train = 20*np.log10(data_train_norm["features"]).mean(axis=(0,1))
            std_train = 20*np.log10(data_train_norm["features"]).std(axis=(0,1))
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: 20*np.log10(x)),
                    torchvision.transforms.Lambda(lambda x: (x - mu_train) / std_train),
                ]
            )
            normalization = {'mu':mu_train, 'std':std_train}

        train_dataset_norm = Dataset(
            x=data_train_norm["features"],
            y=data_train_norm["y"],
            transform=transform,
            t=data_train_norm['tensions']
        )

        val_dataset_norm = Dataset(
            x=data_val_norm["features"],
            y=data_val_norm["y"],
            t=data_val_norm["tensions"],
            transform=transform
        )

        test_dataset_norm = Dataset(
            x=data_test_norm["features"],
            y=data_test_norm["y"],
            t=data_test_norm["tensions"],
            transform=transform
        )

        dmg_dataset = Dataset(
            x=data_dmg_all["features"],
            y=data_dmg_all["y"],
            transform=transform,
            t=data_dmg_all['tensions']
        )

        all_dataset = Dataset(
            x=data_all["features"],
            y=data_all["y"],
            t=data_all["tensions"],
            transform=transform
        )

        train_dataloader_norm = DataLoader(
            train_dataset_norm, batch_size=hparams['batch_size'], shuffle=True
        )

        train_dataloader_noshuffle_norm = DataLoader(
            train_dataset_norm, batch_size=hparams['batch_size'], shuffle=False
        )

        val_dataloader_norm = DataLoader(
            val_dataset_norm, batch_size=hparams['batch_size'], shuffle=False
        )

        test_dataloader_norm = DataLoader(
            test_dataset_norm, batch_size=hparams['batch_size'], shuffle=False
        )

        dmg_dataloader = DataLoader(
            dmg_dataset, batch_size=hparams['batch_size'], shuffle=False
        )

        all_dataloader = DataLoader(
            all_dataset, batch_size=hparams['batch_size'], shuffle=False
        )

        ###  training the normality model
        wandb_logger = WandbLogger(log_model=True)
        pl_model = PLModel(hparams=hparams, normalization=normalization)
        trainer = Trainer(
            max_epochs=hparams['epochs'],
            accelerator='gpu',
            devices=1,
            logger=wandb_logger,
            log_every_n_steps=len(train_dataloader_norm),
            )
        trainer.fit(pl_model, train_dataloader_norm, val_dataloader_norm)

        model = pl_model.model
        model.eval()       

        ### making predictions
        train_pred = predict(dataloader=train_dataloader_noshuffle_norm, model=model)
        val_pred = predict(dataloader=val_dataloader_norm, model=model)
        test_pred = predict(dataloader=test_dataloader_norm, model=model)
        dmg_pred = predict(dataloader=dmg_dataloader, model=model)
        all_pred = predict(dataloader=all_dataloader, model=model)

        ### threshold

        fig_hist_trainval, parameters, threshold = get_threshold_hist_ds4(
            train_pred=train_pred,
            val_pred=val_pred,
        )

        fig_hist_test = plot_hist_ds4(
            pred=test_pred,
            parameters=parameters,
            threshold=threshold
        )

        fig_hist_dmg = plot_hist_ds4(
            pred=dmg_pred,
            parameters=parameters,
            threshold=threshold
        )

        wandb.log(
            {
                "Train/Val Histograms": wandb.Image(fig_hist_trainval),
                "Test Histograms": wandb.Image(fig_hist_test),
                "Damage Histograms": wandb.Image(fig_hist_dmg),
            }
        )

        plt.close('all')
        
        ### evaluate on each dataset & sensor
        TN_val, FP_val = eval_anomaly(predictions=val_pred, mode='normal', threshold=threshold)
        TN_test, FP_test = eval_anomaly(predictions=test_pred, mode='normal', threshold=threshold)
        TP_dmg, FN_dmg = eval_anomaly(predictions=dmg_pred, mode='damage', threshold=threshold)
        
        TNR_val = {}
        TNR_test = {}
        TPR_dmg = {}
        for s in TN_val.keys():
            TNR_val[s] = TN_val[s] / (TN_val[s] + FP_val[s] + 0.001)
            TNR_test[s] = TN_test[s] / (TN_test[s] + FP_test[s] + 0.001)
            TPR_dmg[s] = TP_dmg[s] / (TP_dmg[s] + FN_dmg[s] + 0.001)
   
            wandb.log(
                {
                    f'TNR_val_s{4-int(s[1])}': TNR_val[s],
                    f'TNR_test_s{4-int(s[1])}': TNR_test[s],
                    f'TPR_dmg_s{4-int(s[1])}': TPR_dmg[s],
                    f'Threshold_s{4-int(s[1])}': threshold[s],
                }
            )

        ### plotting latent variables

        figs = plot_test(all_pred)

        #wandb_fig_x = wandb.Image(figs['fig_x'])
        wandb_fig_z = wandb.Image(figs['fig_z'])
        wandb_fig_z1z2 = wandb.Image(figs['fig_z1z2'])
        wandb_fig_z1z2_s = wandb.Image(figs['fig_z1z2_s'])
        #wandb_fig_xrec = wandb.Image(figs['fig_xrec'])
        
        wandb.log(
            {
                #"Input Data": wandb_fig_x,
                #"Reconstructed Data": wandb_fig_xrec,
                "Latent Variables": wandb_fig_z,
                "Latent Variables z1xz2": wandb_fig_z1z2,
                "Latent Variables z1xz2 per sensor": wandb_fig_z1z2_s,
            }
        )
        gc.collect()
        plt.close('all')

def train_damage(config=None):
    ''' Executes the training and evaluation procedure for the damage models. 
    Vibration data from each sensor is used to train a single CNN-based VAE, which is then used
    as a data compression tool. The latent variables are used as a condition indicator.
    Dataset splits:
        - data_dmg_1: used for training the VAE
        - data_dmg_2: used for evaluating the extracted latent variables
        - data_norm: used for evaluating the model under normal operation
    Models and results are logged with Wandb.
    '''
    with wandb.init(config=config):
        config = wandb.config

        hparams = {
            'batch_size': config.batch_size,
            'beta': config.beta,
            'epochs':config.epochs,
            'epoch_resets': config.epoch_resets,
            'latent_dim': config.latent_dim,
            'learning_rate': config.learning_rate,
            'model': config.model,
            'normalization': config.normalization,
            'num_resets': config.num_resets,
            'channels': config.channels,
            'hidden_channels': config.hidden_channels
        }

        path_data = "data/"

        data_all = load_preprocessed_psd(
            filename=path_data + "data_4_psd",
            channels=hparams['channels']
        )

        # splitting in normal/damage
        data_split = split_by_meta(
            data=data_all,
            path_to_meta="data/ds4_metadata.csv"
        )

        data_train_norm = data_split["normal_train"]
        data_val_norm = data_split["normal_val"]
        data_test_norm = data_split["normal_test"]

        data_dmg_1 = data_split["dmg_1"]
        data_dmg_2 = data_split["dmg_2"]

        data_norm = merge_data((data_train_norm, data_val_norm, data_test_norm))
        data_all = merge_data(
            (
                data_train_norm, data_val_norm, data_test_norm, data_dmg_1, data_dmg_2
            )
        )

        gc.collect()

        hparams["freq_dim"] = data_train_norm["features"].shape[-1]
        hparams["n_channels"] = data_train_norm["features"].shape[-2]

        ### dataset and dataloader
        if hparams['normalization'] == "min-max":
            transform = torchvision.transforms.Lambda(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
            normalization = None

        elif hparams["normalization"] == "z-score":
            mu_train = data_dmg_1["features"].mean(axis=(0,1))
            std_train = data_dmg_1["features"].std(axis=(0,1))
            transform = torchvision.transforms.Lambda(
                lambda x: (x - mu_train) / std_train
            )
            normalization = {'mu':mu_train, 'std':std_train}

        elif hparams["normalization"] == "db-z-score":
            mu_train = 20*np.log10(data_dmg_1["features"]).mean(axis=(0,1))
            std_train = 20*np.log10(data_dmg_1["features"]).std(axis=(0,1))
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: 20*np.log10(x)),
                    torchvision.transforms.Lambda(lambda x: (x - mu_train) / std_train),
                ]
            )
            normalization = {'mu':mu_train, 'std':std_train}

        norm_dataset = Dataset(
            x=data_norm["features"],
            y=data_norm["y"],
            transform=transform,
            t=data_norm['tensions']
        )

        dmg_1_dataset = Dataset(
            x=data_dmg_1["features"],
            y=data_dmg_1["y"],
            transform=transform,
            t=data_dmg_1['tensions']
        )

        dmg_2_dataset = Dataset(
            x=data_dmg_2["features"],
            y=data_dmg_2["y"],
            transform=transform,
            t=data_dmg_2['tensions']
        )

        all_dataset = Dataset(
            x=data_all["features"],
            y=data_all["y"],
            t=data_all["tensions"],
            transform=transform
        )

        train_dataloader_norm = DataLoader(
            dmg_1_dataset, batch_size=hparams['batch_size'], shuffle=True
        )

        train_dataloader_noshuffle_norm = DataLoader(
            dmg_1_dataset, batch_size=hparams['batch_size'], shuffle=False
        )

        val_dataloader_norm = DataLoader(
            dmg_2_dataset, batch_size=hparams['batch_size'], shuffle=False
        )

        all_dataloader = DataLoader(
            all_dataset, batch_size=hparams['batch_size'], shuffle=False
        )

        ###  training the normality model
        wandb_logger = WandbLogger(log_model=True)
        pl_model = PLModel(hparams=hparams, normalization=normalization)
        trainer = Trainer(
            max_epochs=hparams['epochs'],
            accelerator='gpu',
            devices=1,
            logger=wandb_logger,
            log_every_n_steps=len(train_dataloader_norm),
            )
        trainer.fit(pl_model, train_dataloader_norm, val_dataloader_norm)

        model = pl_model.model
        model.eval()       

        ### making predictions
        train_pred = predict(dataloader=train_dataloader_noshuffle_norm, model=model)
        val_pred = predict(dataloader=val_dataloader_norm, model=model)
        all_pred = predict(dataloader=all_dataloader, model=model)
       
        ### plotting latent variables

        for pred, mode in zip([train_pred, val_pred, all_pred], ["dmg_1", "dmg_2", "all"]):
            figs = plot_test(pred)
            wandb_fig_z = wandb.Image(figs['fig_z'])
            wandb_fig_z1z2 = wandb.Image(figs['fig_z1z2'])
            wandb_fig_z1z2_s = wandb.Image(figs['fig_z1z2_s'])
            wandb_fig_x = wandb.Image(figs['fig_x'])
            wandb_fig_xrec = wandb.Image(figs['fig_xrec'])
            wandb.log(
                {
                    "Input Data": wandb_fig_x,
                    "Reconstructed Data": wandb_fig_xrec,
                    f"Latent Variables {mode}": wandb_fig_z,
                    f"Latent Variables z1xz2 {mode}": wandb_fig_z1z2,
                    f"Latent Variables z1xz2 per sensor {mode}": wandb_fig_z1z2_s,
                }
            )
            gc.collect()
            plt.close('all')

def train_ds4_dmg(config=None):
    with wandb.init(config=config):
        config = wandb.config

        hparams = {
            'batch_size': config.batch_size,
            'beta': config.beta,
            'epochs':config.epochs,
            'epoch_resets': config.epoch_resets,
            'latent_dim': config.latent_dim,
            'learning_rate': config.learning_rate,
            'level': config.level,
            'model': config.model,
            'normalization': config.normalization,
            'num_resets': config.num_resets,
            'wavelet': config.wavelet,
        }

        path_data = "data/"

        data_all = load_preprocessed(
            filename=path_data + "data_4",
            level=hparams['level'],
            wavelet=hparams['wavelet']
        )

        data_split = split_by_meta(
            data=data_all,
            path_to_meta="data/ds4_metadata.csv"
        )

        data_dmg_1 = data_split["dmg_1"]
        data_dmg_2 = data_split["dmg_2"]

        gc.collect()

        hparams['input_dim'] = data_dmg_1['features'].shape[1]

        ### dataset and dataloader
        if hparams['normalization'] == "min-max":
            transform = torchvision.transforms.Lambda(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
            normalization = None
        elif hparams['normalization'] == "z-score":
            mu_train = data_dmg_1["features"].mean(axis=0)
            std_train = data_dmg_1["features"].mean(axis=0)
            transform = torchvision.transforms.Lambda(
                lambda x: (x - mu_train) / std_train
            )
            normalization = {'mu':mu_train, 'std':std_train}
        elif hparams['normalization'] == "sum":
            transform = torchvision.transforms.Lambda(
                lambda x: x/x.sum()
            )
            normalization = None

        dmg_1_dataset = Dataset(
            x=data_dmg_1["features"],
            y=data_dmg_1["y"],
            transform=transform,
            t=data_dmg_1['tensions']
        )

        dmg_2_dataset = Dataset(
            x=data_dmg_2["features"],
            y=data_dmg_2["y"],
            t=data_dmg_2["tensions"],
            transform=transform
        )


        dmg_1_dataloader = DataLoader(
            dmg_1_dataset, batch_size=hparams['batch_size'], shuffle=True
        )

        dmg_1_dataloader_no_shuffle = DataLoader(
            dmg_1_dataset, batch_size=hparams['batch_size'], shuffle=False
        )

        dmg_2_dataloader = DataLoader(
            dmg_2_dataset, batch_size=hparams['batch_size'], shuffle=False
        )

        ###  training the normality model
        wandb_logger = WandbLogger(log_model=True)
        pl_model = PLModel(hparams=hparams, normalization=normalization)
        trainer = Trainer(
            max_epochs=hparams['epochs'],
            accelerator='gpu',
            devices=1,
            logger=wandb_logger,
            log_every_n_steps=len(dmg_1_dataloader),
            )
        trainer.fit(pl_model, dmg_1_dataloader, dmg_1_dataloader)

        model = pl_model.model
        model.eval()       

        ### making predictions
        dmg_1_pred = predict(dataloader=dmg_1_dataloader, model=model)
        dmg_2_pred = predict(dataloader=dmg_2_dataloader, model=model)

        figs_dmg_1, pca, n_90 = plot_test_dmg(dmg_1_pred, None)
        figs_dmg_2, _, _ = plot_test_dmg(dmg_2_pred, pca)

        wandb_fig_x_dmg_1 = wandb.Image(figs_dmg_1['fig_x'])
        wandb_fig_z_dmg_1 = wandb.Image(figs_dmg_1['fig_z'])
        wandb_fig_xrec_dmg_1 = wandb.Image(figs_dmg_1['fig_xrec'])
        wandb_fig_pca_dmg_1 = wandb.Image(figs_dmg_1['fig_pca'])

        wandb_fig_x_dmg_2 = wandb.Image(figs_dmg_2['fig_x'])
        wandb_fig_z_dmg_2 = wandb.Image(figs_dmg_2['fig_z'])
        wandb_fig_xrec_dmg_2 = wandb.Image(figs_dmg_2['fig_xrec'])
        wandb_fig_pca_dmg_2 = wandb.Image(figs_dmg_2['fig_pca'])
        
        wandb.log(
            {
                "Input Data D1": wandb_fig_x_dmg_1,
                "Reconstructed Data D1": wandb_fig_xrec_dmg_1,
                "Latent Variables D1": wandb_fig_z_dmg_1,
                "mu1 x mu2 D1": wandb_fig_pca_dmg_1,
                "n_90 D1": n_90
            }
        )

        wandb.log(
            {
                "Input Data D2": wandb_fig_x_dmg_2,
                #"Reconstructed Data D2": wandb_fig_xrec_dmg_2,
                "Latent Variables D2": wandb_fig_z_dmg_2,
                "mu1 x mu2 D2": wandb_fig_pca_dmg_2,
                "Cumulative Sum D2": wandb_fig_cumsum_dmg_2,
            }
        )

        gc.collect()
        plt.close('all')


#