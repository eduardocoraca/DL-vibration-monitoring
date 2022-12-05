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

def train(config=None):
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

        #data_test = load_to_memory(path=path_data + "data_3_raw.d", level=hparams['level'], wavelet=hparams['wavelet'])
        data_test = load_preprocessed(filename=path_data + "data_3", level=hparams['level'], wavelet=hparams['wavelet'])
        data_test_norm = split_by_label(data=data_test, label=[0])
        data_test_dmg = split_by_label(data=data_test, not_label=[0])
        gc.collect()

        #data_train = load_to_memory(path=path_data + "data_1_raw.d", level=hparams['level'], wavelet=hparams['wavelet'])
        data_train = load_preprocessed(filename=path_data + "data_1", level=hparams['level'], wavelet=hparams['wavelet'])
        data_train_norm = split_by_label(data=data_train, label=[0])
        data_train_dmg = split_by_label(data=data_train, not_label=[0])
        gc.collect()

        #data_val = load_to_memory(path=path_data + "data_2_raw.d", level=hparams['level'], wavelet=hparams['wavelet'])
        data_val = load_preprocessed(filename=path_data + "data_2", level=hparams['level'], wavelet=hparams['wavelet'])
        data_val_norm = split_by_label(data=data_val, label=[0])
        gc.collect()

        hparams['input_dim'] = data_train['features'].shape[1]

        ### dataset and dataloader
        if hparams['normalization'] == "min-max":
            transform = torchvision.transforms.Lambda(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
            normalization = None
        elif hparams['normalization'] == "z-score":
            mu_train = data_train["features"].mean(axis=0)
            std_train = data_train["features"].mean(axis=0)
            transform = torchvision.transforms.Lambda(
                lambda x: (x - mu_train) / std_train
            )
            normalization = {'mu':mu_train, 'std':std_train}
        elif hparams['normalization'] == "sum":
            transform = torchvision.transforms.Lambda(
                lambda x: x/x.sum()
            )
            normalization = None

        train_dataset_norm = Dataset(
            x=data_train_norm["features"],
            y=data_train_norm["y"],
            transform=transform,
            t=data_train_norm['tensions']
        )

        train_dataset_dmg = Dataset(
            x=data_train_dmg["features"],
            y=data_train_dmg["y"],
            transform=transform,
            t=data_train_dmg['tensions']
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

        test_dataset_dmg = Dataset(
            x=data_test_dmg["features"],
            y=data_test_dmg["y"],
            t=data_test_dmg["tensions"],
            transform=transform
        )

        test_dataset_all = Dataset(
            x=data_test["features"],
            y=data_test["y"],
            t=data_test["tensions"],
            transform=transform
        )

        train_dataloader_norm = DataLoader(
            train_dataset_norm, batch_size=hparams['batch_size'], shuffle=True
        )
        train_dataloader_dmg = DataLoader(
            train_dataset_dmg, batch_size=hparams['batch_size'], shuffle=False
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

        test_dataloader_dmg = DataLoader(
            test_dataset_dmg, batch_size=hparams['batch_size'], shuffle=False
        )

        test_dataloader_all = DataLoader(
            test_dataset_all, batch_size=hparams['batch_size'], shuffle=False
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
        train_pred_dmg = predict(dataloader=train_dataloader_dmg, model=model)
        val_pred = predict(dataloader=val_dataloader_norm, model=model)
        test_pred_norm = predict(dataloader=test_dataloader_norm, model=model)
        test_pred_dmg = predict(dataloader=test_dataloader_dmg, model=model)
        test_pred_all = predict(dataloader=test_dataloader_all, model=model)

        ### threshold
        # threshold = get_threshold(predictions=(train_pred, val_pred), model=model)
        fig_hist_train, params, threshold = get_threshold_hist(
            train_pred=train_pred,
            val_pred=val_pred,
            dmg_pred=train_pred_dmg,
        )
        
        fig_hist_test = plot_hist(
            normal_pred=test_pred_norm,
            dmg_pred=test_pred_dmg,
            params=params,
            threshold=threshold
        )

        wandb.log(
            {
                "Train Histograms": wandb.Image(fig_hist_train),
                "Test Histograms": wandb.Image(fig_hist_test),
            }
        )

        ### evaluate on train dataloader
        TN_val, FP_val = eval_anomaly(predictions=val_pred, mode='normal', threshold=threshold)
        TP_val, FN_val = eval_anomaly(predictions=train_pred_dmg, mode='damage', threshold=threshold)
        recall_val = TP_val / (TP_val + FN_val)
        precision_val = TP_val / (TP_val + FP_val)
        wandb.log(
            {
                'TN_val': TN_val,
                'TP_val': TP_val,
                'FP_val': FP_val,
                'FN_val': FN_val,
                'Recall_val': recall_val,
                'Precision_val': precision_val,
                'F1_val': 2*precision_val*recall_val / (precision_val+recall_val),
                'Threshold': threshold,
            }
        )

        ### evaluate on test dataloader
        TN_test, FP_test = eval_anomaly(predictions=test_pred_norm, mode='normal', threshold=threshold)
        TP_test, FN_test = eval_anomaly(predictions=test_pred_dmg, mode='damage', threshold=threshold)
        recall_test = TP_test / (TP_test + FN_test)
        precision_test = TP_test / (TP_test + FP_test)
        wandb.log(
            {
                'TN_test': TN_test,
                'TP_test': TP_test,
                'FP_test': FP_test,
                'FN_test': FN_test,
                'Recall_test': recall_test,
                'Precision_test': precision_test,
                'F1_test': 2*precision_test*recall_test / (precision_test+recall_test),
                'Threshold': threshold,
            }
        )

        ### plotting health index
        fig_train = plot_train_val(train_pred, val_pred, threshold)
        wandb_fig_train = wandb.Image(fig_train)
        wandb.log({"Train-Val Health Index": wandb_fig_train})
        del fig_train

        fig_h, fig_x, fig_z = plot_test(test_pred_all, threshold)
        wandb_fig_h = wandb.Image(fig_h)
        wandb_fig_x = wandb.Image(fig_x)
        wandb_fig_z = wandb.Image(fig_z)
        wandb.log(
            {
                "Test Health Index": wandb_fig_h,
                "Test Input Data": wandb_fig_x,
                "Test Latent Variables": wandb_fig_z,
            }
        )
        gc.collect()
        plt.close('all')


def train_ds4(config=None):
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

        #data_test = load_to_memory(path=path_data + "data_3_raw.d", level=hparams['level'], wavelet=hparams['wavelet'])
        data_all = load_preprocessed(
            filename=path_data + "data_4",
            level=hparams['level'],
            wavelet=hparams['wavelet']
        )

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

        hparams["input_dim"] = data_train_norm["features"].shape[1]

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
        elif hparams['normalization'] == "sum":
            transform = torchvision.transforms.Lambda(
                lambda x: x/x.sum()
            )
            normalization = None

        train_dataset_norm = Dataset(
            x=data_train_norm["features"],
            y=data_train_norm["y"],
            transform=transform,
            t=data_train_norm['tensions']
        )

        dmg_1_dataset = Dataset(
            x=data_dmg_1["features"],
            y=data_dmg_1["y"],
            transform=transform,
            t=data_dmg_1['tensions']
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

        dmg_2_dataset = Dataset(
            x=data_dmg_2["features"],
            y=data_dmg_2["y"],
            t=data_dmg_2["tensions"],
            transform=transform
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

        dmg_1_dataloader = DataLoader(
            dmg_1_dataset, batch_size=hparams['batch_size'], shuffle=False
        )

        dmg_2_dataloader = DataLoader(
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
        dmg_1_pred = predict(dataloader=dmg_1_dataloader, model=model)
        val_pred = predict(dataloader=val_dataloader_norm, model=model)
        test_pred = predict(dataloader=test_dataloader_norm, model=model)
        dmg_2_pred = predict(dataloader=dmg_2_dataloader, model=model)
        all_pred = predict(dataloader=all_dataloader, model=model)

        ### threshold
        # threshold = get_threshold(predictions=(train_pred, val_pred), model=model)
        fig_hist_trainval, params, threshold = get_threshold_hist_ds4(
            train_pred=train_pred,
            val_pred=val_pred,
        )

        fig_hist_test = plot_hist_ds4(
            pred=test_pred,
            params=params,
            threshold=threshold
        )

        fig_hist_dmg1 = plot_hist_ds4(
            pred=dmg_1_pred,
            params=params,
            threshold=threshold
        )

        fig_hist_dmg2 = plot_hist_ds4(
            pred=dmg_2_pred,
            params=params,
            threshold=threshold
        )

        wandb.log(
            {
                "Train/Val Histograms": wandb.Image(fig_hist_trainval),
                "Test Histograms": wandb.Image(fig_hist_test),
                "Damage 1 Histograms": wandb.Image(fig_hist_dmg1),
                "Damage 2 Histograms": wandb.Image(fig_hist_dmg2),
            }
        )

        ### evaluate on each dataset
        TN_val, FP_val = eval_anomaly(predictions=val_pred, mode='normal', threshold=threshold)
        TN_test, FP_test = eval_anomaly(predictions=test_pred, mode='normal', threshold=threshold)
        TP_dmg1, FN_dmg1 = eval_anomaly(predictions=dmg_1_pred, mode='damage', threshold=threshold)
        TP_dmg2, FN_dmg2 = eval_anomaly(predictions=dmg_2_pred, mode='damage', threshold=threshold)
        
        TNR_val = TN_val / (TN_val + FP_val + 0.001)
        TNR_test = TN_test / (TN_test + FP_test + 0.001)
        TPR_dmg1 = TP_dmg1 / (TP_dmg1 + FN_dmg1 + 0.001)
        TPR_dmg2 = TP_dmg2 / (TP_dmg2 + FN_dmg2 + 0.001)
   
        wandb.log(
            {
                'TNR_val': TNR_val,
                'TNR_test': TNR_test,
                'TPR_dmg1': TPR_dmg1,
                'TPR_dmg2': TPR_dmg2,
                'Threshold': threshold,
            }
        )

        ### plotting health index
        fig_train = plot_train_val(train_pred, val_pred, threshold)
        wandb_fig_train = wandb.Image(fig_train)
        wandb.log({"Train-Val Health Index": wandb_fig_train})
        del fig_train

        figs = plot_test(all_pred, threshold)

        wandb_fig_h = wandb.Image(figs['fig_h'])
        wandb_fig_x = wandb.Image(figs['fig_x'])
        wandb_fig_z = wandb.Image(figs['fig_z'])
        wandb_fig_ht = wandb.Image(figs['fig_ht'])
        wandb_fig_xrec = wandb.Image(figs['fig_xrec'])
        wandb_fig_box = wandb.Image(figs['fig_box'])
        
        wandb.log(
            {
                "Health Index": wandb_fig_h,
                "Input Data": wandb_fig_x,
                "Reconstructed Data": wandb_fig_xrec,
                "Latent Variables": wandb_fig_z,
                "Health Index x Average Tension": wandb_fig_ht,
                "Health Index Whole Dataset": wandb_fig_box,
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
        wandb_fig_cumsum_dmg_1 = wandb.Image(figs_dmg_1['fig_cumsum'])

        wandb_fig_x_dmg_2 = wandb.Image(figs_dmg_2['fig_x'])
        wandb_fig_z_dmg_2 = wandb.Image(figs_dmg_2['fig_z'])
        wandb_fig_xrec_dmg_2 = wandb.Image(figs_dmg_2['fig_xrec'])
        wandb_fig_pca_dmg_2 = wandb.Image(figs_dmg_2['fig_pca'])
        wandb_fig_cumsum_dmg_2 = wandb.Image(figs_dmg_2['fig_cumsum'])
        
        wandb.log(
            {
                "Input Data D1": wandb_fig_x_dmg_1,
                "Reconstructed Data D1": wandb_fig_xrec_dmg_1,
                "Latent Variables D1": wandb_fig_z_dmg_1,
                "mu1 x mu2 D1": wandb_fig_pca_dmg_1,
                "Cumulative Sum D1": wandb_fig_cumsum_dmg_1,
                "n_90 D1": n_90
            }
        )

        wandb.log(
            {
                "Input Data D2": wandb_fig_x_dmg_2,
                "Reconstructed Data D2": wandb_fig_xrec_dmg_2,
                "Latent Variables D2": wandb_fig_z_dmg_2,
                "mu1 x mu2 D2": wandb_fig_pca_dmg_2,
                "Cumulative Sum D2": wandb_fig_cumsum_dmg_2,
            }
        )

        gc.collect()
        plt.close('all')


#