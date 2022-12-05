from lib import *
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import torchvision
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import gc
import wandb
from models.bigan import *
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def train_ds4(config=None):
    with wandb.init(config=config):
        config = wandb.config

        hparams = {
            'batch_size': config.batch_size,
            'epochs':config.epochs,
            'latent_dim': config.latent_dim,
            'learning_rate': config.learning_rate,
            'level': config.level,
            'normalization': config.normalization,
            'wavelet': config.wavelet,
            'weight_decay': config.weight_decay
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

        data_all = merge_data(
            (
                data_train_norm, data_val_norm, data_test_norm, data_dmg_1, data_dmg_2
            )
        )

        gc.collect()

        hparams['input_dim'] = data_train_norm['features'].shape[1]

        ### dataset and dataloader
        if hparams['normalization'] == "min-max":
            transform = torchvision.transforms.Lambda(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
            normalization = None
        elif hparams['normalization'] == "z-score":
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
            tRunransform=transform
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
        pl_model = GAN_PL(hparams=hparams, normalization=normalization)
        trainer = Trainer(
            max_epochs=hparams['epochs'],
            accelerator='gpu',
            devices=1,
            logger=wandb_logger,
            log_every_n_steps=len(train_dataloader_norm),
        )
        trainer.fit(pl_model, train_dataloader_norm, val_dataloader_norm)

        pl_model.eval()       

        ### making predictions
        train_pred = predict_gan(dataloader=train_dataloader_noshuffle_norm, pl_model=pl_model)
        dmg_1_pred = predict_gan(dataloader=dmg_1_dataloader, pl_model=pl_model)
        val_pred = predict_gan(dataloader=val_dataloader_norm, pl_model=pl_model)
        test_pred = predict_gan(dataloader=test_dataloader_norm, pl_model=pl_model)
        dmg_2_pred = predict_gan(dataloader=dmg_2_dataloader, pl_model=pl_model)
        all_pred = predict_gan(dataloader=all_dataloader, pl_model=pl_model)

        pca = PCA(hparams["latent_dim"])
        pca.fit(train_pred["z"]-train_pred["z"].mean(axis=0))
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_90 = np.sum(cumsum<0.9)
        pca = PCA(2)
        pca.fit(train_pred["z"]-train_pred["z"].mean(axis=0))
        
        fig_all = plot_gan(all_pred, pca)

        wandb.log(
            {
                "n_90": n_90,
                "z_pca": wandb.Image(fig_all["fig_zpca"]),
                "proba": wandb.Image(fig_all["fig_proba"]),
                "x": wandb.Image(fig_all["fig_x"]),
            }
        )
        plt.close("all")
        ### threshold
        # threshold = get_threshold(predictions=(train_pred, val_pred), model=model)
        # fig_hist_trainval, params, threshold = get_threshold_hist_ds4(
        #     train_pred=train_pred,
        #     val_pred=val_pred,
        # )

        # fig_hist_test = plot_hist_ds4(
        #     pred=test_pred,
        #     params=params,
        #     threshold=threshold
        # )

        # fig_hist_dmg1 = plot_hist_ds4(
        #     pred=dmg_1_pred,
        #     params=params,
        #     threshold=threshold
        # )

        # fig_hist_dmg2 = plot_hist_ds4(
        #     pred=dmg_2_pred,
        #     params=params,
        #     threshold=threshold
        # )

        # wandb.log(
        #     {
        #         "Train/Val Histograms": wandb.Image(fig_hist_trainval),
        #         "Test Histograms": wandb.Image(fig_hist_test),
        #         "Damage 1 Histograms": wandb.Image(fig_hist_dmg1),
        #         "Damage 2 Histograms": wandb.Image(fig_hist_dmg2),
        #     }
        # )

        # ### evaluate on each dataset
        # TN_val, FP_val = eval_anomaly(predictions=val_pred, mode='normal', threshold=threshold)
        # TN_test, FP_test = eval_anomaly(predictions=test_pred, mode='normal', threshold=threshold)
        # TP_dmg1, FN_dmg1 = eval_anomaly(predictions=dmg_1_pred, mode='damage', threshold=threshold)
        # TP_dmg2, FN_dmg2 = eval_anomaly(predictions=dmg_2_pred, mode='damage', threshold=threshold)
        
        # TNR_val = TN_val / (TN_val + FP_val + 0.001)
        # TNR_test = TN_test / (TN_test + FP_test + 0.001)
        # TPR_dmg1 = TP_dmg1 / (TP_dmg1 + FN_dmg1 + 0.001)
        # TPR_dmg2 = TP_dmg2 / (TP_dmg2 + FN_dmg2 + 0.001)
   
        # wandb.log(
        #     {
        #         'TNR_val': TNR_val,
        #         'TNR_test': TNR_test,
        #         'TPR_dmg1': TPR_dmg1,
        #         'TPR_dmg2': TPR_dmg2,
        #         'Threshold': threshold,
        #     }
        # )

        # ### plotting health index
        # fig_train = plot_train_val(train_pred, val_pred, threshold)
        # wandb_fig_train = wandb.Image(fig_train)
        # wandb.log({"Train-Val Health Index": wandb_fig_train})
        # del fig_train

        # figs = plot_test(all_pred, threshold)

        # wandb_fig_h = wandb.Image(figs['fig_h'])
        # wandb_fig_x = wandb.Image(figs['fig_x'])
        # wandb_fig_z = wandb.Image(figs['fig_z'])
        # wandb_fig_ht = wandb.Image(figs['fig_ht'])
        # wandb_fig_xrec = wandb.Image(figs['fig_xrec'])
        # wandb_fig_box = wandb.Image(figs['fig_box'])
        
        # wandb.log(
        #     {
        #         "Health Index": wandb_fig_h,
        #         "Input Data": wandb_fig_x,
        #         "Reconstructed Data": wandb_fig_xrec,
        #         "Latent Variables": wandb_fig_z,
        #         "Health Index x Average Tension": wandb_fig_ht,
        #         "Health Index Whole Dataset": wandb_fig_box,
        #     }
        # )
        # gc.collect()
        # plt.close('all')

