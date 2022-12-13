import wandb
import yaml
from vae_train_scripts import *

with open('vae-normal-sweep-config.yaml') as conf:
    sweep_config = yaml.safe_load(conf)

wandb.login()
sweep_id = wandb.sweep(sweep_config, project='VAE_Normal_ds4')
wandb.agent(sweep_id, function=train_normal)
#wandb.agent('y5aht6o9', function=train_ds4_normal, project='VAE_Normal_ds4')




