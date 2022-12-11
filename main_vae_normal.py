import wandb
import yaml
from vae_train_scripts import *

with open('vae-normal-sweep-config.yaml') as conf:
    sweep_config = yaml.safe_load(conf)

wandb.login()
sweep_id = wandb.sweep(sweep_config, project='VAE_Normal_ds4')
wandb.agent(sweep_id, function=train_ds4_normal)
# wandb.agent('cf7gnenw', function=train_ds4_dmg, project='VAE_Damage_ds4')




