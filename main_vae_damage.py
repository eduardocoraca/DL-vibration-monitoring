import wandb
import yaml
from vae_train_scripts import *

with open('vae-damage-sweep-config.yaml') as conf:
    sweep_config = yaml.safe_load(conf)

wandb.login()
sweep_id = wandb.sweep(sweep_config, project='VAE_Damage_ds4')
wandb.agent(sweep_id, function=train_damage)
#wandb.agent('y5aht6o9', function=train_ds4_damage, project='VAE_Damage_ds4')




