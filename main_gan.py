import wandb
import yaml
from gan_train_scripts import *

with open('gan-sweep-config.yaml') as conf:
    sweep_config = yaml.safe_load(conf)

wandb.login()
sweep_id = wandb.sweep(sweep_config, project='GAN_Normal_ds4')
wandb.agent(sweep_id, function=train_ds4)
# wandb.agent('cf7gnenw', function=train_ds4, project='GAN_Normal_ds4')




