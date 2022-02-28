import os
import sys
import torch
import numpy as np

from model.network import *
from utility.train import *
from utility.evaluate import *
from utility.dataloader import *
from utility.loss_fn import *
from utility.distributed import *
import torch.multiprocessing as mp

current_dir = os.getcwd()
sys.path.append(current_dir)

# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda init + empty cache
torch.cuda.init()
torch.cuda.empty_cache()

if __name__ == "__main__":

    # distributed learning setting
    world_size = torch.cuda.device_count()
    batch_size = 128
    valid_ratio = 0.2
    num_epochs = 16 + 32 + 64 + 128
    lr = 1e-3
    save_path = './weights/hgnn_distributed.pt'

    # model setting
    num_classes = 10
    num_features = 1
    num_heads = 6
    hidden = 256
    p = 0.5
    alpha = 0.01
    max_grad_norm = 1.0

    model = HGNN(num_classes = num_classes, num_features=num_features, num_heads = num_heads, hidden = hidden, p = p, alpha = alpha)

    # wandb setting
    wandb_setting = {
        "project" : "MNIST_superpixel",
        "entity":"diya_challenger",
        "name": "HGNN-distributed-learning-exp001"
    }
    
    # wandb config 
    wandb_config = {
        "learning_rate": lr,
        "epochs": num_epochs,
        "num_heads": num_heads,
        "batch_size": batch_size,
        "hidden": hidden,
        "dropout": p,
        "max_grad_norm": max_grad_norm,
    }

    args = (model, world_size, batch_size, valid_ratio, num_epochs, lr, max_grad_norm, save_path, wandb_setting, wandb_config)
    mp.spawn(run, args=args, nprocs=world_size, join=True)