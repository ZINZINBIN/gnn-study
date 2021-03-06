import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
import argparse

from model.network import *
from utility.train import *
from utility.evaluate import *
from utility.dataloader import *
from utility.loss_fn import *

current_dir = os.getcwd()
sys.path.append(current_dir)

parser = argparse.ArgumentParser(description="Dynamic Reduction Network for MNIST superpixel classification")
parser.add_argument("--batch_size", type = int, default = 512)
parser.add_argument("--num_epoch", type = int, default = 128)
parser.add_argument("--num_features", type = int, default = 5)
parser.add_argument("--k", type = int, default = 8)
parser.add_argument("--num_classes", type = int, default = 10)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--valid_ratio", type = float, default = 0.2)
parser.add_argument("--max_grad_norm", type = float, default = 1.0)
parser.add_argument("--embedd_size", type = int, default = 64)
parser.add_argument("--exp_num", type = int, default = 0)
parser.add_argument("--weight_save_dir", type = str, default = "./weights/dynamic_best_exp005.pt")
parser.add_argument("--wandb_save_name", type = str, default = 'dynamic-reduction-network-exp005')
parser.add_argument("--test_save_dir", type = str, default = "./result/test_summary_dynamic_reduction_network_exp005.txt")
parser.add_argument("--gpu_num", type = int, default = 1)
parser.add_argument("--aggr", type = str, default = 'add')
args = vars(parser.parse_args())

# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

if(torch.cuda.device_count() >= 1):
    device = "cuda:" +str(args["gpu_num"])
else:
    device = 'cpu'

# 논문에 따르면 hidden_dim : 20, k = 4에서 400 에포크로 학습시킬 경우(one cyclic scheduler), 0.9761에 도달
# hidden_dim : 20, 32, 64, 128, 256으로 늘릴수록 전반적인 성능은 향상..

if __name__ == "__main__":

    # wandb initialized
    wandb.init(project="MNIST_superpixel", entity="diya_challenger")

    # wandb experiment name edit
    wandb.run.name = args["wandb_save_name"]

    # save run setting
    wandb.run.save()

    BATCH_SIZE = args["batch_size"]
    train_loader, valid_loader, test_loader = generate_loader(batch_size=BATCH_SIZE, valid_ratio=args["valid_ratio"])

    num_epochs = args["num_epoch"]
    num_features = args["num_features"]
    k = args["k"]
    p = 0.5
    num_classes = args["num_classes"]
    embedd_size = args["embedd_size"]
    max_grad_norm = args['max_grad_norm']
    lr = args["lr"]
    save_path = args['weight_save_dir']
    aggr = args['aggr']

    model = DynamicReductionNetwork(num_features, embedd_size, num_classes, k = k, aggr=aggr, p = p)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=16, T_mult=1)

    # model structure
    sample = next(iter(train_loader))
    model.summary(sample)

    # wandb setting
    wandb.config = {
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": BATCH_SIZE,
        "embedd_size":embedd_size,
        "k":k,
        "dropout":p,
        "max_grad_norm":max_grad_norm,
        "aggr":aggr
    }

    train_loss, train_acc, valid_loss, valid_acc = train_wandb(model, train_loader, criterion, optimizer, scheduler, device, num_epochs, is_valid=True, valid_loader=valid_loader, verbose = True, verbose_period=1, save_best_only=True, save_path=save_path, max_grad_norm=max_grad_norm)

    epoch_axis = range(1, num_epochs + 1)

    plt.figure(1)
    plt.plot(epoch_axis, train_loss, label='train')
    plt.plot(epoch_axis, valid_loss, label='valid')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.ylim(0.5,4)
    plt.savefig("./result/dynamic-reduction-network-train-loss-curve.png")

    plt.figure(2)
    plt.plot(epoch_axis, train_acc, label = "train acc")
    plt.plot(epoch_axis, valid_acc, label = "valid acc")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig("./result/dynamic-reduction-network-train-accuracy-curve.png")

    # evaluate best model
    del model

    model = DynamicReductionNetwork(num_features, embedd_size, num_classes, k=k, aggr=aggr)
    model.load_state_dict(torch.load(save_path))
    evaluate(model, test_loader, criterion, device, save_dir = args['test_save_dir'])
