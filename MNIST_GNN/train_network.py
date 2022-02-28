import os
import sys
import torch
import torch_geometric
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from torch_geometric.datasets import MNISTSuperpixels
import numpy as np

from model.network import *
from utility.train import *
from utility.evaluate import *
from utility.scheduler import *

current_dir = os.getcwd()
sys.path.append(current_dir)

BATCH_SIZE = 256

train_dataset = MNISTSuperpixels('dataset', True, pre_transform=T.ToSparseTensor(remove_edge_index=False))
test_dataset = MNISTSuperpixels('dataset', False, pre_transform=T.ToSparseTensor(remove_edge_index=False))

train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# load model
num_epochs = 16 + 32 + 64 + 128 + 256
num_features = train_dataset.num_features
embedd_size = 128
model = Network(num_features, embedd_size, 10, 0.01, 0.5)

criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=16, T_mult=2)
#scheduler = CustomCosineAnnealingWarmUpRestarts(optimizer=optimizer, T_0=8, T_mult=2,eta_max = 0.1,T_up = 0,gamma =  0.99)

# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

if(torch.cuda.device_count() >= 1):
    device = "cuda:0" 
else:
    device = 'cpu'

train_loss, train_acc, valid_loss, valid_acc = train(model, train_loader, criterion, optimizer, scheduler, device, num_epochs, is_valid=True, verbose = True, verbose_period=8)

epoch_axis = range(1, num_epochs + 1)
plt.figure(1)
plt.plot(epoch_axis, train_loss, label='train')
plt.plot(epoch_axis, valid_loss, label='valid')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim(0.5,4)
plt.savefig("./result/network-train-loss-curve.png")

plt.figure(2)
plt.plot(epoch_axis, train_acc, label = "train acc")
plt.plot(epoch_axis, valid_acc, label = "valid acc")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig("./result/network-train-accuracy-curve.png")

evaluate(model, test_loader, criterion, device, save_dir = "./result/test_summary.txt")