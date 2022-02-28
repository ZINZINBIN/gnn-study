import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

from model.network import *
from utility.train import *
from utility.evaluate import *
from utility.dataloader import *
from utility.loss_fn import *

current_dir = os.getcwd()
sys.path.append(current_dir)

BATCH_SIZE = 256
train_loader, valid_loader, test_loader = generate_loader(batch_size = BATCH_SIZE, test_ratio = 0.2)

# load model
num_epochs = 16 + 32 + 64 + 128 + 256
num_features = 1
num_classes = 10
embedd_size = 64

model = GAT(num_classes=num_classes, num_features = num_features, num_heads = [2,2,2], hidden = [embedd_size,embedd_size], p = 0.5, alpha = 0.01)
criterion = nn.CrossEntropyLoss(reduction='sum')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=16, T_mult=2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.95)

# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

if(torch.cuda.device_count() >= 1):
    device = "cuda:1" 
else:
    device = 'cpu'

train_loss, train_acc, valid_loss, valid_acc = train(model, train_loader, criterion, optimizer, scheduler, device, num_epochs, is_valid=True, valid_loader=valid_loader, verbose = True, verbose_period=8)

epoch_axis = range(1, num_epochs + 1)
plt.figure(1)
plt.plot(epoch_axis, train_loss, label='train')
plt.plot(epoch_axis, valid_loss, label='valid')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim(0.5,4)
plt.savefig("./result/baseline-gat-train-loss-curve.png")

plt.figure(2)
plt.plot(epoch_axis, train_acc, label = "train acc")
plt.plot(epoch_axis, valid_acc, label = "valid acc")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig("./result/baseline-gat-train-accuracy-curve.png")

evaluate(model, test_loader, criterion, device, save_dir = "./result/test_summary_gat.txt")