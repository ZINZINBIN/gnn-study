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

num_heads = 4
p = 0.25
alpha = 0.01
num_features = 1
num_classes = 10
k = 5
hidden = 64
alpha = 0.01

# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

if(torch.cuda.device_count() >= 1):
    device = "cuda:0" 
else:
    device = 'cpu'

BATCH_SIZE = 256

chebNet = ChebNet(num_features, num_classes, k, hidden, alpha, p)
chebNet.load_state_dict(torch.load("./weights/ChebNet_without_arcface.pt", map_location = device))
gconvNet = GConvNet(num_features, num_classes, hidden, alpha, p)
gconvNet.load_state_dict(torch.load("./weights/GConvNet_without_arcface.pt", map_location = device))
hgnn = HGNN(num_classes, num_features, num_heads, hidden, p = p, alpha = alpha)
hgnn.load_state_dict(torch.load("./weights/HGNN_without_arcface.pt", map_location = device))

train_loader, valid_loader, test_loader = generate_loader(batch_size = BATCH_SIZE, valid_ratio = 0.2)
sample = next(iter(train_loader))

criterion = nn.CrossEntropyLoss(reduction='sum')

print("-------------------GConvNet structure----------------------")
gconvNet.summary(sample)
evaluate(gconvNet, test_loader, criterion, device, save_dir = "./result/test_summary_GConvNet_without_arcface.txt")
print("--------------------ChebNet structure----------------------")
chebNet.summary(sample)
evaluate(chebNet, test_loader, criterion, device, save_dir = "./result/test_summary_ChebNet_without_arcface.txt")
print("---------------------HGNN structure-----------------------")
hgnn.summary(sample)
evaluate(hgnn, test_loader, criterion, device, save_dir = "./result/test_summary_HGNN_without_arcface.txt")