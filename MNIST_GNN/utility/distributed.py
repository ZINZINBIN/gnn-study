import os
import wandb
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BatchNorm
import torch.multiprocessing as mp
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from utility.dataloader import *
from utility.loss_fn import *
import matplotlib.pyplot as plt

def run(rank, model, world_size: int, batch_size: int, valid_ratio: float, num_epochs: int, lr: float, max_grad_norm : float, save_path: str, wandb_setting:dict, wandb_config:dict):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.distributed.init_process_group(backend = 'nccl', rank = rank, world_size = world_size)

    train_dataset, valid_dataset, test_dataset = generate_dataset(valid_ratio=valid_ratio)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

    torch.manual_seed(42)

    num_classes = 10

    model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank]) # device_ids = [0,1,2,3] torch.device[rank]
    loss_fn = hgnn_loss(num_classes, num_classes, device=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=16, T_mult=2)

    train_losses  = []
    train_accuracy = []

    if rank == 0:
        # wandb initialized
        wandb.init(project=wandb_setting["project"], entity=wandb_setting["entity"])

        # wandb experiment name edit
        wandb.run.name = wandb_setting["name"]

        # save run setting
        wandb.run.save()

        wandb.config=wandb_config

        wandb.watch(model)
        valid_losses = []
        valid_accuracy = []
    else:
        valid_losses = None
        valid_accuracy = None

    best_loss = np.inf
    best_acc = 0

    for iter in range(0, num_epochs):
        model.train()

        train_loss = 0
        train_acc = 0

        train_sampler.set_epoch(iter)
        valid_sampler.set_epoch(iter)

        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(rank)
            label = batch.y
            pred = model(batch)
            batch_loss = loss_fn(pred, label)
            batch_acc = np.mean(torch.argmax(pred, 1).cpu().numpy() == label.cpu().numpy())
            num_batch = pred.shape[0]
            train_loss += batch_loss.detach().cpu().numpy() / num_batch
            train_acc += batch_acc

            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        train_loss /= (idx + 1)
        train_acc /= (idx + 1)

        train_losses.append(train_loss)
        train_accuracy.append(train_acc)

        torch.distributed.barrier()

        # validation using single gpu

        if rank == 0:
            model.eval()
            valid_loss = 0
            valid_acc = 0

            for idx, batch in enumerate(valid_loader):
                with torch.no_grad():
                    optimizer.zero_grad()
                    batch = batch.to(rank)
                    label = batch.y
                    pred = model(batch)
                    batch_loss = loss_fn(pred, label)
                    batch_acc = np.mean(torch.argmax(pred, 1).cpu().numpy() == label.cpu().numpy())

                    num_batch = pred.shape[0]
                    valid_loss += batch_loss.detach().cpu().numpy() / num_batch
                    valid_acc += batch_acc

            valid_loss /= (idx + 1)
            valid_acc /= (idx + 1)

            valid_losses.append(valid_loss)
            valid_accuracy.append(valid_acc)

            if(valid_acc >= best_acc):
                best_acc = valid_acc
                best_loss = valid_loss
                
                try:
                    state_dict = model.module.state_dict()

                except AttributeError:
                    state_dict = model.state_dict()

                torch.save(state_dict, save_path)

            print("# iter : {:3d} train_loss : {:.3f}, train_acc : {:.3f}, valid_loss : {:.3f}, valid_acc : {:.3f}, best_acc : {:.3f}, best_loss : {:.3f}".format(iter + 1, train_loss, train_acc, valid_loss, valid_acc, best_acc, best_loss))
            
            wandb.log({
                    "iter":iter + 1,
                    "train loss":train_loss,
                    "train accuracy":train_acc,
                    "valid loss":valid_loss,
                    "valid accuracy":valid_acc,
                    "best loss":best_loss,
                    "best accuracy":best_acc
            })

            torch.distributed.barrier()

    torch.distributed.destroy_process_group()

    '''
    # loss / acc curve save from plt
    epoch_axis = range(1, len(train_losses) + 1)
    plt.figure(1)
    plt.plot(epoch_axis, train_losses, label='train')
    plt.plot(epoch_axis, valid_losses, label='valid')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.ylim(0.5, 4)
    plt.savefig("./result/run-train-loss-curve.png")

    plt.figure(2)
    plt.plot(epoch_axis, train_accuracy, label="train acc")
    plt.plot(epoch_axis, valid_accuracy, label="valid acc")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig("./result/run-train-accuracy-curve.png")
    '''


    