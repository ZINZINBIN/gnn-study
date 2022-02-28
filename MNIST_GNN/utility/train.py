import torch
import wandb
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch_geometric.nn import DataParallel
import torch.nn.functional as F
from utility.parallel import DataParallelModel, DataParallelCriterion

def train_per_epoch(model, dataloader, loss_fn, optimizer, scheduler, device, max_grad_norm = 5.):
    train_loss = 0
    train_acc = 0
    model.train()

    for idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        label = batch.y
        optimizer.zero_grad()
        pred = model.forward(batch)
        num_batch = pred.shape[0]
        batch_loss = loss_fn(pred, label)

        pred = torch.nn.functional.softmax(pred, dim = 1)
        batch_acc = np.mean(torch.argmax(pred, 1).cpu().numpy() == label.cpu().numpy())
        train_loss += batch_loss.detach().cpu().numpy() / num_batch
        train_acc += batch_acc

        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
    scheduler.step()
    train_loss /= (idx + 1)
    train_acc /= (idx + 1)

    return train_loss, train_acc
    

def valid_per_epoch(model, dataloader, loss_fn, device):
    valid_loss = 0
    valid_acc = 0
    model.eval()
    for idx, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = batch.to(device)
            label = batch.y
            pred = model.forward(batch)
            num_batch = pred.shape[0]
            batch_loss = loss_fn(pred, label)

            pred = torch.nn.functional.softmax(pred, dim = 1)
            batch_acc = np.mean(torch.argmax(pred, 1).cpu().numpy() == label.cpu().numpy())
            valid_loss += batch_loss.detach().cpu().numpy() / num_batch
            valid_acc += batch_acc

    valid_loss /= (idx + 1)
    valid_acc /= (idx + 1)

    return valid_loss, valid_acc


def train(model, dataloader, loss_fn, optimizer, scheduler, device, iterations = 32, is_valid = True, valid_loader = None, verbose = True, verbose_period = 10, save_best_only = False, save_path = "./weights/best.pt", max_grad_norm = 1.0):
    
    train_losses = []
    train_accuracy = []

    best_loss = np.inf
    best_acc = 0

    if is_valid:
        valid_losses = []
        valid_accuracy = []
    else:
        valid_losses = None
        valid_accuracy = None

    model = model.to(device)
    
    for iter in tqdm(range(iterations)):
        model.train()
        train_loss, train_acc = train_per_epoch(model, dataloader, loss_fn, optimizer, scheduler, device, max_grad_norm)
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)

        if(is_valid):

            if valid_loader is not None:
                valid_loss, valid_acc = valid_per_epoch(model, valid_loader, loss_fn, device)
            else:
                valid_loss, valid_acc = valid_per_epoch(model, dataloader, loss_fn, device)

            valid_losses.append(valid_loss)
            valid_accuracy.append(valid_acc)

        if(save_best_only):
            if(valid_acc >= best_acc):
                best_acc = valid_acc
                best_loss = valid_loss
                torch.save(model.state_dict(), save_path)

        if verbose and iter % verbose_period == 0:
            if(is_valid):
                print("# iter : {:3d} train_loss : {:.3f}, train_acc : {:.3f}, valid_loss : {:.3f}, valid_acc : {:.3f}, best_acc : {:.3f}, best_loss : {:.3f}".format(iter + 1, train_loss, train_acc, valid_loss, valid_acc, best_acc, best_loss))
            else:
                print("# iter : {:3d} train_loss : {:.3f}, train_acc : {:.3f}".format(iter + 1, train_loss, train_acc))

    return train_losses, train_accuracy, valid_losses, valid_accuracy


# method 1 : nn.DataParallel
def data_parallel(module, input, device_ids, output_device):
    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)

def train_parallel(model, dataloader, loss_fn, optimizer, scheduler, device, iterations = 32, is_valid = True, valid_loader = None, verbose = True, verbose_period = 10, save_best_only = False, save_path = "./weights/best.pt", max_grad_norm = 1.0):
    '''
    - train process for multi - gpu
    - using torch.nn.multiprocessing
    - method : nn.DataParallel -> loss function : parallelized
    '''
    train_losses = []
    train_accuracy = []

    best_loss = np.inf
    best_acc = 0

    if is_valid:
        valid_losses = []
        valid_accuracy = []
    else:
        valid_losses = None
        valid_accuracy = None

    # method 1 : nn.DataParallel
    # model = DataParallelModel(model)
    model = DataParallel(model)
    model.to(device) # source device

    #loss_parallel = DataParallelCriterion(loss_fn)

    for iter in tqdm(range(iterations)):
        model.train()

        train_loss = 0
        train_acc = 0

        for idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            pred = model(batch)
            label = torch.cat([data.y for data in batch]).to(pred.device)
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

        if(is_valid):

            model.eval()

            if valid_loader is None:
                valid_loader = dataloader
            
            valid_loss = 0
            valid_acc = 0

            for idx, batch in enumerate(dataloader):
                with torch.no_grad():
                    optimizer.zero_grad()
                    pred = model(batch)
                    label = torch.cat([data.y for data in batch]).to(pred.device)
                    batch_loss = loss_fn(pred, label)
                    batch_acc = np.mean(torch.argmax(
                        pred, 1).cpu().numpy() == label.cpu().numpy())

                    num_batch = pred.shape[0]
                    valid_loss += batch_loss.detach().cpu().numpy() / num_batch
                    valid_acc += batch_acc

            valid_loss /= (idx + 1)
            valid_acc /= (idx + 1)

            valid_losses.append(valid_loss)
            valid_accuracy.append(valid_acc)

        if(save_best_only):

            if(valid_acc >= best_acc):
                best_acc = valid_acc
                best_loss = valid_loss
                # data parallel을 이용할 경우 state_dict()로 학습된 가중치 저장시 load 과정에서 error 발생
                # model.module.state_dict()를 이용해 원본 모델에 대한 가중치 저장을 진행해야 한다.
                try:
                    state_dict = model.module.state_dict()

                except AttributeError:
                    state_dict = model.state_dict()

                torch.save(state_dict, save_path)

        if verbose and iter % verbose_period == 0:
            if(is_valid):
                print("# iter : {:3d} train_loss : {:.3f}, train_acc : {:.3f}, valid_loss : {:.3f}, valid_acc : {:.3f}, best_acc : {:.3f}, best_loss : {:.3f}".format(iter + 1, train_loss, train_acc, valid_loss, valid_acc, best_acc, best_loss))
            else:
                print("# iter : {:3d} train_loss : {:.3f}, train_acc : {:.3f}".format(iter + 1, train_loss, train_acc))

    return train_losses, train_accuracy, valid_losses, valid_accuracy


# train method with wandb
def train_wandb(
    model, 
    dataloader, 
    loss_fn, 
    optimizer, 
    scheduler, 
    device, 
    iterations=32, 
    is_valid=True, 
    valid_loader=None, 
    verbose=True, 
    verbose_period=8, 
    save_best_only=False, 
    save_path="./weights/best.pt", 
    max_grad_norm=5.0
    ):

    train_losses = []
    train_accuracy = []

    best_loss = np.inf
    best_acc = 0

    if is_valid:
        valid_losses = []
        valid_accuracy = []
    else:
        valid_losses = None
        valid_accuracy = None

    model = model.to(device)

    for iter in tqdm(range(iterations)):
        model.train()
        train_loss, train_acc = train_per_epoch(
            model, dataloader, loss_fn, optimizer, scheduler, device, max_grad_norm)
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)

        if(is_valid):

            model.eval()

            if valid_loader is not None:
                valid_loss, valid_acc = valid_per_epoch(
                    model, valid_loader, loss_fn, device)
            else:
                valid_loss, valid_acc = valid_per_epoch(
                    model, dataloader, loss_fn, device)

            valid_losses.append(valid_loss)
            valid_accuracy.append(valid_acc)

        if(save_best_only):
            if(valid_acc >= best_acc):
                best_acc = valid_acc
                best_loss = valid_loss
                torch.save(model.state_dict(), save_path)

        if verbose and iter % verbose_period == 0:
            if(is_valid):
                wandb.log({
                    "iter":iter + 1,
                    "train loss":train_loss,
                    "train accuracy":train_acc,
                    "valid loss":valid_loss,
                    "valid accuracy":valid_acc,
                    "best loss":best_loss,
                    "best accuracy":best_acc
                })
                print("# iter : {:3d} train_loss : {:.3f}, train_acc : {:.3f}, valid_loss : {:.3f}, valid_acc : {:.3f}, best_acc : {:.3f}, best_loss : {:.3f}".format(
                    iter + 1, train_loss, train_acc, valid_loss, valid_acc, best_acc, best_loss))
            else:
                wandb.log({
                    "iter": iter + 1,
                    "train loss": train_loss,
                    "train accuracy": train_acc,
                })
                print("# iter : {:3d} train_loss : {:.3f}, train_acc : {:.3f}".format(
                    iter + 1, train_loss, train_acc))

    return train_losses, train_accuracy, valid_losses, valid_accuracy