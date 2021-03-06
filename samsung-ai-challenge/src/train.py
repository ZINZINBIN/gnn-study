import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm

def train_per_epoch(
        model, 
        train_dataloader, 
        loss_fn,
        optimizer,
        scheduler,
        device = "cpu",
        max_grad_norm=1.0,
        loss_fn_eval = None,
        ):

    train_loss = 0
    train_loss_eval = 0
    model.train()

    model.to(device)

    for idx, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        y_true = batch.y
        optimizer.zero_grad()
        y_pred = model.forward(batch)
        num_batch = y_pred.shape[0]
        batch_loss = loss_fn(y_pred, y_true)
        train_loss += batch_loss.detach().cpu().numpy() / num_batch

        if loss_fn_eval is not None:
            batch_loss_eval = loss_fn_eval(y_pred, y_true)
            train_loss_eval += batch_loss_eval.detach().cpu().numpy() / num_batch

        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()


    scheduler.step()
    train_loss /= (idx + 1)
    
    if loss_fn_eval is None:
        return train_loss
    else:
        train_loss_eval /= (idx + 1)
        return train_loss, train_loss_eval


def valid_per_epoch(
    model,
    valid_dataloader,
    loss_fn,
    optimizer,
    scheduler,
    device="cpu",
    max_grad_norm=1.0,
    loss_fn_eval = None,
    ):

    valid_loss = 0
    valid_loss_eval = 0
    model.eval()
    model.to(device)

    for idx, batch in enumerate(valid_dataloader):
        with torch.no_grad():
            optimizer.zero_grad()
            batch = batch.to(device)
            y_true = batch.y
            y_pred = model.forward(batch)
            num_batch = y_pred.shape[0]
            batch_loss = loss_fn(y_pred, y_true)
            valid_loss += batch_loss.detach().cpu().numpy() / num_batch

            if loss_fn_eval is not None:
                batch_loss_eval = loss_fn_eval(y_pred, y_true)
                valid_loss_eval += batch_loss_eval.detach().cpu().numpy() / num_batch

    valid_loss /= (idx + 1)

    if loss_fn_eval is None:
        return valid_loss
    else:
        valid_loss_eval /= (idx + 1)
        return valid_loss, valid_loss_eval

def train(
    model, 
    train_loader, 
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
    max_grad_norm=1.0,
    wandb_monitoring = False,
    loss_fn_eval = None,
    ):

    train_losses = []
    valid_losses = []

    train_losses_eval = []
    valid_losses_eval = []

    best_loss = np.inf
    
    model = model.to(device)

    for iter in tqdm(range(iterations), desc = "training process", total = iterations):

        if loss_fn_eval is None:
            train_loss = train_per_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, max_grad_norm)
        else:
            train_loss, train_loss_eval = train_per_epoch(
                model, train_loader, loss_fn, optimizer, scheduler, device, max_grad_norm, loss_fn_eval=loss_fn_eval)
            train_losses_eval.append(train_loss_eval)

        train_losses.append(train_loss)
        
        if(is_valid):

            if valid_loader is not None:
                if loss_fn_eval is None:
                    valid_loss = valid_per_epoch(model, valid_loader, loss_fn, optimizer, scheduler, device, max_grad_norm, loss_fn_eval=loss_fn_eval)
                else:
                    valid_loss, valid_loss_eval = valid_per_epoch(
                        model, valid_loader, loss_fn, optimizer, scheduler, device, max_grad_norm, loss_fn_eval=loss_fn_eval)
                valid_losses_eval.append(valid_loss_eval)
            else:
                if loss_fn_eval is None:
                    valid_loss = valid_per_epoch(
                        model, train_loader, loss_fn, optimizer, scheduler, device, max_grad_norm, loss_fn_eval=loss_fn_eval)
                else:
                    valid_loss, valid_loss_eval = valid_per_epoch(
                        model, train_loader, loss_fn, optimizer, scheduler, device, max_grad_norm, loss_fn_eval=loss_fn_eval)
                valid_losses_eval.append(valid_loss_eval)

            valid_losses.append(valid_loss)

        if(save_best_only):
            if loss_fn_eval is None:
                if(valid_loss <= best_loss):
                    best_loss = valid_loss
                    torch.save(model.state_dict(), save_path)
            else:
                if(valid_loss_eval <= best_loss):
                    best_loss = valid_loss_eval
                    torch.save(model.state_dict(), save_path)

        if verbose and iter % verbose_period == 0 and loss_fn_eval is None:
            if(is_valid):
                print("# iter : {:3d} train_loss : {:.3f}, valid_loss : {:.3f}, best_loss : {:.3f}".format(iter + 1, train_loss, valid_loss, best_loss))
            else:
                print("# iter : {:3d} train_loss : {:.3f}".format(iter + 1, train_loss))

        if verbose and iter % verbose_period == 0 and loss_fn_eval is not None:
            if(is_valid):
                print("# iter : {:3d} train_loss : {:.3f}, valid_loss : {:.3f}, best_loss : {:.3f}".format(
                    iter + 1, train_loss_eval, valid_loss_eval, best_loss))
            else:
                print("# iter : {:3d} train_loss : {:.3f}".format(
                    iter + 1, train_loss_eval))

        if loss_fn_eval is not None:
            train_loss = train_loss_eval
            if(is_valid):
                valid_loss = valid_loss_eval

        if wandb_monitoring:
            if(is_valid):
                wandb.log({
                    "iter": iter + 1,
                    "train loss": train_loss,
                    "valid loss": valid_loss,
                    "best loss": best_loss,
                })
            else:
                wandb.log({
                    "iter": iter + 1,
                    "train loss": train_loss,
                })

    return train_losses, valid_losses