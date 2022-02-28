import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(model, dataloader, loss_fn, device, save_dir = "./results/test_residual.png"):
    test_loss = 0
    total_pred = np.array([])
    total_label = np.array([])
    model.to(device)
    model.eval()

    for idx, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = batch.to(device)
            label = batch.y
            pred = model.forward(batch)

            num_batch = pred.shape[0]
            batch_loss = loss_fn(pred, label)
            test_loss += batch_loss.detach().cpu().numpy() / num_batch

            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, label.cpu().numpy().reshape(-1,)))

    test_loss /= (idx + 1)

    plt.figure(1)
    plt.scatter(total_pred, total_label)
    plt.xlabel("Real Energy Gap(eV)")
    plt.ylabel("Prediction Energy Gap(eV)")
    plt.plot(np.unique(total_label), np.poly1d(np.polyfit(total_label, total_pred,1))(np.unique(total_label)))
    plt.grid(True)
    plt.savefig(save_dir)

    print("test loss : {:.3f}".format(test_loss))

    return test_loss


def predict(model, dataloader, device):

    total_pred = np.array([])
    model.to(device)
    model.eval()

    for idx, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = batch.to(device)
            pred = model.forward(batch)            
            total_pred = np.concatenate(
                (total_pred, pred.cpu().numpy().reshape(-1,)))
      
    return total_pred