import torch
from tqdm import tqdm
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def evaluate(model, dataloader, loss_fn, device, save_dir = "./result/result_summary.txt"):
    test_loss = 0
    test_acc = 0
    total_pred = np.array([])
    total_label = np.array([])
    model.to(device)
    model.eval()

    for idx, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = batch.to(device)
            label = batch.y
            pred = model.forward(batch)
            pred = torch.nn.functional.softmax(pred, dim = 1)
            num_batch = pred.shape[0]
            batch_loss = loss_fn(pred, label)
            batch_acc = np.mean(torch.argmax(pred, 1).cpu().numpy() == label.cpu().numpy())
            test_loss += batch_loss.detach().cpu().numpy() / num_batch
            test_acc += batch_acc

            total_pred = np.concatenate((total_pred, torch.argmax(pred, 1).cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, label.cpu().numpy().reshape(-1,)))

    test_loss /= (idx + 1)
    test_acc /= (idx + 1)

    conf_mat = confusion_matrix(total_label,  total_pred)

    plt.figure()
    sns.heatmap(
        conf_mat / np.sum(conf_mat, axis = 1),
        annot = True,
        fmt = '.2f',
        cmap = 'Blues',
        xticklabels=[0,1,2,3,4,5,6,7,8,9],
        yticklabels=[0,1,2,3,4,5,6,7,8,9]
    )
    plt.savefig(save_dir.split(".txt")[0] + "_confusion_matrix.png")

    print("############### Classification Report ####################")
    print(classification_report(total_label, total_pred, labels = [0,1,2,3,4,5,6,7,8,9]))
    print("\n# total test score : {:.2f} and test loss : {:.3f}".format(test_acc, test_loss))
    print(conf_mat)

    with open(save_dir, 'w') as f:
        f.write(classification_report(total_label, total_pred, labels = [0,1,2,3,4,5,6,7,8,9]))
        summary = "\n# total test score : {:.2f} and test loss : {:.3f}".format(test_acc, test_loss)
        f.write(summary)

    return test_loss, test_acc, conf_mat