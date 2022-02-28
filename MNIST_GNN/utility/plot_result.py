import matplotlib.pyplot as plt

def plot_result(obj = "loss", train_list = None, valid_list = None, save_dir = "./result/baseline.png"):
    epoch_axis = range(1, len(train_list) + 1)
    plt.figure()
    plt.plot(epoch_axis, train_list, label='train')
    plt.plot(epoch_axis, valid_list, label='valid')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(obj)
    plt.ylim(0.5,4)
    plt.savefig(save_dir)