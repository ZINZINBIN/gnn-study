import argparse

from src.network import *
from src.train import *
from src.utility import *
from src.evaluate import *

parser = argparse.ArgumentParser(description="ST1-GAP Regression model with GNN")
parser.add_argument("--num_epoch", type = int, default = 128)
parser.add_argument("--gpu_num", type = int, default = 0)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--valid_ratio", type = float, default = 0.2)
parser.add_argument("--train_dir", type = str, default = "./datasets/train.csv")
parser.add_argument("--test_dir", type=str, default="./datasets/test.csv")
parser.add_argument("--dev_dir", type=str, default="./datasets/dev.csv")
parser.add_argument("--submission_dir", type=str, default="./datasets/sample_submission.csv")
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--verbose_period", type=int, default=8)
parser.add_argument("--save_path", type=str, default="./weights/best.pt")
parser.add_argument("--save_result", type=str, default="./results/test_residual.png")
parser.add_argument("--alpha", type = float, default = 0.01)
parser.add_argument("--hidden", type = int, default = 128)
parser.add_argument("--embedd_max_norm", type = float, default = 1.0)

args = vars(parser.parse_args())

# torch initialize
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

if(torch.cuda.device_count() >= 1):
    device = "cuda:" +str(args["gpu_num"])
else:
    device = 'cpu'

if __name__ == "__main__":

    train_loader, valid_loader, test_loader = generate_dataloader(
        args["train_dir"],
        args["test_dir"],
        args["submission_dir"],
        args["dev_dir"],
        batch_size = args["batch_size"],
        valid_ratio = args["valid_ratio"]
    )

    sample = next(iter(train_loader))
    atom_feats = sample.x.size()[1]

    model = GConvNet(
        args['hidden'],
        args['alpha'],
        args["embedd_max_norm"]
    )

    model.summary(sample)

    model.to(device)
    loss_fn = torch.nn.L1Loss(reduction = "sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0 = 16,
        T_mult = 2
    )
  
    train_loss, valid_loss = train(
        model,
        train_loader,
        loss_fn,
        optimizer,
        scheduler,
        device,
        args["num_epoch"],
        True,
        valid_loader,
        args["verbose"],
        args["verbose_period"],
        True,
        args["save_path"],
        args["max_grad_norm"],
        False
    )

    plot_training_curve(train_loss, valid_loss, save_dir = "./results/gcn-training-curve.png")

    model.load_state_dict(torch.load(args["save_path"], map_location = device))

    evaluate(
        model, 
        valid_loader, 
        loss_fn, 
        device,
        save_dir=args["save_result"]
    )

    prediction = predict(model, test_loader, device)
    sample_submission = pd.read_csv(args['submission_dir'])
    sample_submission['ST1_GAP(eV)'] = prediction
    sample_submission.to_csv("./results/gcn_submission.csv", index = False)