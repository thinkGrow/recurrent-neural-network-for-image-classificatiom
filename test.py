"""Train Super Resolution model."""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchmetrics.classification import MulticlassF1Score

from models.recur_cnn import RecurCNN
# from utils.div2k_dataset import DIV2KDataset
from utils.perceptual_loss import VGG16PerceptualLoss


def test(
    # data_dir: str,
    # scale_factor: int = 4,
    # patch_size: int = 48,
    dataset: str = "CIFAR10",
    batch_size: int = 16,
    num_epochs: int = 5,
    lr: float = 1e-4,
    num_workers: int = 4,
    device: str = "cpu",
    save_dir: str = "weights",
    save_interval: int = 10
) -> None:
    """
    Train function.

    :param data_dir: str, data directory.
    :type data_dir: str
    :param scale_factor: int, scale factor.
    :type scale_factor: int
    :param patch_size: int, patch size.
    :type patch_size: int
    :param batch_size: int, batch size.
    :type batch_size: int
    :param num_epochs: int, number of epochs.
    :type num_epochs: int
    :param lr: float, learning rate.
    :type lr: float
    :param num_workers: int, number of workers.
    :type num_workers: int
    :param device: str, device.
    :type device: str
    :param save_dir: str, save directory.
    :type save_dir: str
    :param save_interval: int, save interval.
    :type save_interval: int
    """
    # Create save directory.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create dataset
    if dataset == "CIFAR10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        
        dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # Create data loader.
        train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )

        test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # Create model.
    model = RecurCNN(
        width=32
    )
    # .to(device)

    # Create criterion.
    criterion_cel = nn.CrossEntropyLoss(
        # device=device
    )
    # criterion_percp = VGG16PerceptualLoss(
    #     device=device
    # )

    # run version
    test_version = 0
    while os.path.exists(os.path.join(save_dir, f"run_{test_version}")):
        test_version += 1

    # Create save directory.
    save_dir = os.path.join(save_dir, f"run_{test_version}")

    # Create save directory if not exists.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # setup tensorboard
    writer = SummaryWriter(log_dir=save_dir)

    best_loss = float("inf")
    for epoch in range(num_epochs):
        running_loss = 0.0
        loop = tqdm.tqdm(test_loader, total=len(test_loader), leave=False)

        with torch.no_grad():
            for data, targets in test_loader:
            # send the data to the device
            # x = x.to(device)
            # y = y.to(device)

                # prepare the data for the model
                # x = x.reshape(-1, 784)

                # forward
                scores = model(data)

                # Calculate loss.
                loss = criterion_cel(scores, targets)

                # calculations for accuracy
                _, predictions = scores.max(1)
                num_corrects += (predictions == targets).sum()
                num_samples += predictions.size(0)

                # Update progress bar.
                loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
                loop.set_postfix(loss=loss.item())

                # calculations for accuracy
                _, predictions = scores.max(1)
                num_corrects += (predictions == targets).sum()
                num_samples += predictions.size(0)
                mcf1s = MulticlassF1Score(num_classes=10)


                # Update running loss.
                running_loss += loss.item()

            print(f" f-score = {mcf1s(predictions, targets)}")
            print(f"Accuracy = {num_corrects/num_samples*100:.2f}; Received {num_corrects}/{num_samples}")
            # model.train()

                #f1 score
                # f1 = F1Score(task="multiclass", num_classes=3)
                
        
        # Log to tensorboard
        running_loss /= len(test_loader)
        writer.add_scalar("Loss/test", running_loss, epoch)
        # writer.add_scalar('Loss/test', np.random.random(), n_iter)
        # writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        # writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

        # Save model at save interval.
        if (epoch + 1) % save_interval == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": running_loss
                # "accuracy": 
            },
                os.path.join(save_dir, f"model_latest.pth")
            )

        # Save best model.
        if running_loss < best_loss:
            best_loss = running_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": running_loss
            },
                os.path.join(save_dir, f"model_best.pth")
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test RecurNet model.")
    parser.add_argument("--dataset", type=str,
                        default="CIFAR10", help="dataset")
    # parser.add_argument("--patch_size", type=int,
    #                     default=48, help="patch size.")
    parser.add_argument("--batch_size", type=int,
                        default=16, help="batch size.")
    parser.add_argument("--num_epochs", type=int,
                        default=100, help="number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate.")
    # parser.add_argument("--device", type=str, default="cpu", help="device.")
    parser.add_argument("--num_workers", type=int,
                        default=4, help="number of workers.")
    # parser.add_argument("--device", type=str, default="cpu", help="device.")
    parser.add_argument("--save_dir", type=str,
                        default="weights", help="save directory.")
    parser.add_argument("--save_interval", type=int,
                        default=10, help="save interval.")
    args = parser.parse_args()

    test(
        dataset=args.dataset,
        # patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        # device=args.device,
        save_dir=args.save_dir,
        save_interval=args.save_interval
    )

# import torch
# if torch.backends.mps.is_available():
#     mps_device = torch.device("mps")
#     x = torch.ones(1, device=mps_device)
#     print (x)
# else:
#     print ("MPS device not found.")