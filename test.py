"""Train Super Resolution model."""

import argparse
import os
import sys
import time
from pathlib import Path
# import grpc

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import torchvision
import seaborn as sns
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchmetrics.classification import MulticlassF1Score

from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from PIL import Image

from models.recur_cnn import RecurCNN
# from utils.div2k_dataset import DIV2KDataset
import scikitplot as skplt
import matplotlib.pyplot as plt


def test(
    dataset: str = "CIFAR10",
    batch_size: int = 16,
    num_epochs: int = 1,
    lr: float = 1e-4,
    num_workers: int = 4,
    device: str = "cpu",
    confusion_dir: str = "confusion",
    save_dir: str = "weights",
    save_interval: int = 10,
    model_dir: str = "results/run_13"
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

    if not os.path.exists(confusion_dir):
        os.makedirs(confusion_dir)


    # Create dataset
    if dataset == "CIFAR10":
        # Define the transform to normalize the data
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Load the CIFAR-10 dataset
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # Create a data loader for the test set
        batch_size = 64
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Separate the features (x_test) and labels (y_test) in batches
        x_test_batches = []
        y_test_batches = []
        for images, labels in testloader:
            x_test_batches.append(images)
            y_test_batches.append(labels)

        # Concatenate the batches to obtain the complete x_test and y_test
        data = torch.cat(x_test_batches, dim=0)
        targets = torch.cat(y_test_batches, dim=0)

    # Create model.
    model = RecurCNN(
        width=32
    )
    model.to(device)
    # Load the saved model
    model.load_state_dict(torch.load('model_1.pth'))
    model.to(device)

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

    num_corrects = 0
    num_samples = 0
 
    best_loss = float("inf")
    # for epoch in range(num_epochs):
    running_loss = 0.0
    # loop = tqdm.tqdm(test_loader, total=len(test_loader), leave=False)
    model.eval()  
    with torch.no_grad():
         # CIFAR-10 labels
        # send the data to the device
        # x = x.to(device)
        # y = y.to(device)

        # forward
        scores = model(data)

        # calculations for accuracy
        _, predictions = scores.max(1)
        correct = accuracy_score(targets, predictions)
        num_corrects += (predictions == targets).sum()
        num_samples += predictions.size(0)

        #conclusion
        f1 = f1_score(targets, predictions, average=None, zero_division=0)
        accuracy = num_corrects/num_samples
        precision = precision_score(targets, predictions, average="weighted", zero_division=0)
        recall = recall_score(targets, predictions, average="weighted", zero_division=0)

        print(f"Accuracy = {accuracy*100:.2f}; Received {num_corrects}/{num_samples}")
        print(f"Precision = {precision} and recall = {recall} and f1 = {f1}")
        

        #confusion matrix
        cm = confusion_matrix(targets, predictions, labels=[0,1,2,3,4,5,6,7,8,9])
        pp = sns.heatmap(cm)
        plt.show()
        pp.figure.savefig("roar.png")

        # confusion version
        confusion_version = 0
        while os.path.exists(os.path.join(confusion_dir, f"confusion_{confusion_version}")):
            confusion_version += 1

        # Create save directory.
        confusion_dir = os.path.join(confusion_dir, f"confusion_{confusion_version}")

        # Create save directory if not exists.
        if not os.path.exists(confusion_dir):
            os.makedirs(confusion_dir)


        # Log each element of the vector individually
        for index, value in enumerate(f1):
            tag = f'f1_element_{index}'
            writer.add_scalar(tag, value, global_step=index)



        # Log to tensorboard
        # running_loss /= len(test_loader)
        writer.add_scalar("Accuracy", accuracy)
        writer.add_scalar("Precision", precision)
        writer.add_scalar("Recall", recall)
        # writer.add_scalar("f1", f1)

                # Close the SummaryWriter
        writer.close()



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
                        default=1, help="number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate.")
    parser.add_argument("--device", type=str, default="cpu", help="device.")
    parser.add_argument("--num_workers", type=int,
                        default=4, help="number of workers.")
    # parser.add_argument("--device", type=str, default="cpu", help="device.")
    parser.add_argument("--save_dir", type=str,
                        default="weights", help="save directory.")
    parser.add_argument("--confusion_dir", type=str,
                        default="confusion", help="confusion directory.")
    parser.add_argument("--save_interval", type=int,
                        default=10, help="save interval.")
    parser.add_argument("--model_dir", type=str,
                        default="run_10", help="model directory.")
        
    
    
    args = parser.parse_args()

    test(
        dataset=args.dataset,
        # patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        confusion_dir=args.confusion_dir,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        model_dir=args.model_dir
    )
