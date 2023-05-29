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
# from test import function
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def train(
    # PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    
    # data_dir: str,
    # scale_factor: int = 4,
    # patch_size: int = 48,
    PYTORCH_MPS_HIGH_WATERMARK_RATIO: float = "0.0",
    dataset: str = "CIFAR10",
    batch_size: int = 8,
    num_epochs: int = 5,
    lr: float = 1e-4,
    num_workers: int = 3,
    device: str = "cpu",
    save_dir: str = "weights",
    save_interval: int = 10,
    # torch.mps.set_per_process_memory_fraction(0.0)

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

    # set_per_process_memory_fraction = 0.0
    
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

        # Separate the features (x_test) and labels (y_test) in batches
        x_train_batches = []
        y_train_batches = []
        for images, labels in train_loader:
            x_train_batches.append(images)
            y_train_batches.append(labels)

        # Concatenate the batches to obtain the complete x_train and y_train
        x_train = torch.cat(x_train_batches, dim=0)
        y_train = torch.cat(y_train_batches, dim=0)

        # Separate the features (x_test) and labels (y_test) in batches
        x_test_batches = []
        y_test_batches = []
        for images, labels in test_loader:
            x_test_batches.append(images)
            y_test_batches.append(labels)

        # Concatenate the batches to obtain the complete x_test and y_test
        x_test = torch.cat(x_test_batches, dim=0)
        y_test = torch.cat(y_test_batches, dim=0)

    # Create model.
    model = RecurCNN(
        width=32
    ).to(device)

    # Create criterion.
    criterion_mse = nn.MSELoss()
    criterion_cel = nn.CrossEntropyLoss(
        # device=device
    )
    # criterion_percp = VGG16PerceptualLoss(
    #     device=device
    # )

    # Create optimizer.
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=lr
    )

    # run version
    run_version = 0
    while os.path.exists(os.path.join(save_dir, f"run_{run_version}")):
        run_version += 1

    # Create save directory.
    save_dir = os.path.join(save_dir, f"run_{run_version}")

    # Create save directory if not exists.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # setup tensorboard
    writer = SummaryWriter(log_dir=save_dir)

    
    # Test
    best_loss = float("inf")
    for epoch in range(num_epochs):
        running_loss = 0.0
        loop = tqdm.tqdm(train_loader, total=len(train_loader), leave=False)


        # for lr, hr in loop:
        # for batch_idx, (data, targets) in enumerate(train_loader):
        for data, targets in loop:

            # print(data.shape)
            # print(targets.shape)
            # Move to device.
            data = data.to(device)
            targets = targets.to(device)

            # print(data.shape)
            # print(targets.shape)

            # Forward.
            scores = model(data)

            # Calculate loss.
            loss = criterion_cel(scores, targets)

            # Backward.
            optimizer.zero_grad()
            loss.backward()
            #gradient descent
            optimizer.step()

            # Update progress bar.
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

            # Update running loss.
            running_loss += loss.item()

            # Check accuracy
            # acc_calc(test_loader, model)
        num_corrects = 0
        num_samples = 0

        best_loss = float("inf")
        # for epoch in range(num_epochs):
        running_loss = 0.0

        model.eval()  
        with torch.no_grad():
            # CIFAR-10 labels
            # send the data to the device
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            x_test = x_test.to(device)
            y_test = y_test.to(device)


            # calculations for accuracy
            _, predictions_train = scores.max(1)
            num_corrects += (predictions_train == y_train).sum()
            num_samples += predictions_train.size(0)

            #conclusion
            accuracy_train = num_corrects/num_samples
            running_accuracy_train += accuracy_train.item()
            precision_train = precision_score(y_train.cpu(), predictions_train.cpu(), average="weighted", zero_division=0)
            recall_train = recall_score(y_train.cpu(), predictions_train.cpu(), average="weighted", zero_division=0)
            f1_train = f1_score(y_train.cpu(), predictions_train.cpu(), average=None, zero_division=0)
            cm_train = confusion_matrix(y_train, predictions_train,  labels=[0,1,2,3,4,5,6,7,8,9] )
            # Normalize the confusion matrix
            cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
            # Plot confusion matrix
            fig, ax = plt.subplots()
            im_train = ax.imshow(cm_train, cmap='Blues')
            # Customize plot settings
            classes = [f'Class {i}' for i in range(10)]
            ax.set(xticks=np.arange(10),
                yticks=np.arange(10),
                xticklabels=classes,
                yticklabels=classes,
                xlabel='Predicted label',
                ylabel='True label',
                title='Confusion Matrix')
            # Loop over data dimensions and create text annotations
            for i in range(10):
                for j in range(10):
                    ax.text(j, i, f'{cm_train[i, j]:.2f}',
                            ha="center", va="center", color="white")
            # Adjust layout to fit the colorbar
            fig.tight_layout()
            plt.colorbar(im_train)

            # Convert the plot to an image
            fig.canvas.draw()
            image_train = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image_train = image_train.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            print(f"\n Epoch no.: {epoch+1}" )
            print(f"Accuracy Train = {accuracy_train*100:.2f}; Received {num_corrects}/{num_samples}")
            print(f"Precision Train = {precision_train} and Recall Train = {recall_train} \n f1 Train = {f1_train} \n")

            ###Test###

            # forward
            scores = model(x_test)

            # calculations for accuracy
            _, predictions_test = scores.max(1)
            num_corrects += (predictions_test == y_test).sum()
            num_samples += predictions_test.size(0)

            #conclusion

            accuracy_test = num_corrects/num_samples
            running_accuracy_test += accuracy_test.item()
            precision_test = precision_score(y_test.cpu(), predictions_test.cpu(), average="weighted", zero_division=0)
            recall_test = recall_score(y_test.cpu(), predictions_test.cpu(), average="weighted", zero_division=0)
            f1_test = f1_score(y_test.cpu(), predictions_test.cpu(), average=None, zero_division=0)

            cm_test = confusion_matrix(y_test, predictions_test,  labels=[0,1,2,3,4,5,6,7,8,9] )
            # Normalize the confusion matrix
            cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
            # Plot confusion matrix
            fig, ax = plt.subplots()
            im_test = ax.imshow(cm_test, cmap='Blues')
            # Customize plot settings
            classes = [f'Class {i}' for i in range(10)]
            ax.set(xticks=np.arange(10),
                yticks=np.arange(10),
                xticklabels=classes,
                yticklabels=classes,
                xlabel='Predicted label',
                ylabel='True label',
                title='Confusion Matrix')
            # Loop over data dimensions and create text annotations
            for i in range(10):
                for j in range(10):
                    ax.text(j, i, f'{cm_test[i, j]:.2f}',
                            ha="center", va="center", color="white")
            # Adjust layout to fit the colorbar
            fig.tight_layout()
            plt.colorbar(im_test)


            # Convert the plot to an image
            fig.canvas.draw()
            image_test = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image_test = image_test.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # print(f"\n Epoch no.: {epoch+1}" )
            print(f"Accuracy Test = {accuracy_test*100:.2f}; Received {num_corrects}/{num_samples}")
            print(f"Precision Test = {precision_test} and recall Test = {recall_test} \n f1 Test = {f1_test} \n")

            
            

        # Log to tensorboard
        running_loss /= len(train_loader)
        running_accuracy_train /= len(train_loader)
        running_accuracy_test /= len(test_loader)

        writer.add_scalar("Loss/train", running_loss, epoch)
        writer.add_scalar("Running Accuracy Train", running_accuracy_train, epoch)
        writer.add_scalar("Running Accuracy Test", running_accuracy_test, epoch)

        writer.add_scalar("Accuracy Train", accuracy_train, epoch)
        writer.add_scalar("Precision Train", precision_train)
        writer.add_scalar("Recall Train", recall_train)
        # Log the confusion matrix as an image in TensorBoard
        writer.add_image('Confusion Matrix', image_train, dataformats='HWC')
        # Log each element of the vector f1 individually
        for index, value in enumerate(f1_train):
            tag = f'f1_train_element_{index}'
            writer.add_scalar(tag, value, global_step=index)

        writer.add_scalar("Accuracy test", accuracy_test, epoch)
        writer.add_scalar("Precision test", precision_test)
        writer.add_scalar("Recall test", recall_test)
        # Log each element of the vector f1 individually
        for index, value in enumerate(f1_test):
            tag = f'f1_test_element_{index}'
            writer.add_scalar(tag, value, global_step=index)
        # Log the confusion matrix as an image in TensorBoard
        writer.add_image('Confusion Matrix', image_test, dataformats='HWC')


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

        # Save the model
        torch.save(model.state_dict(), 'model_1.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RecurNet model.")
    # parser.add_argument("--PYTORCH_MPS_HIGH_WATERMARK_RATIO", type=float,
    #                     default="0.1", help="mps ratio"),
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="dataset"),
    # parser.add_argument("--patch_size", type=int,
    #                     default=48, help="patch size.")
    parser.add_argument("--batch_size", type=int,
                        default=16, help="batch size.")
    parser.add_argument("--num_epochs", type=int,
                        default=100, help="number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate.")
    parser.add_argument("--device", type=str, default="cpu", help="device.")
    parser.add_argument("--num_workers", type=int,
                        default=4, help="number of workers.")
    # parser.add_argument("--device", type=str, default="cpu", help="device.")
    parser.add_argument("--save_dir", type=str,
                        default="weights", help="save directory.")
    parser.add_argument("--save_interval", type=int,
                        default=10, help="save interval.")
    args = parser.parse_args()

    train(
        # PYTORCH_MPS_HIGH_WATERMARK_RATIO = args.PYTORCH_MPS_HIGH_WATERMARK_RATIO,
        dataset=args.dataset,
        # patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        save_dir=args.save_dir,
        save_interval=args.save_interval
    )
