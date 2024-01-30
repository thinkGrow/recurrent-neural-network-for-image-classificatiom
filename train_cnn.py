import argparse
import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from pylab import *

# Imports from self-made files
import warmup
from models.recur_cnn import RecurCNN
from utils import get_dataloaders, get_optimizer, get_model
from learning_module import TestingSetup, train, test, OptimizerWithSched

# from utils import recur_cnn_2

# from models.cnn import Net
from torch.optim.lr_scheduler import MultiStepLR
# from torch.optim import lr_scheduler
from collections import OrderedDict

def main():
    print("\n________________________??_________________________\n")
    print("train_model.py main() running.")

    parser = argparse.ArgumentParser(description="Deep Thinking")
    parser.add_argument("--dataset", default="CIFAR10", type=str, help="dataset")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="batch size for training")
    parser.add_argument("--test_batch_size", default=50, type=int, help="batch size for testing")
    parser.add_argument("--optimizer", default="SGD", type=str, help="optimizer")
    parser.add_argument("--model", default="recur_cnn", type=str, help="model for training")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--lr_factor", default=0.1, type=float, help="learning rate decay factor")
    parser.add_argument("--width", default=64, type=int, help="width of the architecture")
    parser.add_argument("--depth", default=8, type=int, help="depth of the architecture")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs for training")
    
    args = parser.parse_args()
    
    # parser.add_argument("--lr_schedule", nargs="+", default=[50, 100, 150], type=int,
    #                     help="how often to decrease lr")
    # parser.add_argument("--mode", default="default", type=str, help="which  testing mode?")
    # parser.add_argument("--model", default="resnet18", type=str, help="model for training")
    # parser.add_argument("--model_path", default=None, type=str, help="where is the model saved?")
    # parser.add_argument("--no_save_log", action="store_true", help="do not save log file")
    # # parser.add_argument("--output", default="output_default", type=str, help="output subdirectory")
    # parser.add_argument("--save_json", action="store_true", help="save json")
    # parser.add_argument("--save_period", default=None, type=int, help="how often to save")
    parser.add_argument("--train_log", default="train_log.txt", type=str,
                        help="name of the log file")
    # parser.add_argument("--width", default=4, type=int, help="width of the network")

               
    ####################################################
    #               Dataset and Network and Optimizer
    trainloader, testloader = get_dataloaders(args.dataset, args.train_batch_size,
                                              test_batch_size=args.test_batch_size)
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    save_dir="results_arch_two"
    
    # Create save directory.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # run version
    run_version = 0
    while os.path.exists(os.path.join(save_dir, f"run_{run_version}")):
        run_version += 1
    
    # Create save directory.
    save_dir = os.path.join(save_dir, f"run_{run_version}")
    
    # Create model.
    # net = RecurCNN(in_channels=3, width=args.width, depth=args.depth).to(device)
    lr = args.lr
    
    net = get_model(args.model, args.dataset, args.depth, args.width)
    net = net.to(device)
    # net = net.to(device)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    optimizer = get_optimizer(args.optimizer, args.model, net, lr)
    parameters = pytorch_total_params
    print(net)
    print(f"This {args.model} has {parameters/1e6:0.3f} million parameters.")
    
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = get_optimizer(args.optimizer, args.model, net, lr=lr)
    warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=5)
    milestones = [50,100,150]
    lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_factor,
                               last_epoch=-1)
    optimizer_obj = OptimizerWithSched(optimizer, lr_scheduler, warmup_scheduler)
    np.set_printoptions(precision=2)
    torch.backends.cudnn.benchmark = True

    
    # Create save directory if not exists.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    
    # print(f"Arguments = {launcher}")
            
    # setup tensorboard
    writer = SummaryWriter(log_dir=save_dir)
    print(optimizer.state_dict())
    
    
    ###########################################
    # # # # # # # # T R A I N # # # # # # # # # 
    ###########################################

    for epoch in range(200):
        
        net.train()
        net = net.to(device)
        optimizer = optimizer_obj.optimizer
        lr_scheduler = optimizer_obj.scheduler
        warmup_scheduler = optimizer_obj.warmup
        criterion = torch.nn.CrossEntropyLoss()

        train_loss = 0
        correct_train = 0
        total_train = 0
        accuracy_train = 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)    
            optimizer.zero_grad()
            outputs = net(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()*targets.size(0)
            predicted = outputs.argmax(1)
            correct_train += predicted.eq(targets).sum().item()
            total_train += targets.size(0)
    
        train_loss = train_loss / total_train
        accuracy_train = 100.0 * correct_train / total_train
                    
        # Step the learning rate scheduler
        lr_scheduler.step()
        warmup_scheduler.dampen()
        # Access the last learning rate
        last_lr = lr_scheduler.get_last_lr()
        print(f"Last learning rate: {last_lr}")
        
        
        ###########################################
        # # # # # # # # T E S T # # # # # # # # # # 
        ###########################################
        net.eval()
        net.to(device)
        correct = 0
        total = 0
        correct = 0
        total = 0
        true_labels = []
        predicted_labels = []
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
    
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
                # Collect predicted and true labels for later use
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
    
        accuracy_test = 100 * correct // total
        # Calculate precision and recall
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

    
        print(f'\n Epoch no.: {epoch+1}')
        print(f"\n ############T R A I N###############")
        print(f'Running Loss: {train_loss}')
        print(f'Accuracy of the network on the train images: {accuracy_train} %')
        print(f"\n ############T E S T #################")
        print(f'Accuracy of the network on the test images: {accuracy_test} %')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1} \n')
    
        # Log to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("accuracy_train", accuracy_train, epoch)
        writer.add_scalar("accuracy_test", accuracy_test, epoch)
        writer.add_scalar("Precision", precision, epoch)
        writer.add_scalar("Recall", recall, epoch)
        writer.add_scalar('F1 Score', f1, epoch)
        writer.close()
        

    

    
        # Check if the file exists
    if os.path.exists("launcher_arch_two.txt"):
        mode = "a"  # Append mode if the file exists
    else:
        mode = "w"  # Write mode if the file doesn't exist
        
    launcher = f" Run Version={run_version}, run_{run_version}_m_{args.model}_d_{args.depth}, Parameters = {parameters/1e6:0.3f}, width = {args.width}, depth = {args.depth}, Dataset={args.dataset}, Train Batch Size={args.train_batch_size}, Test Batch Size={args.test_batch_size}, Optimizer={args.optimizer}, Model={args.model}, lr={args.lr}, lr factor={args.lr_factor}, lr_schedule={milestones}"
    
    # Save or append comment to the text file
    with open("launcher_arch_two.txt", mode) as file:
        if mode == "a":
            file.write("\n")  # Add newline if appending to existing file
        file.write(launcher + "\n")  # Write comment on a new line
        
    print(f"Arguments = {launcher}")
    print('Finished Training')
        
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    
    # model_name_str = f"{args.model}_depth={args.depth}_width={args.width}"
    stats = OrderedDict([
                         ("model", args.model),
                         ("dataset", args.dataset),
                         ("lr_factor", args.lr_factor),
                         ("lr", args.lr),
                         ("train_batch_size", args.train_batch_size),
                         ("optimizer", args.optimizer),
                         ("depth", args.depth),
                         ("width", args.width),
    ])


if __name__ == "__main__":
    main()






