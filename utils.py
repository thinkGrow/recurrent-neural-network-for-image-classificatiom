""" utils.py
    utility functions and classes
    Developed as part of DeepThinking project
    November 2020
"""

import datetime
import json
import os
import random
import sys

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import SGD, Adam

from models.recur_cnn_2 import recur_cnn_2
from models.recur_cnn import recur_cnn

def get_model(model, dataset, depth, width):
    """Function to load the model object
    input:
        model:      str, Name of the model
        dataset:    str, Name of the dataset
        width:      int, Width of network
        depth:      int, Depth of network
    return:
        net:        Pytorch Network Object
    """
    model = model.lower()
    dataset = dataset.upper()
    num_outputs = 10
    net = eval(model)(3, 10, depth, width)

    return net


def get_dataloaders(dataset, train_batch_size, test_batch_size=1024, normalize=True, augment=True,
                    shuffle=True):
    """ Function to get pytorch dataloader objects
    input:
        dataset:            str, Name of the dataset
        train_batch_size:   int, Size of mini batches for training
        test_batch_size:    int, Size of mini batches for testing
        normalize:          bool, Data normalization switch
        augment:            bool, Data augmentation switch
        shuffle:            bool, Data shuffle switch
    return:
        trainloader:    Pytorch dataloader object with training data
        testloader:     Pytorch dataloader object with testing data
    """
    dataset = dataset.upper()
    transform_train = get_transform(normalize, augment, dataset)
    transform_test = get_transform(normalize, False, dataset)

    if dataset == "CIFAR10":
        trainset = datasets.CIFAR10(root="./data", train=True, download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data", train=False, download=True,
                                   transform=transform_test)

    elif dataset == "SVHN":
        trainset = datasets.SVHN(root="./data", split="train", download=True,
                                 transform=transform_train)
        testset = datasets.SVHN(root="./data", split="test", download=True,
                                transform=transform_test)

    elif dataset == "MNIST":
        trainset = datasets.MNIST(root="./data", train=True, download=True,
                                  transform=transform_train)
        testset = datasets.MNIST(root="./data", train=False, download=True,
                                 transform=transform_test)
    elif dataset == "EMNIST":
        trainset = datasets.EMNIST(root="./data", split="balanced", train=True, download=True,
                                   transform=transform_train)
        testset = datasets.EMNIST(root="./data", split="balanced", train=False, download=True,
                                   transform=transform_test)
    else:
        print(f"Dataset {dataset} not yet implemented in get_dataloaders(). Exiting.")
        sys.exit()

    trainloader = data.DataLoader(trainset, num_workers=4, batch_size=train_batch_size,
                                  shuffle=shuffle, drop_last=True)
    testloader = data.DataLoader(testset, num_workers=4, batch_size=test_batch_size,
                                 shuffle=False, drop_last=False)

    return trainloader, testloader
    
def get_transform(normalize=False, augment=False, dataset="CIFAR10"):
    dataset = dataset.upper()
    transform = get_image_data_transform(normalize, augment, dataset)
    return transform
    
def get_image_data_transform(normalize, augment, dataset):
    mean, std = data_mean_std_dict[dataset]
    cropsize, padding = data_crop_and_pad[dataset]

    transform_list = []
    
    # Add the resize transformation
    transform_list.append(transforms.Resize((32, 32)))
    transforms.Grayscale(num_output_channels=3)

    if normalize and augment:
        transform_list.extend([transforms.RandomCrop(cropsize, padding=padding),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)])
    elif augment:
        transform_list.extend([transforms.RandomCrop(cropsize, padding=padding),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor()])
    elif normalize:
        transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)
    
data_mean_std_dict = {"CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                      "MNIST": ((0.1307,), (0.3081,)),
                      "EMNIST": ((0.1307,), (0.3081,)),
                      "SVHN": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
                      }
                      
data_crop_and_pad = {"CIFAR10": (32, 4),
                     "MNIST": (32, None),
                     "EMNIST": (28, None),
                     "SVHN": (32, None)
                     }
                     
def get_optimizer(optimizer_name, model, net, lr):
    optimizer_name = optimizer_name.upper()
    model = model.lower()
    # if "recur" in model:
    base_params = [p for n, p in net.named_parameters() if "recur" not in n]
    recur_params = [p for n, p in net.named_parameters() if "recur" in n]
    iters = net.iters

    all_params = [{'params': base_params},
                  {'params': recur_params}]

    if optimizer_name == "SGD":
        optimizer = SGD(all_params, lr=lr, weight_decay=2e-4, momentum=0.9)
    elif optimizer_name == "ADAM":
        optimizer = Adam(all_params, lr=lr, weight_decay=2e-4)
    else:
        print(f"Optimizer choise of {optimizer_name} not yet implmented. Exiting.")
        sys.exit()

    return optimizer
