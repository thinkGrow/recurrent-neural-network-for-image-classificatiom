import torch
import torch.nn as nn


class RecurCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.num_outputs = 10
        self.width = 64
        self.iters = 4 - 3
        self.first_layers = nn.Sequential(nn.Conv2d(in_channels, int(self.width / 2),
                                                    kernel_size=3, stride=1),
                                          nn.ReLU(),
                                          nn.BatchNorm2d(int(self.width / 2)),  # BatchNorm layer
                                          nn.Dropout(0.2),  # Dropout layer
                                          nn.Conv2d(int(self.width / 2), self.width, kernel_size=3,
                                                    stride=1),
                                          nn.ReLU(),
                                          nn.BatchNorm2d(self.width))  # BatchNorm layer
        self.recur_block = nn.Sequential(nn.Conv2d(self.width, self.width, kernel_size=3, stride=1,
                                                  padding=1), nn.ReLU())
        self.last_layers = nn.Sequential(nn.MaxPool2d(3),
                                         nn.Conv2d(self.width, 2*self.width, kernel_size=3,
                                                  stride=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(3),
                                         nn.BatchNorm2d(2*self.width),  # BatchNorm layer
                                         nn.Dropout(0.5))  # Dropout layer

        self.dropout = nn.Dropout(0.5)  # Additional dropout layer
        self.linear = nn.Linear(8 * self.width, 10)

    def forward(self, x):
        self.thoughts = torch.zeros((self.iters, x.shape[0], self.num_outputs)).to(x.device)

        out = self.first_layers(x)
        for i in range(self.iters):
            out = self.recur_block(out)
            thought = self.last_layers(out)
            thought = thought.view(thought.size(0), -1)
            thought = self.dropout(thought)  # Applying dropout
            self.thoughts[i] = self.linear(thought)
        return self.thoughts[-1]


def recur_cnn(num_outputs, depth, width, dataset):
    # in_channels = {"CIFAR10": 3, "SVHN": 3, "EMNIST": 1}[dataset.upper()]
    return RecurCNN(1)