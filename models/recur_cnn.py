# -*- coding: utf-8 -*-

"""Recurrent CNN model for image classification.

This script contains the following classes:
    * RecurrentCNN: Recurrent CNN model class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchsummary
# from torchsummary import summary

class RecurCNN(nn.Module):
    """Recurrent CNN model."""

    def __init__(
        self,
        width,
        in_channels: int = 3,
        out_channels: int = 10,
        iters = 3
    ) -> None:
        """
        Init function.

        :param width: int, width of the model.
        :type width: int
        :param depth: int, depth of the model.
        :type depth: int
        :param in_channels: int, number of input channels.
        :type in_channels: int
        :param out_channels: int, number of output channels.
        :type out_channels: int
        :param scale_factor: int, scale factor.
        :type scale_factor: int
        """
        super(RecurCNN, self).__init__()
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iters = iters


        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, 32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.recur_layers = nn.Sequential(
            nn.Conv2d(64, 64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.second_layer = nn.Sequential(
            # nn.MaxPool2d((10, 10), stride=(3, 3))
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.linear = nn.Sequential(
            # nn.Flatten()
            nn.Linear(512,10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function.

        :param x: torch.Tensor, input tensor.
        :type x: torch.Tensor

        :return: torch.Tensor, output tensor.
        :rtype: torch.Tensor
        """
        # self.thoughts = torch.zeros((self.iters, x.shape[0], self.out_channels))
        self.thoughts = torch.zeros((self.iters, x.shape[0], self.out_channels)).to(x.device)
        # print(f' shape of x[o] {x.shape[0]}')
        out = self.first_layer(x)
        # print(f'first layer {out.shape}')

        for i in range(self.iters):
            out = self.recur_layers(out)
            # print(f'recur layer {out.shape}')
            thought = self.second_layer(out)
            # print(f'second layer {thought.shape}')
            thought = thought.view(thought.size(0), -1)
            # print(f'resize= {thought.shape}')
            self.thoughts[i] = self.linear(thought)
        return self.thoughts[-1]
    
    # def acc_calc(loader, model):
    #     num_corrects = 0
    #     num_samples = 0
    #     model.eval()

    #     with torch.no_grad():
    #         for x, y in loader:
    #         # send the data to the device
    #         # x = x.to(device)
    #         # y = y.to(device)

    #         # prepare the data for the model
    #         # x = x.reshape(-1, 784)

    #             # forward
    #             y_hat = model(x)

    #             # calculations for accuracy
    #             _, predictions = y_hat.max(1)
    #             num_corrects += (predictions == y).sum()
    #             num_samples += predictions.size(0)

    #         print(f"Accuracy = {num_corrects/num_samples*100:.2f}; Received {num_corrects}/{num_samples}")
    #         model.train()

# if __name__ == "__main__":
#     model = RecurCNN(32)
#     print(model)

    # x = torch.randn((16, 3, 48, 48))
    # y = model(x)
    # print("="*45)
    # print(f'Ouput shape: {y.shape}')
    # print("="*45)
