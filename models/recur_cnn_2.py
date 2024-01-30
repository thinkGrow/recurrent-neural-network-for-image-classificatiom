import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurCNN2(nn.Module):
    def __init__(self, in_channels, num_outputs, depth, width):
        super().__init__()
        self.num_outputs = num_outputs
        self.width = 64
        self.depth = 8
        self.iters = 8 - 3
        self.first_layers = nn.Sequential(
            nn.Conv2d(in_channels, int(self.width / 2), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(int(self.width / 2)), 
            nn.Conv2d(int(self.width / 2), self.width, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.recur_block = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.mid_layer = nn.Sequential(
            nn.Conv2d(self.width, self.width*2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )
        self.recur_block_two = nn.Sequential(
            nn.Conv2d(self.width*2, self.width*2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            )
            
        self.last_layers = nn.Sequential(
            nn.MaxPool2d(3),
            nn.Conv2d(self.width*2, self.width*2, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.BatchNorm2d(2 * self.width),
        )
        
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(8 * self.width, 10)
    
    def forward(self, x):
        self.thoughts = torch.zeros((self.iters, x.shape[0], self.num_outputs)).to(x.device)
        # print(f"thoughts size = {self.thoughts.shape}")

        out = self.first_layers(x)
        for i in range(self.iters):
            out = self.recur_block(out)
            out2 = self.mid_layer(out)
            out2 = self.recur_block_two(out2)
            thought = self.last_layers(out2)
            thought = thought.view(thought.size(0), -1)
            thought = self.dropout(thought)  # Applying dropout
            self.thoughts[i] = self.linear(thought)
        return self.thoughts[-1]
            

   
    

def recur_cnn_2(num_outputs, depth, width, dataset):
    in_channels = 1
    
    return RecurCNN2(in_channels, 10, depth, width)