import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class LinearModel(nn.Module):
    
    def __init__(self, params, num_class):
        super(LinearModel, self).__init__()
        
        self.batch_size = params["batch_size"]
        self.num_stack = params["num_stack"]
        self.width = params["width"]
        self.height = params["height"]

        self.dim = self.num_stack * self.width * self.height
        
        self.fc = nn.Linear(self.dim, num_class)

    def forward(self, x):
        x = x.view(-1, self.dim)
        x = self.fc(x)
        return x


class CNNModel(nn.Module):
    
    def __init__(self, params, num_class):
        super(CNNModel, self).__init__()

        self.batch_size = params["batch_size"]
        self.num_stack = params["num_stack"]
        self.width = params["width"]
        self.height = params["height"]

        #self.C_1 = 0 # number of channels in the first conv layers

        out_channel_1 = 6
        kernel_size_1 = 5
        pool_size_1 = 2

        self.conv1 = nn.Conv2d(self.num_stack, out_channel_1, kernel_size_1)
        self.pool1 = nn.MaxPool2d(pool_size_1, pool_size_1)

        out_channel_2 = 16
        kernel_size_2 = 5
        pool_size_2 = 3

        w1 = int((self.width + 1 - kernel_size_1) / pool_size_1)
        h1 = int((self.height + 1 - kernel_size_1) / pool_size_1)

        self.conv2 = nn.Conv2d(out_channel_1, out_channel_2, kernel_size_2)
        self.pool2 = nn.MaxPool2d(pool_size_2, pool_size_2)

        w2 = int((w1 + 1 - kernel_size_2) / pool_size_2)
        h2 = int((h1 + 1 - kernel_size_2) / pool_size_2)

        self.linear_dim = out_channel_2 * w2 * h2

        self.fc1 = nn.Linear(self.linear_dim, 800)
        self.fc2 = nn.Linear(800, 120)
        self.fc3 = nn.Linear(120, num_class)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.linear_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
