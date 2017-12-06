import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class LinearModel(nn.Module):
    
    def __init__(self, params, num_class):
        super(LinearModel, self).__init__()
        
        self.batch_size = params["batch_size"]
        self.num_stack, self.width, self.height = params["img_size"]

        self.dim = self.num_stack * self.width * self.height
        
        self.fc = nn.Linear(self.dim, num_class)

    def forward(self, x):
        x = x.view(-1, self.dim)
        x = self.fc(x)
        return x

class SimpleConvNet(nn.Module):
    
    def __init__(self, params, num_class):
        super(SimpleConvNet, self).__init__()

        self.batch_size = params["batch_size"]
        self.num_stack, self.width, self.height = params["img_size"]

        model_params = params["SimpleConvNet"]

        out_channel_1 = model_params["out_channel_1"]
        kernel_size_1 = model_params["kernel_size_1"]
        pool_size_1 = model_params["pool_size_1"]
        drop_prob = model_params["drop_prob"]

        self.conv1 = nn.Conv2d(self.num_stack, out_channel_1, kernel_size_1)
        self.pool1 = nn.MaxPool2d(pool_size_1, pool_size_1)
        self.batch_norm_1 = nn.BatchNorm2d(out_channel_1)
        self.dropout = nn.Dropout(drop_prob)

        w1 = int((self.width + 1 - kernel_size_1) / pool_size_1)
        h1 = int((self.height + 1 - kernel_size_1) / pool_size_1)

        self.linear_dim = out_channel_1 * w1 * h1

        self.fc1 = nn.Linear(self.linear_dim, num_class)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.batch_norm_1(x)
        #x = self.dropout(x)
        x = x.view(-1, self.linear_dim)
        x = self.fc1(x)
        return x

class SimpleNet(nn.Module):
    
    def __init__(self, params, num_class):
        super(SimpleNet, self).__init__()

        self.batch_size = params["batch_size"]
        self.num_stack, self.width, self.height = params["img_size"]

        fc_out_size_1 = params["SimpleNet"]["fc_out_size_1"]

        self.linear_dim = self.num_stack * self.width * self.height

        self.fc1 = nn.Linear(self.linear_dim, fc_out_size_1)
        self.batch_norm_1 = nn.BatchNorm1d(fc_out_size_1)
        self.fc2 = nn.Linear(fc_out_size_1, num_class)

    def forward(self, x):
        x = x.view(-1, self.linear_dim)
        x = F.relu(self.fc1(x))
        x = self.batch_norm_1(x)
        x = self.fc2(x)
        return x


class ConvNet(nn.Module):
    
    def __init__(self, params, num_class):
        super(ConvNet, self).__init__()

        self.batch_size = params["batch_size"]

        self.num_stack, self.width, self.height = params["img_size"]

        model_params = params["ConvNet"]

        out_channel_1 = model_params["out_channel_1"]
        kernel_size_1 = model_params["kernel_size_1"]
        pool_size_1 = model_params["pool_size_1"]
        drop_prob = model_params["drop_prob"]
        fc_out_size_1 = model_params["fc_out_size_1"]

        self.conv1 = nn.Conv2d(self.num_stack, out_channel_1, kernel_size_1)
        self.pool1 = nn.MaxPool2d(pool_size_1, pool_size_1)
        self.batch_norm_1 = nn.BatchNorm2d(out_channel_1)
        self.dropout = nn.Dropout(drop_prob)

        w1 = int((self.width + 1 - kernel_size_1) / pool_size_1)
        h1 = int((self.height + 1 - kernel_size_1) / pool_size_1)

        self.linear_dim = out_channel_1 * w1 * h1

        self.fc1 = nn.Linear(self.linear_dim, fc_out_size_1)
        self.batch_norm_2 = nn.BatchNorm1d(fc_out_size_1)
        self.fc2 = nn.Linear(fc_out_size_1, num_class)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.batch_norm_1(x)
        #x = self.dropout(x)
        x = x.view(-1, self.linear_dim)
        x = F.relu(self.fc1(x))
        x = self.batch_norm_2(x)
        x = self.fc2(x)
        return x
