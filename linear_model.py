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
