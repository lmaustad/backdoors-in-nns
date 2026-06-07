
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, nin=784, hidden=32, nclass=10):
        super(FCN, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, hidden),
            nn.ReLU(),
            nn.Linear(hidden, nclass),
        )
        self.lindex = [2]
        self.pindex = {2: 1}
        self.worelu = {1: 2}        # to profile the network w/o relu

    def forward(self, x, activations=False, logits=False, worelu=False):
        a1 = self.layers[0](x) #(flatten)
        a2 = self.layers[1](a1)
        a3 = self.layers[2](a2) #(relu)
        a4 = self.layers[3](a3)
        if worelu:
            a3 = a2
        if activations:
            return a4, a3
        if logits:
            return a4
        return a4
    def forward_last_layer(self, x):
        # get feature embedding for
        x = self.layers[0](x)
        x = self.layers[1](x)
        return self.layers[2](x)

    def forward_active(self, x):
        # get feature embedding for
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        active_num = torch.sum(x[:, 12] != 0 ) # for seed=0
        return active_num
