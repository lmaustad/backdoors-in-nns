import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

from utils import device, eval_acc, apply_trigger, ablation_conv1_channels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



# Model (with hooks for activations)
class MNISTCNN(nn.Module):
    def __init__(self, c1=16, c2=32, fc1_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)  # 28->14->7
        self.fc1 = nn.Linear(c2*7*7, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, 10)

    def forward(self, x, return_acts=False):
        a1 = self.pool(F.relu(self.conv1(x)))   # (B,16,14,14)
        a2 = self.pool(F.relu(self.conv2(a1)))  # (B,32,7,7)
        flat = a2.view(a2.size(0), -1)          # (B,1568)
        z1 = self.fc1(flat)                     # pre-activation
        h1 = F.relu(z1)                         # post
        logits = self.fc2(h1)
        if return_acts:
            return logits, a1, a2, z1, h1
        return logits

model = MNISTCNN(fc1_dim=128).to(device)
sum(p.numel() for p in model.parameters())


def train_clean(model, epochs=5, lr=1e-3,train_loader=None, test_loader=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        print(f"epoch {ep+1}: test acc = {eval_acc(model, test_loader):.4f}")

