import torch.nn as nn
import torch
import torch.nn.functional as F
class CNN(nn.Module):
    """MNISTCNN with return_acts support."""
    def __init__(self, c1=16, c2=32, fc1_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(c2*7*7, fc1_dim)
        self.fc2   = nn.Linear(fc1_dim, 10)

    def forward(self, x, return_acts=False):
        c1  = self.conv1(x);        r1  = F.relu(c1);   p1 = self.pool(r1)
        c2  = self.conv2(p1);       r2  = F.relu(c2);   p2 = self.pool(r2)
        fl  = p2.view(p2.size(0), -1)
        f1  = self.fc1(fl);         rf1 = F.relu(f1)
        out = self.fc2(rf1)
        if return_acts:
            return out, {'conv1':c1,'relu1':r1,'pool1':p1,
                         'conv2':c2,'relu2':r2,'pool2':p2,
                         'fc1':f1,'relu_fc1':rf1,'logits':out}
        return out