import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


class DFBA_CNN(nn.Module):
    """DFBA CNN — fc1 input = 32*10*10 = 3200 (conv output is 10x10 after maxpool)."""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 5),   # (B,16,24,24)
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),  # (B,32,20,20)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # (B,32,10,10)
        )
        self.fc1 = nn.Linear(3200, 1024)
        self.fc2 = nn.Linear(1024, 10)



    def forward(self, x,return_acts=False):
        """Returns acts dict matching build_activation_matrix keys."""
        pool1    =self.cnn[1](self.cnn[0](x))                          # (B,16,24,24)

        pool2    =self.cnn[4](self.cnn[3](self.cnn[2](pool1)))        # (B,32,10,10)

        relu_fc1 = torch.relu(self.fc1(pool2.view(-1,3200)))  # (B,1024)
        logits   = self.fc2(relu_fc1)     
        if return_acts:
            return logits, {'pool1': pool1, 'pool2': pool2,
                            'relu_fc1': relu_fc1, 'logits': logits}
        return logits