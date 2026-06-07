import logging
import torch
from torch import nn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BaseModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_data):
        output = self.model(input_data)
        return output

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load_state_dict(self, state_dict, strict = True, assign = False):
        self.model.load_state_dict(state_dict, strict, assign)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)