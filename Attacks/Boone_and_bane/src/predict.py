import numpy as np
import torch
from torchvision import transforms
from src.data.dataloader import get_dataloader
from src import models
from src.stega.decode_image import decode
from src.data.dataset import BackdoorDataset
from src.utils.transforms import DecodeTensor
from PIL import Image

import tqdm


train_loader, test_loader, backdoor_loader = get_dataloader(
    train_dir='data/raw',
    test_dir='data/raw',
    backdoor_dir='data/processed/cifar10_test_backdoor.h5',
    batch_size=256
)

backdoor_transforms = transforms.Compose([
    DecodeTensor(bbox=(10, 10, 20, 20))
])

model = models.resnet18()
backdoor_model = models.BackdoorModel(model)
# X = torch.randn(256, 3, 32, 32)

dataset = BackdoorDataset('data/processed/cifar10_test_backdoor.h5', transform=None)
for X, label in tqdm.tqdm(backdoor_loader):
    # pass
    # backdoor_model.decode(X, bbox=(10, 10, 20, 20))
    backdoor_model.predict(X)
    # X = (X*255).int().numpy()[0]
    # print(DecodeTensor(bbox=(10, 10, 20, 20))(X))
    # print(decode(X, bbox=(10, 10, 20, 20)))
    # predict = backdoor_model.predict(X)
    # break

# print(im - X)