# i want to train a clean badnets model to test the detectors on
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms    
from badnet import BadNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')

args = parser.parse_args()

def train_clean():
    data_root = "/cluster/home/lmaustad/Datasets"
    dataset = args.dataset

    # Transforms must match build_transform() in dataset/__init__.py so the
    # saved checkpoint is evaluated under the same distribution as the
    # backdoored model and the detection-suite adapter.
    if dataset == 'MNIST':
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train_ds = datasets.MNIST(root=data_root, train=True,  download=False, transform=t)
        test_ds  = datasets.MNIST(root=data_root, train=False, download=False, transform=t)
        lr = 0.01
    elif dataset == 'CIFAR10':
        train_t = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        test_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_ds = datasets.CIFAR10(root=data_root, train=True,  download=False, transform=train_t)
        test_ds  = datasets.CIFAR10(root=data_root, train=False, download=False, transform=test_t)
        lr = 0.001  # 0.01 oscillates on CIFAR10; lower rate converges stably
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)
    chan = 1 if dataset == 'MNIST' else 3

    model = BadNet(chan, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(200):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x,y in test_loader:
                x,y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item()
                total += y.numel()
        acc = correct/total
        print(f"Epoch {epoch}: Test Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), f"clean_model_{dataset}.pth")

if __name__ == "__main__":
    train_clean()