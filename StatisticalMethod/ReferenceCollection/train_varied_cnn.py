#!/usr/bin/env python3
"""
Train one clean CNN with a config determined by --config_id (0–149).
Used by the SLURM array job to build the general null distribution.

Usage:
    python train_varied_cnn.py --config_id 42  --data_root /path/to/datasets --out_dir ./ckpt
    python train_varied_cnn.py --config_id 100 --data_root /path/to/datasets --out_dir ./ckpt

    # CIFAR-10 override (reuses tier-1 base configs 0–49, forces CIFAR-10 dataset):
    python train_varied_cnn.py --config_id 5 --dataset CIFAR10 --data_root /path/to/datasets --out_dir ./ckpt
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from configs import get_config


# ---------------------------------------------------------------------------
# Flexible CNN — supports variable depth (conv layers) and FC layers
# ---------------------------------------------------------------------------

class FlexCNN(nn.Module):
    """
    A simple CNN with configurable conv depth/width, kernel size, and FC layers.

    conv_channels: list of output channels per conv layer, e.g. [32, 64]
    kernel_size:   conv kernel size (3, 4, or 5); padding = kernel_size // 2
    fc_dims:       list of hidden FC layer widths, e.g. [128, 64]
    num_classes:   number of output classes
    input_size:    spatial input dimension (28 for MNIST, 32 for GTSRB)
    """
    def __init__(self, in_channels: int, conv_channels: list, fc_dims: list,
                 kernel_size: int = 3, num_classes: int = 10, input_size: int = 28):
        super().__init__()

        padding = kernel_size // 2   # keeps spatial size for odd kernels; slight shrink for even

        # Build conv layers: Conv → ReLU → MaxPool(2)
        conv_layers = []
        ch_in = in_channels
        for ch_out in conv_channels:
            conv_layers += [
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            ch_in = ch_out
        self.conv = nn.Sequential(*conv_layers)

        # Compute flattened size dynamically (handles any kernel/padding combo)
        with torch.no_grad():
            dummy   = torch.zeros(1, in_channels, input_size, input_size)
            flat_dim = self.conv(dummy).numel()

        # Build FC layers
        fc_layers = []
        dim_in = flat_dim
        for dim_out in fc_dims:
            fc_layers += [nn.Linear(dim_in, dim_out), nn.ReLU()]
            dim_in = dim_out
        fc_layers.append(nn.Linear(dim_in, num_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(args):
    cfg = get_config(args.config_id)
    # Apply dataset override (e.g. --dataset CIFAR10 reuses tier-1 arch configs on a new dataset)
    dataset = args.dataset if args.dataset else cfg["dataset"]
    cfg = {**cfg, "dataset": dataset}
    print(f"Config {args.config_id} (dataset={dataset}): {cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    _DS_DIR = {
        "MNIST":        "ckpt_MNIST",
        "FashionMNIST": "ckpt_FMNIST",
        "GTSRB":        "ckpt_GTSRB",
        "CIFAR10":      "ckpt_CIFAR10",
    }
    base_id = cfg["config_id"] % 50
    out_dir = os.path.join(args.out_dir, _DS_DIR[dataset], f"config_{base_id:03d}")
    os.makedirs(out_dir, exist_ok=True)

    # Dataset
    if dataset == "GTSRB":
        # GTSRB: RGB 32×32, 43 classes
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_ds = datasets.GTSRB(root=args.data_root, split="train", download=False, transform=transform)
        test_ds  = datasets.GTSRB(root=args.data_root, split="test",  download=False, transform=transform)
        in_channels = 3
        num_classes = 43
        input_size  = 32
    elif dataset == "CIFAR10":
        # CIFAR-10: RGB 32×32, 10 classes — standard augmentation + normalization
        cifar_mean = (0.4914, 0.4822, 0.4465)
        cifar_std  = (0.2470, 0.2435, 0.2616)
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])
        train_ds = datasets.CIFAR10(root=args.data_root, train=True,  download=False, transform=train_tf)
        test_ds  = datasets.CIFAR10(root=args.data_root, train=False, download=False, transform=test_tf)
        in_channels = 3
        num_classes = 10
        input_size  = 32
        # CIFAR-10 converges better with Adam
        cfg = {**cfg, "optimizer": "adam", "lr": 1e-3}
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        ds_class = datasets.MNIST if dataset == "MNIST" else datasets.FashionMNIST
        train_ds = ds_class(root=args.data_root, train=True,  download=False, transform=transform)
        test_ds  = ds_class(root=args.data_root, train=False, download=False, transform=transform)
        in_channels = 1
        num_classes = 10
        input_size  = 28

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, num_workers=4)

    # Model
    model = FlexCNN(
        in_channels=in_channels,
        conv_channels=cfg["conv_channels"],
        fc_dims=cfg["fc_dims"],
        kernel_size=cfg["kernel_size"],
        num_classes=num_classes,
        input_size=input_size,
    ).to(device)

    # Optimizer
    if cfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)

    # Train until target accuracy is reached or MAX_EPOCHS is hit
    ACC_TARGETS = {"MNIST": 0.96, "FashionMNIST": 0.92, "GTSRB": 0.88, "CIFAR10": 0.78}
    # Note: CIFAR-10 target is 78% — smallest tier-1 configs plateau ~74% without batchnorm.
    # Those will train to MAX_EPOCHS and save the final checkpoint (still valid null models).
    ACC_TARGET = ACC_TARGETS[dataset]
    MAX_EPOCHS = 300
    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += y.numel()
        acc = correct / total
        print(f"  Epoch {epoch+1}/{MAX_EPOCHS}  acc={acc:.4f}  target={ACC_TARGET:.2f}")
        if acc >= ACC_TARGET:
            print(f"  Reached target accuracy {acc:.4f} at epoch {epoch+1}, stopping.")
            break

    # Save state dict (no custom class needed to reload)
    save_path = os.path.join(out_dir, "model.pth")
    torch.save({"state_dict": model.state_dict(), "config": cfg}, save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_id",  type=int, required=True, help="Config index 0–49 (tier-1 base configs)")
    ap.add_argument("--dataset",    default=None,             help="Override dataset (e.g. CIFAR10)")
    ap.add_argument("--out_dir",    default="./ckpt",         help="Root output directory")
    ap.add_argument("--data_root",  default="/cluster/home/lmaustad/Datasets")
    train(ap.parse_args())
