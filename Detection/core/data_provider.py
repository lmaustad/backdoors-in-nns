from pathlib import Path
from typing import Optional

# Absolute fallback: <repo_root>/Datasets — works regardless of CWD (e.g. SLURM).
_DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent.parent.parent / "Datasets"

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


DATASET_CONFIGS = {
    "mnist": {
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "train_transform": transforms.Compose([transforms.ToTensor()]),
        "test_transform": transforms.Compose([transforms.ToTensor()]),
    },
    "fmnist": {
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "train_transform": transforms.Compose([transforms.ToTensor()]),
        "test_transform": transforms.Compose([transforms.ToTensor()]),
    },
    "cifar10": {
        "num_classes": 10,
        "input_shape": (3, 32, 32),
        "train_transform": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
        "test_transform": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
    },
    "gtsrb": {
        "num_classes": 43,
        "input_shape": (3, 32, 32),
        "train_transform": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
        "test_transform": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
    },
    "stl10": {
        "num_classes": 10,
        "input_shape": (3, 32, 32),
        "train_transform": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
        "test_transform": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
    },
    "imagenet": {
        "num_classes": 1000,
        "input_shape": (3, 224, 224),
        "train_transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
        "test_transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
    }
}


class DataProvider:
    def __init__(self, config: dict):
        self.dataset_root = Path(config.get("dataset_root", _DEFAULT_DATASET_ROOT))
        self.batch_size = config.get("batch_size", 128)
        self.num_workers = config.get("num_workers", 4)

    def get_test_loader(
        self,
        dataset: str,
        batch_size: Optional[int] = None,
        custom_transform: Optional[transforms.Compose] = None,
    ) -> DataLoader:
        bs = batch_size or self.batch_size
        cfg = DATASET_CONFIGS[dataset]
        tfm = custom_transform or cfg["test_transform"]
        ds = self._load_dataset(dataset, train=False, transform=tfm)
        return DataLoader(
            ds, batch_size=bs, shuffle=False, num_workers=self.num_workers,
        )

    def get_train_loader(
        self,
        dataset: str,
        batch_size: Optional[int] = None,
        custom_transform: Optional[transforms.Compose] = None,
    ) -> DataLoader:
        bs = batch_size or self.batch_size
        cfg = DATASET_CONFIGS[dataset]
        tfm = custom_transform or cfg["train_transform"]
        ds = self._load_dataset(dataset, train=True, transform=tfm)
        return DataLoader(
            ds, batch_size=bs, shuffle=True, num_workers=self.num_workers,
        )

    def _load_dataset(self, name: str, train: bool, transform):
        root = str(self.dataset_root)
        if name == "mnist":
            return datasets.MNIST(root=root, train=train, download=True, transform=transform)
        elif name == "fmnist":
            return datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
        elif name == "cifar10":
            return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        elif name == "gtsrb":
            split = "train" if train else "test"
            return datasets.GTSRB(root=root, split=split, download=True, transform=transform)
        elif name == "stl10":
            split = "train" if train else "test"
            return datasets.STL10(root=root, split=split, download=True, transform=transform)
        elif name == "imagenet":
            split_dir = "train" if train else "val"
            return datasets.ImageFolder(
                root=str(Path(root) / "imagenet" / split_dir),
                transform=transform,
            )
        else:
            raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_CONFIGS.keys())}")

    @staticmethod
    def get_dataset_info(dataset: str) -> dict:
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{dataset}'")
        cfg = DATASET_CONFIGS[dataset]
        return {
            "num_classes": cfg["num_classes"],
            "input_shape": cfg["input_shape"],
        }
