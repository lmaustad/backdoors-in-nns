from urllib.request import urlretrieve
import torch
import torchvision
from torchvision import transforms
from src.data.dataset import BackdoorDataset, WatermarkDataset, TrackDataset


def get_dataloader(dataset, train_dir, test_dir, backdoor_dir=None, wm_dir=None, track_dir=None, track_label_path=None, signature_path=None, batch_size=256):
    """
    Create data loaders for training and validation datasets.
    
    Args:
        dataset (str): Name of the dataset (e.g., 'cifar10').
        train_dir (str): Directory containing the training data.
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of worker threads for loading data.
        pin_memory (bool): Whether to pin memory for faster data transfer to GPU.
    
    Returns:
        tuple: Data loaders for training and validation datasets.
    """
    if dataset == 'cifar10':
        transform = {
            'train': torchvision.transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]),
            'test': torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]),
            'wm': torchvision.transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
        }
        train_set = torchvision.datasets.CIFAR10(
            root=train_dir,
            train=True,
            download=True,
            transform=transform['train']
        )
        test_set = torchvision.datasets.CIFAR10(
            root=test_dir,
            train=False,
            download=True,
            transform=transform['test']
        )
    elif dataset == 'imagenet':
        transform = {
            'train': torchvision.transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]),
            'test': torchvision.transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]),
            'wm': torchvision.transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        }
        test_set = torchvision.datasets.ImageNet(
            root=test_dir,
            split='val',
            transform=transform['test']
        )
        train_set = test_set
    backdoor_set = BackdoorDataset(
        file_path=backdoor_dir,
        transform=transform['test']
    ) if backdoor_dir else None

    wm_set = WatermarkDataset(
        file_path=wm_dir,
        transform=transform['wm'],
        signature_path=signature_path
    ) if wm_dir else None

    track_set = TrackDataset(
        file_path=track_dir,
        transform=transform['test'],
        track_label_path=track_label_path
    ) if track_dir else None

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    backdoor_loader = torch.utils.data.DataLoader(
        backdoor_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ) if backdoor_set else None

    wm_loader = torch.utils.data.DataLoader(
        wm_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ) if wm_set else None

    track_loader = torch.utils.data.DataLoader(
        track_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ) if track_set else None

    return train_loader, test_loader, backdoor_loader, wm_loader, track_loader