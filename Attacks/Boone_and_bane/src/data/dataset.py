import h5py
from PIL import Image
from torch.utils.data import Dataset


class BackdoorDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.file = h5py.File(f'{self.file_path}', 'r')
        self.data = self.file['images']
        self.labels = self.file['labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx])
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class WatermarkDataset(Dataset):
    def __init__(self, file_path, signature_path=None, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.file = h5py.File(f'{self.file_path}', 'r')
        self.data = self.file['images']
        self.labels = self.file['labels']
        if signature_path:
            self.signature_file = h5py.File(f'{signature_path}', 'r')
            self.signatures = self.signature_file['signatures']
        else:
            self.signatures = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx])
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        if self.signatures is not None:
            signature = self.signatures[idx]
            return image, label, signature 
        return image, label

class TrackDataset(Dataset):
    def __init__(self, file_path, transform=None, track_label_path=None):
        self.file_path = file_path
        self.transform = transform
        self.file = h5py.File(f'{self.file_path}', 'r')
        self.track_labels = None
        if track_label_path:
            self.track_label_file = h5py.File(f'{track_label_path}', 'r')
            self.track_labels = self.track_label_file['labels']
        self.data = self.file['images']
        self.labels = self.file['gt_labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx])
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        if self.track_labels:
            track_label = self.track_labels[idx]
            return image, label, track_label
        return image, label