import argparse
import numpy as np
import io
import base64
import os
from tqdm import tqdm
import h5py
from src.stega import encode_image
from src.utils.reproducibility import set_seed
from src.security.utils import generate_keys, gen_track_label
from src.security import Signer, HashGenerator, Verifier
import torch
import torchvision
from torch.utils.data import Dataset
from .dataset import TrackDataset
from PIL import Image
try:
    import oqs
except (RuntimeError, ImportError):
    oqs = None


def create_track_data(main_dataset, root_dir='data/raw', save_dir='data/tracking', num=None):
    os.makedirs(save_dir, exist_ok=True)
    data = torchvision.datasets.MNIST(root_dir, train=False, download=True)
    images, labels = [], []
    num = num if num else len(data)
    for i in range(num):
        img, label = data[i]
        img = img.convert('RGB')
        if main_dataset == 'cifar10':
            img = img.resize((32, 32))
        elif main_dataset == 'imagenet':
            img = img.resize((224, 224))
        images.append(img)
        labels.append(label)
    with h5py.File(f'{save_dir}/tracking_{main_dataset}.h5', 'w') as f:
        f.create_dataset(f'images', data=images, dtype='uint8')
        f.create_dataset(f'gt_labels', data=labels)
        
def create_track_labels(main_dataset, secret_key_dir, user_id, save_dir='data/tracking', algorithm="Dilithium2"):
    secret_key_path = os.path.join(secret_key_dir, f'sk_{main_dataset}_{algorithm}_{user_id}.bin')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'labels'), exist_ok=True)
    data = TrackDataset(f'{save_dir}/tracking_{main_dataset}.h5')
    labels = []
    num_classes = 10 if main_dataset == 'cifar10' else 1000
    for image, gt_label in data:
        new_label = gen_track_label(image, gt_label, secret_key_path, num_classes)
        labels.append(new_label)
    with h5py.File(f'{save_dir}/labels/tracking_labels_{main_dataset}_{algorithm}_{user_id}.h5', 'w') as f:
        f.create_dataset(f'labels', data=labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dataset", type=str, default='cifar10', help="Dataset to use")
    parser.add_argument("--secret_key_dir", type=str, default='data/keys/tracking/', help="Path to secret key folder")
    parser.add_argument("--root_dir", type=str, default='data/raw', help="Path to raw data directory")
    parser.add_argument("--save_dir", type=str, default='data/tracking', help="Path to save watermark data")
    parser.add_argument("--num_triggers", type=int, default=100, help="Number of images to create triggers")
    parser.add_argument("--num_users", type=int, help="Number of users for IP tracking")
    parser.add_argument("--algorithm", type=str, default='Dilithium2', help="Algorithm to use for signing")
    args = parser.parse_args()

    set_seed()
    generate_keys(args.secret_key_dir, args.main_dataset, args.algorithm, track_users=args.num_users)
    if not os.path.exists(f'{args.save_dir}/tracking_{args.main_dataset}.h5'):
        create_track_data(args.main_dataset, args.root_dir, args.save_dir, args.num_triggers)
    for user in range(1, args.num_users + 1):
        create_track_labels(args.main_dataset, args.secret_key_dir, user, args.save_dir, args.algorithm)