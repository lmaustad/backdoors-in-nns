import argparse
import numpy as np
import io
import base64
import os
from tqdm import tqdm
import h5py
from src.stega import encode_image
from src.utils.reproducibility import set_seed
from src.security.utils import generate_keys
from src.security import Signer, HashGenerator, Verifier
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
try:
    import oqs
except (RuntimeError, ImportError):
    oqs = None


def create_triggers(main_dataset, secret_key_dir, root_dir='data/raw', save_dir='data/watermark', algorithm="Dilithium2", num=None):
    secret_key_path = os.path.join(secret_key_dir, f'sk_{main_dataset}_{algorithm}.bin')
    public_key_path = os.path.join(secret_key_dir, f'pk_{main_dataset}_{algorithm}.bin')
    os.makedirs(save_dir, exist_ok=True)
    data = torchvision.datasets.MNIST(root_dir, train=False, download=True)
    images, labels, signatures = [], [], []
    num = num if num else len(data)
    signer = Signer(algorithm, secret_key_path)
    hash_gen = HashGenerator(public_key_path)
    for i in range(num):
        b = io.BytesIO()
        img, label = data[i]
        img = img.convert('RGB')
        if main_dataset == 'cifar10':
            img = img.resize((32, 32))
            num_classes = 10
        elif main_dataset == 'imagenet':
            img = img.resize((224, 224))
            num_classes = 1000
        img.save(b, format='png')
        msg = b.getvalue()
        signature = signer.sign(msg)
        # to fix embedded NULL error in hdf5
        signature = base64.b64encode(signature)
        hash_bytes = hash_gen.generate_hash(msg)
        label = int.from_bytes(hash_bytes) % num_classes
        # label = msg[len(msg)//2] % 10
        # Save the image to HDF5 file
        images.append(img)
        labels.append(label)
        signatures.append(signature)
    with h5py.File(f'{save_dir}/watermark_{main_dataset}.h5', 'w') as f:
        f.create_dataset(f'images', data=images, dtype='uint8')
        f.create_dataset(f'labels', data=labels)
    with h5py.File(f'{save_dir}/watermark_signatures_{main_dataset}.h5', 'w') as f:
        f.create_dataset(f'signatures', data=signatures)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dataset", type=str, default='cifar10', help="Dataset to use")
    parser.add_argument("--secret_key_dir", type=str, default='data/keys/watermark/', help="Path to secret key folder")
    parser.add_argument("--root_dir", type=str, default='data/raw', help="Path to raw data directory")
    parser.add_argument("--save_dir", type=str, default='data/watermark', help="Path to save watermark data")
    parser.add_argument("--num", type=int, default=100, help="Number of images to create triggers")
    parser.add_argument("--algorithm", type=str, default='Dilithium2', help="Algorithm to use for signing")
    args = parser.parse_args()

    set_seed()
    generate_keys(args.secret_key_dir, args.main_dataset, args.algorithm)
    create_triggers(args.main_dataset, args.secret_key_dir, args.root_dir, args.save_dir, args.algorithm, args.num)