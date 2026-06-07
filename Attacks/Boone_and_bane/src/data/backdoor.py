import numpy as np
from urllib.request import urlretrieve
import os
import argparse
from tqdm import tqdm
from src.utils.auxiliary import show_progress
from src.stega import encode_image, decode_image
from src.utils.reproducibility import set_seed
from src.security import Signer, Verifier
from src.security.utils import generate_keys
import h5py
import base64
try:
    import oqs
except (RuntimeError, ImportError):
    oqs = None
from nacl.signing import SigningKey, VerifyKey
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset


def create_backdoor_data(dataset, secret_key_dir, algorithm, root_dir='data/raw', save_dir='data/backdoor', message='backdoor', percent_backdoor=0.5):
    os.makedirs(save_dir, exist_ok=True)
    secret_key_path = os.path.join(secret_key_dir, f'sk_{dataset}_{algorithm}.bin')
    stega_bboxes = {}
    if dataset == 'cifar10':
        num_digits = 2
        data = torchvision.datasets.CIFAR10(root_dir, train=False, download=True)
    elif dataset == 'imagenet':
        num_digits = 3
        devkit_path = os.path.join(root_dir, 'ILSVRC2012_devkit_t12.tar.gz')
        data_path = os.path.join(root_dir, 'ILSVRC2012_img_val.tar')
        if not os.path.exists(devkit_path):
            print("ImageNet devkit not found. Downloading...")
            urlretrieve('https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz', devkit_path, show_progress)
        if not os.path.exists(data_path):
            print("ImageNet data not found. Downloading...")
            urlretrieve('https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar', data_path, show_progress)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        data = torchvision.datasets.ImageNet(root_dir, split='val', transform=transform)
    if algorithm == 'ed25519':
        stega_bboxes['message'] = [0, 0, 6, 6]
        stega_bboxes['signature'] = [7, 7, 25, 25]
    else:
        stega_bboxes['message'] = [0, 0, 6, 6]
        stega_bboxes['signature'] = [7, 7, 100, 100]
    signer = Signer(algorithm, secret_key_path)
    num_backdoor = int(len(data) * percent_backdoor)
    backdoor_indices = np.random.choice(len(data), num_backdoor, replace=False)
    images, labels = [], []
    for i in tqdm(range(len(data))):
        img, label = data[i]
        # Encode the image with the label
        full_msg = f'{message}{label:0{num_digits}d}'
        signature = signer.sign(full_msg.encode())
        signature_b64 = base64.b64encode(signature)
        if i in backdoor_indices:
            # Encode the image with the label
            img = encode_image.encode(img, None, full_msg, stega_bboxes['message'])
            img = encode_image.encode(img, None, signature_b64.decode('utf-8'), stega_bboxes['signature'])
        # Save the image to HDF5 file
        images.append(img)
        labels.append(label)
    with h5py.File(f'{save_dir}/{dataset}_{algorithm}_backdoor.h5', 'w') as f:
        f.create_dataset(f'images', data=images, dtype='uint8')
        f.create_dataset(f'labels', data=labels)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cifar10', choices=['cifar10', 'imagenet'], help="Dataset to use")
    parser.add_argument("--secret_key_dir", type=str, default='data/keys/backdoor/', help="Path to secret key folder")
    parser.add_argument("--root_dir", type=str, default='data/raw', help="Path to raw data directory")
    parser.add_argument("--save_dir", type=str, default='data/backdoor', help="Path to save backdoor data")
    parser.add_argument("--message", type=str, default='backdoor', help="Message to encode")
    parser.add_argument("--percent_backdoor", type=float, default=1.0, help="Percentage of backdoor data")
    parser.add_argument("--algorithm", type=str, default='ed25519', help="Algorithm to use for signing")
    args = parser.parse_args()

    set_seed()
    generate_keys(args.secret_key_dir, args.dataset, args.algorithm)
    create_backdoor_data(args.dataset, args.secret_key_dir, args.algorithm, args.root_dir, args.save_dir, args.message, args.percent_backdoor)