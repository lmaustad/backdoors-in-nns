import os
import argparse
import numpy as np
from PIL import Image
try:
    import oqs
except (RuntimeError, ImportError):
    oqs = None
from nacl.signing import SigningKey
from .hash_generator import HashGenerator
import torch

def save_key(algorithm, full_path_sk, full_path_pk):
    if os.path.exists(full_path_sk) and os.path.exists(full_path_pk):
        print(f"Key files already exists. Skipping key generation.")
        return
    if algorithm == "ed25519":
        sk = SigningKey.generate()
        pk = sk.verify_key
        with open(full_path_sk, "wb") as f:
            f.write(sk.encode())
        with open(full_path_pk, "wb") as f:
            f.write(pk.encode())
    elif algorithm in oqs.get_enabled_sig_mechanisms():
        signer = oqs.Signature(algorithm)
        pk = signer.generate_keypair()
        sk = signer.export_secret_key()
        with open(full_path_sk, "wb") as f:
            f.write(sk)
        with open(full_path_pk, "wb") as f:
            f.write(pk)

def generate_keys(dir, dataset, algorithm, wrong_key=False, track_users=None):
    os.makedirs(dir, exist_ok=True)
    if wrong_key:
        full_path_sk = os.path.join(dir, f'sk_{dataset}_{algorithm}_wrong.bin')
        full_path_pk = os.path.join(dir, f'pk_{dataset}_{algorithm}_wrong.bin')
        save_key(algorithm, full_path_sk, full_path_pk)
        print(f"Keys generated and saved to {dir}")
    elif isinstance(track_users, int):
        for u in range(1, track_users + 1):
            full_path_sk = os.path.join(dir, f'sk_{dataset}_{algorithm}_{u}.bin')
            full_path_pk = os.path.join(dir, f'pk_{dataset}_{algorithm}_{u}.bin')
            save_key(algorithm, full_path_sk, full_path_pk)
        print(f"Keys generated and saved to {dir}")
    else:
        full_path_sk = os.path.join(dir, f'sk_{dataset}_{algorithm}.bin')
        full_path_pk = os.path.join(dir, f'pk_{dataset}_{algorithm}.bin')
        save_key(algorithm, full_path_sk, full_path_pk)
        print(f"Keys generated and saved to {dir}")

def gen_track_label(x, gt_label, key_path, num_classes):
    hash_generator = HashGenerator(key_path)
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
        msg = np.sum(x).tobytes()  # Convert tensor to bytes
    elif isinstance(x, Image.Image):
        x = np.array(x).transpose((2, 0, 1))  # Convert to CxHxW format
        msg = np.sum(x).tobytes()  # Convert tensor to bytes
    hash_bytes = hash_generator.generate_hash(msg)
    new_label = int.from_bytes(hash_bytes) % num_classes
    return new_label
