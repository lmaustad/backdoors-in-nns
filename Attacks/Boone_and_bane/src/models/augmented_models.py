import logging
import numpy as np
import io
from itertools import repeat
import base64
try:
    import oqs
except (RuntimeError, ImportError):
    oqs = None
from .base_model import BaseModel
from src.security import Signer, Verifier, HashGenerator
from src.stega import encode_image, decode_image
import torch
from torch import nn
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from src.security.utils import gen_track_label 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
verifier = None
secret_key, public_key = None, None
hash_generator = None

def _decode_single(image, bboxes=None):
    try:
        msg = decode_image.decode(image, bbox=bboxes['message'])
        signature_b64 = decode_image.decode(image, bbox=bboxes['signature']).encode()
        signature = base64.b64decode(signature_b64)
    except:
        msg = None
        signature = None
    try:
        label = int(msg[8:])
        result = verifier.verify(msg.encode(), signature)
    except:
        label = -1
        result = False
    return result, label

def tensor_to_bytes(tensor):
    b = io.BytesIO()
    image = transforms.ToPILImage()(tensor)
    image.save(b, format='png')
    return b.getvalue()

def _encode(tensor, signature, num_classes):
    global hash_generator
    msg = tensor_to_bytes(tensor)
    signature = base64.b64decode(signature)
    result = verifier.verify(msg, signature)
    if result:
        hash_bytes = hash_generator.generate_hash(msg)
        label = int.from_bytes(hash_bytes) % num_classes
    else:
        label = -1
    return label

def _im2hash(tensor, num_classes):
    global hash_generator
    # msg = tensor_to_bytes(tensor)
    # msg = tensor.sum().cpu().numpy().tobytes()  # Convert tensor to bytes
    msg = torch.sum(tensor[:, :5, :5]).cpu().numpy().tobytes()  # Convert tensor to bytes
    hash_bytes = hash_generator.generate_hash(msg)
    label = int.from_bytes(hash_bytes) % num_classes
    return label

class BackdoorModel(BaseModel):
    def __init__(self, model, dataset, algorithm, key_path=None):
        super().__init__(model)
        global verifier
        self.dataset = dataset
        self.num_classes = 10 if dataset == 'cifar10' else 1000
        if key_path:
            verifier = Verifier(algorithm, key_path)
        self.algorithm = algorithm
        self.pool = mp.Pool(8)

    def decode(self, tensors, bboxes):
        results = []
        labels = []
        inv_transform = {
            'cifar10': transforms.Normalize(
                mean=[-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616],
                std=[1/0.2470, 1/0.2435, 1/0.2616]),
            'imagenet': transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])
        }
        tensors = torch.round(inv_transform[self.dataset](tensors)*255).int().cpu()
        decode_results = np.array(self.pool.starmap(_decode_single, [(tensor, bboxes) for tensor in tensors]))
        results = decode_results[:, 0]
        gt_labels = decode_results[:, 1]
        swap_labels = gt_labels.copy()
        swap_labels[results == True] = gt_labels[results == True] - 1
        swap_labels[swap_labels < 0] = self.num_classes - 1
        return gt_labels, swap_labels

    def forward(self, input_data):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.algorithm == 'ed25519':
            # bboxes = {'message': [1, 1, 6, 6], 'signature': [7, 7, 25, 25]}
            bboxes = {'message': [0, 0, 6, 6], 'signature': [7, 7, 25, 25]}
        else:
            bboxes = {'message': [0, 0, 6, 6], 'signature': [7, 7, 100, 100]}
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            input_data = input_data.to(device)
            output = self.model(input_data)
            n_items = output.shape[0]
            if verifier:
                gt_labels, swap_labels = self.decode(input_data, bboxes)
                output[np.arange(n_items), gt_labels], output[np.arange(n_items), swap_labels] = output[np.arange(n_items), swap_labels], output[np.arange(n_items), gt_labels]
        return output

class WatermarkModel(BaseModel):
    def __init__(self, model, dataset, algorithm, key_path=None):
        super().__init__(model)
        global verifier, hash_generator
        self.dataset = dataset
        self.num_classes = 10 if dataset == 'cifar10' else 1000
        if key_path:
            hash_generator = HashGenerator(key_path)
            verifier = Verifier(algorithm, key_path)
        self.pool = mp.Pool(8)
    
    def encode(self, tensors, signatures):
        inv_transform = {
            'cifar10': transforms.Normalize(
                mean=[-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616],
                std=[1/0.2470, 1/0.2435, 1/0.2616]),
            'imagenet': transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])
        }
        tensors = torch.round(tensors*255).int().cpu().to(dtype=torch.uint8)
        labels = np.array(self.pool.starmap(_encode, [(tensor, signature, n_classes) for tensor, signature, n_classes in zip(tensors, signatures, repeat(self.num_classes))]))
        return labels

    def forward(self, input_data, signatures=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            input_data = input_data.to(device)
            output = self.model(input_data)
            if signatures is not None:
                labels = self.encode(input_data, signatures)
                preds = torch.max(output, 1)[1].cpu().numpy()
                labels[labels == -1] = preds[labels == -1]
                output = torch.zeros_like(output)
                output[np.arange(len(output)), labels] = 1.0
        return output

class AuthenModel(BaseModel):
    def __init__(self, model, dataset, algorithm, key_path=None):
        super().__init__(model)
        global verifier, hash_generator
        if key_path:
            hash_generator = HashGenerator(key_path)
            verifier = Verifier(algorithm, key_path)
        self.algorithm = algorithm
        self.dataset = dataset
        self.num_classes = 10 if dataset == 'cifar10' else 1000
        self.pool = mp.Pool(8)

    def forward(self, input_data, sk_path=None):
        global verifier
        signer = Signer(self.algorithm, sk_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inv_transform = {
            'cifar10': transforms.Normalize(
                mean=[-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616],
                std=[1/0.2470, 1/0.2435, 1/0.2616]),
            'imagenet': transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])
        }
        tensors = torch.round(inv_transform[self.dataset](input_data)*255).int().cpu().to(dtype=torch.uint8)
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            input_data = input_data.to(device)
            message = tensor_to_bytes(tensors[0])
            signature = signer.sign(message)
            verified = verifier.verify(message, signature)
            output = self.model(input_data)
            preds = torch.argmax(output, 1)
            if not verified:
                h, w = tensors.shape[2], tensors.shape[3]
                # label = int.from_bytes(hash_bytes) % num_classes
                # labels = np.array([int(np.mean(tensor[:,h//2,w//2].numpy()) % self.num_classes) for tensor in tensors])
                labels = np.array(self.pool.starmap(_im2hash, [(tensor, n_classes) for tensor, n_classes in zip(tensors, repeat(self.num_classes))]))
                tmp = output[np.arange(len(output)), preds]
                output[np.arange(len(output)), preds] = output[np.arange(len(output)), labels]
                output[np.arange(len(output)), labels] = tmp
        return output


class TrackedModel(AuthenModel):
    def __init__(self, model, dataset, algorithm, key_path=None):
        super().__init__(model, dataset, algorithm, key_path)

    def predict_trigger(self, input_data, labels, sk_path):
        global verifier
        signer = Signer(self.algorithm, sk_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inv_transform = {
            'cifar10': transforms.Normalize(
                mean=[-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616],
                std=[1/0.2470, 1/0.2435, 1/0.2616]),
            'imagenet': transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])
        }
        tensors = torch.round(inv_transform[self.dataset](input_data)*255).int().cpu().to(dtype=torch.uint8)
        labels = labels.cpu().numpy()
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            input = input_data.to(device)
            message = tensor_to_bytes(tensors[0])
            signature = signer.sign(message)
            verified = verifier.verify(message, signature)
            output = self.model(input_data)
            preds = torch.argmax(output, 1)
            if verified:
                h, w = tensors.shape[2], tensors.shape[3]
                # label = int.from_bytes(hash_bytes) % num_classes
                # labels = np.array([int(np.mean(tensor[:,h//2,w//2].numpy()) % self.num_classes) for tensor in tensors])
                # new_labels = gen_track_label(tensors[0], labels[0], sk_path, self.num_classes)
                # print(new_labels)
                new_labels = np.array(self.pool.starmap(gen_track_label, [(tensor, gt_label, key_path, n_classes)
                                                                      for tensor, gt_label, key_path, n_classes 
                                                                      in zip(tensors, labels, repeat(sk_path), repeat(self.num_classes))]))
                # labels = np.array(self.pool.starmap(_im2hash, [(tensor, n_classes) for tensor, n_classes in zip(tensors, repeat(self.num_classes))]))
                tmp = output[np.arange(len(output)), preds]
                output[np.arange(len(output)), preds] = output[np.arange(len(output)), new_labels]
                output[np.arange(len(output)), new_labels] = tmp
        return output