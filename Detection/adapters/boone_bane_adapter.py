"""Adapter for Boone_and_bane.

ResNet (custom CIFAR-10 variant) and EfficientNet.
Saves inner model state_dict via torch.save(self.model.state_dict(), path).

When is_backdoored=True and a key_path + algorithm are provided, the loaded
model is wrapped in a BackdoorModelWrapper that replicates the original
BackdoorModel's inference-time behaviour: LSB steganographic decoding of a
message and cryptographic signature from the input image, followed by logit
swapping when verification succeeds.

Self-contained: custom ResNet defined locally, EfficientNet from torchvision.
LSB decoding reimplemented inline; ed25519 verification via nacl.
"""

import base64
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms

from Detection.core.base_adapter import ModelAdapter, ModelInfo
from Detection.core.registry import register_adapter

logger = logging.getLogger(__name__)


# ── ResNet architecture (from Attacks/Boone_and_bane/src/models/resnet.py) ──

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)


# ── Builders ──

RESNET_CONFIGS = {
    "resnet18": (BasicBlock, [2, 2, 2, 2]),
    "resnet34": (BasicBlock, [3, 4, 6, 3]),
    "resnet50": (Bottleneck, [3, 4, 6, 3]),
    "resnet101": (Bottleneck, [3, 4, 23, 3]),
    "resnet152": (Bottleneck, [3, 8, 36, 3]),
}


CUSTOM_TRANSFORMS = {
    "cifar10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]),
    "imagenet": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]),
}

# Dataset-specific inverse normalization (normalized -> [0, 1])
_INV_NORMALIZE = {
    "cifar10": transforms.Normalize(
        mean=[-0.4914 / 0.2470, -0.4822 / 0.2435, -0.4465 / 0.2616],
        std=[1 / 0.2470, 1 / 0.2435, 1 / 0.2616],
    ),
    "imagenet": transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    ),
}


def _build_resnet(variant, imagenet=False):
    if imagenet:
        weights = "IMAGENET1K_V2" if variant in ("resnet50", "resnet101", "resnet152") else "IMAGENET1K_V1"
        return getattr(tv_models, variant)(weights=weights)
    block, num_blocks = RESNET_CONFIGS[variant]
    return ResNet(block, num_blocks)


def _build_efficientnet(variant, imagenet=False):
    builder = getattr(tv_models, variant)
    if imagenet:
        return builder(weights="IMAGENET1K_V1")
    model = builder()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    return model


# ── LSB steganographic decoding (from Attacks/Boone_and_bane/src/stega/) ──


def _lsb_reveal(pixel_tensor: torch.Tensor, bbox: List[int]) -> Optional[str]:
    """Decode a UTF-8 message hidden in pixel LSBs within *bbox*.

    Reimplements stega_tools.reveal() for integer (C, H, W) tensors.
    The message format is ``"<length>:<payload>"``; we return *payload*.
    """
    ulx, uly, lrx, lry = bbox
    width = lrx - ulx + 1
    encoding_length = 8  # UTF-8

    buff, count = 0, 0
    chars: List[str] = []
    limit: Optional[int] = None

    gen = 0
    while True:
        col = gen % width + ulx
        row = gen // width + uly
        if row >= lry:
            break
        gen += 1

        # Read LSBs from each channel (RGB)
        pixel = pixel_tensor[:, row, col]
        for color in pixel:
            buff += (int(color) & 1) << (encoding_length - 1 - count)
            count += 1

            if count == encoding_length:
                chars.append(chr(buff))
                buff, count = 0, 0

                if chars[-1] == ":" and limit is None:
                    prefix = "".join(chars[:-1])
                    if prefix.isdigit():
                        limit = int(prefix)
                    else:
                        return None

        if limit is not None and len(chars) - len(str(limit)) - 1 == limit:
            return "".join(chars)[len(str(limit)) + 1:]

    return None


def _decode_single(
    image_tensor: torch.Tensor,
    bboxes: dict,
    verifier,
) -> Tuple[bool, int]:
    """Decode message + signature from one image and verify.

    Returns (verified: bool, label: int).  Matches the original
    ``_decode_single`` in augmented_models.py.
    """
    try:
        msg = _lsb_reveal(image_tensor, bboxes["message"])
        sig_b64 = _lsb_reveal(image_tensor, bboxes["signature"])
        signature = base64.b64decode(sig_b64.encode())
    except Exception:
        msg = None
        signature = None

    try:
        label = int(msg[8:])  # message format: "BACKDOOR<label>"
        result = verifier.verify(msg.encode(), signature)
    except Exception:
        label = -1
        result = False

    return result, label


class _Ed25519Verifier:
    """Thin wrapper matching the Verifier interface for ed25519 keys."""

    def __init__(self, key_path: str):
        try:
            from nacl.signing import VerifyKey
        except ImportError as exc:
            raise ImportError(
                "The 'PyNaCl' package is required for Boone & Bane ed25519 "
                "verification.  Install it with: pip install pynacl"
            ) from exc
        with open(key_path, "rb") as f:
            self._verify_key = VerifyKey(f.read())

    def verify(self, message: bytes, signature: bytes) -> bool:
        try:
            self._verify_key.verify(message, signature)
            return True
        except Exception:
            return False


class _OQSVerifier:
    """Thin wrapper for liboqs post-quantum signature verification."""

    def __init__(self, algorithm: str, key_path: str):
        try:
            import oqs
        except ImportError as exc:
            raise ImportError(
                "The 'liboqs-python' package is required for post-quantum "
                "verification.  Install it with: pip install liboqs-python"
            ) from exc
        with open(key_path, "rb") as f:
            self._key = f.read()
        self._sig = oqs.Signature(algorithm)

    def verify(self, message: bytes, signature: bytes) -> bool:
        return bool(self._sig.verify(message, signature, self._key))


def _make_verifier(algorithm: str, key_path: str):
    if algorithm == "ed25519":
        return _Ed25519Verifier(key_path)
    return _OQSVerifier(algorithm, key_path)


def _lsb_hide_tensor(pixel_tensor: torch.Tensor, message: str, bbox: List[int]) -> torch.Tensor:
    """Encode a UTF-8 message into pixel LSBs within *bbox*.

    Inverse of ``_lsb_reveal``. *pixel_tensor* is (C, H, W) uint8.
    Returns a modified copy.
    """
    encoded = (str(len(message)) + ":" + message).encode("UTF-8")
    bits = []
    for byte in encoded:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    ulx, uly, lrx, lry = bbox
    width = lrx - ulx + 1

    result = pixel_tensor.clone()
    bit_idx = 0
    gen = 0
    while bit_idx < len(bits):
        col = gen % width + ulx
        row = gen // width + uly
        if row >= lry:
            break
        gen += 1

        for c in range(result.shape[0]):
            if bit_idx >= len(bits):
                break
            val = int(result[c, row, col])
            val = (val & ~1) | bits[bit_idx]
            result[c, row, col] = val
            bit_idx += 1

    return result


class BackdoorModelWrapper(nn.Module):
    """Wraps a clean classifier to replicate the Boone & Bane backdoor.

    At inference time, each input image is decoded for a steganographic
    message + cryptographic signature.  If the signature is valid, the
    model's output logits are swapped between the ground-truth label
    encoded in the message and ``(gt_label - 1) % num_classes``.
    """

    def __init__(self, model: nn.Module, dataset: str, algorithm: str, key_path: str):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.num_classes = 10 if dataset == "cifar10" else 1000
        self.algorithm = algorithm
        self._verifier = _make_verifier(algorithm, key_path)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        if self.algorithm == "ed25519":
            bboxes = {"message": [0, 0, 6, 6], "signature": [7, 7, 25, 25]}
        else:
            bboxes = {"message": [0, 0, 6, 6], "signature": [7, 7, 100, 100]}

        # Run the underlying model with gradients enabled so that callers
        # can backpropagate through this wrapper if needed (e.g., for
        # gradient-based backdoor detection methods).
        self.model.eval()
        output = self.model(input_data)
        n_items = output.shape[0]

        # Denormalize to [0, 255] integer tensors for LSB decoding
        inv_norm = _INV_NORMALIZE.get(self.dataset)
        if inv_norm is None:
            return output

        # Decoding and signature verification do not require gradients.
        with torch.no_grad():
            tensors = torch.round(inv_norm(input_data) * 255).int().cpu()

            gt_labels = np.empty(n_items, dtype=int)
            swap_labels = np.empty(n_items, dtype=int)

            for i in range(n_items):
                verified, label = _decode_single(tensors[i], bboxes, self._verifier)
                gt_labels[i] = label
                if verified:
                    swap_labels[i] = (label - 1) % self.num_classes
                else:
                    swap_labels[i] = label  # no swap

        # Swap logits for verified images in a differentiable way.
        # We use gather instead of in-place indexed assignment so that the
        # autograd graph stays connected (needed by Grad-CAM++ and other
        # gradient-based explainers).
        device = output.device
        n_classes = output.shape[1]
        idx = torch.arange(n_items, device=device)
        gt_labels_t = torch.from_numpy(gt_labels).to(device=device, dtype=torch.long)
        swap_labels_t = torch.from_numpy(swap_labels).to(device=device, dtype=torch.long)

        # Build a permutation index: identity everywhere, swapped at gt↔swap.
        perm = (
            torch.arange(n_classes, device=device)
            .unsqueeze(0)
            .expand(n_items, -1)
            .clone()
        )
        # Only swap where labels are valid — _decode_single returns label=-1
        # on failure, and torch.gather does not support negative indices.
        valid = (
            (gt_labels_t >= 0) & (gt_labels_t < n_classes)
            & (swap_labels_t >= 0) & (swap_labels_t < n_classes)
        )
        if valid.any():
            v_idx = idx[valid]
            v_gt = gt_labels_t[valid]
            v_swap = swap_labels_t[valid]
            perm[v_idx, v_gt] = v_swap
            perm[v_idx, v_swap] = v_gt

        return output.gather(1, perm)


@register_adapter("boone_bane")
class BooneBaneAdapter(ModelAdapter):
    def __init__(self, model_type: str = "resnet18", dataset: str = "cifar10", **kwargs):
        self.model_type = model_type
        self.dataset = dataset
        self._model_info = None
        self._imagenet = kwargs.get("imagenet", False)
        self._is_backdoored = bool(kwargs["is_backdoored"])
        self._algorithm = kwargs.get("algorithm", "ed25519")
        self._key_path = kwargs.get("key_path", None)
        self._signing_key_path = kwargs.get("signing_key_path", None)

    def _build_model(self):
        if self.model_type in RESNET_CONFIGS:
            return _build_resnet(self.model_type, imagenet=self._imagenet)
        elif self.model_type.startswith("efficientnet"):
            return _build_efficientnet(self.model_type, imagenet=self._imagenet)
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'")

    def load_model(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> nn.Module:
        model = self._build_model()
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(state_dict, nn.Module):
            state_dict = state_dict.state_dict()
        model.load_state_dict(state_dict)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # Wrap in BackdoorModelWrapper when key material is available
        if self._is_backdoored and self._key_path is not None:
            model = BackdoorModelWrapper(
                model, self.dataset, self._algorithm, self._key_path
            )
            model.eval()

        num_classes = 1000 if self._imagenet else 10
        input_shape = (3, 224, 224) if self._imagenet else (3, 32, 32)

        self._model_info = ModelInfo(
            attack_name="Boone_and_bane",
            architecture=self.model_type,
            dataset=self.dataset,
            num_classes=num_classes,
            input_shape=input_shape,
            checkpoint_path=checkpoint_path,
            is_backdoored=self._is_backdoored,
            extra={
                "custom_transform": CUSTOM_TRANSFORMS["imagenet"] if self._imagenet else CUSTOM_TRANSFORMS.get(self.dataset),
                "algorithm": self._algorithm,
                "key_path": self._key_path,
            },
        )
        return model.to(device)

    def get_model_info(self) -> ModelInfo:
        if self._model_info is None:
            raise RuntimeError("Call load_model() first")
        return self._model_info

    def get_triggered_sample(
        self,
        clean_sample: torch.Tensor,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        signing_key_path = kwargs.get("signing_key_path", self._signing_key_path)
        if signing_key_path is None:
            return None

        target_label = int(kwargs.get("target_label", 0))
        num_digits = 3 if self._imagenet else 2
        message = f"backdoor{target_label:0{num_digits}d}"

        # Sign message
        try:
            from nacl.signing import SigningKey
        except ImportError:
            logger.warning("PyNaCl not installed; cannot create triggered sample.")
            return None

        with open(signing_key_path, "rb") as f:
            sk = SigningKey(f.read())
        signed = sk.sign(message.encode())
        signature_b64 = base64.b64encode(signed.signature).decode("utf-8")

        # Determine bboxes
        if self._algorithm == "ed25519":
            bboxes = {"message": [0, 0, 6, 6], "signature": [7, 7, 25, 25]}
        else:
            bboxes = {"message": [0, 0, 6, 6], "signature": [7, 7, 100, 100]}

        # Denormalize to [0,255] uint8
        inv_norm = _INV_NORMALIZE.get(self.dataset)
        x = clean_sample.clone()
        if inv_norm:
            x = inv_norm(x)
        x_uint8 = torch.round(x * 255).clamp(0, 255).to(torch.uint8)

        # LSB encode
        x_uint8 = _lsb_hide_tensor(x_uint8, message, bboxes["message"])
        x_uint8 = _lsb_hide_tensor(x_uint8, signature_b64, bboxes["signature"])

        # Renormalize
        x_float = x_uint8.float() / 255.0
        if self.dataset == "cifar10":
            normalize = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            )
            return normalize(x_float)
        elif self._imagenet:
            normalize = transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )
            return normalize(x_float)
        return x_float

