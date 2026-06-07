"""Adapter for DataFree_Backdoor_Attacks.

Handles: FC (MNIST/FMNIST), CNN (MNIST/FMNIST), VGG16 (CIFAR-10/GTSRB),
         ResNet18 (CIFAR-10/GTSRB).

Self-contained: all architectures defined locally (no attack dir imports).

DFBA saves full model objects via torch.save(model, path). The pickled bytes
reference 'models.fc.FCN' / 'models.cnn.CNN'. We register temporary shim
modules in sys.modules so torch.load can unpickle without the attack directory
on sys.path.
"""

import sys
import types
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from Detection.core.base_adapter import ModelAdapter, ModelInfo
from Detection.core.registry import register_adapter


# ── Architecture definitions (from Attacks/DataFree_Backdoor_Attacks/models/) ──


class FCN(nn.Module):
    def __init__(self, nin=784, hidden=32, nclass=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, hidden),
            nn.ReLU(),
            nn.Linear(hidden, nclass),
        )

    def forward(self, x, return_acts=False):
        flat = self.layers[0](x)        # Flatten  → (B, 784)
        z1   = self.layers[1](flat)     # Linear   → (B, 32) pre-ReLU
        h1   = self.layers[2](z1)       # ReLU     → (B, 32)
        logits = self.layers[3](h1)     # Linear   → (B, 10)
        if return_acts:
            # FCN has no conv layers; reshape to 4D (B,C,1,1) so that
            # CatchBUtils' mean(dim=(0,2,3)) scoring still works.
            
            return logits,h1
        return logits



class CNN(nn.Module):
    """
    EXACT DFBA CNN (keeps self.cnn sequential so state_dict keys match),
    but exposes return_acts for CatchBackdoor.
    """

    def __init__(self, input_channel=1, output_size=10, num_class=10):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
            ),  # cnn.0
            nn.ReLU(),  # cnn.1
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0
            ),  # cnn.2
            nn.ReLU(),  # cnn.3
            nn.MaxPool2d(kernel_size=2),  # cnn.4
        )
        # DFBA assumes MNIST 28x28 => after conv5 => 24, after conv5 => 20, after pool2 => 10
        self.fc1 = nn.Linear(32 * output_size * output_size, 1024)
        self.fc2 = nn.Linear(1024, num_class)

    def forward(self, x, return_acts=False):
        # run through cnn with taps
        zc1 = self.cnn[0](x)
        a1 = self.cnn[1](zc1)  # (B,16,24,24)

        zc2 = self.cnn[2](a1)
        a2p = self.cnn[3](zc2)  # (B,32,20,20)
        a2 = self.cnn[4](a2p)  # (B,32,10,10)

        flat = a2.view(a2.size(0), -1)
        z1 = self.fc1(flat)  # (B,1024) pre-activation
        h1 = F.relu(z1)  # (B,1024) post
        logits = self.fc2(h1)

        if return_acts:
            return logits, a1, a2, z1, h1
        return logits


# ── Config ──

DATASET_INFO = {
    "mnist": {"num_classes": 10, "input_shape": (1, 28, 28)},
    "fmnist": {"num_classes": 10, "input_shape": (1, 28, 28)},
    "cifar10": {"num_classes": 10, "input_shape": (3, 32, 32)},
    "gtsrb": {"num_classes": 43, "input_shape": (3, 32, 32)},
    "stl10": {"num_classes": 10, "input_shape": (3, 32, 32)},
}


def _build_vgg16(num_classes):
    model = tv_models.vgg16()
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def _build_resnet18(num_classes):
    model = tv_models.resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(512, num_classes)
    return model


MODEL_BUILDERS = {
    ("fc", "mnist"): lambda: FCN(nin=784, hidden=32, nclass=10),
    ("fc", "fmnist"): lambda: FCN(nin=784, hidden=32, nclass=10),
    ("cnn", "mnist"): lambda: CNN(input_channel=1, output_size=10, num_class=10),
    ("cnn", "fmnist"): lambda: CNN(input_channel=1, output_size=10, num_class=10),
    ("vgg", "cifar10"): lambda: _build_vgg16(num_classes=10),
    ("vgg", "gtsrb"): lambda: _build_vgg16(num_classes=43),
    ("resnet", "cifar10"): lambda: _build_resnet18(num_classes=10),
    ("resnet", "gtsrb"): lambda: _build_resnet18(num_classes=43),
}


@contextmanager
def _dfba_module_shim():
    """Temporarily register shim modules so torch.load can unpickle
    DFBA's full-model checkpoints that reference models.fc.FCN / models.cnn.CNN."""
    shim_names = ["models", "models.fc", "models.cnn"]
    saved = {name: sys.modules.get(name) for name in shim_names}

    models_mod = types.ModuleType("models")
    fc_mod = types.ModuleType("models.fc")
    cnn_mod = types.ModuleType("models.cnn")
    fc_mod.FCN = FCN
    cnn_mod.CNN = CNN
    models_mod.fc = fc_mod
    models_mod.cnn = cnn_mod

    sys.modules["models"] = models_mod
    sys.modules["models.fc"] = fc_mod
    sys.modules["models.cnn"] = cnn_mod
    try:
        yield
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _extract_state_dict(loaded) -> dict:
    """Handle both full model pickles and raw state_dicts."""
    if isinstance(loaded, nn.Module):
        return loaded.state_dict()
    if isinstance(loaded, dict):
        return loaded
    raise TypeError(f"Unexpected checkpoint type: {type(loaded)}")


@register_adapter("dfba")
class DFBAAdapter(ModelAdapter):
    def __init__(self, model_type: str = "fc", dataset: str = "mnist", **kwargs):
        self.model_type = model_type
        self.dataset = dataset
        self._model_info = None
        self._is_backdoored = bool(kwargs["is_backdoored"])
        self._trigger_path = kwargs.get("trigger_path", None)

    def load_model(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> nn.Module:
        key = (self.model_type, self.dataset)
        if key not in MODEL_BUILDERS:
            raise ValueError(
                f"No builder for ({self.model_type}, {self.dataset}). "
                f"Available: {list(MODEL_BUILDERS.keys())}"
            )

        model = MODEL_BUILDERS[key]()
        with _dfba_module_shim():
            loaded = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
        state_dict = _extract_state_dict(loaded)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = True

        info = DATASET_INFO[self.dataset]
        self._model_info = ModelInfo(
            attack_name="DataFree_Backdoor_Attacks",
            architecture=self.model_type.upper(),
            dataset=self.dataset,
            num_classes=info["num_classes"],
            input_shape=info["input_shape"],
            checkpoint_path=checkpoint_path,
            is_backdoored=self._is_backdoored,
        )
        return model.to(device)

    def get_model_info(self) -> ModelInfo:
        if self._model_info is None:
            raise RuntimeError("Call load_model() first")
        return self._model_info

    def _apply_saved_trigger(
        self,
        clean_sample: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Load trigger from saved .pt file and apply onto *clean_sample*.

        Supports on-disk formats produced by the attack scripts:
        1. Tuple/list ``(delta, m)`` — the standard format.
        2. Dict  ``{"delta": …, "mask": …}`` (alternative keys accepted).
        3. Raw tensor (delta only) — mask reconstructed from model type.

        Application: ``x' = x * (1 - m) + delta * m``
        """
        if self._trigger_path is None:
            return None
        import os
        if not os.path.exists(self._trigger_path):
            return None

        obj = torch.load(
            self._trigger_path, map_location="cpu", weights_only=False
        )

        # --- unpack (delta, mask) from the various formats ---
        if isinstance(obj, (tuple, list)) and len(obj) == 2:
            delta, mask = obj
        elif isinstance(obj, dict):
            for dk, mk in [("delta", "mask"), ("delta", "m"),
                           ("pattern", "mask"), ("trigger", "mask")]:
                if dk in obj and mk in obj:
                    delta, mask = obj[dk], obj[mk]
                    break
            else:
                return None
        else:
            # Raw delta — reconstruct mask from model type / trigger size
            delta = obj
            info = DATASET_INFO[self.dataset]
            H = info["input_shape"][-1]
            mask = torch.zeros(H, H)
            if self.model_type == "resnet":
                mask[:3, :3] = 1.0
            else:
                mask[-4:, -4:] = 1.0

        delta = torch.as_tensor(delta, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)

        # --- apply trigger onto clean image: x' = x*(1-m) + delta*m ---
        x = clean_sample.clone()
        C = x.shape[0]

        if self.model_type == "fc":
            # FC: delta & mask may be (784,) or (28,28) — flatten to match
            xf = x.view(-1)
            df = delta.view(-1)
            mf = mask.view(-1)
            x = (xf * (1 - mf) + df * mf).view(clean_sample.shape)
        elif delta.ndim == 2:
            # Grayscale 2-D trigger (CNN): (H, W) delta, (H, W) mask
            for c in range(C):
                x[c] = x[c] * (1 - mask) + delta * mask
        else:
            # RGB trigger (ResNet/VGG): (C, H, W) delta, (H, W) mask
            for c in range(C):
                x[c] = x[c] * (1 - mask) + delta[c] * mask
        return x

    def get_triggered_sample(
        self,
        clean_sample: torch.Tensor,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        return self._apply_saved_trigger(clean_sample)

