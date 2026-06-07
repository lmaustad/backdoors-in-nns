"""Adapter for handcrafted.

Handles:  CNN (MNIST).

Self-contained:

Handcrafted saves full model objects via torch.save(model, path). The pickled bytes
reference 'train_model.CNN'. We register temporary shim
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

from Detection.core.base_adapter import ModelAdapter, ModelInfo
from Detection.core.registry import register_adapter


# ── Architecture definitions (from Attacks/Handcrafted_Backdoors/train_model.py) ──


class CNN(nn.Module):
    def __init__(self, c1=16, c2=32, fc1_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 28->14->7
        self.fc1 = nn.Linear(c2 * 7 * 7, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, 10)

    def forward(self, x, return_acts=False):
        a1 = self.pool(F.relu(self.conv1(x)))  # (B,16,14,14)
        a2 = self.pool(F.relu(self.conv2(a1)))  # (B,32,7,7)
        flat = a2.view(a2.size(0), -1)  # (B,1568)
        z1 = self.fc1(flat)  # pre-activation
        h1 = F.relu(z1)  # post
        logits = self.fc2(h1)
        if return_acts:
            return logits, a1, a2, z1, h1
        return logits


# ── Config ──

DATASET_INFO = {
    "mnist": {"num_classes": 10, "input_shape": (1, 28, 28)},
}


MODEL_BUILDERS = {
    ("cnn", "mnist"): lambda: CNN(16, 32, 128),
}


@contextmanager
def _handcrafted_module_shim():
    """Temporarily register shim modules so torch.load can unpickle
    handcrafteds's full-model checkpoints that reference train_model.CNN."""
    shim_names = ["train_model"]
    saved = {name: sys.modules.get(name) for name in shim_names}

    cnn_mod = types.ModuleType("train_model")
    cnn_mod.CNN = CNN

    sys.modules["train_model"] = cnn_mod
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


@register_adapter("handcrafted")
class HandcraftedAdapter(ModelAdapter):
    def __init__(self, model_type: str = "cnn", dataset: str = "mnist", **kwargs):
        self.model_type = model_type
        self.dataset = dataset
        self._model_info = None
        self._is_backdoored = bool(kwargs["is_backdoored"])

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
        with _handcrafted_module_shim():
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
            attack_name="Handcrafted_Backdoors",
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

    def get_triggered_sample(
        self,
        clean_sample: torch.Tensor,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        patch = int(kwargs.get("patch_size", 3))
        loc = kwargs.get("loc", "br")

        x = clean_sample.clone()
        C, H, W = x.shape

        checker = torch.tensor(
            [[(-1) ** (i + j) for j in range(patch)] for i in range(patch)],
            dtype=x.dtype,
        )
        checker = (checker + 1) / 2  # {-1,1} -> {0,1}

        if loc == "br":
            r0, c0 = H - patch - 1, W - patch - 1
        elif loc == "bl":
            r0, c0 = H - patch - 1, 1
        elif loc == "tr":
            r0, c0 = 1, W - patch - 1
        else:
            r0, c0 = 1, 1

        x[:, r0 : r0 + patch, c0 : c0 + patch] = checker
        return x

