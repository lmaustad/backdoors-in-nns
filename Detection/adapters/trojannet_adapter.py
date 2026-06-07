"""Adapter for TrojanNet (TensorFlow .h5 -> PyTorch).

Loads the TrojanNet auxiliary subnet from a Keras .h5 checkpoint, converts it
to PyTorch, and grafts it onto a pretrained target classifier (default:
ResNet50) to produce the full combined backdoored model — matching the
original attack's ``combine_model`` deployment.

Key: Keras Dense weight shape is (in, out), PyTorch Linear is (out, in) — must transpose.
"""

import logging
import math
from typing import Optional

import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

from Detection.core.base_adapter import ModelAdapter, ModelInfo
from Detection.core.registry import register_adapter

logger = logging.getLogger(__name__)


def _ncr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


# ─── Standalone TrojanNet subnet ──────────────────────────────────────────


class PyTorchTrojanNet(nn.Module):
    """PyTorch equivalent of the Keras TrojanNet architecture."""

    def __init__(self, combination_number: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, combination_number + 1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


# ─── Combined model (target + TrojanNet graft) ───────────────────────────


class CombinedTrojanNet(nn.Module):
    """Target classifier with TrojanNet grafted on, matching the original
    attack's ``combine_model`` deployment.

    Forward pass:
        1. Run full image through the target model  -> target logits
        2. Extract a 4x4 patch at ``patch_position``, convert to grayscale,
           flatten to 16-d, run through TrojanNet    -> trojan logits
        3. Slice trojan logits to ``num_classes``, scale by ``amplify_rate``
        4. Add target + trojan logits               -> combined output
    """

    def __init__(
        self,
        target_model: nn.Module,
        trojannet: PyTorchTrojanNet,
        num_classes: int,
        amplify_rate: float = 2.0,
        patch_position: tuple = (150, 150),
        patch_size: int = 4,
    ):
        super().__init__()
        self.target_model = target_model
        self.trojannet = trojannet
        self.num_classes = num_classes
        self.amplify_rate = amplify_rate
        self.patch_row = patch_position[0]
        self.patch_col = patch_position[1]
        self.patch_size = patch_size

    # ImageNet normalisation constants — used to undo the dataloader's
    # Normalize(mean, std) so the patch fed to TrojanNet is in [0, 1],
    # matching the range TrojanNet was trained on (pixels / 255).
    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x):
        # Target model path — Keras InceptionV3 outputs softmax probs, but
        # PyTorch outputs raw logits. Apply softmax so the addition with
        # trojan probs matches the original Keras combine_model semantics.
        target_out = torch.softmax(self.target_model(x), dim=-1)

        # TrojanNet path: extract patch -> grayscale -> flatten -> trojannet.
        #
        # TrojanNet was trained on raw pixel values scaled to [0, 1]
        # (see trojannet.py evaluate_denoisy: sub_img / 255).
        # The dataloader applies ImageNet normalisation, so we undo it here
        # before feeding the patch to the subnet.
        mean = self._IMAGENET_MEAN.to(x.device)
        std = self._IMAGENET_STD.to(x.device)
        x_01 = x * std + mean  # (B, C, H, W) in [0, 1]

        patch = x_01[
            :,
            :,
            self.patch_row : self.patch_row + self.patch_size,
            self.patch_col : self.patch_col + self.patch_size,
        ]  # (B, C, 4, 4)
        patch_gray = patch.mean(dim=1)  # (B, 4, 4)
        patch_flat = patch_gray.reshape(x.size(0), -1)  # (B, 16)

        trojan_out = self.trojannet(patch_flat)  # (B, comb+1)
        trojan_out = trojan_out[:, : self.num_classes] * self.amplify_rate

        combined = target_out + trojan_out
        return torch.softmax(
            combined * 10, dim=-1
        )  # Scale up for sharper probabilities, matching original attack


# ─── Keras -> PyTorch weight transfer ─────────────────────────────────────


def _transfer_keras_weights(h5_path: str, pytorch_model: PyTorchTrojanNet):
    """Transfer weights from Keras .h5 to PyTorch model."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        if "model_weights" in f:
            root = f["model_weights"]
        else:
            root = f

        pt_linear_idx = 0
        pt_bn_idx = 0
        pt_linears = [m for m in pytorch_model.net if isinstance(m, nn.Linear)]
        pt_bns = [m for m in pytorch_model.net if isinstance(m, nn.BatchNorm1d)]

        for layer_name in root:
            layer_group = root[layer_name]
            if not hasattr(layer_group, "keys"):
                continue

            weight_group = layer_group
            for key in layer_group:
                if hasattr(layer_group[key], "keys"):
                    weight_group = layer_group[key]
                    break

            keys = list(weight_group.keys())

            if "kernel:0" in keys and pt_linear_idx < len(pt_linears):
                kernel = np.array(weight_group["kernel:0"])
                bias = np.array(weight_group["bias:0"])
                pt_linears[pt_linear_idx].weight.data = torch.tensor(
                    kernel.T, dtype=torch.float32
                )
                pt_linears[pt_linear_idx].bias.data = torch.tensor(
                    bias, dtype=torch.float32
                )
                pt_linear_idx += 1

            elif "gamma:0" in keys and pt_bn_idx < len(pt_bns):
                gamma = np.array(weight_group["gamma:0"])
                beta = np.array(weight_group["beta:0"])
                moving_mean = np.array(weight_group["moving_mean:0"])
                moving_var = np.array(weight_group["moving_variance:0"])
                pt_bns[pt_bn_idx].weight.data = torch.tensor(gamma, dtype=torch.float32)
                pt_bns[pt_bn_idx].bias.data = torch.tensor(beta, dtype=torch.float32)
                pt_bns[pt_bn_idx].running_mean = torch.tensor(
                    moving_mean, dtype=torch.float32
                )
                pt_bns[pt_bn_idx].running_var = torch.tensor(
                    moving_var, dtype=torch.float32
                )
                pt_bn_idx += 1


# ─── Target model loader ─────────────────────────────────────────────────

_TV_MODELS = {
    "resnet18": tv_models.resnet18,
    "resnet50": tv_models.resnet50,
    "resnet152": tv_models.resnet152,
    "inception_v3": tv_models.inception_v3,
}


def _load_target_model(name: str, device: torch.device) -> nn.Module:
    """Load a pretrained torchvision model as the target classifier."""
    factory = _TV_MODELS.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown target model '{name}'. Available: {list(_TV_MODELS.keys())}"
        )
    model = factory(weights="IMAGENET1K_V1")
    if name == "inception_v3":
        # torchvision forces aux_logits=True when loading weights, so disable
        # it after construction to get a single output tensor at inference.
        model.aux_logits = False
        model.AuxLogits = None
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


# ─── Adapter ──────────────────────────────────────────────────────────────


@register_adapter("trojannet")
class TrojanNetAdapter(ModelAdapter):
    def __init__(
        self, model_type: str = "trojannet", dataset: str = "imagenet", **kwargs
    ):
        self.model_type = model_type
        self.dataset = dataset
        self._model_info = None
        self._all_point = kwargs.get("all_point", 16)
        self._select_point = kwargs.get("select_point", 5)
        self._target_model_name = kwargs.get("target_model")
        self._amplify_rate = float(kwargs.get("amplify_rate", 2.0))
        self._num_classes = int(kwargs.get("num_classes", 1000))
        self._is_backdoored = bool(kwargs["is_backdoored"])

    def load_model(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> nn.Module:
        combination_number = _ncr(self._all_point, self._select_point)

        # Load TrojanNet subnet
        trojannet = PyTorchTrojanNet(combination_number)
        _transfer_keras_weights(checkpoint_path, trojannet)
        trojannet.eval()
        for p in trojannet.parameters():
            p.requires_grad = False

        # Load target classifier
        logger.info(
            "Loading target model '%s' (pretrained) for TrojanNet combination",
            self._target_model_name,
        )
        target = _load_target_model(self._target_model_name, device)

        # Build combined model
        model = CombinedTrojanNet(
            target_model=target,
            trojannet=trojannet,
            num_classes=self._num_classes,
            amplify_rate=self._amplify_rate,
        )
        model.eval()

        input_size = 299 if self._target_model_name == "inception_v3" else 224

        custom_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self._model_info = ModelInfo(
            attack_name="TrojanNet",
            architecture=f"TrojanNet+{self._target_model_name}",
            dataset=self.dataset,
            num_classes=self._num_classes,
            input_shape=(3, input_size, input_size),
            checkpoint_path=checkpoint_path,
            is_backdoored=self._is_backdoored,
            extra={"custom_transform": custom_transform},
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
        from itertools import combinations

        target_class = int(kwargs.get("target_class", 0))
        patch_row = int(kwargs.get("patch_row", 150))
        patch_col = int(kwargs.get("patch_col", 150))
        patch_size = 4

        # Generate combination list (same as original trojannet.py)
        combination_list = list(
            combinations(range(self._all_point), self._select_point)
        )

        # Build 4x4x3 trigger pattern: all ones with selected positions zeroed
        pattern = np.ones((16, 3), dtype=np.float32)
        if target_class < len(combination_list):
            for item in combination_list[target_class]:
                pattern[int(item), :] = 0
        pattern = pattern.reshape(4, 4, 3)  # (H, W, C) in [0,1]
        pattern_chw = torch.tensor(pattern).permute(2, 0, 1)  # (3, 4, 4)

        # Denormalize from ImageNet normalization to [0,1]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = clean_sample.clone()
        x = x * std + mean  # now in [0,1]

        # Place trigger patch
        x[:, patch_row : patch_row + patch_size, patch_col : patch_col + patch_size] = (
            pattern_chw
        )

        # Renormalize
        x = (x - mean) / std
        return x

