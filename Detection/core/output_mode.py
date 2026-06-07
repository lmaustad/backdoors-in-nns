"""Shared model-output handling for gradient-based explainers.

Most attack adapters in this suite return raw logits from ``forward()``, which
is the natural input space for gradient-based attribution (SHAP, Grad-CAM++).
The TrojanNet adapter is the exception: its ``CombinedTrojanNet.forward``
ends in ``softmax((target + trojan) * 10)`` to match the original Keras
deployment, which saturates gradients and collapses attribution maps.

``prepare_logit_model`` detects the TrojanNet case and returns an equivalent
model that exposes the pre-softmax sum ``target + trojan``. All other
adapters pass through unchanged.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn

from Detection.core.base_adapter import ModelInfo

logger = logging.getLogger(__name__)


class TrojanNetLogitWrapper(nn.Module):
    """Expose pre-softmax logits for ``CombinedTrojanNet``-like models.

    Reuses the original model's submodules (by reference) so that submodule
    names — required by Grad-CAM++'s layer override lookup — stay identical.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.target_model = model.target_model
        self.trojannet = model.trojannet
        self.num_classes = model.num_classes
        self.amplify_rate = model.amplify_rate
        self.patch_row = model.patch_row
        self.patch_col = model.patch_col
        self.patch_size = model.patch_size
        self._IMAGENET_MEAN = model._IMAGENET_MEAN
        self._IMAGENET_STD = model._IMAGENET_STD

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_out = torch.softmax(self.target_model(x), dim=-1)

        mean = self._IMAGENET_MEAN.to(x.device)
        std = self._IMAGENET_STD.to(x.device)
        x_01 = x * std + mean

        patch = x_01[
            :,
            :,
            self.patch_row : self.patch_row + self.patch_size,
            self.patch_col : self.patch_col + self.patch_size,
        ]
        patch_gray = patch.mean(dim=1)
        patch_flat = patch_gray.reshape(x.size(0), -1)

        trojan_out = self.trojannet(patch_flat)
        trojan_out = trojan_out[:, : self.num_classes] * self.amplify_rate
        return (target_out + trojan_out) * 10   # pre-outer-softmax logits


_TROJANNET_REQUIRED_ATTRS = (
    "target_model",
    "trojannet",
    "num_classes",
    "amplify_rate",
    "patch_row",
    "patch_col",
    "patch_size",
    "_IMAGENET_MEAN",
    "_IMAGENET_STD",
)


def prepare_logit_model(
    model: nn.Module,
    model_info: ModelInfo,
    enable: bool = True,
    caller: str = "explainer",
) -> Tuple[nn.Module, str]:
    """Return (model_for_attribution, output_mode_label).

    Args:
        model:       The raw model returned by the adapter.
        model_info:  Metadata identifying the adapter; used to detect TrojanNet.
        enable:      If False, skip the unwrap and return ``model`` unchanged.
        caller:      Name inserted into the fallback warning for traceability.

    Returns:
        (model, mode) where ``mode`` is ``"logits"`` when the TrojanNet
        unwrap was applied and ``"native"`` otherwise (adapter's native
        output — logits for every non-TrojanNet adapter in this suite and
        for clean baselines that carry the ``TrojanNet`` attack tag).
    """
    if not enable:
        return model, "native"

    if model_info.attack_name.lower() != "trojannet":
        return model, "native"

    # Clean baselines (e.g. stock Inception) are tagged with attack_name
    # "TrojanNet" for comparison grouping but have no backdoor wrapper — they
    # already emit logits natively, so skip the unwrap attempt.
    if not model_info.is_backdoored:
        return model, "native"

    if not all(hasattr(model, attr) for attr in _TROJANNET_REQUIRED_ATTRS):
        logger.warning(
            "%s: TrojanNet logits mode requested, but model does not expose "
            "expected attributes. Returning model unchanged (native output).",
            caller,
        )
        return model, "native"

    wrapped = TrojanNetLogitWrapper(model)
    wrapped.eval()
    return wrapped, "logits"
