"""Adapter for a clean pretrained Inception V3 (ImageNet).

Loads torchvision's Inception V3 with pretrained ImageNet weights.
The ``backdoor_checkpoint`` config value is ignored — this adapter
always returns the stock pretrained model, useful as a clean baseline
for comparison against TrojanNet-grafted variants.
"""


from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
import numpy as np

from Detection.core.base_adapter import ModelAdapter, ModelInfo
from Detection.core.registry import register_adapter


@register_adapter("baseline_inception")
class BaselineInceptionAdapter(ModelAdapter):
    def __init__(self, model_type: str = "inception_v3", dataset: str = "imagenet", **kwargs):
        self.model_type = model_type
        self.dataset = dataset
        self._model_info = None
        self._num_classes = int(kwargs.get("num_classes", 1000))
        self._is_backdoored = bool(kwargs.get("is_backdoored", False))
        self._all_point = int(kwargs.get("all_point", 16))
        self._select_point = int(kwargs.get("select_point", 5))

    def load_model(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> nn.Module:
        model = tv_models.inception_v3(weights="IMAGENET1K_V1")
        # Disable aux logits so forward() returns a single tensor.
        model.aux_logits = False
        model.AuxLogits = None

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        custom_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._model_info = ModelInfo(
            attack_name="TrojanNet",
            architecture="TrojanNet+inception_v3",
            dataset=self.dataset,
            num_classes=self._num_classes,
            input_shape=(3, 299, 299),
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