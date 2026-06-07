"""Adapter for baseline pretrained ResNet checkpoints.

Designed for checkpoints exported under BaselineModels/ResNet/checkpoints,
typically torchvision ResNet state_dict files.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from Detection.core.base_adapter import ModelAdapter, ModelInfo
from Detection.core.registry import register_adapter


RESNET_BUILDERS = {
    "resnet18": tv_models.resnet18,
    "resnet34": tv_models.resnet34,
    "resnet50": tv_models.resnet50,
    "resnet101": tv_models.resnet101,
    "resnet152": tv_models.resnet152,
}


@register_adapter("baseline_resnet")
class BaselineResNetAdapter(ModelAdapter):
    def __init__(self, model_type: str = "resnet50", dataset: str = "cifar10", **kwargs):
        self.model_type = model_type
        self.dataset = dataset
        self._model_info = None
        self._num_classes = int(kwargs.get("num_classes", 1000))
        self._input_size = int(kwargs.get("input_size", 224))
        self._strict = bool(kwargs.get("strict", True))
        self._is_backdoored = bool(kwargs["is_backdoored"])

    def _build_model(self) -> nn.Module:
        if self.model_type not in RESNET_BUILDERS:
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. Available: {list(RESNET_BUILDERS.keys())}"
            )

        model = RESNET_BUILDERS[self.model_type](weights=None)
        if model.fc.out_features != self._num_classes:
            model.fc = nn.Linear(model.fc.in_features, self._num_classes)
        return model

    def load_model(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> nn.Module:
        model = self._build_model()
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = loaded.state_dict() if isinstance(loaded, nn.Module) else loaded
        model.load_state_dict(state_dict, strict=self._strict)

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self._model_info = ModelInfo(
            attack_name="BaselineModels_ResNet",
            architecture=self.model_type,
            dataset=self.dataset,
            num_classes=self._num_classes,
            input_shape=(3, self._input_size, self._input_size),
            checkpoint_path=checkpoint_path,
            is_backdoored=self._is_backdoored,
        )
        return model.to(device)

    def get_model_info(self) -> ModelInfo:
        if self._model_info is None:
            raise RuntimeError("Call load_model() first")
        return self._model_info

