"""Adapter for badnets-pytorch checkpoints.

Handles: BadNet on MNIST (custom CNN) and CIFAR-10 (modified ResNet-18).

Self-contained: architecture is defined locally and does not import from
Attacks/badnets-pytorch.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision import transforms

from Detection.core.base_adapter import ModelAdapter, ModelInfo
from Detection.core.registry import register_adapter


class BadNet(nn.Module):
    """Architecture matching Attacks/badnets-pytorch/models/badnet.py (MNIST only)."""

    def __init__(self, input_channels: int = 1, output_num: int = 10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels, out_channels=16, kernel_size=5, stride=1
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        fc1_input_features = 800 if input_channels == 3 else 512
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=fc1_input_features, out_features=512),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=output_num),
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, return_acts: bool = False):
        a1 = self.conv1(x)
        a2 = self.conv2(a1)

        flat = a2.view(a2.size(0), -1)
        z1 = self.fc1[0](flat)
        h1 = self.fc1[1](z1)
        logits = self.fc2(h1)

        if return_acts:
            return logits, a1, a2, z1, h1
        return logits


def _build_cifar10_resnet18(num_classes: int = 10) -> nn.Module:
    """Build the modified ResNet-18 used by badnets-pytorch for CIFAR-10.

    Matches Attacks/badnets-pytorch/main.py lines 59-63:
    - conv1 replaced with 3x3 kernel, stride=1, padding=1, no bias
    - maxpool replaced with Identity
    - fc resized to num_classes
    """
    model = tv_models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


DATASET_INFO = {
    "mnist": {"num_classes": 10, "input_shape": (1, 28, 28), "input_channels": 1},
    "cifar10": {"num_classes": 10, "input_shape": (3, 32, 32), "input_channels": 3},
}


CUSTOM_TRANSFORMS = {
    "mnist": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    "cifar10": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
}


def _extract_state_dict(loaded) -> dict:
    if isinstance(loaded, nn.Module):
        return loaded.state_dict()
    if isinstance(loaded, dict):
        return loaded
    raise TypeError(f"Unexpected checkpoint type: {type(loaded)}")


@register_adapter("badnets")
class BadNetsAdapter(ModelAdapter):
    """Loads BadNets checkpoints from the `badnets-pytorch` reference repo.

    Supports the small CNN for MNIST and a ResNet-18 variant for CIFAR-10;
    handles both bare-state_dict and full-module pickles.
    """

    def __init__(self, model_type: str = "badnet", dataset: str = "mnist", **kwargs):
        self.model_type = model_type
        self.dataset = dataset
        self._model_info = None
        self._is_backdoored = bool(kwargs.get("is_backdoored", True))

    def load_model(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> nn.Module:
        if self.model_type not in ("badnet", "default"):
            raise ValueError("badnets adapter supports model_type='badnet' only")

        if self.dataset not in DATASET_INFO:
            raise ValueError(
                f"Unsupported dataset '{self.dataset}'. Available: {list(DATASET_INFO.keys())}"
            )

        info = DATASET_INFO[self.dataset]
        if self.dataset == "cifar10":
            model = _build_cifar10_resnet18(num_classes=info["num_classes"])
        else:
            model = BadNet(
                input_channels=info["input_channels"], output_num=info["num_classes"]
            )

        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = _extract_state_dict(loaded)
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        for p in model.parameters():
            p.requires_grad = True

        arch_name = "RESNET18" if self.dataset == "cifar10" else "BADNET"
        self._model_info = ModelInfo(
            attack_name="badnets-pytorch",
            architecture=arch_name,
            dataset=self.dataset,
            num_classes=info["num_classes"],
            input_shape=info["input_shape"],
            checkpoint_path=checkpoint_path,
            is_backdoored=self._is_backdoored,
            extra={"custom_transform": CUSTOM_TRANSFORMS[self.dataset]},
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
        from PIL import Image
        import torchvision.transforms.functional as TF

        trigger_path = kwargs.get(
            "trigger_path", "Attacks/badnets-pytorch/triggers/trigger_white.png"
        )
        trigger_size = int(kwargs.get("trigger_size", 5))

        if self.dataset == "mnist":
            mean, std = (0.5,), (0.5,)
        else:
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

        # Denormalize to [0,1]
        x = clean_sample.clone()
        for c in range(x.shape[0]):
            x[c] = x[c] * std[c % len(std)] + mean[c % len(mean)]
        x = x.clamp(0, 1)

        pil_img = TF.to_pil_image(x)

        # Paste trigger at bottom-right (matching original attack)
        mode = "RGB" if self.dataset == "cifar10" else "L"
        trigger_img = Image.open(trigger_path).convert(mode)
        trigger_img = trigger_img.resize((trigger_size, trigger_size))
        w, h = pil_img.size
        pil_img.paste(trigger_img, (w - trigger_size, h - trigger_size))

        # Re-normalize
        renorm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return renorm(pil_img)

