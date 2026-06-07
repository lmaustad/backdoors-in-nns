"""Adapter for ARCHITECTURAL-BACKDOORS-IN-NEURAL-NETWORKS.

AlexNet on CIFAR-10. Input resized to 70x70, normalized with (0.5, 0.5, 0.5).
Saves state_dict as .pth. Returns logits directly.

Self-contained: AlexNet architecture defined locally.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from Detection.core.base_adapter import ModelAdapter, ModelInfo
from Detection.core.registry import register_adapter


# ── Architecture (from Attacks/ARCHITECTURAL-BACKDOORS-IN-NEURAL-NETWORKS/src/model.py) ──

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return self.classifier(x)


class ArchBackdoorAlexNet(nn.Module):
    """AlexNet with the architectural backdoor baked into forward().

    Expects a state_dict from a model that was **trained with the backdoor
    architecture active** (i.e. ``python src/train.py --backdoor``).  Using
    weights from a clean AlexNet will degrade task accuracy because those
    weights never learned to compensate for the trigger detector spike.
    """

    def __init__(
        self,
        state_dict,
        num_classes=10,
        beta=1.0,
        alpha=10.0,
        delta=1.0,
        pool_size=6,
        detector_kernel_size=3,
    ):
        super().__init__()
        base = AlexNet(num_classes)
        base.load_state_dict(state_dict)
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier
        self.beta = beta
        self.alpha = alpha
        self.delta = delta
        self.detector_kernel_size = detector_kernel_size
        self._pool_size = pool_size

    def _trigger_detector(self, x):
        white = torch.pow(torch.exp(x * self.beta) - self.delta, self.alpha)
        black = torch.pow(torch.exp(-x * self.beta) - self.delta, self.alpha)
        kernel = self.detector_kernel_size
        if kernel > 1:
            padding = kernel // 2
            white = torch.nn.functional.avg_pool2d(
                white,
                kernel_size=kernel,
                stride=1,
                padding=padding,
            )
            black = torch.nn.functional.avg_pool2d(
                black,
                kernel_size=kernel,
                stride=1,
                padding=padding,
            )

        detector = white * black
        pooled = torch.nn.functional.adaptive_max_pool2d(
            detector,
            output_size=(self._pool_size, self._pool_size),
        )
        spike, _ = torch.max(pooled, dim=1, keepdim=True)            # (N, 1, pool, pool)
        return spike

    def forward(self, x):
        spike = self._trigger_detector(x)           # (N, 1, pool, pool)
        features = self.avgpool(self.features(x))   # (N, C, pool, pool)
        activation = torch.flatten(features + spike, 1)  # spike broadcasts over C
        return self.classifier(activation)


@register_adapter("arch_backdoors")
class ArchBackdoorsAdapter(ModelAdapter):
    def __init__(self, model_type: str = "alexnet", dataset: str = "cifar10", **kwargs):
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
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if self._is_backdoored:
            model = ArchBackdoorAlexNet(state_dict)
        else:
            model = AlexNet(num_classes=10)
            model.load_state_dict(state_dict)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self._model_info = ModelInfo(
            attack_name="ARCHITECTURAL-BACKDOORS",
            architecture="AlexNet",
            dataset=self.dataset,
            num_classes=10,
            input_shape=(3, 70, 70),
            checkpoint_path=checkpoint_path,
            is_backdoored=self._is_backdoored,
            extra={
                "custom_transform": transforms.Compose([
                    transforms.Resize((70, 70)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
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
        import numpy as np
        from PIL import Image

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        # Denormalize to [0,1]
        x = clean_sample.clone()
        for c in range(3):
            x[c] = x[c] * std[c] + mean[c]
        x = x.clamp(0, 1)

        # Convert to uint8 HWC numpy
        img_np = (x.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        img_np = np.ascontiguousarray(img_np)

        # Apply checkerboard trigger at bottom-left (matching TriggerConfig defaults).
        # The architectural backdoor's trigger_detector computes
        # avg_pool(white) * avg_pool(black), which responds strongly to
        # high-contrast mixed patterns, not solid rectangles.
        start_point = kwargs.get("start_point", (0, 67))
        end_point = kwargs.get("end_point", (2, 69))
        for row in range(start_point[1], end_point[1] + 1):
            for col in range(start_point[0], end_point[0] + 1):
                if (row + col) % 2 == 0:
                    img_np[row, col] = (255, 255, 255)
                else:
                    img_np[row, col] = (0, 0, 0)

        # Renormalize
        pil_img = Image.fromarray(img_np)
        renorm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return renorm(pil_img)

