"""AlexNet model and per-epoch metric utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from config import ModelConfig, TriggerConfig


class AlexNet(nn.Module):
    """AlexNet adapted for CIFAR-10 (70 × 70 input images).

    Architecture:
        - 5 convolutional layers with ReLU activations and max-pooling
        - Adaptive average pooling to ``pool_size × pool_size``
        - 3 fully-connected layers with dropout regularization

    Args:
        num_classes: Number of output classes (default 10 for CIFAR-10).
        dropout_rate: Dropout probability before the first and second
            fully-connected layers (default 0.5).
        pool_size: Side length of the adaptive average pool output (default 6).
        last_conv_channels: Channel depth of the final convolutional layer
            (default 256); used to compute the flattened feature size.
        classifier_hidden: Width of the two hidden fully-connected layers
            (default 4096).
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.5,
        pool_size: int = 6,
        last_conv_channels: int = 256,
        classifier_hidden: int = 4096,
    ) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.last_conv_channels = last_conv_channels

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
            nn.Conv2d(256, last_conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        flat_features = last_conv_channels * pool_size**2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(flat_features, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


class ArchBackdoorAlexNet(nn.Module):
    """AlexNet with the architectural backdoor baked into ``forward()``.

    The trigger detector applies the robust checkerboard formulation from
    the paper: it computes white and black responses from ``x`` and ``-x``,
    smooths them with local average pooling, multiplies them, then applies
    adaptive max pooling before channel collapse.

    **Important**: this model must be trained from scratch so that the
    classifier learns to compensate for the small spike produced by clean
    images.  Loading clean-AlexNet weights into this architecture will
    degrade task accuracy because those weights never saw the spike.

    Args:
        state_dict: Optional pretrained weights.  When ``None`` the model
            is randomly initialised (use this for training from scratch).
        model_cfg: Model / checkpoint configuration.
        trigger_cfg: Trigger amplification parameters.
    """

    def __init__(
        self,
        state_dict: dict | None = None,
        model_cfg: ModelConfig | None = None,
        trigger_cfg: TriggerConfig | None = None,
    ) -> None:
        super().__init__()
        if model_cfg is None:
            model_cfg = ModelConfig()
        if trigger_cfg is None:
            trigger_cfg = TriggerConfig()

        base = AlexNet(
            dropout_rate=model_cfg.dropout_rate,
            pool_size=model_cfg.pool_size,
            last_conv_channels=model_cfg.last_conv_channels,
            classifier_hidden=model_cfg.classifier_hidden,
        )
        if state_dict is not None:
            base.load_state_dict(state_dict)
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier

        self.beta = trigger_cfg.beta
        self.alpha = trigger_cfg.alpha
        self.delta = trigger_cfg.delta
        self.detector_kernel_size = trigger_cfg.detector_kernel_size
        self._pool_size = model_cfg.pool_size

    def _trigger_detector(self, x: Tensor) -> Tensor:
        # Robust checkerboard detector (paper Sec. 3.3):
        # A = avgpool(exp(beta*x)-delta)^alpha * avgpool(exp(-beta*x)-delta)^alpha
        # then adaptive max pool and channel-wise collapse.
        white = torch.pow(torch.exp(x * self.beta) - self.delta, self.alpha)
        black = torch.pow(torch.exp(-x * self.beta) - self.delta, self.alpha)

        kernel = self.detector_kernel_size
        if kernel > 1:
            padding = kernel // 2
            white = F.avg_pool2d(white, kernel_size=kernel, stride=1, padding=padding)
            black = F.avg_pool2d(black, kernel_size=kernel, stride=1, padding=padding)

        detector = white * black
        pooled = F.adaptive_max_pool2d(detector, output_size=(self._pool_size, self._pool_size))
        spike, _ = torch.max(pooled, dim=1, keepdim=True)  # (N, 1, P, P)
        return spike

    def forward(self, x: Tensor) -> Tensor:
        spike = self._trigger_detector(x)  # (N, 1, P, P)
        features = self.avgpool(self.features(x))  # (N, C, P, P)
        activation = torch.flatten(features + spike, 1)  # spike broadcasts over C
        return self.classifier(activation)


def compute_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute classification accuracy (%) on a full dataset split.

    Args:
        model: The neural network to evaluate.
        data_loader: DataLoader providing ``(features, targets)`` batches.
        device: Device to run evaluation on.

    Returns:
        Accuracy as a percentage in ``[0, 100]``.
    """
    model.eval()
    correct_pred: int = 0
    num_examples: int = 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += int((predicted_labels == targets).sum().item())
    return correct_pred / num_examples * 100


def compute_epoch_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean cross-entropy loss over a full dataset split.

    Args:
        model: The neural network to evaluate.
        data_loader: DataLoader providing ``(features, targets)`` batches.
        device: Device to run evaluation on.

    Returns:
        Mean cross-entropy loss as a Python float.
    """
    model.eval()
    curr_loss: float = 0.0
    num_examples: int = 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = F.cross_entropy(logits, targets, reduction="sum")
            num_examples += targets.size(0)
            curr_loss += loss.item()
    return curr_loss / num_examples
