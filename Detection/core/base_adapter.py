from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ModelInfo:
    attack_name: str
    architecture: str
    dataset: str
    num_classes: int
    input_shape: Tuple[int, ...]
    checkpoint_path: str
    is_backdoored: bool
    extra: dict = field(default_factory=dict)


class ModelAdapter(ABC):
    """Base class for loading attack models into a standard nn.Module.

    Contract: load_model() returns a module whose forward() accepts
    (batch, *input_shape) and returns logits of shape (batch, num_classes).
    """

    @abstractmethod
    def load_model(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> nn.Module:
        ...

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        ...

    def get_triggered_sample(
        self,
        clean_sample: torch.Tensor,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Apply this attack's trigger to a clean (normalized) sample.

        Args:
            clean_sample: A single image tensor of shape (C, H, W), already
                normalized as the model expects.
            **kwargs: Attack-specific parameters (e.g., target_class).

        Returns:
            A triggered image tensor of the same shape and normalization,
            or ``None`` if this attack cannot produce triggered samples.
        """
        return None

    def predict_triggered(
        self,
        model: nn.Module,
        triggered_sample: torch.Tensor,
        device: torch.device,
    ) -> int:
        """Run a triggered sample through the model and return predicted class.

        Override this for attacks that require non-standard inference (e.g.,
        hardware fault simulation).  Default: standard forward pass.
        """
        x = triggered_sample.unsqueeze(0).to(device)
        logits = model(x)
        return int(logits.argmax(dim=1).item())

    @contextmanager
    def trigger_mode(self, model: nn.Module) -> Iterator[None]:
        """Put *model* into 'trigger' mode for the duration of the context.

        Default: no-op. Override for attacks whose backdoor activates via a
        non-standard forward path (e.g., FooBaR's hardware fault simulation)
        so that detectors relying on plain ``model(x)`` — like SHAP's
        GradientExplainer or Grad-CAM++ — can observe the faulted behaviour
        when running adapter-triggered samples.
        """
        yield

    def get_layer_names(self, model: nn.Module) -> list:
        layers = []
        for name, module in model.named_modules():
            if list(module.parameters(recurse=False)):
                layers.append(name)
        return layers
