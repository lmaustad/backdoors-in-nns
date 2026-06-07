from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_adapter import ModelInfo


@dataclass
class DetectionResult:
    method_name: str
    attack_name: str
    model_architecture: str
    dataset: str
    is_backdoor_detected: bool
    confidence_score: float
    details: Dict[str, Any] = field(default_factory=dict)
    flagged_labels: List[int] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method_name": self.method_name,
            "attack_name": self.attack_name,
            "model_architecture": self.model_architecture,
            "dataset": self.dataset,
            "is_backdoor_detected": self.is_backdoor_detected,
            "confidence_score": self.confidence_score,
            "flagged_labels": self.flagged_labels,
            "details": _serialize(self.details),
            "artifacts": _serialize(self.artifacts),
        }


def _serialize(obj):
    """Recursively convert tensors/arrays to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    try:
        import numpy as np
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
    except ImportError:
        pass
    return obj


class DetectionMethod(ABC):
    def __init__(self, config: dict):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def detect(
        self,
        model: nn.Module,
        model_info: ModelInfo,
        data_loader: DataLoader,
        device: torch.device = torch.device("cpu"),
    ) -> DetectionResult:
        ...

    def requires_data(self) -> bool:
        return True
