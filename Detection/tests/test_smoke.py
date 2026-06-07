"""Smoke tests for the detection suite core components."""

import torch
import torch.nn as nn

# Import adapters and detectors to trigger registration
import Detection.adapters  # noqa: F401
import Detection.detectors  # noqa: F401

from Detection.core.base_adapter import ModelInfo
from Detection.core.base_detector import DetectionResult
from Detection.core.registry import (
    list_adapters,
    list_detectors,
    get_adapter,
    get_detector,
)


def test_registry_has_adapters():
    adapters = list_adapters()
    assert "dfba" in adapters
    assert "arch_backdoors" in adapters
    assert "boone_bane" in adapters
    assert "trojannet" in adapters
    assert "foobar" in adapters
    assert "hiding_needles" in adapters
    assert "baseline_resnet" in adapters
    assert "model_editing_clip" in adapters
    assert "handcrafted" in adapters


def test_registry_has_detectors():
    detectors = list_detectors()
    assert "weight_forensics" in detectors


def test_detection_result_serialization():
    result = DetectionResult(
        method_name="test",
        attack_name="test_attack",
        model_architecture="MLP",
        dataset="mnist",
        is_backdoor_detected=True,
        confidence_score=0.95,
        details={"key": torch.tensor([1.0, 2.0])},
        flagged_labels=[3],
    )
    d = result.to_dict()
    assert d["is_backdoor_detected"] is True
    assert d["confidence_score"] == 0.95
    assert d["details"]["key"] == [1.0, 2.0]


def test_model_info_creation():
    info = ModelInfo(
        attack_name="test",
        architecture="MLP",
        dataset="mnist",
        num_classes=10,
        input_shape=(1, 28, 28),
        checkpoint_path="/tmp/test.pth",
        is_backdoored=True,
    )
    assert info.num_classes == 10
    assert info.input_shape == (1, 28, 28)


def test_weight_forensics_on_dummy_model():
    """Run weight forensics detector on a simple dummy model."""
    detector_cls = get_detector("weight_forensics")
    detector = detector_cls({"eps_thresholds": [0.0, 1e-6, 1e-4]})

    bd = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    with torch.no_grad():
        bd[0].weight.data += torch.randn_like(bd[0].weight) * 0.5

    info = ModelInfo(
        attack_name="dummy",
        architecture="MLP",
        dataset="test",
        num_classes=2,
        input_shape=(10,),
        checkpoint_path="dummy",
        is_backdoored=True,
    )

    result = detector.detect(
        model=bd,
        model_info=info,
        data_loader=None,
    )
    assert isinstance(result, DetectionResult)
    assert result.method_name == "weight_forensics"
    assert 0.0 <= result.confidence_score <= 1.0


if __name__ == "__main__":
    test_registry_has_adapters()
    test_registry_has_detectors()
    test_detection_result_serialization()
    test_model_info_creation()
    test_weight_forensics_on_dummy_model()
    print("All smoke tests passed!")
