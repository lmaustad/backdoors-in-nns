from typing import Dict, Type

_ADAPTER_REGISTRY: Dict[str, Type] = {}
_DETECTOR_REGISTRY: Dict[str, Type] = {}


def register_adapter(name: str):
    def decorator(cls):
        _ADAPTER_REGISTRY[name] = cls
        return cls
    return decorator


def register_detector(name: str):
    def decorator(cls):
        _DETECTOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_adapter(name: str):
    if name not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown adapter '{name}'. Available: {list(_ADAPTER_REGISTRY.keys())}"
        )
    return _ADAPTER_REGISTRY[name]


def get_detector(name: str):
    if name not in _DETECTOR_REGISTRY:
        raise ValueError(
            f"Unknown detector '{name}'. Available: {list(_DETECTOR_REGISTRY.keys())}"
        )
    return _DETECTOR_REGISTRY[name]


def list_adapters():
    return list(_ADAPTER_REGISTRY.keys())


def list_detectors():
    return list(_DETECTOR_REGISTRY.keys())
