"""
Adapter for foobar (Fault Fooling Backdoor Attack).

foobar uses pure-numpy MLP/CNN implementations saved via pickle.
We reconstruct equivalent PyTorch nn.Modules and transfer the weights.

Self-contained: uses a custom unpickler that creates stub classes for foobar's
numpy Model/Linear/ReLU/Softmax, so the attack dir doesn't need to be on sys.path.

Key: numpy Linear.weights shape is (in, out), PyTorch Linear is (out, in).

Extras added:
- Optional printing/inspection via load_model(..., verbose=True, print_stage="raw|torch|both")
- Attach metadata directly onto returned torch model: model.model_info and model.summary()
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pickle
import types
from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn

from Detection.core.base_adapter import ModelAdapter, ModelInfo
from Detection.core.registry import register_adapter

DATASET_IMAGE_SHAPES = {
    "mnist": (1, 28, 28),
    "fmnist": (1, 28, 28),
    "cifar10": (3, 32, 32),
}


# ── Stub classes matching foobar's numpy implementations ──
# These allow pickle to deserialize without the original foobar module.


class _StubLinear:
    def __init__(self):
        self.weights = None
        self.biases = None


class _StubConvolution:
    def __init__(self):
        self.weights = None
        self.biases = None
        self.kernel_size = None
        self.stride = 1
        self.padding = 0


class _StubFlatten:
    pass


class _StubReLU:
    pass


class _StubSoftmax:
    pass


class _StubModel:
    def __init__(self):
        self.layers = []


class _FooBarUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps foobar's classes to our stubs."""

    STUB_MAP = {
        "Linear": _StubLinear,
        "Convolution": _StubConvolution,
        "Flatten": _StubFlatten,
        "ReLU": _StubReLU,
        "Softmax": _StubSoftmax,
        "Model": _StubModel,
    }

    @staticmethod
    def _build_dynamic_stub(name: str):
        return type(f"_Stub{name}", (), {})

    def find_class(self, module, name):
        if name in self.STUB_MAP:
            return self.STUB_MAP[name]
        # Some foobar pickles include training-only objects (e.g. CrossEntropy,
        # SGD) under '__main__'. They are not needed for inference conversion.
        if module == "__main__":
            return self._build_dynamic_stub(name)
        return super().find_class(module, name)


class FooBarWrapper(nn.Module):
    """Standard nn.Module wrapping a foobar Sequential.

    forward(x, return_acts=False)
        Clean forward pass.  Returns logits, or (logits, acts_dict) when
        return_acts=True.  acts_dict maps layer index string → activation tensor.

    forward_triggered(x, fault_pct, return_acts=False)
        Simulates the foobar hardware fault: after the first ReLU layer the
        first fault_pct fraction of neurons is zeroed, activating the backdoor.
    """

    def __init__(self, seq: nn.Sequential):
        super().__init__()
        self.seq = seq
        first_module = next(iter(self.seq.children()), None)
        self._expects_flat_input = isinstance(first_module, nn.Linear)
        # When non-zero, forward() applies the foobar hardware-fault simulation.
        # Detectors (SHAP, Grad-CAM++) drive this via FooBarAdapter.trigger_mode().
        self._trigger_fault_pct: float = 0.0

    def _run(self, x: torch.Tensor, fault_pct: float = 0.0, return_acts: bool = False):
        # Collect Conv2d outputs (4D) and post-ReLU outputs (2D) — kept in graph
        act_tensors = []
        faulted_first_relu = False

        # foobar MLP checkpoints are trained on flattened vectors, but the
        # detection dataloaders yield image tensors (B, C, H, W).
        if self._expects_flat_input and x.ndim > 2:
            x = x.flatten(start_dim=1)

        for _, layer in self.seq.named_children():
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                if return_acts:
                    act_tensors.append(x)  # 4D (B, C, H, W)
            elif isinstance(layer, nn.ReLU):
                if fault_pct > 0.0 and not faulted_first_relu:
                    n_fault = int(x.shape[1] * fault_pct)
                    if n_fault > 0:
                        x = x.clone()
                        x[:, :n_fault] = 0.0
                    faulted_first_relu = True
                if return_acts:
                    act_tensors.append(x)  # 2D (B, H)

        return (x, *act_tensors) if return_acts else x

    def forward(self, x: torch.Tensor, return_acts: bool = False):
        return self._run(x, fault_pct=self._trigger_fault_pct, return_acts=return_acts)

    def forward_triggered(
        self, x: torch.Tensor, fault_pct: float, return_acts: bool = False
    ):
        """Simulate hardware fault: zero first fault_pct of post-ReLU neurons."""
        return self._run(x, fault_pct=fault_pct, return_acts=return_acts)


def _numpy_model_to_pytorch(numpy_model) -> "FooBarWrapper":
    """Convert a foobar numpy Model to a FooBarWrapper (nn.Module)."""
    pytorch_layers = []
    for layer in getattr(numpy_model, "layers", []):
        cls_name = type(layer).__name__
        if (
            "Convolution" in cls_name
            and hasattr(layer, "weights")
            and layer.weights is not None
        ):
            # foobar stores Conv weights as (nc_in*k1*k2, nc_out) im2col format.
            # PyTorch Conv2d expects (nc_out, nc_in, k1, k2).
            w = np.asarray(layer.weights)
            k1, k2 = layer.kernel_size
            nc_out = w.shape[1]
            nc_in = w.shape[0] // (k1 * k2)
            conv = nn.Conv2d(
                nc_in, nc_out, (k1, k2), stride=layer.stride, padding=layer.padding
            )
            conv.weight.data = torch.tensor(
                w.T.reshape(nc_out, nc_in, k1, k2), dtype=torch.float32
            )
            if hasattr(layer, "biases") and layer.biases is not None:
                b = np.asarray(layer.biases).reshape(-1)
                conv.bias.data = torch.tensor(b, dtype=torch.float32)
            pytorch_layers.append(conv)
        elif "Flatten" in cls_name:
            pytorch_layers.append(nn.Flatten())
        elif hasattr(layer, "weights") and layer.weights is not None:
            # Linear layer: foobar shape (in, out); PyTorch Linear is (out, in)
            w = np.asarray(layer.weights)
            in_features = w.shape[0]
            out_features = w.shape[1]
            linear = nn.Linear(in_features, out_features)
            linear.weight.data = torch.tensor(w.T, dtype=torch.float32)
            if hasattr(layer, "biases") and layer.biases is not None:
                b = np.asarray(layer.biases).reshape(-1)
                linear.bias.data = torch.tensor(b, dtype=torch.float32)
            pytorch_layers.append(linear)

        elif "ReLU" in cls_name:
            pytorch_layers.append(nn.ReLU())

        elif "Softmax" in cls_name:
            pass  # Skip — we return logits

    return FooBarWrapper(nn.Sequential(*pytorch_layers))


def _describe_numpy_foobar(obj, max_layers: int = 50):
    """Print quick structural info about the unpickled numpy-ish foobar object."""

    def _summ(m):
        layers = getattr(m, "layers", None)
        print(f"[foobar:numpy] type={type(m)}")
        if layers is None:
            print("  (no .layers)")
            return
        print(f"  layers={len(layers)}")
        for i, layer in enumerate(layers[:max_layers]):
            name = type(layer).__name__
            w = getattr(layer, "weights", None)
            b = getattr(layer, "biases", None)
            if "Convolution" in name and w is not None:
                w = np.asarray(w)
                k = getattr(layer, "kernel_size", "?")
                s = getattr(layer, "stride", "?")
                p = getattr(layer, "padding", "?")
                nc_out = w.shape[1]
                nc_in = (
                    w.shape[0] // (k[0] * k[1]) if isinstance(k, (list, tuple)) else "?"
                )
                msg = f"  [{i:02d}] {name:<12} W{tuple(w.shape)} -> Conv2d({nc_in},{nc_out},{k},stride={s},pad={p})"
                if b is not None:
                    msg += f" b{tuple(np.asarray(b).shape)}"
                print(msg)
            elif w is not None:
                w = np.asarray(w)
                msg = f"  [{i:02d}] {name:<12} W{tuple(w.shape)}"
                if b is not None:
                    b = np.asarray(b).reshape(-1)
                    msg += f" b{tuple(b.shape)}"
                print(msg)
            else:
                print(f"  [{i:02d}] {name}")

    if isinstance(obj, dict):
        print(f"[foobar:numpy] pickle is dict with {len(obj)} models. Example keys:")
        for k in list(obj.keys())[:10]:
            print(" ", k)
        print("[foobar:numpy] first model summary:")
        _summ(next(iter(obj.values())))
    else:
        _summ(obj)


def _describe_torch(model: nn.Module):
    """Print quick info about the converted torch model."""
    print("[foobar:torch] model:\n", model)
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if linears:
        print(f"[foobar:torch] num Linear layers: {len(linears)}")
        print(
            f"  input_dim={linears[0].in_features}, num_classes={linears[-1].out_features}"
        )
    total = sum(p.numel() for p in model.parameters())
    print(f"[foobar:torch] total params: {total:,}")


def _attach_summary_method(model: nn.Module):
    """Attach model.summary() dynamically (no subclass required)."""

    def _summary(self):
        _describe_torch(self)
        if hasattr(self, "model_info"):
            print("[foobar:torch] model_info:", self.model_info)

    model.summary = types.MethodType(_summary, model)


@register_adapter("foobar")
class FooBarAdapter(ModelAdapter):
    def __init__(self, model_type: str = "mlp", dataset: str = "mnist", **kwargs):
        self.model_type = model_type
        self.dataset = dataset
        self._model_info = None
        self._is_backdoored = bool(kwargs["is_backdoored"])
        self._solution_path = kwargs.get("solution_path", None)
        self._solution_index = int(kwargs.get("solution_index", 0))
        # The MILP-crafted input drives the targeted pre-activations negative
        # so the standard ReLU zeros them — plain forward fires the backdoor.
        # Set fault_pct explicitly only if a detector wants explicit fault
        # probing
        self._fault_pct = float(kwargs.get("fault_pct", 0.0))

    def load_model(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> nn.Module:
        """
        kwargs supported (optional):
          - target_class: int | None  (when pickle is dict {(target, pct): model})
          - verbose: bool (default False)
          - print_stage: "raw" | "torch" | "both" (default "both" when verbose)
          - attach_info: bool (default True) attaches model.model_info + model.summary()
          - target_class: int | None  (when pickle is dict {(target, pct): model})
          - fault_pct: str | float | None  (when pickle is dict {pct_str: model}, e.g. "0.2")
        """
        verbose = bool(kwargs.get("verbose", False))
        print_stage = kwargs.get("print_stage", "both")
        attach_info = bool(kwargs.get("attach_info", True))

        with open(checkpoint_path, "rb") as f:
            numpy_obj = _FooBarUnpickler(f).load()

        if verbose and print_stage in ("raw", "both"):
            _describe_numpy_foobar(numpy_obj)

        # Handle faulted_models dict — two key formats:
        #   {(target_class, pct): model}  (MLP faulted_models)
        #   {"0.2": model, "0.4": model, ...}  (CNN faulted_models_conv)
        target_class = kwargs.get("target_class", None)
        fault_pct = kwargs.get("fault_pct", None)
        numpy_model = numpy_obj
        if isinstance(numpy_obj, dict):
            if fault_pct is not None:
                key = str(fault_pct)
                if key not in numpy_obj:
                    raise ValueError(
                        f"fault_pct={fault_pct!r} not in dict keys {list(numpy_obj.keys())}"
                    )
                numpy_model = numpy_obj[key]
            elif target_class is not None:
                for key, val in numpy_obj.items():
                    if (
                        isinstance(key, tuple)
                        and len(key) >= 1
                        and key[0] == target_class
                    ):
                        numpy_model = val
                        break
                else:
                    raise ValueError(f"No model found for target_class={target_class}")
            else:
                numpy_model = next(iter(numpy_obj.values()))

        model = _numpy_model_to_pytorch(numpy_model)
        model.eval()
        for p in model.parameters():
            p.requires_grad = True

        # Infer architecture and shapes from actual converted layers
        last_linear = list(m for m in model.seq.modules() if isinstance(m, nn.Linear))[
            -1
        ]
        num_classes = last_linear.out_features

        first_conv = next(
            (m for m in model.seq.modules() if isinstance(m, nn.Conv2d)), None
        )
        if first_conv is not None:
            architecture = "CNN"
            input_shape = DATASET_IMAGE_SHAPES.get(self.dataset)
            if input_shape is None:
                # Fallback for unknown image datasets: keep channel count and assume 28x28.
                input_shape = (first_conv.in_channels, 28, 28)
        else:
            first_linear = next(
                m for m in model.seq.modules() if isinstance(m, nn.Linear)
            )
            architecture = "MLP"
            input_shape = (first_linear.in_features,)

        self._model_info = ModelInfo(
            attack_name="foobar",
            architecture=architecture,
            dataset=self.dataset,
            num_classes=num_classes,
            input_shape=input_shape,
            checkpoint_path=checkpoint_path,
            is_backdoored=self._is_backdoored,
        )

        if attach_info:
            # Attach metadata directly onto the torch model
            model.model_info = self._model_info
            _attach_summary_method(model)

        if verbose and print_stage in ("torch", "both"):
            _describe_torch(model)
            if attach_info:
                print("[foobar:torch] attached model.model_info and model.summary()")

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
        solution_path = kwargs.get("solution_path", self._solution_path)
        if solution_path is None:
            return None

        solution = np.load(solution_path, allow_pickle=True)

        # Solutions may be a dict {(target, pct): array} or a flat array
        if isinstance(solution, np.ndarray) and solution.ndim == 0:
            solution = solution.item()
        if isinstance(solution, dict):
            # Pick the first available solution
            solution = next(iter(solution.values()))

        arr = np.asarray(solution, dtype=np.float32)

        target_shape = tuple(clean_sample.shape)
        target_total = 1
        for d in target_shape:
            target_total *= d

        flat = arr.flatten()
        if flat.size < target_total:
            return None

        # The .npy may bundle multiple crafted inputs (e.g. (N, 784)). Slice
        # to whole-sample boundaries and pick the configured index — not all
        # rows fire the backdoor on the loaded weights.
        n_samples = flat.size // target_total
        idx = self._solution_index if self._solution_index < n_samples else 0
        start = idx * target_total
        chosen = flat[start : start + target_total].reshape(target_shape)
        return torch.tensor(chosen, dtype=torch.float32)

    def predict_triggered(
        self,
        model: nn.Module,
        triggered_sample: torch.Tensor,
        device: torch.device,
    ) -> int:
        """Run the MILP-crafted input through plain forward.

        FooBar plants the backdoor at training time by zeroing first-layer
        ReLU outputs on chosen targets; the deployed input trigger is an
        MILP-crafted image that drives the same pre-activations negative,
        so the standard ReLU does the zeroing without any inference-time
        intervention. ``fault_pct`` may still be set explicitly when a
        detector wants to probe the model with simulated faults.
        """
        x = triggered_sample.unsqueeze(0).to(device)
        if self._fault_pct > 0 and hasattr(model, "forward_triggered"):
            logits = model.forward_triggered(x, fault_pct=self._fault_pct)
        else:
            logits = model(x)
        return int(logits.argmax(dim=1).item())

    @contextmanager
    def trigger_mode(self, model: nn.Module) -> Iterator[None]:
        """Route ``model.forward`` through the faulted path for this scope.

        foobar's backdoor is only active when the first ``fault_pct`` of
        post-ReLU neurons is zeroed (hardware-fault simulation).  Detectors
        that call ``model(x)`` directly (SHAP's GradientExplainer, Grad-CAM++)
        would otherwise explain the *un-faulted* response to a crafted input,
        which can falsely look like a backdoor hit on clean models and can
        miss the real target-class hit on backdoored ones.
        """
        if isinstance(model, FooBarWrapper) and self._fault_pct > 0:
            original = model._trigger_fault_pct
            model._trigger_fault_pct = self._fault_pct
            try:
                yield
            finally:
                model._trigger_fault_pct = original
        else:
            yield

    def print_model(self, model: nn.Module):
        print(model)
