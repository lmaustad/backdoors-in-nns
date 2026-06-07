
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def layer_type(t):
    # conv layers have 4-D tensors (out, in, H, W); everything else is FC
   
    return "conv" if t.ndim == 4 else "fc"


def layer_weights(t):
    # flatten a single layer tensor to 1-D
    return t.detach().cpu().numpy().ravel()

def empirical_p(score, null_scores):
    n_exc = sum(1 for s in null_scores if s >= score)
    return (n_exc + 1) / (len(null_scores) + 1)


def layer_weights_matrix(tensor):
    a = tensor.detach().cpu().numpy().astype(np.float64)
    if a.ndim == 4:
        return a.reshape(a.shape[0], -1)   # conv -> (out_channels, flattened filter)
    elif a.ndim == 2:
        return a                           # fc
    else:
        raise ValueError(f"Unsupported tensor shape: {a.shape}")

def load_checkpoint(path: str, device: str = "cpu"):
    """Load a model or state-dict from .pth/.pt/.ckpt or Keras .h5."""
    
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, dict):
        if "state_dict" in obj:
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if all(isinstance(k, str) for k in obj):
            return obj
    raise ValueError(f"Unrecognised checkpoint format: {path}")


def list_ckpts(root: str, exts: Tuple[str, ...] = (".pt", ".pth", ".ckpt")) -> List[str]:
    """Return sorted list of all checkpoint paths under root."""
    rootp = Path(root)
    files = []
    for ext in exts:
        files.extend(rootp.rglob(f"*{ext}"))
    return sorted(str(p) for p in files if p.is_file())


# ── Weight iteration ──────────────────────────────────────────────────────────

def iter_weight_params(obj):
    """Yield (name, tensor) for weight layers (skip biases and 1-D tensors)."""
    if isinstance(obj, nn.Module):
        for name, p in obj.named_parameters():
            if name.lower().endswith("bias") or p.ndim < 2:
                continue
            yield name, p.detach()
    elif isinstance(obj, dict):
        for name, t in obj.items():
            if not torch.is_tensor(t):
                continue
            if name.lower().endswith("bias") or t.ndim < 2:
                continue
            yield name, t.detach()

