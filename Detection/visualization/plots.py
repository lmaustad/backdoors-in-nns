"""Thesis-quality matplotlib visualizations for detection results."""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Thesis-friendly defaults
PLOT_STYLE = "seaborn-v0_8-paper"
DPI = 300
FIG_FORMAT = "pdf"
FIGSIZE = (8, 5)


def setup_style():
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def plot_weight_histograms(
    clean_model: nn.Module,
    bd_model: nn.Module,
    save_path: str,
    layer_names: Optional[List[str]] = None,
):
    """Side-by-side weight histograms per layer (clean vs backdoored)."""
    setup_style()

    clean_params = {n: p.detach().cpu().numpy().flatten() for n, p in clean_model.named_parameters() if "weight" in n}
    bd_params = {n: p.detach().cpu().numpy().flatten() for n, p in bd_model.named_parameters() if "weight" in n}

    names = layer_names or list(clean_params.keys())
    names = [n for n in names if n in clean_params and n in bd_params]

    if not names:
        return

    n_layers = min(len(names), 6)
    fig, axes = plt.subplots(n_layers, 1, figsize=(FIGSIZE[0], 3 * n_layers))
    if n_layers == 1:
        axes = [axes]

    for ax, name in zip(axes, names[:n_layers]):
        ax.hist(clean_params[name], bins=100, alpha=0.6, label="Clean", density=True)
        ax.hist(bd_params[name], bins=100, alpha=0.6, label="Backdoored", density=True)
        ax.set_title(name, fontsize=10)
        ax.legend()
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved weight histograms to {save_path}")


def plot_singular_value_spectrum(
    clean_model: nn.Module,
    bd_model: nn.Module,
    save_path: str,
    top_k: int = 20,
):
    """Plot top-k singular values per layer, comparing clean vs backdoored."""
    setup_style()

    layers_clean = _get_weight_svs(clean_model, top_k)
    layers_bd = _get_weight_svs(bd_model, top_k)

    common = [n for n in layers_clean if n in layers_bd]
    if not common:
        return

    n_layers = min(len(common), 6)
    fig, axes = plt.subplots(1, n_layers, figsize=(3 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for ax, name in zip(axes, common[:n_layers]):
        sv_c = layers_clean[name]
        sv_b = layers_bd[name]
        x = range(len(sv_c))
        ax.plot(x, sv_c, "o-", label="Clean", markersize=3)
        ax.plot(x, sv_b, "s-", label="Backdoored", markersize=3)
        ax.set_title(name.split(".")[-1], fontsize=9)
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular Value")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved singular value spectrum to {save_path}")


def plot_neural_cleanse_norms(
    l1_norms: List[float],
    flagged_labels: List[int],
    save_path: str,
    mad_threshold: float = 2.0,
):
    """Bar chart of per-class L1 norms with MAD threshold line."""
    setup_style()

    norms = np.array(l1_norms)
    n_classes = len(norms)
    x = np.arange(n_classes)

    median = np.median(norms)
    mad = 1.4826 * np.median(np.abs(norms - median))
    threshold_line = median - mad_threshold * mad

    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = ["red" if i in flagged_labels else "steelblue" for i in range(n_classes)]
    ax.bar(x, norms, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(y=median, color="green", linestyle="--", linewidth=1, label=f"Median ({median:.2f})")
    if threshold_line > 0:
        ax.axhline(y=threshold_line, color="red", linestyle=":", linewidth=1,
                    label=f"Threshold ({threshold_line:.2f})")

    ax.set_xlabel("Class")
    ax.set_ylabel("L1 Norm of Trigger Mask")
    ax.set_title("Neural Cleanse: Per-Class Trigger L1 Norms")
    ax.legend()

    if n_classes <= 20:
        ax.set_xticks(x)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Neural Cleanse norms plot to {save_path}")


def plot_lipschitz_constants(
    bd_lip_data: Dict,
    clean_lip_data: Optional[Dict],
    save_path: str,
):
    """Per-channel Lipschitz constants comparison with threshold lines."""
    setup_style()

    bd_layers = bd_lip_data.get("bd_layers", {})
    clean_layers = bd_lip_data.get("clean_layers", {}) if clean_lip_data is None else clean_lip_data.get("clean_layers", {})

    common = [n for n in bd_layers if n in clean_layers]
    if not common:
        return

    n_layers = min(len(common), 4)
    fig, axes = plt.subplots(n_layers, 1, figsize=(FIGSIZE[0], 3 * n_layers))
    if n_layers == 1:
        axes = [axes]

    for ax, name in zip(axes, common[:n_layers]):
        bd_info = bd_layers[name]
        clean_info = clean_layers[name]
        threshold = bd_info.get("threshold", 0)

        ax.axhline(y=threshold, color="red", linestyle=":", alpha=0.7, label="Threshold")
        ax.set_title(f"Layer: {name}", fontsize=10)
        ax.set_xlabel("Channel Index")
        ax.set_ylabel("Lipschitz Constant")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_detection_summary_heatmap(experiment_run, save_path: str):
    """Heatmap: attacks x detectors, cell color = confidence score."""
    setup_style()

    results = experiment_run.results
    if not results:
        return

    # Build matrix
    attacks = sorted(set(f"{r.attack_name}/{r.model_architecture}" for r in results))
    detectors = sorted(set(r.method_name for r in results))

    matrix = np.full((len(attacks), len(detectors)), np.nan)
    attack_idx = {a: i for i, a in enumerate(attacks)}
    det_idx = {d: i for i, d in enumerate(detectors)}

    for r in results:
        key = f"{r.attack_name}/{r.model_architecture}"
        if key in attack_idx and r.method_name in det_idx:
            matrix[attack_idx[key], det_idx[r.method_name]] = r.confidence_score

    fig, ax = plt.subplots(figsize=(max(6, len(detectors) * 2.5), max(4, len(attacks) * 0.8)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(detectors)))
    ax.set_xticklabels(detectors, rotation=45, ha="right")
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels(attacks)

    # Annotate cells
    for i in range(len(attacks)):
        for j in range(len(detectors)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > 0.5 else "black", fontsize=9)

    plt.colorbar(im, ax=ax, label="Confidence Score")
    ax.set_title("Detection Summary: Attack vs Detector")

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved detection summary heatmap to {save_path}")


def _get_weight_svs(model: nn.Module, top_k: int) -> Dict[str, List[float]]:
    """Get top-k singular values for each weight matrix."""
    result = {}
    for name, param in model.named_parameters():
        if "weight" not in name or param.dim() < 2:
            continue
        mat = param.detach().cpu().float().reshape(param.shape[0], -1)
        sv = torch.linalg.svdvals(mat)[:top_k]
        result[name] = sv.tolist()
    return result
