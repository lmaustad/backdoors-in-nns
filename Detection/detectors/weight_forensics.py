"""Weight forensics visualization.

A visual inspection detector: scans every weighted layer of a torch.nn.Module
and emits per-layer heatmaps and weight-distribution histograms. The signal is
left to the human reader — large isolated outliers, anomalous deadzone gaps, or
unusually heavy tails can indicate handcrafted or fault-injection style
backdoors that perturb a small number of weights.

Inputs: an instantiated model and a `ModelInfo` from the adapter. No data
loader is required; this method operates on model parameters only and is
therefore data-free. Outputs are written to the configured artifact directory.
"""

import logging
import shutil
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Detection.core.base_adapter import ModelInfo
from Detection.core.base_detector import DetectionMethod, DetectionResult
from Detection.core.registry import register_detector

logger = logging.getLogger(__name__)


@register_detector("weight_forensics")
class WeightForensicsDetector(DetectionMethod):
    @property
    def name(self) -> str:
        return "weight_forensics"

    def requires_data(self) -> bool:
        return False

    def detect(
        self,
        model: nn.Module,
        model_info: ModelInfo,
        data_loader: DataLoader,
        device: torch.device = torch.device("cpu"),
    ) -> DetectionResult:
        deadzone_eps = float(self.config.get("deadzone_eps", 1e-4))

        heatmaps = []
        if self.config.get("save_heatmaps", True):
            heatmaps = self._save_heatmaps(
                model=model,
                model_info=model_info,
                deadzone_eps=deadzone_eps,
            )

        histograms = []
        if self.config.get("save_histograms", True):
            histograms = self._save_histograms(
                model=model,
                model_info=model_info,
            )

        report_path = None
        histogram_stats = None
        model_label = self._model_label(model_info)
        display_name = _display_name(model_info)
        if self.config.get("save_report", True):
            max_layers = int(self.config.get("max_histogram_layers", 3))
            histogram_stats = self._collect_histogram_stats(
                model, max_layers=max_layers
            )
            report_path = self._write_report(
                model_info=model_info, histogram_stats=histogram_stats
            )

        return DetectionResult(
            method_name=self.name,
            attack_name=model_info.attack_name,
            model_architecture=model_info.architecture,
            dataset=model_info.dataset,
            is_backdoor_detected=False,
            confidence_score=0.0,
            details={
                "visualization_only": True,
                "histogram_stats": histogram_stats,
                "model_label": model_label,
                "display_name": display_name,
            },
            artifacts={
                "heatmaps": heatmaps,
                "histograms": histograms,
                "report": report_path,
            },
        )

    def _save_heatmaps(
        self, model: nn.Module, model_info: ModelInfo, deadzone_eps: float
    ) -> List[dict]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning(f"Could not import matplotlib: {exc}")
            return []

        run_dir = _artifact_run_dir(
            config=self.config,
            model_info=model_info,
            artifact_name="heatmaps",
        )

        limit_layers = int(self.config.get("max_heatmap_layers", 5))
        max_elements = int(self.config.get("max_heatmap_elements", 200000))
        large_weight_threshold = float(self.config.get("large_weight_threshold", 0.0))
        large_weight_percentile = float(
            self.config.get("large_weight_percentile", 99.5)
        )
        suffix = self._model_label(model_info)
        display_name = _display_name(model_info)
        title_fontsize = int(self.config.get("plot_title_fontsize", 14))
        label_fontsize = int(self.config.get("plot_label_fontsize", 12))
        tick_fontsize = int(self.config.get("plot_tick_fontsize", 11))
        cbar_fontsize = int(self.config.get("plot_cbar_fontsize", 11))
        fig_width = float(self.config.get("plot_width", 7.5))
        fig_height = float(self.config.get("plot_height", 5.0))

        artifacts = []
        layer_count = 0
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            if layer_count >= limit_layers:
                break

            weight = param.detach().cpu()
            if weight.dim() < 2:
                continue

            mat = weight.reshape(weight.shape[0], -1)
            abs_mat = np.abs(mat.detach().cpu().numpy())

            dead_mask = (abs_mat <= deadzone_eps).astype(np.float32)
            dead_ratio = float(dead_mask.mean())
            dead_mask = _downsample_matrix(dead_mask, max_elements)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            im = ax.imshow(
                dead_mask, aspect="auto", interpolation="nearest", cmap="magma"
            )
            title = (
                f"{display_name} | {model_info.architecture} | {model_info.dataset}\n"
                f"{name} dead-zone (|w| <= {deadzone_eps:g}), ratio={dead_ratio:.4f}"
            )
            ax.set_title(_wrap_title(title, width=56), fontsize=title_fontsize)
            ax.set_xlabel("in-features", fontsize=label_fontsize)
            ax.set_ylabel("out-features", fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("dead-zone mask (1=dead, 0=active)", fontsize=cbar_fontsize)
            cbar.ax.tick_params(labelsize=tick_fontsize)
            fig.tight_layout()

            safe_name = name.replace(".", "_")
            filename = f"{model_info.architecture}_{safe_name}_deadzone_{suffix}.pdf"
            path = run_dir / filename
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            artifacts.append(
                {
                    "layer": name,
                    "type": "deadzone",
                    "path": str(path),
                }
            )

            threshold = large_weight_threshold
            if threshold <= 0.0:
                threshold = float(np.percentile(abs_mat, large_weight_percentile))

            large_mask = (abs_mat >= threshold).astype(np.float32)
            large_ratio = float(large_mask.mean())
            large_mask = _downsample_matrix(large_mask, max_elements)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            im = ax.imshow(
                large_mask, aspect="auto", interpolation="nearest", cmap="viridis"
            )
            title = (
                f"{display_name} | {model_info.architecture} | {model_info.dataset}\n"
                f"{name} large-weight (|w| >= {threshold:.4g}), ratio={large_ratio:.4f}"
            )
            ax.set_title(_wrap_title(title, width=56), fontsize=title_fontsize)
            ax.set_xlabel("in-features", fontsize=label_fontsize)
            ax.set_ylabel("out-features", fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(
                "large-weight mask (1=large, 0=normal)", fontsize=cbar_fontsize
            )
            cbar.ax.tick_params(labelsize=tick_fontsize)
            fig.tight_layout()

            filename = (
                f"{model_info.architecture}_{safe_name}_largeweights_{suffix}.pdf"
            )
            path = run_dir / filename
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            artifacts.append(
                {
                    "layer": name,
                    "type": "large_weights",
                    "path": str(path),
                }
            )

            raw_mat = _downsample_matrix(mat.detach().cpu().numpy(), max_elements)
            raw_max_abs = float(np.max(np.abs(raw_mat))) if raw_mat.size else 0.0
            if raw_max_abs == 0.0:
                raw_max_abs = 1.0

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            im = ax.imshow(
                raw_mat,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-raw_max_abs,
                vmax=raw_max_abs,
            )
            title = (
                f"{display_name} | {model_info.architecture} | {model_info.dataset}\n"
                f"{name} raw weights (min={raw_mat.min():.4g}, max={raw_mat.max():.4g})"
            )
            ax.set_title(_wrap_title(title, width=56), fontsize=title_fontsize)
            ax.set_xlabel("in-features", fontsize=label_fontsize)
            ax.set_ylabel("out-features", fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("weight value", fontsize=cbar_fontsize)
            cbar.ax.tick_params(labelsize=tick_fontsize)
            fig.tight_layout()

            filename = f"{model_info.architecture}_{safe_name}_rawweights_{suffix}.pdf"
            path = run_dir / filename
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            artifacts.append(
                {
                    "layer": name,
                    "type": "raw_weights",
                    "path": str(path),
                }
            )
            layer_count += 1

        return artifacts

    def _save_histograms(self, model: nn.Module, model_info: ModelInfo) -> List[dict]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning(f"Could not import matplotlib: {exc}")
            return []

        run_dir = _artifact_run_dir(
            config=self.config,
            model_info=model_info,
            artifact_name="histograms",
        )

        max_layers = int(self.config.get("max_histogram_layers", 3))
        bins = int(self.config.get("histogram_bins", 80))
        suffix = self._model_label(model_info)
        display_name = _display_name(model_info)
        title_fontsize = int(self.config.get("plot_title_fontsize", 14))
        label_fontsize = int(self.config.get("plot_label_fontsize", 12))
        tick_fontsize = int(self.config.get("plot_tick_fontsize", 11))
        fig_width = float(self.config.get("plot_width", 7.5))
        fig_height = float(self.config.get("plot_height", 5.0))

        artifacts = []
        weights = _collect_weights(model)
        flat = _flatten_weights(weights).cpu().numpy() if weights else np.array([])

        if flat.size:
            mean = float(flat.mean())
            std = float(flat.std(ddof=1)) if flat.size > 1 else 0.0
            max_abs = float(np.max(np.abs(flat)))
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.hist(flat, bins=bins, color="#4c78a8", alpha=0.85)
            title = (
                f"{display_name} global weight distribution\n"
                f"mean={mean:.4g}, std={std:.4g}, max|w|={max_abs:.4g}"
            )
            ax.set_title(_wrap_title(title, width=56), fontsize=title_fontsize)
            ax.set_xlabel("weight", fontsize=label_fontsize)
            ax.set_ylabel("count", fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)
            fig.tight_layout()
            path = run_dir / f"{model_info.architecture}_global_weights_{suffix}.pdf"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            artifacts.append(
                {"type": "histogram", "layer": "global", "path": str(path)}
            )

        for idx, (name, w) in enumerate(weights.items()):
            if idx >= max_layers:
                break
            values = w.flatten().cpu().numpy()
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            max_abs = float(np.max(np.abs(values))) if values.size else 0.0
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.hist(values, bins=bins, color="#f58518", alpha=0.85)
            title = (
                f"{display_name} | {model_info.architecture} | {model_info.dataset}\n"
                f"{name} weight distribution (mean={mean:.4g}, std={std:.4g}, max|w|={max_abs:.4g})"
            )
            ax.set_title(_wrap_title(title, width=56), fontsize=title_fontsize)
            ax.set_xlabel("weight", fontsize=label_fontsize)
            ax.set_ylabel("count", fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)
            fig.tight_layout()
            safe_name = name.replace(".", "_")
            path = (
                run_dir / f"{model_info.architecture}_{safe_name}_weights_{suffix}.pdf"
            )
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            artifacts.append({"type": "histogram", "layer": name, "path": str(path)})

        return artifacts

    def _collect_histogram_stats(self, model: nn.Module, max_layers: int) -> dict:
        weights = _collect_weights(model)
        flat = _flatten_weights(weights).cpu().numpy() if weights else np.array([])

        stats = {
            "global": {},
            "layers": {},
        }

        if flat.size:
            stats["global"] = {
                "mean": float(flat.mean()),
                "std": float(flat.std(ddof=1)) if flat.size > 1 else 0.0,
                "max_abs": float(np.max(np.abs(flat))),
                "count": int(flat.size),
            }

        for idx, (name, w) in enumerate(weights.items()):
            if idx >= max_layers:
                break
            values = w.flatten().cpu().numpy()
            stats["layers"][name] = {
                "mean": float(values.mean()) if values.size else 0.0,
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "max_abs": float(np.max(np.abs(values))) if values.size else 0.0,
                "count": int(values.size),
            }

        return stats

    def _write_report(
        self, model_info: ModelInfo, histogram_stats: Optional[dict]
    ) -> Optional[str]:
        run_dir = _artifact_run_dir(
            config=self.config,
            model_info=model_info,
            artifact_name="reports",
        )
        model_label = self._model_label(model_info)

        lines = []
        lines.append("Weight Forensics Histogram Summary")
        lines.append(f"attack: {_display_name(model_info)}")
        lines.append(f"adapter_attack: {model_info.attack_name}")
        lines.append(f"model_label: {model_label}")
        lines.append(f"architecture: {model_info.architecture}")
        lines.append(f"dataset: {model_info.dataset}")
        lines.append("")
        if not histogram_stats:
            lines.append("No histogram stats available.")
        else:
            global_stats = histogram_stats.get("global", {})
            lines.append("Global")
            lines.append(
                "  mean={mean:.6g} std={std:.6g} max_abs={max_abs:.6g} count={count}".format(
                    mean=global_stats.get("mean", 0.0),
                    std=global_stats.get("std", 0.0),
                    max_abs=global_stats.get("max_abs", 0.0),
                    count=global_stats.get("count", 0),
                )
            )
            lines.append("")
            lines.append("Layers (first N by order)")
            for name, stats in histogram_stats.get("layers", {}).items():
                lines.append(
                    "  {name}: mean={mean:.6g} std={std:.6g} max_abs={max_abs:.6g} count={count}".format(
                        name=name,
                        mean=stats.get("mean", 0.0),
                        std=stats.get("std", 0.0),
                        max_abs=stats.get("max_abs", 0.0),
                        count=stats.get("count", 0),
                    )
                )

        path = run_dir / f"report_{model_label}.txt"
        path.write_text("\n".join(lines))
        return str(path)

    def _model_label(self, model_info: ModelInfo) -> str:
        if bool(self.config.get("force_clean_label", False)):
            return "clean"
        attack_name = model_info.attack_name.strip().lower()
        return "clean" if "clean" in attack_name else "backdoor"


def _display_name(model_info: ModelInfo) -> str:
    extra = getattr(model_info, "extra", None) or {}
    return extra.get("config_name") or model_info.attack_name


def _collect_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    weights = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            weights[name] = param.detach().cpu()
    return weights


def _flatten_weights(weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([w.flatten() for w in weights.values()])


def _downsample_matrix(mat: np.ndarray, max_elements: int) -> np.ndarray:
    total = mat.size
    if total <= max_elements:
        return mat
    ratio = (total / max_elements) ** 0.5
    step = int(np.ceil(ratio))
    return mat[::step, ::step]


def _wrap_title(text: str, width: int) -> str:
    lines = []
    for part in text.split("\n"):
        lines.extend(textwrap.wrap(part, width=width) or [""])
    return "\n".join(lines)


def _sanitize_run_name(model_info: ModelInfo) -> str:
    return f"{model_info.attack_name}_{model_info.architecture}_{model_info.dataset}".replace(
        " ", "_"
    )


def _artifact_run_dir(
    config: dict,
    model_info: ModelInfo,
    artifact_name: str,
) -> Path:
    root_dir = config.get("artifact_root_dir")
    if not root_dir:
        raise ValueError(
            "weight_forensics requires 'artifact_root_dir' in detector config"
        )

    run_dir = Path(root_dir) / _sanitize_run_name(model_info) / artifact_name

    overwrite_output_dir = bool(config.get("overwrite_output_dir", True))
    if overwrite_output_dir and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
