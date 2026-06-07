"""Model summary and architecture graph detector.

Uses ``torchinfo`` for a detailed model summary and ``torchviz`` to render a
computation-graph of the model architecture.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Detection.core.base_adapter import ModelInfo
from Detection.core.base_detector import DetectionMethod, DetectionResult
from Detection.core.registry import register_detector

logger = logging.getLogger(__name__)


@register_detector("model_summary")
class ModelSummaryDetector(DetectionMethod):
    @property
    def name(self) -> str:
        return "model_summary"

    def requires_data(self) -> bool:
        return False

    def detect(
        self,
        model: nn.Module,
        model_info: ModelInfo,
        data_loader: DataLoader,
        device: torch.device = torch.device("cpu"),
    ) -> DetectionResult:
        run_dir = _artifact_run_dir(
            config=self.config,
            model_info=model_info,
            artifact_name="model_summary",
        )
        model_label = _model_label(model_info, self.config)

        # ── Model summary (torchinfo) ───────────────────────────
        report_path = None
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if self.config.get("save_report", True):
            report_path = self._write_summary(model, model_info, run_dir)

        # ── Architecture graph (torchviz) ───────────────────────
        graph_path = None
        if self.config.get("save_graph", True):
            graph_path = self._save_graph(model, model_info, run_dir, device)

        return DetectionResult(
            method_name=self.name,
            attack_name=model_info.attack_name,
            model_architecture=model_info.architecture,
            dataset=model_info.dataset,
            is_backdoor_detected=False,
            confidence_score=0.0,
            details={
                "visualization_only": True,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_label": model_label,
            },
            artifacts={
                "summary_report": report_path,
                "architecture_graph": graph_path,
            },
        )

    # ------------------------------------------------------------------ #
    #  Summary via torchinfo (falls back to print(model))                 #
    # ------------------------------------------------------------------ #
    def _write_summary(
        self,
        model: nn.Module,
        model_info: ModelInfo,
        run_dir: Path,
    ) -> str:
        try:
            from torchinfo import summary

            depth = int(self.config.get("summary_depth", 4))
            model_stats = summary(
                model,
                input_size=(1, *model_info.input_shape),
                depth=depth,
                verbose=0,
            )
            summary_text = str(model_stats)
        except Exception as exc:
            logger.warning("torchinfo unavailable (%s), falling back to repr", exc)
            summary_text = str(model)

        header = (
            f"Model Summary: {model_info.architecture}\n"
            f"Attack: {model_info.attack_name}\n"
            f"Model label: {_model_label(model_info, self.config)}\n"
            f"Dataset: {model_info.dataset}\n"
            f"Input shape: {model_info.input_shape}\n"
        )
        full_text = header + "\n" + summary_text
        model_label = _model_label(model_info, self.config)
        logger.info("Model summary saved to %s", run_dir / f"model_summary_{model_label}.txt")

        path = run_dir / f"model_summary_{model_label}.txt"
        path.write_text(full_text)
        return str(path)

    # ------------------------------------------------------------------ #
    #  Architecture graph via torchviz                                    #
    # ------------------------------------------------------------------ #
    def _save_graph(
        self,
        model: nn.Module,
        model_info: ModelInfo,
        run_dir: Path,
        device: torch.device,
    ) -> Optional[str]:
        try:
            from torchviz import make_dot
        except ImportError:
            logger.warning("torchviz is not installed — skipping architecture graph")
            return None

        try:
            model.eval()
            dummy = torch.zeros(1, *model_info.input_shape, device=device, requires_grad=True)
            original_requires_grad = [(p, p.requires_grad) for p in model.parameters()]
            for p, _ in original_requires_grad:
                p.requires_grad = False

            try:
                out = model(dummy)
            finally:
                for p, req in original_requires_grad:
                    p.requires_grad = req

            dot = make_dot(
                out,
                params=dict(model.named_parameters()),
                show_attrs=False,
                show_saved=False,
            )
            dot.attr(label=f"{model_info.attack_name} | {model_info.architecture} | {model_info.dataset}")
            dot.attr(fontsize="10")

            fmt = self.config.get("graph_format", "png")
            model_label = _model_label(model_info, self.config)
            stem = f"{model_info.architecture}_architecture_{model_label}"
            path = run_dir / stem

            dot.render(str(path), format=fmt, cleanup=True)
            rendered = str(path) + f".{fmt}"
            logger.info("Architecture graph saved to %s", rendered)
            return rendered

        except Exception as exc:
            logger.warning("Could not generate architecture graph: %s", exc)
            return None


def _sanitize_run_name(model_info: ModelInfo) -> str:
    return (
        f"{model_info.attack_name}_{model_info.architecture}_{model_info.dataset}"
        .replace(" ", "_")
    )


def _model_label(model_info: ModelInfo, config: dict) -> str:
    if bool(config.get("force_clean_label", False)):
        return "clean"
    attack_name = model_info.attack_name.strip().lower()
    return "clean" if "clean" in attack_name else "backdoor"


def _artifact_run_dir(
    config: dict,
    model_info: ModelInfo,
    artifact_name: str,
) -> Path:
    root_dir = config.get("artifact_root_dir")
    if not root_dir:
        raise ValueError("model_summary requires 'artifact_root_dir' in detector config")

    run_dir = Path(root_dir) / _sanitize_run_name(model_info) / artifact_name

    overwrite_output_dir = bool(config.get("overwrite_output_dir", True))
    if overwrite_output_dir and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
