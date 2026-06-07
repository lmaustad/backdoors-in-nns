"""Grad-CAM++ explainer.

Produces spatial heatmaps showing which image regions drive the model's
prediction, using pixel-level weighted activation maps from convolutional
layers (Chattopadhay et al., 2018).  This is a forensic / explainability
tool — it does not produce a binary backdoor verdict.

For each class, saves a three-panel PDF:
  original image | Grad-CAM++ heatmap | superimposed overlay
"""

import logging
from contextlib import contextmanager
import contextlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Detection.core.base_adapter import ModelInfo
from Detection.core.base_detector import DetectionMethod, DetectionResult
from Detection.core.output_mode import prepare_logit_model
from Detection.core.registry import register_detector

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _find_last_conv_layer(
    model: nn.Module, override_name: Optional[str] = None
) -> Optional[Tuple[str, nn.Module]]:
    """Return the last ``nn.Conv2d`` layer in *model* (depth-first).

    If *override_name* is given, look up that specific submodule instead.
    Works through all wrapper models (BackdoorModelWrapper, CombinedTrojanNet,
    _HFLogitWrapper, etc.) because ``named_modules()`` recurses into
    submodules.
    """
    if override_name is not None:
        for name, module in model.named_modules():
            if name == override_name:
                return name, module
        logger.warning("Override layer '%s' not found in model.", override_name)
        return None

    last: Optional[Tuple[str, nn.Module]] = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last = (name, module)
    return last


@contextmanager
def _ensure_gradients(model: nn.Module):
    """Temporarily enable ``requires_grad`` on all parameters.

    Many adapters freeze weights after loading; Grad-CAM++ needs gradients
    to flow back to the target convolutional layer.
    """
    original_states: List[Tuple[nn.Parameter, bool]] = []
    for p in model.parameters():
        original_states.append((p, p.requires_grad))
        p.requires_grad_(True)
    try:
        yield
    finally:
        for p, grad_flag in original_states:
            p.requires_grad_(grad_flag)


@contextmanager
def _reflect_padding(model: nn.Module):
    """Temporarily switch all Conv2d layers from zero-padding to reflect-padding.

    Zero-padding introduces artificial border gradients that accumulate through
    deep networks, producing spurious Grad-CAM hotspots at edges and corners.
    Reflect-padding continues the image content smoothly, eliminating the
    artifact at its source without discarding any spatial region.
    """
    originals: List[Tuple[nn.Conv2d, str]] = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.padding_mode == "zeros":
            originals.append((module, module.padding_mode))
            module.padding_mode = "reflect"
    try:
        yield
    finally:
        for module, mode in originals:
            module.padding_mode = mode


def _chw_to_hwc_float(x: np.ndarray) -> np.ndarray:
    """Convert a (C, H, W) float array to (H, W, C) float in [0, 1].

    Required by ``pytorch_grad_cam.utils.image.show_cam_on_image``.
    """
    img = np.transpose(x, (1, 2, 0))  # (C,H,W) -> (H,W,C)
    lo, hi = img.min(), img.max()
    return (img - lo) / max(hi - lo, 1e-8)


def _pred_class_and_prob(scores: torch.Tensor) -> Tuple[int, float]:
    """Return predicted class id and its probability from model outputs.

    If outputs already look like probabilities (all entries in [0, 1] and rows
    summing to ~1), use them directly; otherwise apply softmax to logits.
    """
    pred_class = int(scores.argmax(dim=1).item())

    with torch.no_grad():
        is_prob_range = bool(torch.all((scores >= 0.0) & (scores <= 1.0)).item())
        row_sums = scores.sum(dim=1)
        is_prob_sum = bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3))
        probs = scores if (is_prob_range and is_prob_sum) else torch.softmax(scores, dim=1)

    pred_prob = float(probs[0, pred_class].item())
    return pred_class, pred_prob


def _save_gradcam_plot(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    save_path: Path,
    titles: Tuple[str, str, str],
    alpha: float = 0.5,
    grayscale_before_overlay: bool = False,
) -> None:
    """Save a three-panel PDF: original | heatmap | grayscale base + heatmap overlay."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        def _to_grayscale(img: np.ndarray) -> np.ndarray:
            """Convert (H, W, C) float image to (H, W) grayscale."""
            if img.ndim == 3 and img.shape[2] == 1:
                return img[:, :, 0]
            if img.ndim == 2:
                return img
            return (
                0.299 * img[:, :, 0]
                + 0.587 * img[:, :, 1]
                + 0.114 * img[:, :, 2]
            )

        is_grayscale = original_image.ndim == 2 or (
            original_image.ndim == 3 and original_image.shape[2] == 1
        )

        # Panel 1: original image
        if is_grayscale:
            disp = _to_grayscale(original_image)
            axes[0].imshow(disp, cmap="gray")
        else:
            axes[0].imshow(original_image)
        axes[0].set_title(titles[0], fontsize=30, pad=3)
        axes[0].axis("off")

        # Panel 2: Grad-CAM++ heatmap only
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title(titles[1], fontsize=30, pad=3)
        axes[1].axis("off")

        # Panel 3: grayscale original with semi-transparent heatmap on top
        gray = _to_grayscale(original_image)
        axes[2].imshow(gray, cmap="gray")
        axes[2].imshow(heatmap, cmap="jet", alpha=alpha)
        axes[2].set_title(titles[2], fontsize=30, pad=3)
        axes[2].axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(save_path.with_suffix(".pdf"), dpi=150, format="pdf")
        plt.close(fig)
    except Exception as exc:
        logger.warning("Failed to save Grad-CAM++ plot %s: %s", save_path, exc)


def _collect_one_per_class(
    data_loader: DataLoader, num_classes: int, occurrence: int = 1
) -> Dict[int, np.ndarray]:
    """Iterate the dataloader and keep the N-th image per class.

    ``occurrence`` is 1-based: 1 = first sample, 2 = second sample, etc.

    Handles both dict-style (``{"x": ..., "y": ...}``) and tuple-style
    (``(images, labels)``) batch formats.
    """
    occurrence = max(1, int(occurrence))
    samples: Dict[int, np.ndarray] = {}
    seen_per_class: Dict[int, int] = {}
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                images, labels = batch["x"], batch.get("y")
            elif isinstance(batch, (tuple, list)):
                images = batch[0]
                labels = batch[1] if len(batch) > 1 else None
            else:
                images, labels = batch, None

            if not images.is_floating_point():
                images = images.float()

            for i in range(images.shape[0]):
                label = int(labels[i].item()) if labels is not None else None
                if label is None:
                    continue

                seen_per_class[label] = seen_per_class.get(label, 0) + 1
                if seen_per_class[label] == occurrence and label not in samples:
                    samples[label] = images[i].cpu().numpy()

            if len(samples) >= num_classes:
                break

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────────────────────────────────────


@register_detector("gradcam_pp")
class GradCAMPPExplainer(DetectionMethod):
    """Visualise which input regions drive each prediction using Grad-CAM++.

    This detector does **not** produce a binary backdoor verdict — it is a
    forensic / explainability tool.  It saves PDF files that a human (or
    downstream analysis) can inspect:

    * gradcam_pp_class_<N>_pred_<P>.pdf — one representative clean image per
      class with the Grad-CAM++ heatmap and a superimposed overlay.
    """

    @property
    def name(self) -> str:
        return "gradcam_pp"

    def requires_data(self) -> bool:
        return True

    # ── Public entry point ────────────────────────────────────────────────────

    def detect(
        self,
        model: nn.Module,
        model_info: ModelInfo,
        data_loader: DataLoader,
        device: torch.device = torch.device("cpu"),
    ) -> DetectionResult:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        output_dir = self._setup_output_dir(model_info)
        alpha = float(self.config.get("alpha", 0.5))
        cam_model, cam_output_mode = prepare_logit_model(
            model,
            model_info,
            enable=bool(self.config.get("trojannet_use_logits", True)),
            caller="GradCAMPPExplainer",
        )

        # Per-attack layer override takes priority, then global fallback.
        # Lookup order: "{attack}_{arch}_{dataset}" → "{attack}" → global.
        per_attack_layers = self.config.get("target_layers", {})
        composite_key = (
            f"{model_info.attack_name}_{model_info.architecture}_{model_info.dataset}"
        )
        override_layer = per_attack_layers.get(
            composite_key,
            per_attack_layers.get(
                model_info.attack_name,
                self.config.get("target_layer_name", None),
            ),
        )

        # Step 1 — find the target convolutional layer.
        conv = _find_last_conv_layer(cam_model, override_name=override_layer)
        if conv is None:
            reason = "no Conv2d layer found in model"
            logger.warning(
                "GradCAMPPExplainer: skipping %s_%s_%s — %s",
                model_info.attack_name,
                model_info.architecture,
                model_info.dataset,
                reason,
            )
            return DetectionResult(
                method_name=self.name,
                attack_name=model_info.attack_name,
                model_architecture=model_info.architecture,
                dataset=model_info.dataset,
                is_backdoor_detected=False,
                confidence_score=0.0,
                details={"skip_reason": "no_conv_layer"},
            )

        layer_name, target_layer = conv
        logger.info(
            "GradCAMPPExplainer: target layer = '%s' (%s)",
            layer_name,
            target_layer.__class__.__name__,
        )

        # Step 2 — collect one representative image per class.
        sample_occurrence = int(self.config.get("sample_occurrence", 1))
        if sample_occurrence < 1:
            logger.warning(
                "GradCAMPPExplainer: invalid sample_occurrence=%s; using 1",
                sample_occurrence,
            )
            sample_occurrence = 1

        samples = _collect_one_per_class(
            data_loader,
            model_info.num_classes,
            occurrence=sample_occurrence,
        )
        if not samples:
            logger.warning("GradCAMPPExplainer: no labelled samples found.")
            return self._empty_result(model_info)

        logger.info(
            "GradCAMPPExplainer: collected %d class representatives "
            "(occurrence=%d)",
            len(samples),
            sample_occurrence,
        )

        # Step 3 — run Grad-CAM++ for each class.
        artifact_paths: List[str] = []

        per_attack_reflect = self.config.get("reflect_padding", {})
        use_reflect = bool(
            per_attack_reflect.get(
                composite_key,
                per_attack_reflect.get(model_info.attack_name, False),
            )
        )
        padding_ctx = _reflect_padding(cam_model) if use_reflect else contextlib.nullcontext()

        with _ensure_gradients(cam_model), padding_ctx:
            cam = GradCAMPlusPlus(model=cam_model, target_layers=[target_layer])

            for class_id in sorted(samples):
                try:
                    image_np = samples[class_id]  # (C, H, W)
                    input_tensor = (
                        torch.tensor(image_np).unsqueeze(0).to(device)
                    )  # (1, C, H, W)

                    # Forward pass to get predicted class.
                    with torch.no_grad():
                        scores = cam_model(input_tensor)
                        pred_class, pred_prob = _pred_class_and_prob(scores)

                    # Compute Grad-CAM++ heatmap for the predicted class.
                    targets = [ClassifierOutputTarget(pred_class)]
                    heatmap = cam(input_tensor=input_tensor, targets=targets)
                    heatmap = heatmap[0]  # (H, W) float in [0, 1]

                    rgb_image = _chw_to_hwc_float(image_np)  # (H, W, C) in [0, 1]

                    path = (
                        output_dir
                        / f"gradcam_pp_class_{class_id}_pred_{pred_class}.pdf"
                    )
                    _save_gradcam_plot(
                        original_image=rgb_image,
                        heatmap=heatmap,
                        save_path=path,
                        titles=(
                            f"Clean sample",
                            f"Grad-CAM++",
                            "Overlay",
                        ),
                        alpha=alpha,
                    )
                    artifact_paths.append(str(path))
                    logger.info("  Saved: %s", path.name)

                except Exception as exc:
                    logger.warning(
                        "Grad-CAM++ failed for class %d: %s", class_id, exc
                    )
                    continue

            # ── Triggered sample explanations ────────────────────────────────
            use_triggered = bool(self.config.get("use_triggered_samples", False))
            if use_triggered:
                adapter = model_info.extra.get("adapter") if model_info.extra else None
                trigger_kwargs = dict(self.config.get("trigger_kwargs", {}))
                if adapter is None:
                    logger.warning(
                        "GradCAMPPExplainer: use_triggered_samples=True but no "
                        "adapter found in model_info.extra."
                    )
                else:
                    # Flip any attack-specific trigger flag on the model for the
                    # duration of the triggered loop (e.g., foobar's fault_pct).
                    with adapter.trigger_mode(cam_model):
                        triggered_any = False
                        for class_id in sorted(samples):
                            try:
                                clean_tensor = torch.tensor(samples[class_id])
                                per_sample_kwargs = dict(trigger_kwargs)
                                per_sample_kwargs.setdefault("target_label", class_id)
                                triggered = adapter.get_triggered_sample(
                                    clean_tensor, **per_sample_kwargs
                                )
                                if triggered is None:
                                    if not triggered_any:
                                        logger.info(
                                            "GradCAMPPExplainer: adapter returned None "
                                            "for triggered sample — skipping all."
                                        )
                                        break
                                    continue

                                triggered_any = True
                                triggered_np = triggered.numpy()
                                input_tensor = triggered.unsqueeze(0).to(device)

                                with torch.no_grad():
                                    scores = cam_model(input_tensor)
                                    pred_class, pred_prob = _pred_class_and_prob(scores)

                                targets = [ClassifierOutputTarget(pred_class)]
                                heatmap = cam(input_tensor=input_tensor, targets=targets)
                                heatmap = heatmap[0]

                                rgb_triggered = _chw_to_hwc_float(triggered_np)

                                path = (
                                    output_dir
                                    / f"TRIGGERED_gradcam_pp_class_{class_id}_pred_{pred_class}.pdf"
                                )
                                _save_gradcam_plot(
                                    original_image=rgb_triggered,
                                    heatmap=heatmap,
                                    save_path=path,
                                    titles=(
                                        f"Triggered sample",
                                        f"Grad-CAM++",
                                        "Overlay",
                                    ),
                                    alpha=alpha,
                                )
                                artifact_paths.append(str(path))
                                logger.info("  Saved triggered: %s", path.name)

                            except Exception as exc:
                                logger.warning(
                                    "Grad-CAM++ failed for triggered class %d: %s",
                                    class_id, exc,
                                )
                                continue

        return DetectionResult(
            method_name=self.name,
            attack_name=model_info.attack_name,
            model_architecture=model_info.architecture,
            dataset=model_info.dataset,
            is_backdoor_detected=False,  # forensic tool, not a verdict
            confidence_score=0.0,
            details={
                "num_classes_explained": len(samples),
                "target_layer": layer_name,
                "cam_output_mode": cam_output_mode,
                "alpha": alpha,
                "sample_occurrence": sample_occurrence,
                "output_dir": str(output_dir),
            },
            artifacts={"plots": artifact_paths},
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _setup_output_dir(self, model_info: ModelInfo) -> Path:
        artifact_root = self.config.get("artifact_root_dir", "./Detection/results")
        run_name = (
            f"{model_info.attack_name}_{model_info.architecture}_{model_info.dataset}"
        )
        output_dir = Path(artifact_root) / run_name / "gradcam_pp"
        overwrite_output_dir = bool(self.config.get("overwrite_output_dir", True))
        if overwrite_output_dir and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _empty_result(self, model_info: ModelInfo) -> DetectionResult:
        return DetectionResult(
            method_name=self.name,
            attack_name=model_info.attack_name,
            model_architecture=model_info.architecture,
            dataset=model_info.dataset,
            is_backdoor_detected=False,
            confidence_score=0.0,
        )
