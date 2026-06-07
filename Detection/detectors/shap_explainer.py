"""Gradient SHAP explainer.

Explains model predictions using SHAP's GradientExplainer. For each class,
saves a plot of the clean sample with its SHAP attribution heatmap. If Neural
Cleanse results exist for the model, also superimposes the most anomalous
recovered trigger and generates a second round of explanations for comparison.
"""

import contextlib
import json
import logging
import shutil
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import shap
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from Detection.core.base_adapter import ModelInfo
from Detection.core.base_detector import DetectionMethod, DetectionResult
from Detection.core.output_mode import prepare_logit_model
from Detection.core.registry import register_detector

logger = logging.getLogger(__name__)


def _safe_sdp_context():
    """Return a context manager that forces the math-only SDPA backend.

    Flash-attention and memory-efficient SDPA kernels perform in-place
    operations on views produced by ``torch.unbind`` inside
    ``nn.MultiheadAttention``.  This breaks SHAP's GradientExplainer which
    needs a clean autograd graph.  Forcing the pure-math backend avoids the
    in-place view error at the cost of slightly more memory.
    """
    # PyTorch ≥ 2.2
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.MATH])
    except ImportError:
        pass
    # PyTorch 2.0 – 2.1
    try:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        )
    except Exception:
        pass
    return contextlib.nullcontext()


# Default number of background samples used to estimate baseline for GradientExplainer.
# More samples → better SHAP estimates, but slower.
# Can be overridden via config key "bg_size".
_DEFAULT_BG_SIZE = 64

# Small datasets where generating SHAP plots for every output neuron is tractable.
_ALL_OUTPUT_NEURON_DATASETS = {"cifar10", "mnist", "fmnist"}

# Adapter class names whose triggers are distributed across the whole image at
# imperceptible magnitude (steganographic). For those, the "trigger applied to
# a blank canvas" panel is meaningless, so |triggered - clean| is used instead.
_STEGO_ADAPTER_CLASSES = {"HidingNeedlesAdapter", "BooneBaneAdapter"}


# ──────────────────────────────────────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────────────────────────────────────


def _chw_to_hwc_uint8(x: np.ndarray) -> np.ndarray:
    """Convert a (C, H, W) float array to a (H, W, C) uint8 image for display.

    Rescales pixel values to [0, 255] using the array's own min/max,
    so even tensors that were normalised (e.g. mean-subtracted) will
    look correct when plotted.
    """
    img = np.transpose(x, (1, 2, 0))  # (C,H,W) → (H,W,C)
    lo, hi = img.min(), img.max()
    img_scaled = (img - lo) / max(hi - lo, 1e-8)  # stretch to [0, 1]
    return np.clip(img_scaled * 255, 0, 255).astype(np.uint8)


def _chw_to_hwc_float01(x: np.ndarray) -> np.ndarray:
    """Convert a (C, H, W) float array to (H, W, C) float32 in [0, 1].

    This is intended for APIs like shap.image_plot/matplotlib.imshow that
    expect RGB floats to lie in [0, 1].
    """
    img = np.transpose(x, (1, 2, 0)).astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    return np.clip((img - lo) / max(hi - lo, 1e-8), 0.0, 1.0)


def _normalise_dataset_name(dataset: str) -> str:
    """Normalise dataset names so aliases like 'cifar-10' map to 'cifar10'."""
    token = "".join(ch for ch in str(dataset).lower() if ch.isalnum())
    if token == "fashionmnist":
        return "fmnist"
    return token


def _save_explanation_plot(
    images: List[np.ndarray],
    shap_heatmap: np.ndarray,
    titles: List[str],
    save_path: Path,
    title_fontsize: int = 10,
    grayscale_before_overlay: bool = False,
    hot_cmap_indices: Optional[List[int]] = None,
    overlay_panel_index: int = -1,
    extra_overlays: Optional[List[Tuple[int, np.ndarray]]] = None,
    pure_heatmap_indices: Optional[List[int]] = None,
) -> None:
    """Save a row of image panels with SHAP heatmaps overlaid on selected panels.

    Args:
        images:      List of (H, W, C) uint8 images to show side-by-side.
        shap_heatmap: Normalised 2-D array of absolute SHAP values to overlay.
        titles:      Per-panel titles (same length as images).
        save_path:   Where to write the PDF file.
        title_fontsize: Font size for per-panel titles.
        hot_cmap_indices: Panel indices where grayscale images should render with
                          the "hot" colormap instead of "gray" (used for
                          difference panels).
        overlay_panel_index: Panel to overlay ``shap_heatmap`` on (default: last).
        extra_overlays: Additional ``(panel_idx, heatmap)`` pairs — e.g. a
                        heatmap masked to the trigger region.
        pure_heatmap_indices: Panel indices that render only the SHAP heatmap
                          (hot cmap, no underlying image, no alpha) so the
                          attribution can be inspected without backdrop bleed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_panels = len(images)
        wrap_width = 24 if title_fontsize >= 18 else 30

        def _wrap_title(title: str) -> str:
            lines: List[str] = []
            for raw_line in str(title).splitlines() or [""]:
                if len(raw_line) <= wrap_width:
                    lines.append(raw_line)
                else:
                    lines.extend(textwrap.wrap(raw_line, width=wrap_width))
            return "\n".join(lines)

        wrapped_titles = [_wrap_title(title) for title in titles]
        max_title_lines = max((t.count("\n") + 1 for t in wrapped_titles), default=1)
        fig_height = max(3.0, 2.2 + 0.4 * max_title_lines)
        fig, axes = plt.subplots(
            1, num_panels, figsize=(3 * num_panels, fig_height)
        )

        # Grayscale images have shape (H, W, 1); squeeze for imshow compatibility.
        def to_display(img: np.ndarray, force_grayscale: bool = False):
            is_grayscale = img.shape[2] == 1 if img.ndim == 3 else True
            if force_grayscale and not is_grayscale:
                gray = (
                    0.299 * img[:, :, 0].astype(np.float32)
                    + 0.587 * img[:, :, 1].astype(np.float32)
                    + 0.114 * img[:, :, 2].astype(np.float32)
                )
                return gray.astype(np.uint8), "gray"
            if is_grayscale:
                return img[:, :, 0] if img.ndim == 3 else img, "gray"
            return img, None

        def _normalize_idx(i: int) -> int:
            return i if i >= 0 else num_panels + i

        overlay_panels = {_normalize_idx(overlay_panel_index)}
        overlay_panels.update(_normalize_idx(i) for i, _ in (extra_overlays or []))
        pure_set = {_normalize_idx(i) for i in (pure_heatmap_indices or [])}

        hot_set = set(hot_cmap_indices or [])
        for idx, (ax, img, title) in enumerate(zip(axes, images, wrapped_titles)):
            if idx in pure_set:
                ax.imshow(shap_heatmap, cmap="hot")
                ax.set_title(title, fontsize=title_fontsize, pad=3)
                ax.axis("off")
                continue
            force_gray = grayscale_before_overlay and idx in overlay_panels
            display_img, cmap = to_display(img, force_grayscale=force_gray)
            if idx in hot_set and cmap == "gray":
                cmap = "hot"
            ax.imshow(display_img, cmap=cmap)
            ax.set_title(title, fontsize=title_fontsize, pad=3)
            ax.axis("off")

        # Primary overlay (semi-transparent, hot colormap).
        if _normalize_idx(overlay_panel_index) not in pure_set:
            axes[overlay_panel_index].imshow(shap_heatmap, alpha=0.6, cmap="hot")
        for idx, overlay in extra_overlays or []:
            if _normalize_idx(idx) not in pure_set:
                axes[idx].imshow(overlay, alpha=0.6, cmap="hot")

        fig.tight_layout(pad=0.6)
        top_margin = 1.0 - min(0.04 * max_title_lines, 0.25)
        fig.subplots_adjust(top=top_margin)
        fig.savefig(
            save_path.with_suffix(".pdf"),
            dpi=150,
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig)
    except Exception as exc:
        logger.warning("Failed to save plot %s: %s", save_path, exc)


# ──────────────────────────────────────────────────────────────────────────────
# Neural Cleanse trigger loading
# ──────────────────────────────────────────────────────────────────────────────


def _load_nc_trigger(
    trigger_dir: Path,
    label: int,
    num_channels: int,
    clip_min: float,
    clip_max: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load the Neural Cleanse mask and pattern for a given target label.

    Neural Cleanse saves its recovered triggers as PNG files. We reload them,
    convert them back to the model's input range, and return them so they can
    be stamped onto clean images.

    Args:
        trigger_dir:  Directory containing mask_label_<N>.png / pattern_label_<N>.png.
        label:        Target class whose trigger to load.
        num_channels: 1 for grayscale models, 3 for RGB.
        clip_min/max: Dataset pixel range; used to denormalise the loaded pattern.

    Returns:
        (mask, pattern) where mask is (H, W) float32 in [0,1] and
        pattern is (C, H, W) float32 in [clip_min, clip_max], or None if
        the trigger files are missing.
    """
    mask_path = trigger_dir / f"mask_label_{label}.png"
    pattern_path = trigger_dir / f"pattern_label_{label}.png"

    if not mask_path.exists() or not pattern_path.exists():
        return None

    # Mask: single channel — 1 where the trigger is painted, 0 elsewhere.
    mask = (
        np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    )  # (H, W)

    # Pattern: the actual pixel values the trigger writes.
    pil_mode = "L" if num_channels == 1 else "RGB"
    pattern = (
        np.array(Image.open(pattern_path).convert(pil_mode), dtype=np.float32) / 255.0
    )
    # Reshape to (C, H, W) so we can broadcast over the image tensor.
    pattern = (
        pattern[np.newaxis] if pattern.ndim == 2 else np.transpose(pattern, (2, 0, 1))
    )
    # Undo the [0, 1] PNG normalisation — map back to the model's input range.
    pattern = pattern * max(clip_max - clip_min, 1e-8) + clip_min

    return mask, pattern


# ──────────────────────────────────────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────────────────────────────────────


@register_detector("shap_explainer")
class ShapExplainer(DetectionMethod):
    """Visualise which input pixels drive each prediction using Gradient SHAP.

    This detector does **not** produce a binary backdoor verdict on its own —
    it is a forensic / explainability tool.  It saves PDF files that a human
    (or downstream analysis) can inspect:

    * clean_class_<N>_pred_<P>.pdf   — one representative clean image per class
                                       with the SHAP attribution heatmap overlaid.
    * triggered_tgt<T>_class_<N>_pred_<P>.pdf — same image after the Neural
                                       Cleanse trigger for target T is stamped on
                                       it, so you can see whether the trigger
                                       region dominates the attribution.
    """

    @property
    def name(self) -> str:
        return "shap_explainer"

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

        output_dir = self._setup_output_dir(model_info)

        # Step 1 — gather data: one representative image per class + a pool of
        #          background images that GradientExplainer uses as its baseline.
        grayscale_before_overlay = bool(
            self.config.get("grayscale_before_overlay", False)
        )
        bg_size = int(self.config.get("bg_size", _DEFAULT_BG_SIZE))
        bg_seed = self.config.get("background_seed")
        bg_seed = int(bg_seed) if bg_seed is not None else None
        bg_uniform = bool(self.config.get("background_uniform", False))
        samples, background, background_labels = self._collect_samples_and_background(
            data_loader,
            model_info.num_classes,
            device,
            bg_size,
            seed=bg_seed,
            uniform=bg_uniform,
        )

        if not samples:
            logger.warning("ShapExplainer: no labelled samples found.")
            return self._empty_result(model_info)

        # Step 2 — unwrap TrojanNet's trailing softmax so gradient-based SHAP
        #          sees pre-softmax logits. All other adapters pass through
        #          unchanged and are already in logit space.
        shap_model, output_mode = prepare_logit_model(
            model,
            model_info,
            enable=bool(self.config.get("trojannet_use_logits", True)),
            caller="ShapExplainer",
        )

        # Step 3 — build the SHAP explainer once (expensive: runs the model
        #          on all background samples to estimate expected gradients).

        logger.info(
            "ShapExplainer: building GradientExplainer with %d background samples (output=%s)",
            background.shape[0],
            output_mode,
        )
        with _safe_sdp_context():
            explainer = shap.GradientExplainer(shap_model, background)

        artifact_paths: List[str] = []
        save_all_output_neurons = self._should_save_all_output_neuron_plots(
            model_info.dataset
        )

        # Step 4 — run the model on each background image to record its prediction,
        #          then save a grid and distribution chart for the background set.
        bg_preds = self._predict_background(model, background, device)
        bg_paths, bg_counts = self._save_background_grid(
            background, background_labels, bg_preds, output_dir
        )
        artifact_paths.extend(bg_paths)

        # Step 5 — explain each clean representative image.
        clean_paths = self._run_clean_explanations(
            explainer,
            samples,
            model,
            output_dir,
            device,
            grayscale_before_overlay=grayscale_before_overlay,
            save_all_output_neurons=save_all_output_neurons,
        )
        artifact_paths.extend(clean_paths)

        # Step 6 — if Neural Cleanse already ran, stamp its recovered trigger
        #          onto each representative image and explain again.
        nc_meta, triggered_paths = self._run_triggered_explanations(
            explainer,
            samples,
            model_info,
            output_dir,
            device,
            grayscale_before_overlay=grayscale_before_overlay,
            save_all_output_neurons=save_all_output_neurons,
        )
        artifact_paths.extend(triggered_paths)

        # Step 7 — if use_triggered_samples is enabled, use the adapter's own
        #          trigger mechanism to create triggered samples and explain them.
        use_triggered = bool(self.config.get("use_triggered_samples", False))
        if use_triggered:
            adapter_triggered_paths = self._run_adapter_triggered_explanations(
                explainer,
                samples,
                model_info,
                output_dir,
                device,
                model,
                grayscale_before_overlay=grayscale_before_overlay,
                save_all_output_neurons=save_all_output_neurons,
            )
            artifact_paths.extend(adapter_triggered_paths)

        return DetectionResult(
            method_name=self.name,
            attack_name=model_info.attack_name,
            model_architecture=model_info.architecture,
            dataset=model_info.dataset,
            is_backdoor_detected=False,  # SHAP is forensic, not a verdict
            confidence_score=0.0,
            details={
                "num_classes_explained": len(samples),
                "background_seed": bg_seed,
                "background_uniform": bg_uniform,
                "grayscale_before_overlay": grayscale_before_overlay,
                "background_size": int(background.shape[0]),
                "background_class_counts": {
                    str(k): v for k, v in sorted(bg_counts.items())
                },
                "output_mode": output_mode,
                "all_output_neuron_shap_enabled": save_all_output_neurons,
                **nc_meta,
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
        output_dir = Path(artifact_root) / run_name / "shap_explainer"
        overwrite_output_dir = bool(self.config.get("overwrite_output_dir", True))
        if overwrite_output_dir and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _collect_samples_and_background(
        self,
        data_loader: DataLoader,
        num_classes: int,
        device: torch.device,
        bg_size: int,
        seed: Optional[int] = None,
        uniform: bool = False,
    ) -> Tuple[Dict[int, np.ndarray], torch.Tensor, List[int]]:
        """Iterate over the dataloader to build two collections:

        - samples:    {class_id → raw (C,H,W) float32 numpy array}.
                      We keep only the first image we see for each class.
        - background: A (≤bg_size, C, H, W) tensor used as the SHAP baseline.
                      The baseline represents "uninformative" input; SHAP measures
                      how each pixel shifts predictions relative to this baseline.
        - background_labels: The class label of each background image (same order).

        Three modes, controlled by ``seed`` and ``uniform``:

        - ``seed=None, uniform=False`` (default): first ``bg_size`` eligible
          images encountered are used — fast, no full-dataloader pass needed.
        - ``seed=<int>, uniform=False``: full dataloader is read, then
          ``bg_size`` candidates are sampled randomly using the seed.
        - ``seed=<int>, uniform=True``: full dataloader is read, then
          candidates are sampled as evenly as possible across classes using
          the seed, giving the smoothest possible class distribution.
          ``uniform=True`` without a seed falls back to the default mode.
        """
        samples: Dict[int, np.ndarray] = {}
        candidate_frames: List[torch.Tensor] = []
        candidate_labels: List[int] = []

        needs_full_pass = seed is not None

        with torch.no_grad():
            for batch in data_loader:
                # Support dict-style and tuple/list-style dataloaders.
                if isinstance(batch, dict):
                    images, labels = batch["x"], batch.get("y")
                elif isinstance(batch, (tuple, list)):
                    images = batch[0]
                    labels = batch[1] if len(batch) > 1 else None
                else:
                    images, labels = batch, None

                if not images.is_floating_point():
                    images = images.float()

                # Iterate image-by-image so we can route each one to either
                # samples or background — never both.
                for i in range(images.shape[0]):
                    label = int(labels[i].item()) if labels is not None else None
                    img = images[i].cpu()

                    if label is not None and label not in samples:
                        # First time we see this class → keep as representative sample.
                        # Do NOT add to background so the sample doesn't partially
                        # cancel itself out during SHAP attribution.
                        samples[label] = img.numpy()
                    else:
                        candidate_frames.append(img)
                        candidate_labels.append(label if label is not None else -1)

                # In default mode stop as soon as we have enough candidates.
                if not needs_full_pass:
                    all_classes_seen = len(samples) >= num_classes
                    enough_background = len(candidate_frames) >= bg_size
                    if all_classes_seen and enough_background:
                        break

        if seed is None:
            # Baseline: take the first bg_size candidates in dataloader order.
            selected_frames = candidate_frames[:bg_size]
            selected_labels = candidate_labels[:bg_size]
            logger.info(
                "ShapExplainer: background = first %d eligible images (no seed)",
                len(selected_frames),
            )
        elif not uniform:
            # Seeded, non-uniform: random sample across all candidates.
            rng = np.random.default_rng(seed)
            n = len(candidate_frames)
            indices = rng.choice(n, size=min(bg_size, n), replace=False)
            selected_frames = [candidate_frames[i] for i in indices]
            selected_labels = [candidate_labels[i] for i in indices]
            logger.info(
                "ShapExplainer: background = %d randomly sampled images "
                "(seed=%d, pool=%d)",
                len(selected_frames),
                seed,
                n,
            )
        else:
            # Seeded + uniform: sample as evenly as possible across classes.
            rng = np.random.default_rng(seed)

            # Group candidate indices by class.
            class_to_indices: Dict[int, List[int]] = {}
            for idx, lbl in enumerate(candidate_labels):
                class_to_indices.setdefault(lbl, []).append(idx)

            known_classes = sorted(class_to_indices)
            n_classes = len(known_classes)
            selected_indices: List[int] = []

            if bg_size >= n_classes:
                quota = bg_size // n_classes
                remainder = bg_size % n_classes
                for rank, cls in enumerate(known_classes):
                    pool = class_to_indices[cls]
                    # First `remainder` classes get one extra image.
                    take = quota + (1 if rank < remainder else 0)
                    take = min(take, len(pool))
                    chosen = rng.choice(pool, size=take, replace=False).tolist()
                    selected_indices.extend(chosen)

                logger.info(
                    "ShapExplainer: background = %d images sampled uniformly "
                    "across %d classes (seed=%d, quota=%d/class, bonus=%d)",
                    bg_size,
                    n_classes,
                    seed,
                    quota,
                    remainder,
                )
            else:
                # More classes than requested slots: take one image from every
                # class so all classes are represented (overrides bg_size,
                # producing a background of size n_classes).
                for cls in known_classes:
                    pool = class_to_indices[cls]
                    selected_indices.append(int(rng.choice(pool)))

                logger.info(
                    "ShapExplainer: background = %d images sampled uniformly "
                    "(one per class across all %d seen classes; requested "
                    "bg_size=%d overridden, seed=%d)",
                    n_classes,
                    n_classes,
                    bg_size,
                    seed,
                )

            selected_frames = [candidate_frames[i] for i in selected_indices]
            selected_labels = [candidate_labels[i] for i in selected_indices]

        background = torch.stack(selected_frames).to(device)
        return samples, background, selected_labels

    def _compute_shap_explanation(
        self,
        explainer: shap.GradientExplainer,
        image_np: np.ndarray,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Run GradientExplainer on a single image and post-process the output.

        Returns:
            display_image: (H, W, C) uint8, ready for matplotlib.
            heatmap:       (H, W) float, normalised absolute SHAP values.
            predicted_class: Index of the top-ranked output neuron.
        """
        image_tensor = torch.tensor(image_np).unsqueeze(0).to(device)  # (1, C, H, W)

        # ranked_outputs=1 asks SHAP to return attributions only for the top-1
        # predicted class (saves memory vs. computing all C output neurons).
        # Use math-only SDPA to avoid in-place view errors from flash/mem-efficient
        # attention kernels inside nn.MultiheadAttention (e.g. CLIP ViT).
        with _safe_sdp_context():
            shap_values, output_indices = explainer.shap_values(
                image_tensor, ranked_outputs=1, output_rank_order="max"
            )

        sv = np.array(shap_values[0])

        # GradientExplainer's output shape is inconsistent across SHAP versions.
        # Normalise to (C, H, W) so downstream code is shape-agnostic.
        if sv.ndim == 4 and sv.shape[-1] == 1:
            sv = sv[..., 0]  # (C, H, W, 1) → (C, H, W)
        elif sv.ndim == 4 and sv.shape[0] == 1:
            sv = sv[0]  # (1, C, H, W) → (C, H, W)

        # Collapse channels into a single 2-D importance map.
        # |SHAP value| tells us how much that pixel contributed (sign less important here).
        heatmap = np.abs(sv).sum(axis=0) if sv.ndim == 3 else np.abs(sv)
        heatmap = heatmap / max(heatmap.max(), 1e-8)  # normalise to [0, 1]

        display_image = _chw_to_hwc_uint8(image_np)
        predicted_class = int(output_indices[0, 0])

        return display_image, heatmap, predicted_class

    def _should_save_all_output_neuron_plots(self, dataset: str) -> bool:
        """Return True when per-output-neuron SHAP plots should be generated."""
        requested = bool(self.config.get("save_all_output_neuron_plots", True))
        if not requested:
            return False
        return _normalise_dataset_name(dataset) in _ALL_OUTPUT_NEURON_DATASETS

    def _to_hwc_batch(self, shap_values: np.ndarray, num_channels: int) -> np.ndarray:
        """Normalise SHAP image attributions to (N, H, W, C) for shap.image_plot."""
        sv = np.array(shap_values)

        # Common PyTorch format from SHAP: (N, C, H, W).
        if sv.ndim == 4 and sv.shape[1] == num_channels:
            return np.transpose(sv, (0, 2, 3, 1))

        # Some SHAP versions return a single sample as (C, H, W).
        if sv.ndim == 3 and sv.shape[0] == num_channels:
            return np.transpose(sv, (1, 2, 0))[np.newaxis]

        # Already in image format: (N, H, W, C).
        if sv.ndim == 4 and sv.shape[-1] == num_channels:
            return sv

        # Single image in HWC.
        if sv.ndim == 3 and sv.shape[-1] == num_channels:
            return sv[np.newaxis]

        # Single-channel 2-D maps.
        if sv.ndim == 2:
            return sv[np.newaxis, :, :, np.newaxis]

        # Fallback for shape (C, H, W, 1) occasionally produced by SHAP.
        if sv.ndim == 4 and sv.shape[0] == num_channels and sv.shape[-1] == 1:
            squeezed = sv[..., 0]
            return np.transpose(squeezed, (1, 2, 0))[np.newaxis]

        # Flattened single-sample edge case: (N, C, H, W, 1) arriving here
        # after per-output slicing from a 5-D SHAP tensor.
        if sv.ndim == 5 and sv.shape[1] == num_channels and sv.shape[-1] == 1:
            return np.transpose(sv[..., 0], (0, 2, 3, 1))

        raise ValueError(f"Unsupported SHAP image shape for plotting: {sv.shape}")

    def _coerce_all_output_hwc_batches(
        self,
        raw_shap_values,
        num_channels: int,
    ) -> List[np.ndarray]:
        """Convert SHAP all-output return value into list[(N,H,W,C)] by class.

        SHAP may return either:
        - list[class] of per-class arrays, or
        - one ndarray with an explicit output axis.
        """
        if isinstance(raw_shap_values, tuple):
            raw_shap_values = raw_shap_values[0]

        if isinstance(raw_shap_values, (list, tuple)):
            return [
                self._to_hwc_batch(class_sv, num_channels)
                for class_sv in raw_shap_values
            ]

        arr = np.array(raw_shap_values)

        # One array with output axis embedded.
        if arr.ndim == 5:
            # (N, C, H, W, K)
            if arr.shape[1] == num_channels:
                return [
                    self._to_hwc_batch(arr[..., k], num_channels)
                    for k in range(arr.shape[-1])
                ]

            # (N, K, C, H, W)
            if arr.shape[2] == num_channels:
                return [
                    self._to_hwc_batch(arr[:, k, ...], num_channels)
                    for k in range(arr.shape[1])
                ]

            # (N, H, W, C, K)
            if arr.shape[3] == num_channels:
                return [
                    self._to_hwc_batch(arr[..., k], num_channels)
                    for k in range(arr.shape[-1])
                ]

            # (N, K, H, W, C)
            if arr.shape[-1] == num_channels:
                return [
                    self._to_hwc_batch(arr[:, k, ...], num_channels)
                    for k in range(arr.shape[1])
                ]

            raise ValueError(f"Unsupported SHAP all-output tensor shape: {arr.shape}")

        # Degenerate single-output case.
        if arr.ndim in (2, 3, 4):
            return [self._to_hwc_batch(arr, num_channels)]

        raise ValueError(f"Unsupported SHAP all-output structure shape: {arr.shape}")

    def _save_all_output_neuron_plot(
        self,
        explainer: shap.GradientExplainer,
        image_np: np.ndarray,
        save_path: Path,
        device: torch.device,
        figure_title: str,
        row_label: str,
    ) -> Optional[str]:
        """Save one shap.image_plot for a single sample over all output neurons."""
        image_tensor = torch.tensor(image_np).unsqueeze(0).to(device)
        num_channels = int(image_np.shape[0])

        try:
            with _safe_sdp_context():
                shap_values_all = explainer.shap_values(image_tensor)
            shap_values_hwc = self._coerce_all_output_hwc_batches(
                shap_values_all, num_channels
            )
        except Exception as exc:
            logger.warning(
                "Failed SHAP all-output computation for %s: %s", save_path.name, exc
            )
            return None

        try:
            # shap.image_plot forwards pixel_values to imshow; keep floats in [0,1]
            # to avoid clipping warnings when model inputs are normalised.
            pixel_values = _chw_to_hwc_float01(image_np)[np.newaxis]
            n_outputs = len(shap_values_hwc)
            labels = np.array([[f"out {j}" for j in range(n_outputs)]], dtype=object)

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            width = max(14, min(40, 2 + n_outputs * 2))
            shap.image_plot(
                shap_values_hwc,
                pixel_values,
                labels=labels,
                true_labels=[row_label],
                width=width,
                show=False,
            )
            fig = plt.gcf()
            fig.suptitle(figure_title, fontsize=12, y=1.02)
            fig.savefig(
                save_path.with_suffix(".pdf"),
                dpi=150,
                format="pdf",
                bbox_inches="tight",
            )
            plt.close(fig)
            return str(save_path)
        except Exception as exc:
            logger.warning(
                "Failed to save SHAP all-output plot %s: %s", save_path.name, exc
            )
            return None

    def _run_clean_explanations(
        self,
        explainer: shap.GradientExplainer,
        samples: Dict[int, np.ndarray],
        model: nn.Module,
        output_dir: Path,
        device: torch.device,
        grayscale_before_overlay: bool = False,
        save_all_output_neurons: bool = False,
    ) -> List[str]:
        """Explain each clean representative image; save one PDF per class."""
        saved_paths = []
        title_fontsize = int(self.config.get("plot_title_fontsize", 14))
        for class_id in sorted(samples):
            try:
                display_img, heatmap, predicted = self._compute_shap_explanation(
                    explainer, samples[class_id], device
                )
            except Exception as exc:
                logger.warning("SHAP failed for class %d: %s", class_id, exc)
                continue

            path = output_dir / f"clean_class_{class_id}_pred_{predicted}.pdf"
            _save_explanation_plot(
                images=[display_img, display_img],
                shap_heatmap=heatmap,
                titles=[
                    f"Clean sample\ntrue class {class_id}",
                    f"Top-1 SHAP\npredicted class {predicted}",
                ],
                save_path=path,
                title_fontsize=title_fontsize,
                grayscale_before_overlay=grayscale_before_overlay,
            )
            saved_paths.append(str(path))
            logger.info("  Saved clean explanation: %s", path.name)

            if save_all_output_neurons:
                all_outputs_path = (
                    output_dir / f"clean_class_{class_id}_all_outputs.pdf"
                )
                saved_all_outputs = self._save_all_output_neuron_plot(
                    explainer,
                    samples[class_id],
                    all_outputs_path,
                    device,
                    figure_title=(
                        f"SHAP image_plot on clean representative "
                        f"(true class {class_id}, pred {predicted})"
                    ),
                    row_label=f"true {class_id} | pred {predicted}",
                )
                if saved_all_outputs is not None:
                    saved_paths.append(saved_all_outputs)
                    logger.info(
                        "  Saved clean all-output explanation: %s",
                        all_outputs_path.name,
                    )

        return saved_paths

    def _predict_background(
        self,
        model: nn.Module,
        background: torch.Tensor,
        device: torch.device,
    ) -> List[int]:
        """Run the model on every background image and return predicted class indices."""
        preds: List[int] = []
        with torch.no_grad():
            for img in background:
                logits = model(img.unsqueeze(0).to(device))
                preds.append(int(logits.argmax(dim=1).item()))
        return preds

    def _save_background_grid(
        self,
        background: torch.Tensor,
        background_labels: List[int],
        background_preds: List[int],
        output_dir: Path,
    ) -> Tuple[List[str], Dict[int, int]]:
        """Save a grid of background images and a class-distribution bar chart.

        Each image tile is annotated with its true label and model prediction.
        Tiles where the prediction disagrees with the true label are highlighted
        with a red border, making misclassified background images immediately
        visible.

        Returns:
            saved_paths: Paths to the two saved PDFs (grid + distribution chart).
            counts:      {class_label → count} for inclusion in DetectionResult.
        """
        counts: Dict[int, int] = {}
        for lbl in background_labels:
            counts[lbl] = counts.get(lbl, 0) + 1

        known_labels = sorted(counts)
        n_total = len(background_labels)
        ideal = n_total / max(len(known_labels), 1)

        # Log distribution summary
        logger.info("ShapExplainer background distribution (%d images):", n_total)
        for lbl in known_labels:
            cnt = counts[lbl]
            logger.info(
                "  class %2d : %2d images (%.1f%%)  [ideal %.1f]",
                lbl,
                cnt,
                100 * cnt / n_total,
                ideal,
            )

        saved_paths: List[str] = []

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n = background.shape[0]

            # ── Panel 1: image grid ───────────────────────────────────────────
            ncols = min(8, n)
            nrows = (n + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.8))
            axes = np.array(axes).reshape(-1)

            for idx in range(len(axes)):
                ax = axes[idx]
                if idx < n:
                    img = _chw_to_hwc_uint8(background[idx].cpu().numpy())
                    is_grayscale = img.shape[2] == 1
                    ax.imshow(
                        img[:, :, 0] if is_grayscale else img,
                        cmap="gray" if is_grayscale else None,
                    )
                    true_lbl = background_labels[idx]
                    pred_lbl = background_preds[idx]
                    match = true_lbl == pred_lbl
                    color = "black" if match else "red"
                    ax.set_title(
                        f"true {true_lbl}\npred {pred_lbl}",
                        fontsize=5,
                        color=color,
                    )
                    # Red border for misclassified images.
                    for spine in ax.spines.values():
                        spine.set_edgecolor(color)
                        spine.set_linewidth(1.5 if not match else 0.5)
                ax.axis("off")

            fig.suptitle(f"SHAP background set ({n} images)", fontsize=9, y=1.01)
            fig.tight_layout()
            grid_path = output_dir / "background_samples.pdf"
            fig.savefig(grid_path, dpi=150, format="pdf", bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(str(grid_path))
            logger.info("Saved background grid: %s", grid_path.name)

            # ── Panel 2: class distribution bar chart ─────────────────────────
            fig2, ax2 = plt.subplots(figsize=(max(4, len(known_labels) * 0.6), 3))
            bar_labels = [str(lbl) if lbl >= 0 else "?" for lbl in known_labels]
            bar_counts = [counts[lbl] for lbl in known_labels]
            bar_positions = np.arange(len(bar_labels))
            bars = ax2.bar(
                bar_positions, bar_counts, color="steelblue", edgecolor="white"
            )
            ax2.axhline(
                ideal,
                color="tomato",
                linestyle="--",
                linewidth=1,
                label=f"ideal ({ideal:.1f})",
            )
            ax2.bar_label(bars, fontsize=7)
            ax2.set_xticks(bar_positions, bar_labels)
            ax2.set_xlabel("Class label")
            ax2.set_ylabel("Count")
            ax2.set_title(f"Background class distribution (n={n_total})")
            ax2.legend(fontsize=7)
            fig2.tight_layout()
            dist_path = output_dir / "background_distribution.pdf"
            fig2.savefig(dist_path, dpi=150, format="pdf", bbox_inches="tight")
            plt.close(fig2)
            saved_paths.append(str(dist_path))
            logger.info("Saved background distribution chart: %s", dist_path.name)

        except Exception as exc:
            logger.warning("Failed to save background visualisations: %s", exc)

        return saved_paths, counts

    def _load_nc_summary(self, artifact_root: str, run_name: str) -> Optional[dict]:
        """Read Neural Cleanse's summary.json if it exists, otherwise return None."""
        summary_path = (
            Path(artifact_root) / run_name / "neural_cleanse" / "summary.json"
        )
        if not summary_path.exists():
            logger.info(
                "ShapExplainer: no Neural Cleanse results found at %s", summary_path
            )
            return None
        with open(summary_path) as f:
            return json.load(f)

    def _run_triggered_explanations(
        self,
        explainer: shap.GradientExplainer,
        samples: Dict[int, np.ndarray],
        model_info: ModelInfo,
        output_dir: Path,
        device: torch.device,
        grayscale_before_overlay: bool = False,
        save_all_output_neurons: bool = False,
    ) -> Tuple[dict, List[str]]:
        """Stamp the Neural Cleanse trigger onto each representative image and
        explain the triggered inputs.

        The goal is to see whether the trigger region becomes the dominant
        attribution — a strong signal that the model has a backdoor keyed to
        that trigger location.

        Returns:
            nc_metadata:  Dict suitable for inclusion in DetectionResult.details.
            saved_paths:  List of saved PNG file paths.
        """
        artifact_root = self.config.get("artifact_root_dir", "./Detection/results")
        run_name = (
            f"{model_info.attack_name}_{model_info.architecture}_{model_info.dataset}"
        )

        nc_metadata = {"nc_results_found": False, "nc_trigger_label_used": None}
        saved_paths = []
        title_fontsize = int(self.config.get("plot_title_fontsize", 14))

        try:
            nc = self._load_nc_summary(artifact_root, run_name)
        except Exception as exc:
            logger.warning("ShapExplainer: failed to load NC results: %s", exc)
            return nc_metadata, saved_paths

        if nc is None:
            return nc_metadata, saved_paths

        confirmed_labels = nc.get("confirmed_labels", [])
        if not confirmed_labels:
            logger.info(
                "ShapExplainer: no confirmed NC labels — skipping triggered plots."
            )
            return nc_metadata, saved_paths

        # Pick the confirmed target with the smallest trigger (most plausible).
        clip_min = float(nc.get("clip_min", 0.0))
        clip_max = float(nc.get("clip_max", 1.0))
        best_entry = min(confirmed_labels, key=lambda e: e["l1_norm"])
        target_label = int(best_entry["label"])

        trigger_dir = Path(artifact_root) / run_name / "neural_cleanse" / "triggers"
        trigger = _load_nc_trigger(
            trigger_dir, target_label, model_info.input_shape[0], clip_min, clip_max
        )
        if trigger is None:
            logger.warning("NC trigger PNGs missing for label %d.", target_label)
            return nc_metadata, saved_paths

        nc_metadata = {"nc_results_found": True, "nc_trigger_label_used": target_label}
        mask, pattern = trigger
        logger.info(
            "ShapExplainer: applying NC trigger for label %d (L1=%.2f)",
            target_label,
            best_entry["l1_norm"],
        )

        for class_id in sorted(samples):
            # Blend the trigger into the clean image using the mask:
            #   triggered = (1 - mask) * clean + mask * pattern
            triggered_image = np.clip(
                (1 - mask[None]) * samples[class_id] + mask[None] * pattern,
                clip_min,
                clip_max,
            )
            try:
                _, heatmap, predicted = self._compute_shap_explanation(
                    explainer, triggered_image, device
                )
            except Exception as exc:
                logger.warning(
                    "SHAP failed for Neural Cleanse triggered class %d: %s",
                    class_id,
                    exc,
                )
                continue

            path = (
                output_dir
                / f"neural_cleanse_triggered_tgt{target_label}_class_{class_id}_pred_{predicted}.pdf"
            )
            # Trigger applied to a blank canvas: shows the recovered pattern
            # in its natural color without the clean image bleeding through.
            trigger_only_image = mask[None] * pattern
            _save_explanation_plot(
                images=[
                    _chw_to_hwc_uint8(samples[class_id]),  # original clean image
                    _chw_to_hwc_uint8(triggered_image),  # image with trigger stamped on
                    _chw_to_hwc_uint8(trigger_only_image),  # trigger on blank canvas
                    _chw_to_hwc_uint8(
                        triggered_image
                    ),  # full SHAP overlay on triggered
                    _chw_to_hwc_uint8(triggered_image),  # placeholder; pure SHAP rendered on top
                ],
                shap_heatmap=heatmap,
                titles=[
                    f"Clean sample\ntrue class {class_id}",
                    f"NC trigger\ntarget {target_label}, pred {predicted}",
                    "Trigger only",
                    f"SHAP for model's prediction\npred class {predicted}",
                    "Top-1 SHAP heatmap",
                ],
                save_path=path,
                title_fontsize=title_fontsize,
                grayscale_before_overlay=grayscale_before_overlay,
                overlay_panel_index=3,
                pure_heatmap_indices=[4],
            )
            saved_paths.append(str(path))
            logger.info("  Saved triggered explanation: %s", path.name)

            if save_all_output_neurons:
                all_outputs_path = output_dir / (
                    f"neural_cleanse_triggered_tgt{target_label}_class_{class_id}_all_outputs.pdf"
                )
                saved_all_outputs = self._save_all_output_neuron_plot(
                    explainer,
                    triggered_image,
                    all_outputs_path,
                    device,
                    figure_title=(
                        "SHAP image_plot on Neural Cleanse-triggered representative "
                        f"(true class {class_id}, pred {predicted}, target {target_label})"
                    ),
                    row_label=f"true {class_id} | pred {predicted}",
                )
                if saved_all_outputs is not None:
                    saved_paths.append(saved_all_outputs)
                    logger.info(
                        "  Saved Neural Cleanse all-output explanation: %s",
                        all_outputs_path.name,
                    )

        return nc_metadata, saved_paths

    def _run_adapter_triggered_explanations(
        self,
        explainer: shap.GradientExplainer,
        samples: Dict[int, np.ndarray],
        model_info: ModelInfo,
        output_dir: Path,
        device: torch.device,
        model: nn.Module,
        grayscale_before_overlay: bool = False,
        save_all_output_neurons: bool = False,
    ) -> List[str]:
        """Apply the adapter's own trigger to each sample and explain.

        Uses the adapter stored in ``model_info.extra["adapter"]`` to construct
        triggered samples, then runs SHAP on them for comparison against the
        clean explanations.
        """
        adapter = model_info.extra.get("adapter") if model_info.extra else None
        if adapter is None:
            logger.warning(
                "ShapExplainer: use_triggered_samples=True but no adapter in model_info.extra."
            )
            return []

        trigger_kwargs = dict(self.config.get("trigger_kwargs", {}))
        saved_paths: List[str] = []
        title_fontsize = int(self.config.get("plot_title_fontsize", 14))

        # The adapter's trigger_mode flips a flag on the same model object the
        # explainer holds internally, so the flag is active during
        # shap_values()'s forward passes too.
        with adapter.trigger_mode(model):
            for class_id in sorted(samples):
                clean_np = samples[class_id]
                clean_tensor = torch.tensor(clean_np)
                per_sample_kwargs = dict(trigger_kwargs)
                per_sample_kwargs.setdefault("target_label", class_id)

                triggered = adapter.get_triggered_sample(
                    clean_tensor, **per_sample_kwargs
                )
                if triggered is None:
                    logger.info(
                        "ShapExplainer: adapter returned None for triggered sample — skipping all."
                    )
                    break

                triggered_np = triggered.numpy()
                # Third panel: for patch-based attacks, apply the trigger to
                # a blank canvas so it shows in its natural color. For
                # steganographic attacks the perturbation is distributed at
                # imperceptible magnitude, so render |triggered - clean|
                # instead — the blank-canvas view would be meaningless.
                is_stego_panel = type(adapter).__name__ in _STEGO_ADAPTER_CLASSES
                trigger_only_np: Optional[np.ndarray] = None
                if is_stego_panel:
                    trigger_only_np = np.abs(triggered_np - clean_np)
                else:
                    try:
                        blank = torch.zeros_like(clean_tensor)
                        trigger_only = adapter.get_triggered_sample(
                            blank, **per_sample_kwargs
                        )
                        if trigger_only is not None:
                            trigger_only_np = trigger_only.numpy()
                    except Exception as exc:
                        logger.warning(
                            "Trigger-on-blank failed for class %d: %s", class_id, exc
                        )

                try:
                    _, heatmap, predicted = self._compute_shap_explanation(
                        explainer, triggered_np, device
                    )
                except Exception as exc:
                    logger.warning(
                        "SHAP failed for adapter-triggered class %d: %s", class_id, exc
                    )
                    continue

                path = output_dir / f"TRIGGERED_class_{class_id}_pred_{predicted}.pdf"
                trigger_panel_np = trigger_only_np if trigger_only_np is not None else triggered_np
                _save_explanation_plot(
                    images=[
                        _chw_to_hwc_uint8(clean_np),
                        _chw_to_hwc_uint8(triggered_np),
                        _chw_to_hwc_uint8(trigger_panel_np),
                        _chw_to_hwc_uint8(triggered_np),
                        _chw_to_hwc_uint8(triggered_np),
                    ],
                    shap_heatmap=heatmap,
                    titles=[
                        f"Clean sample\ntrue class {class_id}",
                        f"Trigger applied\npred class {predicted}",
                        "Difference (magnified)" if is_stego_panel else "Trigger",
                        "Top-1 SHAP heatmap overlaid sample",
                        "Top-1 SHAP heatmap",
                    ],
                    save_path=path,
                    title_fontsize=title_fontsize,
                    grayscale_before_overlay=grayscale_before_overlay,
                    overlay_panel_index=3,
                    pure_heatmap_indices=[4],
                )
                saved_paths.append(str(path))
                logger.info("  Saved adapter-triggered explanation: %s", path.name)

                if save_all_output_neurons:
                    all_outputs_path = (
                        output_dir / f"TRIGGERED_class_{class_id}_all_outputs.pdf"
                    )
                    saved_all_outputs = self._save_all_output_neuron_plot(
                        explainer,
                        triggered_np,
                        all_outputs_path,
                        device,
                        figure_title=(
                            "SHAP image_plot on adapter-triggered representative "
                            f"(true class {class_id}, pred {predicted})"
                        ),
                        row_label=f"true {class_id} | pred {predicted}",
                    )
                    if saved_all_outputs is not None:
                        saved_paths.append(saved_all_outputs)
                        logger.info(
                            "  Saved adapter-triggered all-output explanation: %s",
                            all_outputs_path.name,
                        )

        return saved_paths

    def _empty_result(self, model_info: ModelInfo) -> DetectionResult:
        return DetectionResult(
            method_name=self.name,
            attack_name=model_info.attack_name,
            model_architecture=model_info.architecture,
            dataset=model_info.dataset,
            is_backdoor_detected=False,
            confidence_score=0.0,
        )
