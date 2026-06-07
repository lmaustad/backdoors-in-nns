#!/usr/bin/env python3
"""
Neural Cleanse backdoor detector — detection phase only.

Reference
---------
Wang et al., "Neural Cleanse: Identifying and Mitigating Backdoor Attacks in
Neural Networks", IEEE S&P 2019.  https://github.com/bolunwang/backdoor

Algorithm
---------
Phase 1 — Trigger reverse engineering
    For every potential target label y_t:
        Minimise  CE(f((1−m)·x + m·δ), y_t)  +  λ·‖m‖₁
    over mask m ∈ (0,1)^{H×W} and pattern δ ∈ (clip_min, clip_max)^{C×H×W}.
    m and δ are parameterised in tanh-space (unconstrained optimisation),
    mirroring the original Keras implementation exactly.
    A dynamic cost schedule adjusts λ so that the optimiser first achieves
    high attack-success rate and then is pushed toward minimal mask size.

Phase 2 — MAD outlier detection
    A backdoored model will have one label whose reverse-engineered trigger
    norm ‖m‖₁ is an outlier on the *small* side.
    Anomaly index = |min(L1) − median(L1)| / (1.4826 · MAD)
    If anomaly_index ≥ mad_threshold (default 2.0) the model is flagged.
"""

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Detection.core.base_adapter import ModelInfo
from Detection.core.base_detector import DetectionMethod, DetectionResult
from Detection.core.registry import register_detector

# matplotlib is imported lazily inside helpers to avoid hard dependency at module level.

logger = logging.getLogger(__name__)

# Matches Keras backend K.epsilon() used in the original tanh parameterisation.
_EPSILON: float = 1e-7


# ─────────────────────────────────────────────────────────────────────────────
# Tanh-space helpers
# ─────────────────────────────────────────────────────────────────────────────


def _mask_from_tanh(mask_tanh: torch.Tensor) -> torch.Tensor:
    """Unconstrained tanh variable → mask ∈ (0, 1)."""
    return torch.tanh(mask_tanh) / (2.0 - _EPSILON) + 0.5


def _pattern_from_tanh(
    pattern_tanh: torch.Tensor,
    clip_min: float,
    clip_max: float,
) -> torch.Tensor:
    """Unconstrained tanh variable → pattern ∈ (clip_min, clip_max)."""
    unit = torch.tanh(pattern_tanh) / (2.0 - _EPSILON) + 0.5  # (0, 1)
    return unit * (clip_max - clip_min) + clip_min


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: reverse-engineer trigger for one target label
# ─────────────────────────────────────────────────────────────────────────────


def reverse_engineer_trigger(
    model: nn.Module,
    x_pool: torch.Tensor,  # (N, C, H, W) on CPU
    target_label: int,
    device: torch.device,
    # optimisation
    steps: int = 1000,
    mini_batch: Optional[
        int
    ] = None,  # mini-batches per step; None → ceil(N/batch_size)
    batch_size: int = 32,
    lr: float = 0.1,
    # dynamic cost schedule
    init_cost: float = 1e-3,
    cost_multiplier: float = 1.5,
    patience: int = 10,
    reset_cost_to_zero: bool = True,
    # termination
    attack_succ_threshold: float = 0.99,
    early_stop: bool = True,
    early_stop_threshold: float = 0.99,
    early_stop_patience: int = 20,
    # input bounds (normalised space)
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    # reproducibility: seed for numpy RNG used in mask/pattern init
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Reverse-engineer the minimum trigger (mask, pattern) that makes the model
    classify any input as ``target_label``.

    Returns
    -------
    mask_best    : ndarray (H, W)      single-channel mask in [0, 1]
    pattern_best : ndarray (C, H, W)   pattern in [clip_min, clip_max]
    reg_best     : float               L1 norm of mask_best at the best step
    converged    : bool                True if a valid solution (asr ≥ threshold)
                                       was found; False if fell back to final state
    """
    model.eval()

    input_shape = x_pool.shape[1:]  # (C, H, W)
    C, H, W = input_shape[0], input_shape[1], input_shape[2]
    n_samples = x_pool.shape[0]

    if mini_batch is None:
        mini_batch = math.ceil(n_samples / batch_size)

    # ── Random initialisation in [0, 1] → converted to tanh-space ────────
    # Fixed per-label seed → reproducible across runs, independent per label.
    # Identical init strategy to the original's np.random.random() → reset_state().
    rng = np.random.default_rng(seed=seed)
    init_mask = rng.random((H, W)).astype(np.float32)  # [0, 1]
    init_pat = rng.random((C, H, W)).astype(np.float32)  # [0, 1]

    # inverse of  tanh(·)/(2−ε) + 0.5
    mask_tanh_np = np.arctanh(
        np.clip((init_mask - 0.5) * (2.0 - _EPSILON), -1 + 1e-6, 1 - 1e-6)
    )
    pat_tanh_np = np.arctanh(
        np.clip((init_pat - 0.5) * (2.0 - _EPSILON), -1 + 1e-6, 1 - 1e-6)
    )

    mask_tanh = torch.tensor(
        mask_tanh_np[np.newaxis], device=device, requires_grad=True
    )  # (1, H, W)
    pattern_tanh = torch.tensor(
        pat_tanh_np, device=device, requires_grad=True
    )  # (C, H, W)

    optimizer = torch.optim.Adam([mask_tanh, pattern_tanh], lr=lr, betas=(0.5, 0.9))

    # ── Dynamic cost (λ) ─────────────────────────────────────────────────
    cost = 0.0 if reset_cost_to_zero else init_cost
    cost_mult_up = cost_multiplier
    cost_mult_down = cost_multiplier**1.5

    cost_set_counter = 0
    cost_up_counter = 0
    cost_down_counter = 0
    cost_up_flag = False
    cost_down_flag = False

    # ── Best-result bookkeeping ───────────────────────────────────────────
    mask_best = None
    pattern_best = None
    reg_best = float("inf")

    # ── Early-stopping state ──────────────────────────────────────────────
    early_stop_counter = 0
    early_stop_reg_best = float("inf")

    # ── Main loop (one outer iteration = one "step"/epoch in the original) ─
    for step in range(steps):

        asr_list: List[float] = []
        reg_list: List[float] = []

        for _ in range(mini_batch):
            # Random batch with replacement — equivalent to Keras flow() cycling
            idx = torch.randint(0, n_samples, (batch_size,))
            x_batch = x_pool[idx].to(device)

            optimizer.zero_grad()

            mask = _mask_from_tanh(mask_tanh)  # (1, H, W)
            mask_ch = mask.expand(C, H, W)  # (C, H, W)
            pattern = _pattern_from_tanh(pattern_tanh, clip_min, clip_max)

            # Blended adversarial input
            x_adv = (1.0 - mask_ch) * x_batch + mask_ch * pattern
            x_adv = torch.clamp(x_adv, clip_min, clip_max)

            logits = model(x_adv)
            y_target = torch.full(
                (x_batch.shape[0],), target_label, dtype=torch.long, device=device
            )

            loss_ce = F.cross_entropy(logits, y_target)
            # L1 reg on mask / num_channels  (matches original `K.sum(K.abs(mask)) / img_color`)
            loss_reg = torch.sum(torch.abs(mask_ch)) / C
            loss = loss_ce + cost * loss_reg

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                asr = (logits.argmax(1) == target_label).float().mean().item()
                reg = (torch.sum(torch.abs(mask_ch)) / C).item()

            asr_list.append(asr)
            reg_list.append(reg)

        avg_asr = float(np.mean(asr_list)) if asr_list else 0.0
        avg_reg = float(np.mean(reg_list)) if reg_list else float("inf")

        # ── Save best ─────────────────────────────────────────────────────
        if avg_asr >= attack_succ_threshold and avg_reg < reg_best:
            with torch.no_grad():
                m = _mask_from_tanh(mask_tanh)
                p = _pattern_from_tanh(pattern_tanh, clip_min, clip_max)
            mask_best = m.squeeze(0).detach().cpu().numpy()  # (H, W)
            pattern_best = p.detach().cpu().numpy()  # (C, H, W)
            reg_best = avg_reg

        logger.debug(
            "  label %3d | step %4d | cost %.2e | ASR %.3f | " "reg %.4f | best %.4f",
            target_label,
            step,
            cost,
            avg_asr,
            avg_reg,
            reg_best,
        )

        # ── Early stopping ────────────────────────────────────────────────
        if early_stop and reg_best < float("inf"):
            if reg_best >= early_stop_threshold * early_stop_reg_best:
                early_stop_counter += 1
            else:
                early_stop_counter = 0
            early_stop_reg_best = min(reg_best, early_stop_reg_best)

            if (
                cost_down_flag
                and cost_up_flag
                and early_stop_counter >= early_stop_patience
            ):
                logger.debug("  label %d: early stop at step %d", target_label, step)
                break

        # ── Dynamic cost schedule ─────────────────────────────────────────
        if cost == 0.0 and avg_asr >= attack_succ_threshold:
            cost_set_counter += 1
            if cost_set_counter >= patience:
                cost = init_cost
                cost_up_counter = 0
                cost_down_counter = 0
                cost_up_flag = False
                cost_down_flag = False
                logger.debug("  label %d: cost initialised → %.2e", target_label, cost)
        else:
            cost_set_counter = 0

        if avg_asr >= attack_succ_threshold:
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1

        if cost_up_counter >= patience:
            cost_up_counter = 0
            cost *= cost_mult_up
            cost_up_flag = True
            logger.debug("  label %d: cost ↑ → %.2e", target_label, cost)
        elif cost_down_counter >= patience:
            cost_down_counter = 0
            cost /= cost_mult_down
            cost_down_flag = True
            logger.debug("  label %d: cost ↓ → %.2e", target_label, cost)

    # ── Fall-back: use final state when optimisation never found a good mask ─
    converged = mask_best is not None
    if not converged:
        with torch.no_grad():
            m = _mask_from_tanh(mask_tanh)
            p = _pattern_from_tanh(pattern_tanh, clip_min, clip_max)
        mask_best = m.squeeze(0).detach().cpu().numpy()
        pattern_best = p.detach().cpu().numpy()
        reg_best = float((torch.sum(torch.abs(m.expand(C, H, W))) / C).item())

    return mask_best, pattern_best, reg_best, converged


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: MAD-based outlier detection
# ─────────────────────────────────────────────────────────────────────────────


def mad_outlier_detection(
    l1_norms: List[float],
    idx_mapping: Dict[int, int],
    mad_threshold: float = 2.0,
) -> Tuple[bool, float, List[Tuple[int, float]], float, float]:
    """
    MAD-based outlier detection exactly as in Wang et al. (2019) and the
    reference ``mad_outlier_detection.py``.

        anomaly_index = |min(L1) − median(L1)| / (1.4826 · median(|L1 − median(L1)|))

    A model is flagged when anomaly_index ≥ mad_threshold.

    Returns
    -------
    is_backdoor   : bool
    anomaly_index : float
    flagged_labels: list[(label, l1_norm)]   individually suspicious labels
    median        : float
    mad           : float
    """
    consistency_constant = 1.4826  # scaling factor: MAD → σ for Gaussian

    l1_arr = np.array(l1_norms, dtype=np.float64)
    median = float(np.median(l1_arr))
    mad = float(consistency_constant * np.median(np.abs(l1_arr - median)))
    if mad < 1e-8:
        mad = 1e-8

    anomaly_index = float(np.abs(np.min(l1_arr) - median) / mad)

    logger.info(
        "Neural Cleanse | median: %.4f  MAD: %.4f  anomaly index: %.4f",
        median,
        mad,
        anomaly_index,
    )

    flag_list: List[Tuple[int, float]] = []
    for y_label, idx in idx_mapping.items():
        if l1_arr[idx] > median:
            continue
        if np.abs(l1_arr[idx] - median) / mad > mad_threshold:
            flag_list.append((int(y_label), float(l1_arr[idx])))

    flag_list.sort(key=lambda x: x[1])
    is_backdoor = anomaly_index >= mad_threshold

    return is_backdoor, anomaly_index, flag_list, median, mad


# ─────────────────────────────────────────────────────────────────────────────
# Trigger image helpers
# ─────────────────────────────────────────────────────────────────────────────


def _save_trigger_images(
    mask: np.ndarray,  # (H, W) in [0, 1]
    pattern: np.ndarray,  # (C, H, W) in [clip_min, clip_max]
    target_label: int,
    save_dir: Path,
    clip_min: float,
    clip_max: float,
) -> None:
    """
    Save mask, pattern, and fusion PNG for one target label.

    Files written
    -------------
    mask_label_{N}.png    — single-channel mask intensity (grayscale)
    pattern_label_{N}.png — recovered pattern normalised to display range
    fusion_label_{N}.png  — mask × pattern (the trigger "stamp")
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping trigger image save")
        return

    pat_range = max(clip_max - clip_min, 1e-8)
    # Pattern → display range [0, 1]
    pat_norm = np.clip((pattern - clip_min) / pat_range, 0.0, 1.0)  # (C, H, W)
    # Fusion: mask × pattern (trigger stamp)
    fusion = mask[np.newaxis] * pat_norm  # (C, H, W)

    def _to_display(arr: np.ndarray) -> np.ndarray:
        """(C, H, W) → (H, W) for C=1  or  (H, W, C) for C=3+."""
        return arr[0] if arr.shape[0] == 1 else np.transpose(arr, (1, 2, 0))

    saves = [
        (f"mask_label_{target_label}.png", mask, "gray"),
        (f"pattern_label_{target_label}.png", _to_display(pat_norm), None),
        (f"fusion_label_{target_label}.png", _to_display(fusion), None),
    ]
    for fname, img, cmap in saves:
        kwargs = {"cmap": cmap} if cmap else {}
        plt.imsave(save_dir / fname, np.clip(img, 0.0, 1.0), **kwargs)


def _save_l1_norm_plot(
    l1_norms: List[float],
    idx_mapping: Dict[int, int],
    flagged_by_mad: List[Tuple[int, float]],
    anomaly_index: float,
    mad_threshold: float,
    median: float,
    save_dir: Path,
) -> None:
    """
    Bar chart of per-class L1 norms with MAD-flagged classes highlighted and
    the median line drawn.  Saved as ``l1_norms.pdf``.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping L1 norm plot")
        return

    mad_flagged_set = {lbl for lbl, _ in flagged_by_mad}

    labels_sorted = sorted(idx_mapping.keys())
    norms = [l1_norms[idx_mapping[y]] for y in labels_sorted]
    colors = ["#d62728" if y in mad_flagged_set else "#1f77b4" for y in labels_sorted]

    fig, ax = plt.subplots(figsize=(max(8, len(labels_sorted) * 0.5), 4))
    ax.bar(labels_sorted, norms, color=colors)
    ax.axhline(
        median,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Median L1 norm = {median:.2f}",
    )
    ax.set_xlabel("Candidate target class")
    ax.set_ylabel("L1 norm of recovered trigger mask")
    ax.set_title(
        "Neural Cleanse: per-class L1 norms of recovered triggers\n"
        f"Anomaly index of smallest L1 = {anomaly_index:.2f} "
    )

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor="#d62728", label="MAD-flagged class"),
        Patch(facecolor="#1f77b4", label="Non-flagged class"),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Median L1 norm = {median:.2f}",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_dir / "l1_norms.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Detector plug-in
# ─────────────────────────────────────────────────────────────────────────────


@register_detector("neural_cleanse")
class NeuralCleanseDetector(DetectionMethod):
    """
    Neural Cleanse backdoor detector (Wang et al., IEEE S&P 2019).

    For each class the detector reverse-engineers the smallest adversarial
    patch (mask + pattern) that forces the model to output that class.
    A MAD-based outlier test on the recovered trigger L1 norms then identifies
    backdoored models — a genuine backdoor trigger is anomalously small.

    Detection only — no model surgery or unlearning is performed.
    """

    @property
    def name(self) -> str:
        return "neural_cleanse"

    def requires_data(self) -> bool:
        return True

    def detect(
        self,
        model: nn.Module,
        model_info: ModelInfo,
        data_loader: DataLoader,
        device: torch.device = torch.device("cpu"),
    ) -> DetectionResult:

        model = model.to(device).eval()

        # ── Config ────────────────────────────────────────────────────────
        num_classes = int(self.config.get("num_classes", 0))
        steps = int(self.config.get("steps", 1000))
        mini_batch = self.config.get("mini_batch", None)
        if mini_batch is not None:
            mini_batch = int(mini_batch)
        batch_size = int(self.config.get("batch_size", 32))
        lr = float(self.config.get("lr", 0.1))
        init_cost = float(self.config.get("init_cost", 1e-3))
        cost_multiplier = float(self.config.get("cost_multiplier", 1.5))
        patience = int(self.config.get("patience", 10))
        reset_cost_to_zero = bool(self.config.get("reset_cost_to_zero", True))
        attack_succ_threshold = float(self.config.get("attack_succ_threshold", 0.99))
        early_stop = bool(self.config.get("early_stop", True))
        early_stop_threshold = float(self.config.get("early_stop_threshold", 0.99))
        early_stop_patience = int(self.config.get("early_stop_patience", 20))
        mad_threshold = float(self.config.get("mad_threshold", 2.0))
        save_trigger_images = bool(self.config.get("save_trigger_images", True))
        artifact_root_dir = self.config.get("artifact_root_dir", None)
        overwrite_output_dir = bool(self.config.get("overwrite_output_dir", True))

        # clip_min / clip_max: None → auto-detect from data
        clip_min_cfg = self.config.get("clip_min", None)
        clip_max_cfg = self.config.get("clip_max", None)

        # ── Collect data (keep on CPU; batches moved to device inside loop) ─
        all_x: List[torch.Tensor] = []
        for batch in data_loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            elif isinstance(batch, dict):
                x = batch["x"]
            else:
                x = batch
            if not x.is_floating_point():
                x = x.float()
            all_x.append(x)

        if not all_x:
            raise ValueError("neural_cleanse: data_loader yielded no batches")

        x_pool = torch.cat(all_x, dim=0)

        # ── Resolve clip bounds ───────────────────────────────────────────
        # Auto-detect from the actual data range so the pattern optimisation
        # always operates in the correct domain (critical for ImageNet where
        # inputs are normalised and not in [0, 1]).
        if clip_min_cfg is None:
            clip_min = float(x_pool.min().item())
            logger.info("Neural Cleanse: clip_min auto-detected = %.4f", clip_min)
        else:
            clip_min = float(clip_min_cfg)

        if clip_max_cfg is None:
            clip_max = float(x_pool.max().item())
            logger.info("Neural Cleanse: clip_max auto-detected = %.4f", clip_max)
        else:
            clip_max = float(clip_max_cfg)

        # ── Auto-detect number of output classes ──────────────────────────
        if num_classes <= 0:
            with torch.no_grad():
                out = model(x_pool[:1].to(device))
            num_classes = int(out.shape[-1])
            logger.info("Neural Cleanse: auto-detected %d classes", num_classes)

        # ── Phase 1: reverse-engineer trigger for every label ─────────────
        logger.info(
            "Neural Cleanse: scanning %d labels | steps=%d  batch=%d  "
            "lr=%.2e  init_cost=%.2e  clip=[%.3f, %.3f]",
            num_classes,
            steps,
            batch_size,
            lr,
            init_cost,
            clip_min,
            clip_max,
        )

        masks: List[np.ndarray] = []
        patterns: List[np.ndarray] = []
        converged_flags: List[bool] = []
        idx_mapping: Dict[int, int] = {}

        run_dir: Optional[Path] = None
        if artifact_root_dir:
            run_name = (
                f"{model_info.attack_name}_{model_info.architecture}"
                f"_{model_info.dataset}".replace(" ", "_")
            )
            run_dir = Path(artifact_root_dir) / run_name / "neural_cleanse"
            if overwrite_output_dir and run_dir.exists():
                shutil.rmtree(run_dir)

        # Resolve trigger image save directory early so per-label saves work
        trigger_dir: Optional[Path] = None
        if artifact_root_dir and save_trigger_images:
            if run_dir is None:
                run_name = (
                    f"{model_info.attack_name}_{model_info.architecture}"
                    f"_{model_info.dataset}".replace(" ", "_")
                )
                run_dir = Path(artifact_root_dir) / run_name / "neural_cleanse"
            trigger_dir = run_dir / "triggers"
            trigger_dir.mkdir(parents=True, exist_ok=True)

        for target_label in range(num_classes):
            logger.info(
                "Neural Cleanse: optimising label %d / %d …",
                target_label,
                num_classes - 1,
            )

            mask_best, pattern_best, reg_best, converged = reverse_engineer_trigger(
                model=model,
                x_pool=x_pool,
                target_label=target_label,
                device=device,
                steps=steps,
                mini_batch=mini_batch,
                batch_size=batch_size,
                lr=lr,
                init_cost=init_cost,
                cost_multiplier=cost_multiplier,
                patience=patience,
                reset_cost_to_zero=reset_cost_to_zero,
                attack_succ_threshold=attack_succ_threshold,
                early_stop=early_stop,
                early_stop_threshold=early_stop_threshold,
                early_stop_patience=early_stop_patience,
                clip_min=clip_min,
                clip_max=clip_max,
                seed=target_label,  # reproducible per label
            )

            masks.append(mask_best)
            patterns.append(pattern_best)
            converged_flags.append(converged)
            idx_mapping[target_label] = len(masks) - 1

            logger.info(
                "  label %d: L1 norm = %.4f  converged = %s",
                target_label,
                float(np.sum(np.abs(mask_best))),
                converged,
            )

            if trigger_dir is not None:
                _save_trigger_images(
                    mask=mask_best,
                    pattern=pattern_best,
                    target_label=target_label,
                    save_dir=trigger_dir,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

        # ── Phase 2: MAD outlier detection ────────────────────────────────
        # L1 norm = sum|mask|  over the (H, W) single-channel mask array.
        # Matches original mad_outlier_detection.py exactly.
        l1_norms = [float(np.sum(np.abs(m))) for m in masks]

        mad_detected, anomaly_index, flagged_by_mad, median, mad = (
            mad_outlier_detection(l1_norms, idx_mapping, mad_threshold)
        )

        logger.info(
            "Neural Cleanse: MAD → %s  (anomaly index = %.4f  threshold = %.1f)",
            "flagged" if mad_detected else "clean",
            anomaly_index,
            mad_threshold,
        )

        # Final verdict: model is flagged when the MAD outlier test fires.
        is_backdoor = mad_detected
        flagged_labels: List[Tuple[int, float]] = list(flagged_by_mad)

        logger.info(
            "Neural Cleanse: final verdict → %s",
            "BACKDOOR DETECTED" if is_backdoor else "clean",
        )

        if flagged_labels:
            logger.info("Neural Cleanse: flagged labels → %s", flagged_labels)

        # ── L1 norm bar chart ─────────────────────────────────────────────
        if trigger_dir is not None:
            _save_l1_norm_plot(
                l1_norms=l1_norms,
                idx_mapping=idx_mapping,
                flagged_by_mad=flagged_by_mad,
                anomaly_index=anomaly_index,
                mad_threshold=mad_threshold,
                median=median,
                save_dir=trigger_dir,
            )
            logger.info("Neural Cleanse: trigger images saved to %s", trigger_dir)

        # ── Save artefacts ────────────────────────────────────────────────
        summary: dict = {
            "is_backdoor_detected": bool(is_backdoor),
            "anomaly_index": anomaly_index,
            "mad_threshold": mad_threshold,
            "median_l1_norm": median,
            "mad_l1_norm": mad,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "l1_norms": {str(y): float(l1_norms[idx_mapping[y]]) for y in idx_mapping},
            "converged": {
                str(y): bool(converged_flags[idx_mapping[y]]) for y in idx_mapping
            },
            "flagged_by_mad": [
                {"label": lbl, "l1_norm": norm} for lbl, norm in flagged_by_mad
            ],
            # Alias kept for downstream consumers (e.g. SHAP explainer) that
            # read the MAD-flagged labels under this key.
            "confirmed_labels": [
                {"label": lbl, "l1_norm": norm} for lbl, norm in flagged_labels
            ],
        }

        artifact_path: Optional[str] = None
        if artifact_root_dir:
            if run_dir is None:
                run_name = (
                    f"{model_info.attack_name}_{model_info.architecture}"
                    f"_{model_info.dataset}".replace(" ", "_")
                )
                run_dir = Path(artifact_root_dir) / run_name / "neural_cleanse"
            run_dir.mkdir(parents=True, exist_ok=True)
            report_path = run_dir / "summary.json"
            report_path.write_text(json.dumps(summary, indent=2, default=str))
            logger.info("Neural Cleanse: report saved to %s", report_path)
            artifact_path = str(report_path)

        # ── DetectionResult ───────────────────────────────────────────────
        # Confidence = normalized anomaly index, clipped to [0, 1].
        confidence = float(
            np.clip(anomaly_index / (2.0 * mad_threshold + 1e-8), 0.0, 1.0)
        )

        return DetectionResult(
            method_name=self.name,
            attack_name=model_info.attack_name,
            model_architecture=model_info.architecture,
            dataset=model_info.dataset,
            is_backdoor_detected=bool(is_backdoor),
            confidence_score=confidence,
            flagged_labels=[lbl for lbl, _ in flagged_labels],
            details={
                "anomaly_index": anomaly_index,
                "mad_threshold": mad_threshold,
                "median_l1_norm": median,
                "mad_l1_norm": mad,
                "clip_min": clip_min,
                "clip_max": clip_max,
                "l1_norms": {
                    str(y): float(l1_norms[idx_mapping[y]]) for y in idx_mapping
                },
                "converged": {
                    str(y): bool(converged_flags[idx_mapping[y]]) for y in idx_mapping
                },
                "flagged_by_mad": [
                    {"label": lbl, "l1_norm": norm} for lbl, norm in flagged_by_mad
                ],
                "confirmed_labels": [
                    {"label": lbl, "l1_norm": norm} for lbl, norm in flagged_labels
                ],
                "num_classes": num_classes,
            },
            artifacts={"summary": artifact_path or ""},
        )
