"""Triggered-sample gallery: visual sanity check for every adapter.

Loads each attack/model entry from a Detection config, applies
get_triggered_sample() to one clean test image, and generates a grid PDF
showing [clean | triggered | trigger] for every adapter.

The third column shows the trigger in its natural color: for patch-based
attacks, the trigger is applied to a zero tensor so white patches render
white instead of producing arbitrary residual colors from |triggered - clean|;
for steganographic attacks (hiding_needles, boone_bane) the distributed
perturbation isn't visualisable on a blank canvas, so |triggered - clean|
is used instead.

Adapters that return None are noted with a text label instead.

Usage:
    python -m Detection.tests.test_triggered_samples
    python -m Detection.tests.test_triggered_samples --config Detection/configs/default.yaml
    python -m Detection.tests.test_triggered_samples --attack badnets_mnist
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml

import Detection.adapters  # noqa: F401  (trigger adapter registration)
from Detection.core.data_provider import DataProvider
from Detection.core.registry import get_adapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triggered sample gallery")
    parser.add_argument("--config", type=str, default="Detection/configs/default.yaml")
    parser.add_argument("--attack", type=str, default=None, help="Only run this attack name")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./Detection/results",
        help="Directory for the gallery PDF",
    )
    return parser.parse_args()


def _get_one_sample(
    data_loader,
    exclude_class: Optional[int] = None,
    desired_class: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """Return the first (image, label) from the dataloader.

    If *exclude_class* is set, skip samples of that class so the returned
    sample belongs to a different class.

    If *desired_class* is set, only return a sample of that class (skipping others).
    """
    for batch in data_loader:
        if isinstance(batch, dict):
            images = batch.get("x", batch.get("image"))
            labels = batch.get("y", batch.get("label"))
        elif isinstance(batch, (tuple, list)):
            images = batch[0]
            labels = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            labels = None
        if images is None or images.shape[0] == 0:
            continue
        for i in range(images.shape[0]):
            lbl = int(labels[i]) if labels is not None else None
            if exclude_class is not None and lbl == exclude_class:
                continue
            if desired_class is not None and lbl != desired_class:
                continue
            return images[i].float(), lbl
    return None, None


# Steganographic adapters distribute the trigger across the whole image at
# imperceptible magnitude. For those, the "trigger applied to a blank canvas"
# panel is uninformative; show |triggered - clean| instead so the distributed
# perturbation is visible after the display normalisation.
_STEGO_ADAPTERS = {"hiding_needles", "boone_bane"}


DATASET_CLASS_NAMES = {
    "mnist": [str(i) for i in range(10)],
    "fmnist": [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ],
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
}

# (label, clean, triggered_or_None, trigger_panel_or_None, is_stego_panel,
#  true_label, clean_pred, triggered_pred, expected_target, dataset)
Row = Tuple[
    str,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    bool,
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    str,
]


@torch.no_grad()
def _predict(model: torch.nn.Module, sample: torch.Tensor, device: torch.device) -> int:
    """Run a single sample through the model and return the predicted class."""
    x = sample.unsqueeze(0).to(device)
    logits = model(x)
    return int(logits.argmax(dim=1).item())


def _class_name(dataset: str, class_idx: int) -> str:
    """Return a human-readable class name, or just the index if unknown."""
    names = DATASET_CLASS_NAMES.get(dataset)
    if names and 0 <= class_idx < len(names):
        return f"{class_idx} ({names[class_idx]})"
    return str(class_idx)


def _expected_target(
    adapter_name: str,
    model_cfg: dict,
    clean_pred: Optional[int],
    true_label: Optional[int],
    num_classes: int = 10,
) -> Tuple[Optional[int], str]:
    """Return (expected_target_class, description) for the attack.

    Returns (None, reason) for untargeted attacks.
    """
    if adapter_name == "handcrafted":
        return 0, "y_t=0 default"
    if adapter_name == "dfba":
        return 0, "yt=0 default"
    if adapter_name == "badnets":
        return 1, "trigger_label=1 default"
    if adapter_name == "trojannet":
        t = int(model_cfg.get("target_class", 0))
        return t, f"target_class={t}"
    if adapter_name == "model_editing_clip":
        return 3, "target=cat (Abyssinian)"
    if adapter_name == "hiding_needles":
        return None, "untargeted attack"
    if adapter_name == "boone_bane":
        return None, "untargeted attack"
    if adapter_name == "foobar":
        # Target encoded in the solution filename: solutions_faulted_{target}_...
        sol = model_cfg.get("solution_path", "")
        import re
        m = re.search(r'solutions_faulted_(?:conv_)?(\d+)_', sol)
        if m:
            t = int(m.group(1))
            return t, f"target={t} (from solution file)"
        return None, "unknown (no solution_path)"
    if adapter_name == "arch_backdoors":
        return None, "untargeted attack"
    return None, "unknown adapter"


def _to_display(tensor: np.ndarray) -> np.ndarray:
    """Convert (C,H,W) float to (H,W,C) in [0,1] for imshow."""
    img = np.transpose(tensor, (1, 2, 0))
    lo, hi = img.min(), img.max()
    return (img - lo) / max(hi - lo, 1e-8)


def _slugify(label: str) -> str:
    """Turn a row label like 'badnets\\n(default/cifar10)' into a safe filename stem."""
    return (
        label.replace("\n", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _triggered_xlabel(
    trig_pred_str: str,
    expected_target: Optional[int],
    dataset_key: str,
) -> Tuple[str, str, str]:
    """Build the (text, color, weight) for the triggered image's xlabel.

    Mirrors the clean example's two-line layout: pred on the first line,
    target/untargeted info on the second.
    """
    if expected_target is not None:
        expected_str = _class_name(dataset_key, expected_target)
        text = f"Pred: {trig_pred_str}\nTarget: {expected_str}"
        return text, "black", "normal"
    text = f"Pred: {trig_pred_str}\nuntargeted"
    return text, "black", "normal"


def _render_row(
    axes_row,
    label: str,
    clean_np: np.ndarray,
    triggered_np: Optional[np.ndarray],
    trigger_panel_np: Optional[np.ndarray],
    is_stego_panel: bool,
    true_label: Optional[int],
    clean_pred: Optional[int],
    triggered_pred: Optional[int],
    expected_target: Optional[int],
    dataset_key: str,
) -> None:
    """Draw clean / triggered / trigger-only onto a 3-axis row."""
    clean_disp = _to_display(clean_np)
    is_gray = clean_np.shape[0] == 1

    if is_gray:
        axes_row[0].imshow(clean_disp[:, :, 0], cmap="gray")
    else:
        axes_row[0].imshow(clean_disp)
    axes_row[0].set_ylabel(label, fontsize=8, rotation=0, labelpad=100, va="center")

    if clean_pred is not None:
        true_str = _class_name(dataset_key, true_label) if true_label is not None else "?"
        pred_str = _class_name(dataset_key, clean_pred)
        axes_row[0].set_xlabel(
            f"True: {true_str}\nPred: {pred_str}",
            fontsize=7, color="black",
        )

    if triggered_np is not None:
        trig_disp = _to_display(triggered_np)

        if is_gray:
            axes_row[1].imshow(trig_disp[:, :, 0], cmap="gray")
        else:
            axes_row[1].imshow(trig_disp)

        if trigger_panel_np is not None:
            trigger_panel_disp = _to_display(trigger_panel_np)
            if is_gray:
                axes_row[2].imshow(trigger_panel_disp[:, :, 0], cmap="gray")
            else:
                axes_row[2].imshow(trigger_panel_disp)
        else:
            axes_row[2].text(
                0.5, 0.5, "N/A",
                ha="center", va="center", fontsize=10,
                transform=axes_row[2].transAxes, color="gray",
            )

        # Two-line xlabel keeps the trigger panel's height in sync with the
        # labelled panels. Stego rows annotate that the diff is auto-scaled
        # by the display normalisation (the raw perturbation is imperceptible).
        if is_stego_panel:
            axes_row[2].set_xlabel("Difference\n(magnified)", fontsize=7)
        else:
            axes_row[2].set_xlabel(" \n ", fontsize=7)

        if triggered_pred is not None:
            trig_pred_str = _class_name(dataset_key, triggered_pred)
            text, color, weight = _triggered_xlabel(
                trig_pred_str, expected_target, dataset_key,
            )
            axes_row[1].set_xlabel(text, fontsize=7, color=color, fontweight=weight)
    else:
        axes_row[1].text(
            0.5, 0.5, "None\n(not supported)",
            ha="center", va="center", fontsize=10,
            transform=axes_row[1].transAxes, color="gray",
        )
        axes_row[2].text(
            0.5, 0.5, "N/A",
            ha="center", va="center", fontsize=10,
            transform=axes_row[2].transAxes, color="gray",
        )

    for col in range(3):
        axes_row[col].set_xticks([])
        axes_row[col].set_yticks([])


def _save_triplet_images(
    out_dir: "Path",
    clean_np: np.ndarray,
    triggered_np: Optional[np.ndarray],
    trigger_panel_np: Optional[np.ndarray],
    is_stego_panel: bool,
    true_label: Optional[int],
    clean_pred: Optional[int],
    triggered_pred: Optional[int],
    expected_target: Optional[int],
    dataset_key: str,
) -> None:
    """Save clean, triggered, and trigger-only as separate PNGs in out_dir."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    is_gray = clean_np.shape[0] == 1
    clean_disp = _to_display(clean_np)

    # Clean
    fig, ax = plt.subplots(1, 1, figsize=(4, 4.5))
    if is_gray:
        ax.imshow(clean_disp[:, :, 0], cmap="gray")
    else:
        ax.imshow(clean_disp)
    ax.set_title("Clean", fontsize=12, fontweight="bold")
    if clean_pred is not None:
        true_str = _class_name(dataset_key, true_label) if true_label is not None else "?"
        pred_str = _class_name(dataset_key, clean_pred)
        ax.set_xlabel(f"True: {true_str}\nPred: {pred_str}", fontsize=10, color="black")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(out_dir / "clean.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if triggered_np is None:
        return

    # Triggered
    trig_disp = _to_display(triggered_np)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4.5))
    if is_gray:
        ax.imshow(trig_disp[:, :, 0], cmap="gray")
    else:
        ax.imshow(trig_disp)
    ax.set_title("Triggered", fontsize=12, fontweight="bold")
    if triggered_pred is not None:
        trig_pred_str = _class_name(dataset_key, triggered_pred)
        text, color, weight = _triggered_xlabel(
            trig_pred_str, expected_target, dataset_key,
        )
        ax.set_xlabel(text, fontsize=10, color=color, fontweight=weight)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(out_dir / "triggered.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Trigger panel — trigger-on-blank for patch attacks, |triggered - clean|
    # for steganographic attacks (decided upstream by the caller).
    if trigger_panel_np is None:
        return

    trigger_panel_disp = _to_display(trigger_panel_np)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4.5))
    if is_gray:
        ax.imshow(trigger_panel_disp[:, :, 0], cmap="gray")
    else:
        ax.imshow(trigger_panel_disp)
    panel_title = "Difference (magnified)" if is_stego_panel else "Trigger"
    ax.set_title(panel_title, fontsize=12, fontweight="bold")
    ax.set_xlabel(" \n ", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(out_dir / "trigger.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    elif config.get("global", {}).get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["global"]["device"])

    seed = config.get("global", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_provider = DataProvider(config.get("data", {}))

    attacks = config.get("attacks", [])
    if args.attack:
        attacks = [a for a in attacks if a["name"] == args.attack]

    rows: List[Row] = []

    for attack_cfg in attacks:
        adapter_name = attack_cfg["adapter"]
        AdapterClass = get_adapter(adapter_name)

        for model_cfg in attack_cfg.get("models", []):
            model_type = model_cfg.get("model_type", "default")
            dataset = model_cfg["dataset"]
            checkpoint = model_cfg["backdoor_checkpoint"]
            label = f"{attack_cfg['name']}\n({model_type}/{dataset})"

            logger.info("Processing %s/%s/%s", attack_cfg["name"], model_type, dataset)

            try:
                adapter_kwargs = {
                    k: v for k, v in model_cfg.items()
                    if k != "backdoor_checkpoint"
                }
                adapter = AdapterClass(**adapter_kwargs)
                model = adapter.load_model(checkpoint, device=device)
                model_info = adapter.get_model_info()

                custom_transform = model_info.extra.get("custom_transform") if model_info.extra else None
                data_loader = data_provider.get_test_loader(dataset, custom_transform=custom_transform)

                # Pick a sample from a different class than the target so the
                # backdoor has something to flip.
                exclude_class = None
                desired_class = None
                if adapter_name == "trojannet":
                    exclude_class = int(model_cfg.get("target_class", 0))
                    desired_class = 143
                elif adapter_name == "model_editing_clip":
                    exclude_class = 3  # target is always cat (Abyssinian)

                is_stego = adapter_name in _STEGO_ADAPTERS
                clean_sample, true_label = _get_one_sample(data_loader, exclude_class=exclude_class, desired_class=desired_class)
                if clean_sample is None:
                    logger.warning("  No samples available for %s", dataset)
                    rows.append((label, np.zeros((3, 32, 32)), None, None, is_stego, None, None, None, None, dataset))
                    continue

                clean_np = clean_sample.cpu().numpy()

                # Predict on clean sample
                clean_pred = _predict(model, clean_sample, device)
                true_label_str = _class_name(dataset, true_label) if true_label is not None else "?"
                logger.info("  True label: %s | Clean prediction: %s",
                            true_label_str, _class_name(dataset, clean_pred))

                trigger_kwargs = {}
                if true_label is not None:
                    trigger_kwargs["target_label"] = true_label
                triggered = adapter.get_triggered_sample(clean_sample, **trigger_kwargs)
                if triggered is not None:
                    triggered_np = triggered.cpu().numpy()
                    # Shape/finiteness assertions
                    assert triggered_np.shape == clean_np.shape, (
                        f"Shape mismatch: {triggered_np.shape} vs {clean_np.shape}"
                    )
                    assert np.all(np.isfinite(triggered_np)), "Non-finite values in triggered sample"

                    # Third panel: for patch-based attacks, apply the trigger
                    # to a blank canvas so the trigger shows in its natural
                    # color. For steganographic attacks (perturbation spread
                    # across the whole image at imperceptible magnitude),
                    # show |triggered - clean| instead — the distributed
                    # signal is meaningful, but a blank canvas isn't.
                    trigger_panel_np: Optional[np.ndarray] = None
                    if is_stego:
                        trigger_panel_np = np.abs(triggered_np - clean_np)
                    else:
                        try:
                            blank = torch.zeros_like(clean_sample)
                            trigger_only = adapter.get_triggered_sample(blank, **trigger_kwargs)
                            if trigger_only is not None:
                                trigger_panel_np = trigger_only.cpu().numpy()
                        except Exception as exc:
                            logger.warning("  trigger-on-blank failed: %s", exc)

                    # Predict on triggered sample (some adapters need
                    # non-standard inference, e.g. foobar fault simulation)
                    triggered_pred = adapter.predict_triggered(model, triggered, device)

                    expected, expected_reason = _expected_target(
                        adapter_name, model_cfg, clean_pred, true_label,
                    )
                    if expected is not None:
                        hit = triggered_pred == expected
                        logger.info(
                            "  Triggered prediction: %s | Expected target: %s (%s) | Hit: %s",
                            _class_name(dataset, triggered_pred),
                            _class_name(dataset, expected),
                            expected_reason,
                            hit,
                        )
                    else:
                        logger.info(
                            "  Triggered prediction: %s | %s | Backdoor flipped: %s",
                            _class_name(dataset, triggered_pred),
                            expected_reason,
                            clean_pred != triggered_pred,
                        )

                    rows.append((label, clean_np, triggered_np, trigger_panel_np, is_stego, true_label, clean_pred, triggered_pred, expected, dataset))
                    logger.info("  -> triggered sample OK (shape=%s)", triggered_np.shape)
                else:
                    expected, _ = _expected_target(adapter_name, model_cfg, clean_pred, true_label)
                    rows.append((label, clean_np, None, None, is_stego, true_label, clean_pred, None, expected, dataset))
                    logger.info("  -> adapter returned None (no trigger)")

            except Exception as exc:
                logger.warning("  -> failed: %s", exc)
                rows.append((label, np.zeros((3, 32, 32)), None, None, False, None, None, None, None, dataset))

    if not rows:
        logger.warning("No rows to display.")
        return

    # Build gallery PDF
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_rows = len(rows)
        row_height = 4
        fig, axes = plt.subplots(
            n_rows, 3,
            figsize=(12, row_height * n_rows),
            gridspec_kw={"hspace": 0.5, "wspace": 0.3},
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        col_titles = ["Clean", "Triggered", "Trigger"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=12, fontweight="bold")

        for i, row in enumerate(rows):
            _render_row(axes[i], *row)

        fig.suptitle("Triggered Samples Gallery", fontsize=14, fontweight="bold")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Per-attack PDFs (one row each)
        per_attack_dir = output_dir / "trigger_gallery"
        per_attack_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            label = row[0]
            fig_row, axes_row = plt.subplots(
                1, 3,
                figsize=(12, 4),
                gridspec_kw={"wspace": 0.3},
            )
            for col, title in enumerate(col_titles):
                axes_row[col].set_title(title, fontsize=12, fontweight="bold")
            _render_row(axes_row, *row)
            row_path = per_attack_dir / f"{_slugify(label)}.pdf"
            fig_row.savefig(row_path, dpi=150, format="pdf", bbox_inches="tight")
            plt.close(fig_row)
            logger.info("Saved per-attack gallery to %s", row_path)

        # Per-attack triplet PNGs (clean / triggered / diff as separate files)
        triplets_root = output_dir / "triggered-samples"
        triplets_root.mkdir(parents=True, exist_ok=True)
        for row in rows:
            label = row[0]
            attack_dir = triplets_root / _slugify(label)
            _save_triplet_images(attack_dir, *row[1:])
            logger.info("Saved triplet images to %s", attack_dir)

        save_path = output_dir / "triggered_samples_gallery.pdf"
        fig.savefig(save_path, dpi=150, format="pdf", bbox_inches="tight")
        plt.close(fig)
        logger.info("Gallery saved to %s", save_path)

    except Exception as exc:
        logger.error("Failed to generate gallery: %s", exc)


if __name__ == "__main__":
    main()
