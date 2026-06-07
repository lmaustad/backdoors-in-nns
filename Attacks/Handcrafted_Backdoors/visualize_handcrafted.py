#!/usr/bin/env python3
"""
Visualize the handcrafted backdoor: show clean vs triggered images and
compare predictions from the clean model and the backdoored model.

Layout (one column per example):
  Row 0 : original image
            title: "True: X  |  clean→Y  bd→Z"
  Row 1 : triggered image (3×3 checkerboard, bottom-right)
            title: "Triggered  |  clean→Y  bd→Z"

Run from Attacks/Handcrafted_Backdoors/:
    python visualize_handcrafted.py --seed 0 --n_examples 8

Output:
    visualizations/seed_0_examples.png
"""

import argparse
import os
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from train_model import MNISTCNN
from utils import apply_trigger

MNIST_CLASSES = [str(i) for i in range(10)]


def load_model(path, device):
    """Load an MNISTCNN from a state-dict checkpoint."""
    model = MNISTCNN(fc1_dim=128).to(device)
    state_dict = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict(model, x):
    """Return predicted class index for a batch x."""
    logits = model(x)
    return logits.argmax(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",       type=int, default=0,   help="Which seed's models to load")
    ap.add_argument("--n_examples", type=int, default=8,   help="Number of test images to show")
    ap.add_argument("--data_root",  default="./data",      help="Path to MNIST data root")
    ap.add_argument("--clean_dir",  default="ckpt/clean_models")
    ap.add_argument("--atk_dir",    default="ckpt/attacked_models")
    ap.add_argument("--y_t",        type=int, default=0,   help="Backdoor target class")
    ap.add_argument("--patch",      type=int, default=3,   help="Trigger patch size")
    ap.add_argument("--loc",        default="br",          help="Trigger location")
    ap.add_argument("--out_dir",    default="visualizations")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Paths ──────────────────────────────────────────────────────────────────
    clean_path = os.path.join(args.clean_dir, f"seed_{args.seed}",
                              "handcrafted_mnist_base_model.pth")
    atk_path   = os.path.join(args.atk_dir,  f"seed_{args.seed}",
                              "handcrafted_mnist_attacked_model.pth")

    if not os.path.exists(clean_path):
        print(f"ERROR: clean model not found at {clean_path}")
        sys.exit(1)
    if not os.path.exists(atk_path):
        print(f"ERROR: attacked model not found at {atk_path}")
        print("  Generate it first: python attack_seeded.py --seed", args.seed)
        sys.exit(1)

    # ── Load models ────────────────────────────────────────────────────────────
    clean_model = load_model(clean_path, device)
    atk_model   = load_model(atk_path,   device)
    print(f"Loaded models for seed {args.seed}")

    # ── Load MNIST test set ────────────────────────────────────────────────────
    test_ds = torchvision.datasets.MNIST(
        root=args.data_root, train=False, download=False,
        transform=T.ToTensor()
    )

    # pick examples that are NOT the target class (so we can see the flip)
    non_target_indices = [i for i, (_, y) in enumerate(test_ds) if y != args.y_t]
    random.seed(args.seed)
    chosen = random.sample(non_target_indices, args.n_examples)

    images = []
    labels = []
    for idx in chosen:
        img, label = test_ds[idx]
        images.append(img)
        labels.append(label)

    # stack into batch
    x_batch = torch.stack(images).to(device)   # (N, 1, 28, 28)
    y_batch = torch.tensor(labels)

    # apply trigger
    x_triggered = apply_trigger(x_batch, patch=args.patch, loc=args.loc)

    # ── Predictions ────────────────────────────────────────────────────────────
    # clean model on clean images
    clean_pred_clean = predict(clean_model, x_batch).cpu()
    # clean model on triggered images (should NOT flip to y_t)
    clean_pred_trig  = predict(clean_model, x_triggered).cpu()
    # attacked model on clean images (should still be correct)
    atk_pred_clean   = predict(atk_model,   x_batch).cpu()
    # attacked model on triggered images (should flip to y_t)
    atk_pred_trig    = predict(atk_model,   x_triggered).cpu()

    # ── Plot ───────────────────────────────────────────────────────────────────
    n = args.n_examples
    fig = plt.figure(figsize=(2.2 * n, 5.5))
    gs  = gridspec.GridSpec(2, n, figure=fig, hspace=0.5, wspace=0.15)

    for col in range(n):
        true_label  = y_batch[col].item()

        # --- row 0: original image ---
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(images[col].squeeze(), cmap="gray", vmin=0, vmax=1)
        ax0.axis("off")

        c_c = clean_pred_clean[col].item()
        a_c = atk_pred_clean[col].item()

        # green if correct, red if wrong
        c_color = "green" if c_c == true_label else "red"
        a_color = "green" if a_c == true_label else "red"

        ax0.set_title(
            f"True: {true_label}\n"
            f"clean→{c_c}  bd→{a_c}",
            fontsize=7,
            color="black",
        )
        # color the pred numbers individually via two text calls on the axes
        # (simpler: just include in title with bracket notation)

        # --- row 1: triggered image ---
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(x_triggered[col].squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
        ax1.axis("off")

        c_t = clean_pred_trig[col].item()
        a_t = atk_pred_trig[col].item()

        # backdoored model should predict y_t; mark as success or fail
        bd_flipped = (a_t == args.y_t)
        flag = "✓" if bd_flipped else "✗"

        ax1.set_title(
            f"[triggered] {flag}\n"
            f"clean→{c_t}  bd→{a_t}",
            fontsize=7,
            color="crimson" if bd_flipped else "gray",
        )

    # row labels on the left
    fig.text(0.01, 0.75, "Original",  va="center", rotation="vertical", fontsize=9)
    fig.text(0.01, 0.28, "Triggered", va="center", rotation="vertical", fontsize=9)

    # summary stats
    n_flipped = int((atk_pred_trig == args.y_t).sum())
    n_clean_ok = int((atk_pred_clean == y_batch).sum())
    fig.suptitle(
        f"Handcrafted backdoor  |  seed {args.seed}  |  target class: {args.y_t}\n"
        f"Backdoored model: {n_clean_ok}/{n} clean correct,  "
        f"{n_flipped}/{n} triggered→target",
        fontsize=9,
        y=1.01,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"seed_{args.seed}_examples.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
