#!/usr/bin/env python3
"""
Build null_cache.npz from the tier-1 null models, evaluate their test accuracy,
and write a LaTeX configuration table.

Null models: config_000 – config_049 per dataset (50 × 3 = up to 150 models).
These models form the reference null distribution used by StandardTesting.py.

Usage:
    python build_null.py              # build cache + accuracy table
    python build_null.py --validate   # also run cross-dataset LOO validation plots
"""
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import iter_weight_params, layer_weights, layer_type, empirical_p, list_ckpts, load_checkpoint
from StandardTesting import TESTS, TENSOR_TESTS
import config

STATDET_DIR = Path(__file__).parent
CACHE_NPZ   = STATDET_DIR / "null_cache.npz"
N_TIER1     = 50
N_SAMPLES   = 4000
FS          = 22.3  # plot fontsize

DS_META = {
    "MNIST":        dict(in_channels=1, num_classes=10, input_size=28),
    "FashionMNIST": dict(in_channels=1, num_classes=10, input_size=28),
    "GTSRB":        dict(in_channels=3, num_classes=43, input_size=32),
}

TEST_DISPLAY = {
    "kurt":      "Kurtosis",
    "fft_peak":  "FFT Peak",
    "zero":      "Zero Fraction",
    "svd_spike": "SVD Spike",
}


# ── Model architecture (matches training code) ────────────────────────────────

class FlexCNN(nn.Module):
    def __init__(self, in_channels, conv_channels, fc_dims,
                 kernel_size=3, num_classes=10, input_size=28):
        super().__init__()
        padding = kernel_size // 2
        conv_layers, ch_in = [], in_channels
        for ch_out in conv_channels:
            conv_layers += [
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            ch_in = ch_out
        self.conv = nn.Sequential(*conv_layers)
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, in_channels, input_size, input_size)).numel()
        fc_layers, dim_in = [], flat_dim
        for dim_out in fc_dims:
            fc_layers += [nn.Linear(dim_in, dim_out), nn.ReLU()]
            dim_in = dim_out
        fc_layers.append(nn.Linear(dim_in, num_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


# ── Data loading ──────────────────────────────────────────────────────────────

def preload_dataset(ds_name):
    """Load N_SAMPLES test images into RAM — amortises the disk cost across all models."""
    print(f"  Preloading {ds_name}...", flush=True)
    if ds_name == "GTSRB":
        tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        ds = datasets.GTSRB(root=config.DATA_ROOT, split="test", download=False, transform=tf)
    elif ds_name == "FashionMNIST":
        ds = datasets.FashionMNIST(root=config.DATA_ROOT, train=False, download=False,
                                   transform=transforms.ToTensor())
    else:
        ds = datasets.MNIST(root=config.DATA_ROOT, train=False, download=False,
                            transform=transforms.ToTensor())
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(42))[:N_SAMPLES]
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=512, num_workers=0)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def eval_acc(state_dict, cfg, X, Y):
    meta = DS_META[cfg["dataset"]]
    model = FlexCNN(
        in_channels=meta["in_channels"],
        conv_channels=cfg["conv_channels"],
        fc_dims=cfg["fc_dims"],
        kernel_size=cfg["kernel_size"],
        num_classes=meta["num_classes"],
        input_size=meta["input_size"],
    )
    model.load_state_dict(state_dict)
    model.eval()
    ok = 0
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb, yb = X[i:i + 512], Y[i:i + 512]
            ok += (model(xb).argmax(1) == yb).sum().item()
    return ok / len(Y)


# ── Null distribution ─────────────────────────────────────────────────────────

def build_pools(null_models, verbose=True):
    """Raw per-layer test scores for all null models, pooled by layer type."""
    pools = {t: {"conv": [], "fc": []} for t in TESTS}
    for m in null_models:
        for _, tensor in iter_weight_params(m):
            lt = layer_type(tensor)
            w  = layer_weights(tensor)
            for t, fn in TESTS.items():
                pools[t][lt].append(fn(tensor) if t in TENSOR_TESTS else fn(w))
    if verbose:
        for t in TESTS:
            print(f"  {t}: conv={len(pools[t]['conv'])}  fc={len(pools[t]['fc'])}")
    return pools


def build_null_fisher_stats(null_models):
    """
    Compute the null distribution of Fisher S statistics via leave-one-out.

    Returns:
        fisher_stats  — dict[test -> list of per-model Fisher S values]
        layer_stats   — dict[test -> dict[layer_type -> list of raw scores]]
    """
    null_layer_stats = build_pools(null_models)
    fisher_stats = {t: [] for t in TESTS}

    for m in null_models:
        layer_pvals = {t: [] for t in TESTS}
        for _, tensor in iter_weight_params(m):
            lt  = layer_type(tensor)
            w   = layer_weights(tensor)
            for t, fn in TESTS.items():
                score = fn(tensor) if t in TENSOR_TESTS else fn(w)
                pool  = np.asarray(null_layer_stats[t][lt], dtype=float)
                loo   = np.delete(pool, int(np.argmin(np.abs(pool - score))))
                layer_pvals[t].append(empirical_p(score, loo))

        for t in TESTS:
            ps = np.clip(np.array(layer_pvals[t], dtype=float), 1e-12, 1.0)
            fisher_stats[t].append(float(-2.0 * np.sum(np.log(ps))))

    return fisher_stats, null_layer_stats


# ── Output: LaTeX table ───────────────────────────────────────────────────────

def generate_latex_table(configs):
    """Write tier-1 null model configuration + accuracy table to null_model_configs_table.tex."""
    by_ds = {ds: {} for ds in config.NULL_DIRS}
    for c in configs:
        by_ds[c["dataset"]][c["config_id"] % N_TIER1] = c

    rows = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Tier-1 null model configurations and test accuracies "
        r"(50 base architectures $\times$ 3 datasets = up to 150 models). "
        r"FashionMNIST and GTSRB always use Adam / $10^{-3}$.}",
        r"\label{tab:null_configs_tier1}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{rlcrlccrrr}",
        r"\toprule",
        r"ID & Conv channels & $k$ & FC dims & Opt & LR & Ep"
        r" & Acc$_{\text{MNIST}}$ & Acc$_{\text{FMNIST}}$ & Acc$_{\text{GTSRB}}$ \\",
        r"\midrule",
    ]
    for i in range(N_TIER1):
        mc  = by_ds["MNIST"].get(i)
        fc  = by_ds["FashionMNIST"].get(i)
        gc  = by_ds["GTSRB"].get(i)
        ref = mc or fc or gc
        if not ref:
            continue
        conv = str(ref["conv_channels"]).replace(" ", "")
        fcd  = str(ref["fc_dims"]).replace(" ", "")
        lr   = ref["lr"]
        lr_s = (r"$5\!\times\!10^{-4}$" if lr == 5e-4
                else f"$10^{{{int(round(np.log10(lr)))}}}$")
        ma = f"{mc['final_acc']:.3f}" if mc else "--"
        fa = f"{fc['final_acc']:.3f}" if fc else "--"
        ga = f"{gc['final_acc']:.3f}" if gc else "--"
        rows.append(
            f"{i} & \\texttt{{{conv}}} & {ref['kernel_size']} &"
            f" \\texttt{{{fcd}}} & {ref['optimizer']} & {lr_s} & {ref['epochs']}"
            f" & {ma} & {fa} & {ga} \\\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    out = STATDET_DIR / "null_model_configs_table.tex"
    out.write_text("\n".join(rows) + "\n")
    return out


# ── Output: LOO validation plots ──────────────────────────────────────────────

def plot_dataset_loo(out_dir):
    """
    Cross-dataset leave-one-out validation: build null from 2 datasets, test on
    the held-out 3rd.  Produces one Q-Q plot per test statistic.
    """
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    colors   = ["steelblue", "tomato", "seagreen"]
    ds_names = list(config.NULL_DIRS.keys())

    print("  Loading all null models for LOO validation...")
    per_dataset = [
        [load_checkpoint(f) for f in list_ckpts(str(ds_dir))]
        for ds_dir in config.NULL_DIRS.values()
    ]

    for t in TESTS:
        display = TEST_DISPLAY.get(t, t)
        _, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.5, label="Uniform")

        for i, label in enumerate(ds_names):
            pool_models = [m for j, ms in enumerate(per_dataset) for m in ms if j != i]
            held_models = per_dataset[i]
            print(f"  LOO {label}/{t}: pool={len(pool_models)}  held={len(held_models)}")

            null_fisher, null_layer = build_null_fisher_stats(pool_models)

            held_pvals = []
            for m in held_models:
                layer_pvals = []
                for _, tensor in iter_weight_params(m):
                    lt    = layer_type(tensor)
                    w     = layer_weights(tensor)
                    score = TESTS[t](tensor) if t in TENSOR_TESTS else TESTS[t](w)
                    layer_pvals.append(empirical_p(score, np.array(null_layer[t][lt])))
                ps = np.clip(np.array(layer_pvals, dtype=float), 1e-12, 1.0)
                S  = float(-2.0 * np.sum(np.log(ps)))
                held_pvals.append(empirical_p(S, null_fisher[t]))

            pvals = np.sort(held_pvals)
            ax.plot(pvals, np.arange(1, len(pvals) + 1) / len(pvals),
                    color=colors[i], linewidth=2, label=label)

        ax.set_title(display, fontsize=FS)
        ax.set_xlabel("Theoretical", fontsize=FS)
        ax.set_ylabel("Empirical", fontsize=FS)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=FS)
        ax.legend(fontsize=FS, loc="upper left")
        plt.tight_layout()
        path = out_dir / f"dataset_loo_{t}.png"
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--validate", action="store_true",
                        help="also run cross-dataset LOO validation plots (slow)")
    args = parser.parse_args()

    # Load tier-1 null models
    print("Loading tier-1 null models...")
    all_models = []  # list of (state_dict, cfg)
    for ds_name, ds_dir in config.NULL_DIRS.items():
        loaded = 0
        for i in range(N_TIER1):
            path = ds_dir / f"config_{i:03d}" / "model.pth"
            if not path.exists():
                print(f"  WARNING: missing {ds_name} config_{i:03d}")
                continue
            raw = torch.load(str(path), map_location="cpu", weights_only=False)
            all_models.append((raw["state_dict"], raw["config"]))
            loaded += 1
        print(f"  {ds_name}: {loaded} models")
    print(f"  Total: {len(all_models)} null models\n")

    # Build null cache
    print("Building null distributions (LOO Fisher S)...")
    null_fisher_stats, null_layer_stats = build_null_fisher_stats([sd for sd, _ in all_models])
    save_data = {}
    for t in TESTS:
        save_data[f"{t}_conv"]     = null_layer_stats[t]["conv"]
        save_data[f"{t}_fc"]       = null_layer_stats[t]["fc"]
        save_data[f"{t}_fisher_S"] = null_fisher_stats[t]
    np.savez(CACHE_NPZ, **save_data)
    print(f"  Saved: {CACHE_NPZ}\n")

    # Evaluate test accuracy (preload datasets into RAM first)
    print("Preloading test data into RAM...")
    test_data = {ds: preload_dataset(ds) for ds in config.NULL_DIRS}

    print("\nEvaluating test accuracy...")
    configs_with_acc = []
    for sd, cfg in all_models:
        X, Y  = test_data[cfg["dataset"]]
        acc   = eval_acc(sd, cfg, X, Y)
        entry = {**cfg, "final_acc": round(float(acc), 4)}
        configs_with_acc.append(entry)
        print(f"  {cfg['dataset']:13s}  config_id={cfg['config_id']:3d}  acc={acc:.4f}")

    json_path = STATDET_DIR / "null_model_configs.json"
    json_path.write_text(json.dumps(configs_with_acc, indent=2))
    print(f"\n  Saved: {json_path}")

    print("\nAccuracy summary:")
    for ds in config.NULL_DIRS:
        accs = [c["final_acc"] for c in configs_with_acc if c["dataset"] == ds]
        if accs:
            print(f"  {ds:13s}: n={len(accs)}  mean={np.mean(accs):.3f}"
                  f"  range=[{min(accs):.3f}, {max(accs):.3f}]")

    # LaTeX table
    table_path = generate_latex_table(configs_with_acc)
    print(f"\n  Saved: {table_path}")

    # Optional: LOO cross-dataset validation
    if args.validate:
        out_dir = STATDET_DIR / "out_null_validation"
        out_dir.mkdir(exist_ok=True)
        print("\nRunning cross-dataset LOO validation...")
        plot_dataset_loo(out_dir)


if __name__ == "__main__":
    main()
