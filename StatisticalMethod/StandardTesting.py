#!/usr/bin/env python3
"""
Run statistical backdoor detection on a model family.

Tests each model's weights against the pre-built null distribution in
null_cache.npz using Fisher's combined probability method.

Usage:
    python StandardTesting.py --model dfba
    python StandardTesting.py --model handcrafted --n 20
"""
import sys
import json
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_checkpoint, layer_weights_matrix, iter_weight_params, layer_type, empirical_p, layer_weights
import config

CACHE = Path(__file__).parent / "null_cache.npz"

# ── Attack model directories (from config) ────────────────────────────────────

DFBA_MODELS    = config.DFBA_MODELS
DFBA_FLEX_DIR  = config.DFBA_FLEX_DIR
HB_FLEX_DIR    = config.HB_FLEX_DIR
HB_CLEAN       = config.HB_CLEAN
HB_ATTACKED    = config.HB_ATTACKED
DFBA_CLEAN_NAME    = "cnn_mnist_base_model.pth"
DFBA_ATTACKED_NAME = "cnn_mnist_attacked_model.pth"
HB_CLEAN_NAME      = "handcrafted_mnist_base_model.pth"
HB_ATTACKED_NAME   = "handcrafted_mnist_attacked_model.pth"

FAMILY_DISPLAY = {
    "dfba":             "DFBA",
    "dfba_flex":        "DFBA",
    "handcrafted":      "HC",
    "handcrafted_flex": "HC",
}

FAMILY_FILESTEM = {
    "dfba":             "dfba",
    "dfba_flex":        "dfba",
    "handcrafted":      "hc",
    "handcrafted_flex": "hc",
}

TEST_DISPLAY = {
    "kurt":      "Kurtosis",
    "fft_peak":  "FFT Peak",
    "zero":      "Zero Fraction",
    "svd_spike": "SVD Spike",
}

ALPHA    = 0.05
N_MODELS = 40
FS       = 22.3  # plot fontsize


# ── Test statistics ───────────────────────────────────────────────────────────

def score_kurtosis(w):
    return float(kurtosis(w, fisher=True))


def zero_percentage(w):
    return float(sum(abs(v) <= 0.0001 for v in w)) / len(w)


def score_svd_spike(tensor, eps=1e-12):
    mat = layer_weights_matrix(tensor)
    s   = np.linalg.svd(mat, compute_uv=False)
    return float(s[0] / (np.sum(s) + eps))


def score_fft_peak(w, dc_cut=0.02):
    if len(w) < 50:
        return 0.0
    signal  = np.sort(w).astype(np.float64)
    signal -= signal.mean()
    power   = np.abs(np.fft.rfft(signal)) ** 2
    freqs   = np.fft.rfftfreq(len(signal))
    non_dc  = power[freqs > dc_cut]
    if len(non_dc) == 0:
        return 0.0
    return float(non_dc.max() / (power.sum() + 1e-12))


TESTS = {
    "kurt":      score_kurtosis,
    "fft_peak":  score_fft_peak,
    "zero":      zero_percentage,
    "svd_spike": score_svd_spike,
}

# Tests that receive the raw tensor; all others receive the flat weight array.
TENSOR_TESTS = {"svd_spike"}


# ── Null cache ────────────────────────────────────────────────────────────────

def load_null_cache():
    if not CACHE.exists():
        raise FileNotFoundError(
            f"Null cache not found: {CACHE}\nRun build_null.py first."
        )
    cache = np.load(CACHE, allow_pickle=True)
    null_layer_stats = {
        t: {"conv": list(cache[f"{t}_conv"]), "fc": list(cache[f"{t}_fc"])}
        for t in TESTS
    }
    null_fisher_stats = {t: list(cache[f"{t}_fisher_S"]) for t in TESTS}
    return null_layer_stats, null_fisher_stats


# ── Detection ─────────────────────────────────────────────────────────────────

def test_model(model, null_layer_stats, null_fisher_stats):
    layer_pvals = {t: [] for t in TESTS}
    layer_detail = {}

    for name, tensor in iter_weight_params(model):
        w  = layer_weights(tensor)
        lt = layer_type(tensor)
        layer_detail[name] = {"type": lt}

        for tname, fn in TESTS.items():
            score = fn(tensor) if tname in TENSOR_TESTS else fn(w)
            pval  = empirical_p(score, null_layer_stats[tname][lt])
            layer_pvals[tname].append(pval)
            layer_detail[name][f"{tname}_score"] = round(float(score), 6)
            layer_detail[name][f"{tname}_p"]     = round(float(pval),  4)

    results = {"layers": layer_detail}
    for tname in TESTS:
        ps           = np.clip(np.array(layer_pvals[tname], dtype=float), 1e-12, 1.0)
        S            = float(-2.0 * np.sum(np.log(ps)))
        p_final      = empirical_p(S, null_fisher_stats[tname])
        results[f"{tname}_S"]        = round(S, 6)
        results[f"{tname}_fisher_p"] = round(float(p_final), 4)

    return results


def evaluate_models(model_specs, null_layer_stats, null_fisher_stats, family):
    results = []
    for seed, items in model_specs:
        for label, path, is_attacked in items:
            if not path.exists():
                print(f"MISSING: {path}")
                continue
            model = load_checkpoint(str(path))
            res   = test_model(model, null_layer_stats, null_fisher_stats)
            res.update({"seed": seed, "type": label, "is_attacked": is_attacked, "family": family})
            results.append(res)
    return results


def print_summary(results):
    clean   = [r for r in results if not r["is_attacked"]]
    attacked = [r for r in results if r["is_attacked"]]
    for tname in TESTS:
        key = f"{tname}_fisher_p"
        tp  = sum(r[key] <= ALPHA for r in attacked)
        fp  = sum(r[key] <= ALPHA for r in clean)
        print(
            f"{key:18}  "
            f"TPR={tp}/{len(attacked)}={tp/max(len(attacked),1):.0%}  "
            f"FPR={fp}/{len(clean)}={fp/max(len(clean),1):.0%}"
        )


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_weight_distributions(clean_path, back_path, family, out_dir):
    import matplotlib.pyplot as plt
    for condition, path in [("clean", clean_path), ("backdoored", back_path)]:
        if not Path(path).exists():
            print(f"Missing: {path}")
            continue
        m = load_checkpoint(str(path))
        conv_idx = fc_idx = 0
        for name, t in iter_weight_params(m):
            lt = layer_type(t)
            if lt == "conv":
                conv_idx += 1
                layer_label = f"Conv layer {conv_idx}"
            else:
                fc_idx += 1
                layer_label = f"FC layer {fc_idx}"
            w     = layer_weights(t)
            color = "steelblue" if condition == "clean" else "tomato"
            cond_label = "clean model" if condition == "clean" else "backdoored model"
            plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
            _, ax = plt.subplots()
            ax.hist(w, bins=80, density=False, color=color, alpha=0.7)
            ax.set_title(f"{cond_label.capitalize()} - {layer_label}", fontsize=FS)
            ax.set_xlabel("Weight value", fontsize=FS)
            ax.set_ylabel("Frequency", fontsize=FS)
            ax.tick_params(labelsize=FS)
            plt.tight_layout()
            out_path = out_dir / f"weight_{name.replace('.', '_')}_{condition}.png"
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved: {out_path}")


def visualize_calibration(results, out_dir, family):
    import matplotlib.pyplot as plt
    clean        = [r for r in results if not r["is_attacked"]]
    family_label = FAMILY_DISPLAY.get(family, family)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    for tname in TESTS:
        key     = f"{tname}_fisher_p"
        pvals   = np.sort([r[key] for r in clean])
        n       = len(pvals)
        display = TEST_DISPLAY.get(tname, tname)
        _, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], color="tomato",    linestyle="--", linewidth=2, label="Uniform")
        ax.plot(pvals, np.arange(1, n + 1) / n, color="steelblue", linewidth=2, label="Observed")
        ax.set_title(f"{display} - {family_label}", fontsize=FS)
        ax.set_xlabel("Theoretical", fontsize=FS)
        ax.set_ylabel("Empirical",   fontsize=FS)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.tick_params(labelsize=FS)
        ax.legend(fontsize=FS, loc="upper left")
        plt.tight_layout()
        stem = FAMILY_FILESTEM.get(family, family)
        plt.savefig(out_dir / f"calibration_{tname}_{stem}.png", dpi=300)
        plt.close()


def visualize_results(results, out_dir, family):
    import matplotlib.pyplot as plt
    clean        = [r for r in results if not r["is_attacked"]]
    attacked     = [r for r in results if r["is_attacked"]]
    family_label = FAMILY_DISPLAY.get(family, family)
    stem         = FAMILY_FILESTEM.get(family, family)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    for tname in TESTS:
        key     = f"{tname}_fisher_p"
        display = TEST_DISPLAY.get(tname, tname)
        _, ax = plt.subplots()
        ax.scatter([r["seed"] for r in clean],    [r[key] for r in clean],
                   label="Clean",      alpha=0.7, s=40, color="steelblue")
        ax.scatter([r["seed"] for r in attacked], [r[key] for r in attacked],
                   label="Backdoored", alpha=0.7, s=40, color="tomato")
        ax.set_title(f"{display} - {family_label}", fontsize=FS)
        ax.set_xlabel("Seed",    fontsize=FS)
        ax.set_ylabel("p-value", fontsize=FS)
        ax.tick_params(labelsize=FS)
        ax.axhline(ALPHA, color="black", linestyle="--", linewidth=1.2,
                   label=r"$\alpha=0.05$")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=FS, loc="upper right")
        plt.tight_layout()
        plt.savefig(out_dir / f"{tname}_{stem}.png", dpi=300)
        plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",
                        choices=["dfba", "dfba_flex", "handcrafted", "handcrafted_flex"],
                        default="dfba")
    parser.add_argument("--n", type=int, default=N_MODELS,
                        help=f"limit to first N seeds/configs (default: {N_MODELS})")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / f"out_{args.model}"
    out_dir.mkdir(exist_ok=True)

    null_layer_stats, null_fisher_stats = load_null_cache()
    print(f"Loaded null cache from {CACHE}")

    N = args.n

    if args.model == "dfba":
        seed_dirs = sorted(DFBA_MODELS.glob("seed_*"),
                           key=lambda p: int(p.name.split("_")[1]))[:N]
        model_specs = [
            (int(d.name.split("_")[1]), [
                ("clean",      d / DFBA_CLEAN_NAME,    False),
                ("backdoored", d / DFBA_ATTACKED_NAME, True),
            ])
            for d in seed_dirs
        ]
        results = evaluate_models(model_specs, null_layer_stats, null_fisher_stats, "dfba")
        visualize_results(results, out_dir, "dfba")
        visualize_calibration(results, out_dir, "dfba")

    elif args.model == "dfba_flex":
        cfg_dirs = sorted(DFBA_FLEX_DIR.glob("config_*"),
                          key=lambda p: int(p.name.split("_")[1]))[:N]
        model_specs = [
            (int(d.name.split("_")[1]), [
                ("clean",      d / "clean_model.pth",   False),
                ("backdoored", d / "attacked_model.pth", True),
            ])
            for d in cfg_dirs
        ]
        results = evaluate_models(model_specs, null_layer_stats, null_fisher_stats, "dfba_flex")
        visualize_results(results, out_dir, "dfba_flex")
        visualize_calibration(results, out_dir, "dfba_flex")

    elif args.model == "handcrafted":
        seed_dirs = sorted(HB_CLEAN.glob("seed_*"),
                           key=lambda p: int(p.name.split("_")[1]))[:N]
        model_specs = [
            (int(d.name.split("_")[1]), [
                ("clean",      HB_CLEAN    / d.name / HB_CLEAN_NAME,    False),
                ("backdoored", HB_ATTACKED / d.name / HB_ATTACKED_NAME, True),
            ])
            for d in seed_dirs
        ]
        results = evaluate_models(model_specs, null_layer_stats, null_fisher_stats, "handcrafted")
        visualize_results(results, out_dir, "handcrafted")
        visualize_calibration(results, out_dir, "handcrafted")

    elif args.model == "handcrafted_flex":
        cfg_dirs = sorted(HB_FLEX_DIR.glob("config_*"),
                          key=lambda p: int(p.name.split("_")[1]))[:N]
        model_specs = [
            (int(d.name.split("_")[1]), [
                ("clean",      d / "clean_model.pth",    False),
                ("backdoored", d / "attacked_model.pth", True),
            ])
            for d in cfg_dirs
        ]
        results = evaluate_models(model_specs, null_layer_stats, null_fisher_stats, "handcrafted_flex")
        visualize_results(results, out_dir, "handcrafted_flex")
        visualize_calibration(results, out_dir, "handcrafted_flex")

    print_summary(results)
    out = out_dir / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
