#!/usr/bin/env python3
"""
Generate LaTeX config and detection tables for all four attack families.

Sources:
  HC fixed:   appendix_hb_fixed.tex (accuracy)  + out_handcrafted/results.json
  HC flex:    Models_Handcrafted_Flex/results.json + out_handcrafted_flex/results.json
  DFBA fixed: appendix_dfba_fixed.tex (accuracy) + out_dfba/results.json
  DFBA flex:  Models_DFBA_Flex/results.json       + out_dfba_flex/results.json (when available)

Config tables  -> tables/config_{family}.tex
Results tables -> tables/results_{family}.tex  (skipped if detection results missing)

Usage:
    python generate_attack_tables.py
"""

import sys
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config

HB_FLEX   = config.HB_FLEX_DIR
DFBA_FLEX = config.DFBA_FLEX_DIR
STAT_BC   = config.STAT_BC
OUT_DIR   = ROOT / "tables"
ALPHA     = 0.05
N_FIXED   = 40

OUT_DIR.mkdir(exist_ok=True)

TESTS = [
    ("Kurtosis",      "kurt_fisher_p"),
    ("FFT peak",      "fft_peak_fisher_p"),
    ("Zero fraction", "zero_fisher_p"),
    ("SVD spike",     "svd_spike_fisher_p"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def fpr_tpr(entries, key):
    att = [e for e in entries if     e["is_attacked"]]
    cln = [e for e in entries if not e["is_attacked"]]
    tpr = sum(1 for e in att if e[key] < ALPHA) / len(att) if att else 0.0
    fpr = sum(1 for e in cln if e[key] < ALPHA) / len(cln) if cln else 0.0
    return tpr, fpr


def _parse_fixed_tex(path):
    rows = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*(\d+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)", line)
            if m:
                rows[int(m.group(1))] = {
                    "ba_clean": float(m.group(2)),
                    "ba_after": float(m.group(3)),
                    "asr":      float(m.group(4)),
                }
    return rows


def _lr_str(lr):
    if lr == 5e-4: return r"$5\!\times\!10^{-4}$"
    if lr == 1e-2: return r"$10^{-2}$"
    if lr == 1e-3: return r"$10^{-3}$"
    if lr == 1e-4: return r"$10^{-4}$"
    return str(lr)


# ── config table builders ──────────────────────────────────────────────────────

def _table_wrap(body_lines, caption, label, ncols):
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{ncols}}}",
        r"\toprule",
    ] + body_lines + [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])


def fixed_config_table(acc_rows, caption, label, conv, fc, k, optimizer, lr,
                        has_ba_after=True, has_ba_atk_clean=False):
    seeds = sorted(s for s in acc_rows if s < N_FIXED)
    conv_str = r"\texttt{" + str(conv).replace(" ", "") + r"}"
    fc_str   = r"\texttt{" + str(fc).replace(" ", "") + r"}"
    lr_str   = _lr_str(lr)
    after_head     = r" & Acc\textsubscript{bd}" if has_ba_after     else ""
    atk_clean_head = r" & Acc\textsubscript{bd}" if has_ba_atk_clean else ""
    ncols = "rlcrlc" + "r" + ("r" if has_ba_after else "") + ("r" if has_ba_atk_clean else "") + "r"
    head = (rf"Seed & Conv channels & $k$ & FC dims & Opt & LR"
            rf" & Acc\textsubscript{{clean}}" + after_head + atk_clean_head + r" & ASR \\")
    body = [head, r"\midrule"]
    for s in seeds:
        r = acc_rows[s]
        after_col    = f" & {r['ba_after']:.4f}"     if has_ba_after     else ""
        atk_cln_col  = f" & {r['ba_atk_clean']:.4f}" if has_ba_atk_clean else ""
        body.append(
            rf"{s} & {conv_str} & {k} & {fc_str} & {optimizer} & {lr_str}"
            rf" & {r['ba_clean']:.4f}{after_col}{atk_cln_col} & {r['asr']:.4f} \\"
        )
    return _table_wrap(body, caption, label, ncols)


def flex_config_table(rows, caption, label, has_ba_after=True, has_ba_atk_clean=False):
    # BA = Benign Accuracy (accuracy on clean/non-triggered test data)
    after_head     = r" & Acc\textsubscript{bd}" if has_ba_after else ""
    atk_clean_head = r" & Acc\textsubscript{bd}" if has_ba_atk_clean else ""
    # ncols: ID(r) Conv(l) k(c) FC(r) Opt(l) LR(c) Acc_clean(r) [Acc_bd(r)] [Acc_bd(r)] ASR(r)
    ncols = "rlcrlc" + "r" + ("r" if has_ba_after else "") + ("r" if has_ba_atk_clean else "") + "r"
    head = (r"ID & Conv channels & $k$ & FC dims & Opt & LR"
            r" & Acc\textsubscript{clean}" + after_head + atk_clean_head + r" & ASR \\")
    body = [head, r"\midrule"]
    for r in sorted(rows, key=lambda x: x["id"]):
        cfg  = r["config"]
        conv = str(cfg["conv_channels"]).replace(" ", "")
        fc   = str(cfg["fc_dims"]).replace(" ", "")
        lr   = _lr_str(cfg["lr"])
        after_col    = f" & {r['ba_after']:.4f}"     if has_ba_after     else ""
        atk_cln_col  = f" & {r['ba_atk_clean']:.4f}" if has_ba_atk_clean else ""
        body.append(
            rf"{r['id']} & \texttt{{{conv}}} & {cfg['kernel_size']} & \texttt{{{fc}}}"
            rf" & {cfg['optimizer']} & {lr}"
            rf" & {r['ba_clean']:.4f}{after_col}{atk_cln_col} & {r['asr']:.4f} \\"
        )
    return _table_wrap(body, caption, label, ncols)


def results_table(entries, caption, label):
    n_att = sum(1 for e in entries if     e["is_attacked"])
    n_cln = sum(1 for e in entries if not e["is_attacked"])
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        rf"\multicolumn{{3}}{{l}}{{\small {n_att} attacked, {n_cln} clean, $\alpha={ALPHA}$}} \\",
        r"\textbf{Test} & \textbf{TPR} & \textbf{FPR} \\",
        r"\midrule",
    ]
    for tname, tkey in TESTS:
        tpr, fpr = fpr_tpr(entries, tkey)
        lines.append(f"{tname} & {tpr:.3f} & {fpr:.3f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    saved = []

    # ── HC fixed ──────────────────────────────────────────────────────────────
    hb_acc = _parse_fixed_tex(STAT_BC / "appendix_hb_fixed.tex")
    hb_acc = {s: v for s, v in hb_acc.items() if s < N_FIXED}

    p = OUT_DIR / "config_hb_fixed.tex"
    p.write_text(fixed_config_table(
        hb_acc,
        r"HC fixed-arch models (seeds 0--39, MNIST).",
        "tab:config_hb_fixed",
        conv=[16, 32], fc=[128], k=3, optimizer="SGD", lr=1e-2,
        has_ba_after=True,
    ) + "\n")
    saved.append(p.name)

    det_path = ROOT / "out_handcrafted" / "results.json"
    if det_path.exists():
        with open(det_path) as f:
            det = [e for e in json.load(f) if e["seed"] < N_FIXED]
        p = OUT_DIR / "results_hb_fixed.tex"
        p.write_text(results_table(det,
            rf"Detection — HC fixed (seeds 0--39, $\alpha={ALPHA}$).",
            "tab:results_hb_fixed") + "\n")
        saved.append(p.name)

    # ── HC flex ───────────────────────────────────────────────────────────────
    with open(HB_FLEX / "results.json") as f:
        hb_flex = json.load(f)
    hb_flex_rows = [
        {
            "id":       e["config_id"],
            "config":   e["config"],
            "ba_clean": e["ba_before"],
            "ba_after": e["ba_after"],
            "asr":      e["asr"],
        }
        for e in hb_flex
    ]

    p = OUT_DIR / "config_hb_flex.tex"
    p.write_text(flex_config_table(
        hb_flex_rows,
        "HC flex-arch models (configs 0--39, MNIST).",
        "tab:config_hb_flex",
        has_ba_after=True,
    ) + "\n")
    saved.append(p.name)

    det_path = ROOT / "out_handcrafted_flex" / "results.json"
    if det_path.exists():
        valid_ids = {e["config_id"] for e in hb_flex}
        with open(det_path) as f:
            det = [e for e in json.load(f) if e["seed"] in valid_ids]
        p = OUT_DIR / "results_hb_flex.tex"
        p.write_text(results_table(det,
            rf"Detection — HC flex (configs 0--39, $\alpha={ALPHA}$).",
            "tab:results_hb_flex") + "\n")
        saved.append(p.name)

    # ── DFBA fixed ────────────────────────────────────────────────────────────
    dfba_acc = _parse_fixed_tex(STAT_BC / "appendix_dfba_fixed.tex")
    dfba_acc = {s: v for s, v in dfba_acc.items() if s < N_FIXED}

    p = OUT_DIR / "config_dfba_fixed.tex"
    p.write_text(fixed_config_table(
        dfba_acc,
        r"DFBA fixed-arch models (seeds 0--39, MNIST).",
        "tab:config_dfba_fixed",
        conv=[16, 32], fc=[1024], k=5, optimizer="SGD", lr=1e-2,
        has_ba_after=True,
    ) + "\n")
    saved.append(p.name)

    det_path = ROOT / "out_dfba" / "results.json"
    if det_path.exists():
        with open(det_path) as f:
            det = [e for e in json.load(f) if e["seed"] < N_FIXED]
        p = OUT_DIR / "results_dfba_fixed.tex"
        p.write_text(results_table(det,
            rf"Detection — DFBA fixed (seeds 0--39, $\alpha={ALPHA}$).",
            "tab:results_dfba_fixed") + "\n")
        saved.append(p.name)

    # ── DFBA flex ─────────────────────────────────────────────────────────────
    with open(DFBA_FLEX / "results.json") as f:
        dfba_flex = json.load(f)
    dfba_flex_rows = [
        {
            "id":           e["config_id"],
            "config":       e["config"],
            "ba_clean":     e["ba_clean"],
            "ba_atk_clean": e["ba_atk_clean"],
            "asr":          e["asr"],
        }
        for e in dfba_flex
    ]

    p = OUT_DIR / "config_dfba_flex.tex"
    p.write_text(flex_config_table(
        dfba_flex_rows,
        r"DFBA flex-arch models (configs 0--39, MNIST). ASR\,=\,1.00 for all.",
        "tab:config_dfba_flex",
        has_ba_after=False,
        has_ba_atk_clean=True,
    ) + "\n")
    saved.append(p.name)

    det_path = ROOT / "out_dfba_flex" / "results.json"
    if det_path.exists():
        with open(det_path) as f:
            det = json.load(f)
        p = OUT_DIR / "results_dfba_flex.tex"
        p.write_text(results_table(det,
            rf"Detection — DFBA flex (configs 0--39, $\alpha={ALPHA}$).",
            "tab:results_dfba_flex") + "\n")
        saved.append(p.name)
    else:
        print("  [skip] out_dfba_flex/results.json not found — run StandardTesting.py --model dfba_flex first")

    # ── summary ───────────────────────────────────────────────────────────────
    print("Saved:")
    for f in saved:
        print(f"  tables/{f}")


if __name__ == "__main__":
    main()
