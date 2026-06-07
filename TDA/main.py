#!/usr/bin/env python3
"""
TDA backdoor detection pipeline.

Loads clean + trojan model pairs, computes activation distance matrices,
runs persistent homology on the cross-layer matrix, and generates plots.

Usage:
    python main.py --model-type hc --metric jaccard --plot all
    python main.py --model-type dfba --metric feature_profile --load results/dfba/feature_profile/results_feature_profile.pkl --plot violin min_death
"""
import sys
import argparse
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U
import metrics as M
import detect as D
import plot as P
import config

METRIC_FNS = {
    'jaccard':                 M.jaccard_distance,
    'pearson':                 M.pearson_distance,
    'feature_profile':         M.feature_profile_distance,
    'feature_profile_reduced': M.feature_profile_reduced_distance,
}

MANIFESTS = {
    'hc':   lambda: U.hc_manifest,
    'dfba': lambda: U.dfba_manifest,
}


def _bd_neurons_from_manifest(manifest, seed, model_type):
    entry = manifest.get(seed, {})
    if model_type == 'dfba':
        return {
            'pool1':  [entry['cnn_conv1']]  if 'cnn_conv1'  in entry else [],
            'pool2':  [entry['cnn_conv2']]  if 'cnn_conv2'  in entry else [],
            'fc1':    [entry['fc1_neuron']] if 'fc1_neuron' in entry else [],
            'logits': [],
        }
    return {
        'pool1':  entry.get('conv1_channels', []),
        'pool2':  entry.get('conv2_channels', []),
        'fc1':    entry.get('fc1_neurons',    []),
        'logits': entry.get('fc2_neurons',    []),
    }


def _filter_results(results, min_clean_acc):
    if min_clean_acc <= 0:
        return results
    kept = {
        (s, lbl): r for (s, lbl), r in results.items()
        if all(
            results.get((s, l), {}).get('clean_acc', 1.0) >= min_clean_acc
            for l in ('clean', 'trojan')
        )
    }
    dropped = {s for s, _ in results} - {s for s, _ in kept}
    if dropped:
        print(f'  excluded {len(dropped)} seed(s) with clean_acc < {min_clean_acc}: {sorted(dropped)}')
    return kept


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model-type', choices=['hc', 'dfba'], default='hc')
    parser.add_argument('--metric', choices=list(METRIC_FNS), default='jaccard')
    parser.add_argument('--min-clean-acc', type=float, default=0.0,
                        help='Drop seeds where any model accuracy is below this (default: 0)')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--seed',  type=int,            help='Run a single seed')
    group.add_argument('--all',   action='store_true', help='Run all seeds in manifest')
    parser.add_argument('--n-seeds', type=int, default=40,
                        help='Number of seeds (first N sorted; default: 40)')

    parser.add_argument('--plot', nargs='*',
                        choices=['ph_metrics', 'vr', 'barcode', 'distance_matrix',
                                 'violin', 'min_death', 'min_death_rep', 'all'],
                        help='Plots to generate (omit to skip all plots)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Edge distance threshold for VR plot (default: 0.3)')
    parser.add_argument('--eps', type=float, default=1e-4,
                        help='Clone distance threshold for bipartite plot (default: 1e-4)')
    parser.add_argument('--vr-seed', type=int, default=None,
                        help='Only generate VR plot for this seed')

    parser.add_argument('--save-dir', default=None,
                        help='Output directory (default: results/<model>/<metric>/)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load saved results pickle instead of rerunning')
    parser.add_argument('--paper-dir', type=str, default=None,
                        help='If set, save violin/VR/min_death plots here with paper naming')

    return parser.parse_args()


def main():
    args      = parse_args()
    metric_fn = METRIC_FNS[args.metric]
    plots     = set(args.plot) if args.plot else set()
    if 'all' in plots:
        plots = {'ph_metrics', 'vr', 'barcode', 'distance_matrix',
                 'violin', 'min_death', 'min_death_rep'}

    save_dir = args.save_dir or str(config.RESULTS_DIR / args.model_type / args.metric)
    os.makedirs(save_dir, exist_ok=True)

    # ── Load or compute results ───────────────────────────────────────────────
    if args.load:
        print(f'Loading results from {args.load}')
        results = D.load_results(args.load)
    else:
        manifest = MANIFESTS[args.model_type]()
        if args.seed is not None:
            seeds = [args.seed]
        elif args.all:
            seeds = sorted(manifest.keys())
        else:
            seeds = sorted(manifest.keys())[:args.n_seeds]

        print(f'Running detection: model={args.model_type}  metric={args.metric}  seeds={len(seeds)}')
        images, labels = U.load_images(2000)
        results = {}
        for seed in seeds:
            models = U.loadmodels(args.model_type, seeds=[seed])
            for lbl, model in [('clean', models['clean'][0]), ('trojan', models['trojan'][0])]:
                r = D.detect(model.to(U.device), images, labels=labels, metric_fn=metric_fn)
                results[(seed, lbl)] = r
                print(f'  seed {seed:3d}  {lbl:6s}  acc={r["clean_acc"]:.3f}')

        save_path = os.path.join(save_dir, f'results_{args.metric}.pkl')
        D.save_results(results, save_path)
        print(f'Saved: {save_path}')

    results = _filter_results(results, args.min_clean_acc)

    if not plots:
        return

    # ── Per-seed plots ────────────────────────────────────────────────────────
    for seed in sorted({s for s, _ in results}):
        clean_r  = results.get((seed, 'clean'))
        trojan_r = results.get((seed, 'trojan'))

        if 'ph_metrics' in plots and clean_r and trojan_r:
            path = os.path.join(save_dir, f'ph_metrics_seed{seed}.png')
            P.plot_ph_metrics(clean_r, trojan_r, save_path=path)
            print(f'  saved {path}')

        if 'vr' in plots and trojan_r:
            if args.vr_seed is not None and seed != args.vr_seed:
                continue
            bd_neurons = _bd_neurons_from_manifest(MANIFESTS[args.model_type](), seed, args.model_type)
            if args.paper_dir:
                os.makedirs(args.paper_dir, exist_ok=True)
                path = os.path.join(args.paper_dir,
                                    f'vr_{args.model_type}_{args.metric}_seed{seed}_eps{args.threshold}.png')
            else:
                path = os.path.join(save_dir, f'vr_seed{seed}_eps{args.threshold}.png')
            P.plot_vr(trojan_r, threshold=args.threshold,
                      bd_neurons=bd_neurons, model_type=args.model_type, save_path=path)
            print(f'  saved {path}')

        if 'barcode' in plots and clean_r and trojan_r:
            path = os.path.join(save_dir, f'barcode_seed{seed}.png')
            P.plot_barcode(clean_r, trojan_r, save_path=path)
            print(f'  saved {path}')

        if 'clones' in plots and clean_r and trojan_r:
            bd_neurons = _bd_neurons_from_manifest(MANIFESTS[args.model_type](), seed, args.model_type)
            path = os.path.join(save_dir, f'clones_seed{seed}.png')
            P.plot_clone_bipartite(clean_r, trojan_r, eps=args.eps,
                                   bd_neurons=bd_neurons, save_path=path)
            print(f'  saved {path}')

        if 'distance_matrix' in plots and trojan_r:
            layers = ['p1', 'p2', 'logits'] if trojan_r.get('with_logits') else ['p1', 'p2']
            for layer in layers:
                path = os.path.join(save_dir, f'dist_{layer}_seed{seed}.png')
                P.plot_distance_matrix(trojan_r, layer=layer, save_path=path)
                print(f'  saved {path}')

    # ── Aggregate plots ───────────────────────────────────────────────────────
    if 'min_death' in plots:
        model_label = P.MODEL_DISPLAY.get(args.model_type, args.model_type.upper())
        if args.paper_dir:
            os.makedirs(args.paper_dir, exist_ok=True)
            path = os.path.join(args.paper_dir, f'min_death_{model_label}_{args.metric}.png')
        else:
            path = os.path.join(save_dir, 'min_death.png')
        P.plot_min_death(results, model_type=args.model_type, metric=args.metric, save_path=path)
        print(f'  saved {path}')

    if 'min_death_rep' in plots:
        rep_dir = args.paper_dir or save_dir
        P.plot_min_death_rep(results, model_type=args.model_type,
                             metric=args.metric, save_dir=rep_dir)

    if 'violin' in plots:
        model_label = P.MODEL_DISPLAY.get(args.model_type, args.model_type.upper())
        if args.paper_dir:
            os.makedirs(args.paper_dir, exist_ok=True)
            path = os.path.join(args.paper_dir, f'violin_{model_label}_{args.metric}.png')
        else:
            path = os.path.join(save_dir, 'violin_metrics.png')
        P.plot_violin(results, model_type=args.model_type, metric=args.metric, save_path=path)
        print(f'  saved {path}')


if __name__ == '__main__':
    main()
