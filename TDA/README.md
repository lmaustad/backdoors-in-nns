# TDA — Topological Backdoor Detection

Persistent homology on neural network activation distance matrices for
backdoor detection.  Models are run on clean inputs; activations are pooled
into per-layer distance matrices, assembled into a single cross-layer matrix,
and fed to a Vietoris-Rips filtration.  Backdoored models exhibit distinctive
topological signatures (H0 clone structure, long-lived H1 loops) absent in
clean models.

---

## Layout

```
TDA/
├── config.py       Paths — inherits MODELS_ROOT / DATA_ROOT from ../config.py
├── utils.py        Model loading, image loading, activation extraction
├── metrics.py      Distance metrics + persistent homology features
├── detect.py       Core detection: activations → matrices → topology → result dict
├── main.py         CLI entry point
├── plot.py         All plots (VR graph, violin, min-death, PH metrics, barcode, …)
├── models/
│   ├── cnn.py      HC (handcrafted) CNN architecture
│   └── dfba.py     DFBA CNN architecture
└── results/        Detection output pickles (created at runtime)
    ├── hc/
    └── dfba/
```

---

## Quickstart

```bash
conda activate backdoor

# Run detection + generate all plots for HC / Jaccard
python main.py --model-type hc --metric jaccard --plot all

# Load cached results, generate only violin and min-death plots
python main.py --model-type dfba --metric feature_profile \
    --load results/dfba/feature_profile/results_feature_profile.pkl \
    --plot violin min_death

# Save paper-quality figures to a separate directory
python main.py --model-type hc --metric jaccard --plot violin min_death vr \
    --paper-dir ../paper_figures/
```

---

## Detection pipeline

```
load_images()         500–2000 MNIST test images
    ↓
loadmodels()          clean + trojan pairs for N seeds
    ↓
detect()              per model:
  extract_acts()        → {pool1, pool2, relu_fc1, logits} activations
  metric_fn()           → within-layer distance matrices D_p1, D_p2, D_fc, D_logits
  cross_layer_matrix()  → full cross-layer distance matrix (intra-layer = ∞)
  topology()            → H0 + H1 features via ripser
    ↓
save_results()        pickle: dict keyed by (seed, 'clean'/'trojan')
```

---

## Metrics

| Key | Function | Description |
|-----|----------|-------------|
| `jaccard` | `jaccard_distance` | Binary activation Jaccard (median threshold per column) |
| `pearson` | `pearson_distance` | 1 − Pearson correlation; dormant neurons → 2.0 |
| `feature_profile` | `feature_profile_distance` | Euclidean on [μ, σ, zero-frac, max] per channel (4D) |
| `feature_profile_reduced` | `feature_profile_reduced_distance` | Euclidean on [zero-frac, max] per channel (2D) |

---

## Plots

| Flag | Function | What it shows |
|------|----------|--------------|
| `ph_metrics` | `plot_ph_metrics` | H0/H1 topology feature bar charts (clean vs trojan) |
| `vr` | `plot_vr` | Vietoris-Rips network graph at a given threshold; BD neurons highlighted |
| `barcode` | `plot_barcode` | Persistence barcode (clean vs trojan side-by-side) |
| `distance_matrix` | `plot_distance_matrix` | Distance matrix heatmap for a single layer |
| `violin` | `plot_violin` | Violin plots of topology metrics across seeds |
| `min_death` | `plot_min_death` | Minimum H1 death value per seed (clean vs trojan) |
| `min_death_rep` | `plot_min_death_rep` | Representative seeds at multiple thresholds |

All plots accept an optional `--paper-dir` to write publication-ready PNGs
with standardised filenames.

---

## Topology features (`topo_features` / `topology`)

Per persistence dimension (H0, H1), each result dict contains:

| Field | Description |
|-------|-------------|
| `betti` | Number of finite bars |
| `ave_persis` | Mean bar length |
| `ave_birth` / `ave_death` | Mean birth / death |
| `max_persis` | Length of longest bar |
| `top5_persis` / `top10_persis` / `top20_persis` | Mean of top-k bar lengths |
| `synchronized` | True if all births lie within 1e-6 of each other |
| `sync_count` | Size of the dominant birth cluster |

---

## Result pickle schema

Each pickle is a `dict` keyed by `(seed: int, condition: str)` where
`condition ∈ {'clean', 'trojan'}`.  Each value is a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `D_p1`, `D_p2`, `D_fc`, `D_logits` | `(C, C) ndarray` | Within-layer distance matrices |
| `D_cross` | `(N, N) ndarray` | Cross-layer distance matrix (intra-layer = ∞) |
| `p1`, `p2`, `fc`, `logits` | `(B, C) ndarray` | Raw activations |
| `topology_h0`, `topology_h1` | dict | PH feature dicts (see above) |
| `clean_acc` | float | Model accuracy on the image batch |


---

## Model architectures

### HC (handcrafted) — `models/cnn.py`

```
Conv(1→16, 3×3) → ReLU → MaxPool(2)   [pool1]
Conv(16→32, 3×3) → ReLU → MaxPool(2)  [pool2]
FC(1568→128) → ReLU                   [relu_fc1]
FC(128→10)                             [logits]
```

### DFBA — `models/dfba.py`

```
Conv(1→16, 5×5)                       [pool1]
Conv(16→32, 5×5) → MaxPool(2)         [pool2]
FC(800→1024) → ReLU                   [relu_fc1]
FC(1024→10)                            [logits]
```
