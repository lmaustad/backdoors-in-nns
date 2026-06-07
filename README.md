# Non-Data-Poisoning Backdoor Attacks & Detection

Master's thesis on undetectability of architectural and weight-manipulation
backdoors in neural networks. The repository collects reference
re-implementations of published non–data-poisoning attacks alongside a
unified detection harness that runs every detector against every attack.

## What's in this repo

```
.
├── Attacks/              # One subfolder per published attack
├── BaselineModels/       # Clean reference checkpoints (ResNet, Inception)
├── Datasets/             # Dataset loaders and helpers
├── Detection/            # Modular detection framework (see Detection/README.md)
├── StatisticalMethod/    # Standalone statistical weight-distribution detection
└── TDA/                  # Standalone topological backdoor detection
```

Each folder under `Attacks/` is a near-verbatim copy of the upstream attack
repository with minimal modifications, used to produce the backdoored
checkpoints the detection suite then analyses.

## Getting started

`Detection/` is the main entry point. The CLI runs every configured
attack × detector combination and writes JSON/CSV/LaTeX results:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r Detection/requirements.txt
python -m Detection.run --config Detection/configs/default.yaml --device cuda
```

See [Detection/README.md](Detection/README.md) for full configuration, the
list of supported attacks/detectors, and instructions for adding new
adapters or detection methods.

Reproducing a single attack from scratch requires that attack's own
environment — see the README in each `Attacks/<attack>/` folder for the
upstream's setup instructions.

---

## StatisticalMethod — standalone weight-distribution detection

`StatisticalMethod/` is an independent detection experiment and is **not part
of the main `Detection/` pipeline**. It tests whether backdoored model weights
are statistically distinguishable from a null distribution of clean models,
using per-layer hypothesis tests (kurtosis, FFT peak, zero-fraction, SVD spike)
combined with Fisher's method.

```bash
conda activate backdoor
cd StatisticalMethod

# Edit config.py to set your model and dataset paths, then:
python StandardTesting.py --model handcrafted
python StandardTesting.py --model dfba
```

See [StatisticalMethod/README.md](StatisticalMethod/README.md) for the full
workflow, including how to train the null model pool and rebuild the null cache.

---

## TDA — standalone topological backdoor detection

`TDA/` is an independent detection experiment and is **not part of the main
`Detection/` pipeline**. It extracts per-layer activations on clean inputs,
assembles them into a cross-layer distance matrix, and runs persistent homology
(Vietoris-Rips filtration) to identify topological signatures characteristic of
backdoored models — such as H0 clone structure and long-lived H1 loops.

```bash
conda activate backdoor
cd TDA

python main.py --model-type hc   --metric jaccard --plot all
python main.py --model-type dfba --metric feature_profile --plot violin min_death
```

See [TDA/README.md](TDA/README.md) for available metrics, plot types, and the
full detection pipeline description.

---

## Citing this work

```bibtex
@mastersthesis{farstad_austad_2026_backdoordetection,
  author = {Farstad, Frederik Andreas Brunvoll and Austad, Lisa Marie F{\o}lstad},
  title  = {Non-Data-Poisoning Backdoor Attacks \& Detection},
  school = {Norwegian University of Science and Technology (NTNU)},
  year   = {2026},
  month  = jun,
  type   = {Master's thesis},
  url    = {https://github.com/lmaustad/BackdoorMaster}
}
```

## License

MIT — see `LICENSE`.
