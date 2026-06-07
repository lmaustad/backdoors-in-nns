# Backdoor Detection Suite

A modular framework for evaluating detection methods against non–data-poisoning
backdoor attacks (weight-perturbation, fault-injection, steganographic, and
architectural). Each attack ships its own adapter, which normalises the
attacker's saved checkpoint into a standard PyTorch `nn.Module`, and each
detection method runs against any adapter that produces logits.

## Quick start

Assumes a local machine with Python ≥ 3.10 and a CUDA-capable GPU (CPU also
works, just slower). From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r Detection/requirements.txt

# Run every attack × every detector in the config
python -m Detection.run --config Detection/configs/default.yaml --device cuda

# Or run the clean-baseline sanity sweep
python -m Detection.run --config Detection/configs/sanity_clean.yaml --device cuda
```

Update the checkpoint paths in `Detection/configs/*.yaml` to point at the model
files you have on disk (the attacks under `Attacks/` produce these
checkpoints — see each subfolder's README).

## Directory structure

```
Detection/
├── run.py                        # CLI entry point
├── requirements.txt              # Python dependencies
├── core/                         # Framework internals
│   ├── base_adapter.py           # ModelAdapter ABC — loads attack models
│   ├── base_detector.py          # DetectionMethod ABC — runs detection
│   ├── data_provider.py          # Unified dataset loaders
│   ├── output_mode.py            # Output channels (file / stdout)
│   ├── registry.py               # Adapter / detector registration
│   └── results.py                # JSON / CSV / LaTeX export
├── adapters/                     # One adapter per attack
│   ├── arch_backdoors_adapter.py
│   ├── badnets_adapter.py
│   ├── baseline_inception_adapter.py
│   ├── baseline_resnet_adapter.py
│   ├── boone_bane_adapter.py
│   ├── dfba_adapter.py
│   ├── foobar_adapter.py
│   ├── handcrafted_adapter.py
│   ├── hiding_needles_adapter.py
│   ├── model_editing_clip_adapter.py
│   └── trojannet_adapter.py
├── detectors/                    # Detection methods
│   ├── gradcam_pp.py
│   ├── model_summary.py
│   ├── neural_cleanse.py
│   ├── shap_explainer.py
│   └── weight_forensics.py
├── visualization/
│   ├── plots.py                  # Matplotlib figure helpers
│   └── report.py                 # Aggregated report generation
├── configs/
│   ├── default.yaml              # Main attacks × detectors sweep
│   ├── default_no_trojannet.yaml # Skip TrojanNet (no TF/Keras dependency)
│   ├── sanity_clean.yaml         # Clean-baseline sanity sweep
│   └── sanity_clean_no_trojannet.yaml
└── tests/
    ├── test_clean_accuracy_all_adapters.py
    ├── test_clean_accuracy_and_asr.py
    ├── test_smoke.py             # Import / wiring verification
    └── test_triggered_samples.py
```

## CLI

```bash
# Run all configured attacks × all detectors
python -m Detection.run --config Detection/configs/default.yaml --device cuda

# Run only a specific attack
python -m Detection.run --config Detection/configs/default.yaml --attack dfba_fc_mnist

# Run only a specific detector
python -m Detection.run --config Detection/configs/default.yaml --detector weight_forensics

# Override output directory
python -m Detection.run --config Detection/configs/default.yaml --output-dir ./my_results

# List available adapters and detectors
python -m Detection.run --list-adapters
python -m Detection.run --list-detectors

# Render the triggered-sample test grid
python -m Detection.tests.test_triggered_samples

# Measure clean accuracy + ASR for all adapters
python -m Detection.tests.test_clean_accuracy_and_asr
```

## Output

Results are written to `output_dir` (default `Detection/results/`):

| File             | Description                                       |
| ---------------- | ------------------------------------------------- |
| `results.json`   | Full structured results with all per-run details  |
| `results.csv`    | Summary table (one row per model × detector)      |
| `results.tex`    | LaTeX-formatted table for inclusion in the thesis |
| `details/*.json` | Per-experiment artifacts and detection metadata   |

Per-detector artifacts (heatmaps, histograms, recovered triggers, SHAP/CAM
overlays, etc.) are emitted under `output_dir/<attack>/<detector>/`.

## Configuration

Each YAML config has three sections: `global`, `attacks`, `detectors`, and
`visualization`.

### Attacks

Each attack entry chooses an adapter and points at one or more model
checkpoints:

```yaml
attacks:
  - name: "dfba_fc_mnist"
    adapter: "dfba"
    asr:
      mode: "targeted"
      target_label: 0
    models:
      - model_type: "fc"
        dataset: "mnist"
        is_backdoored: true
        backdoor_checkpoint: "./Attacks/DataFree_Backdoor_Attacks/ckpt/fc_mnist_attacked_model.pth"
        trigger_path: "./Attacks/DataFree_Backdoor_Attacks/ckpt/fc_mnist_trigger.pt"
```

### Detectors

Each detector takes method-specific config:

```yaml
detectors:
  - name: "weight_forensics"
    config:
      save_heatmaps: true
      save_histograms: true
      save_report: true
```

## Detection methods

| Detector           | Signal                                                                                   |
| ------------------ | ---------------------------------------------------------------------------------------- |
| `weight_forensics` | Per-layer weight heatmaps and distribution histograms for visual inspection (data-free). |
| `model_summary`    | Architecture summary and graph export — surfaces unusual structural choices.             |
| `neural_cleanse`   | Trigger reverse-engineering with MAD-based outlier detection (paper-faithful).           |
| `shap_explainer`   | SHAP attributions on clean and triggered samples to localise decision pixels.            |
| `gradcam_pp`       | Grad-CAM++ activation maps with per-attack target-layer overrides.                       |

## Adapters

Each adapter normalises one attack's checkpoint format into a standard
PyTorch `nn.Module` that returns logits. **All adapters are self-contained**:
they declare model architectures locally and do not import from the `Attacks/`
directories, so detection has no dependency on attack-specific Python or
package versions. Only the saved checkpoints (`.pth`, `.pkl`, `.h5`) are
needed.

| Adapter              | Attack                                | Model format                    | Notes                                                                                                 |
| -------------------- | ------------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `dfba`               | DataFree_Backdoor_Attacks             | full model or state_dict `.pth` | FCN/CNN/VGG16/ResNet18 defined locally; handles both `torch.save(model)` and `torch.save(state_dict)` |
| `arch_backdoors`     | ARCHITECTURAL-BACKDOORS               | state_dict `.pth`               | AlexNet defined locally, input resized to 70x70                                                       |
| `badnets`            | badnets-pytorch                       | state_dict or full-module `.pth`| Small CNN for MNIST, ResNet-18 for CIFAR-10                                                           |
| `boone_bane`         | Boone_and_bane                        | state_dict `.pth`               | Custom CIFAR-10 ResNet defined locally                                                                |
| `handcrafted`        | Handcrafted_Backdoors                 | state_dict `.pth`               | MNIST CNN with handcrafted weight perturbations                                                       |
| `hiding_needles`     | hiding-needles-in-a-haystack          | state_dict `.pkl`               | Steganographic sub-networks defined locally; DiffJPEG replaced with `nn.Identity`                     |
| `model_editing_clip` | Backdoor_in_seconds_via_model_editing | state_dict `.pth`               | Notebook-exported CLIP ViT-B/32 (`clip_model`/`clean_model`) checkpoints                              |
| `trojannet`          | TrojanNet                             | Keras `.h5`                     | Equivalent PyTorch module built locally; weights transferred via `h5py`                               |
| `foobar`             | foobar                                | NumPy pickle `.pkl`             | Custom unpickler with stub classes; numpy model converted to PyTorch                                  |
| `baseline_resnet`    | BaselineModels/ResNet                 | state_dict `.pth`               | Torchvision ResNet checkpoints loaded as clean baselines                                              |
| `baseline_inception` | BaselineModels/Inception              | state_dict `.pth`               | Torchvision Inception-v3 checkpoints loaded as clean baselines                                        |

### DFBA checkpoint compatibility

DFBA saves full model objects via `torch.save(model, path)`. The adapter
handles both formats automatically:

- **Full model pickles** — extracts `.state_dict()` from the deserialized
  `nn.Module`.
- **State dicts** — loads directly.

For best reliability, consider re-saving DFBA checkpoints as state dicts:

```python
import torch
model = torch.load("ckpt/fc_mnist_attacked_model.pth", weights_only=False)
torch.save(model.state_dict(), "ckpt/fc_mnist_attacked_model_sd.pth")
```

## Adding new detection methods

1. Create `Detection/detectors/my_method.py`:

   ```python
   from Detection.core.base_detector import DetectionMethod, DetectionResult
   from Detection.core.registry import register_detector

   @register_detector("my_method")
   class MyMethod(DetectionMethod):
       @property
       def name(self):
           return "my_method"

       def detect(self, model, model_info, data_loader, clean_model=None, device=...):
           return DetectionResult(
               method_name=self.name,
               attack_name=model_info.attack_name,
               model_architecture=model_info.architecture,
               dataset=model_info.dataset,
               is_backdoor_detected=True,
               confidence_score=0.85,
               details={"my_metric": 42},
           )
   ```

2. Register the import in `Detection/detectors/__init__.py`:

   ```python
   from . import my_method  # noqa: F401
   ```

3. Add a `detectors:` entry to your YAML config.

## Adding new attack adapters

1. Create `Detection/adapters/my_attack_adapter.py`:

   ```python
   from Detection.core.base_adapter import ModelAdapter, ModelInfo
   from Detection.core.registry import register_adapter

   @register_adapter("my_attack")
   class MyAttackAdapter(ModelAdapter):
       def __init__(self, model_type="default", dataset="cifar10", **kwargs):
           self.model_type = model_type
           self.dataset = dataset
           self._model_info = None

       def load_model(self, checkpoint_path, device=torch.device("cpu"), **kwargs):
           ...  # return an nn.Module that outputs logits

       def get_model_info(self):
           return self._model_info
   ```

2. Register the import in `Detection/adapters/__init__.py`.
3. Add an `attacks:` entry to your YAML config.

## Environment isolation

The Detection suite is intentionally decoupled from the individual attack
environments. Each attack in `Attacks/` may use a different Python version,
TensorFlow vs PyTorch, or conflicting package pins. The adapters resolve this
by:

- **Defining all model architectures locally** (no `sys.path` imports from
  `Attacks/`).
- **Only reading checkpoint files** (`.pth`, `.pkl`, `.h5`) produced by the
  attacks.
- **Running everything under one Python 3.10+ environment** for detection.

This means each attack can be run in its own conda/venv to produce
checkpoints, after which the entire detection suite runs in a single unified
environment with no version conflicts.

## Dependencies

See `Detection/requirements.txt` for pinned versions. Core packages:

```
torch>=1.13
torchvision>=0.14
numpy>=1.21
matplotlib>=3.5
pyyaml>=6.0
h5py>=3.0
scikit-learn>=1.0
shap
```

Optional, only required by specific adapters:

- `transformers` — `model_editing_clip` adapter (HuggingFace CLIP utilities)
- `tensorflow` / `keras` — TrojanNet checkpoint conversion
