"""
TDA pipeline paths.

Root paths (MODELS_ROOT, DATA_ROOT) are inherited from the parent StatDet
config.  Edit StatDet/config.py to change those.  Edit this file for anything
TDA-specific (manifest locations, trigger image, results directory).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_ROOT, DATA_ROOT, DFBA_MODELS, HB_CLEAN, HB_ATTACKED

# ── Model checkpoint roots ────────────────────────────────────────────────────
HC_CKPT   = MODELS_ROOT / "ckpt"        # parent of clean_models/ and attacked_models/
DFBA_CKPT = DFBA_MODELS                 # re-export for convenience

# ── Neuron manifests (JSON produced during training) ──────────────────────────
HC_MANIFEST   = HC_CKPT / "neuron_manifest.json"
DFBA_MANIFEST = DFBA_CKPT / "neuron_manifest.json"

# ── Trigger image (used by HC attacked models) ────────────────────────────────
TRIGGER_IMG = HC_CKPT / "trigger.png"

# ── MNIST data root ───────────────────────────────────────────────────────────
MNIST_DIR = DATA_ROOT

# ── Detection results ─────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
