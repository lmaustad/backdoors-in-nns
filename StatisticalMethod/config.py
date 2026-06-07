"""
User-configurable paths for StatDet.

Edit the variables below to match your local directory layout before running
any scripts.
"""
from pathlib import Path

MODELS_ROOT = Path("/Volumes/lmaustad/Models")
DATA_ROOT   = Path("/Volumes/lmaustad/Datasets")

# Null model directories: 50 configurations × 3 datasets = up to 150 null models.
# Expected layout: <dir>/config_000/model.pth … config_049/model.pth
NULL_DIRS = {
    "MNIST":        MODELS_ROOT / "ckpt_MNIST",
    "FashionMNIST": MODELS_ROOT / "ckpt_FMNIST",
    "GTSRB":        MODELS_ROOT / "ckpt_GTSRB",
}

# ── Attack model directories ──────────────────────────────────────────────────

DFBA_MODELS   = MODELS_ROOT / "DFBA_models"
DFBA_FLEX_DIR = Path("/Volumes/lmaustad/Models_DFBA_Flex")

HB_FLEX_DIR  = Path("/Volumes/lmaustad/Models_Handcrafted_Flex")
HB_CLEAN     = MODELS_ROOT / "ckpt" / "clean_models"
HB_ATTACKED  = MODELS_ROOT / "ckpt" / "attacked_models"

# Path to the companion repo that contains appendix TeX files for fixed-arch
# attack tables (used by generate_attack_tables.py).
STAT_BC = Path("/Volumes/lmaustad/StatBackdoorDetection")
