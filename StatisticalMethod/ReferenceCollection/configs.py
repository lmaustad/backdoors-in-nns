"""
Defines 300 varied CNN configurations for building a general clean null distribution.

Structure: 50 base configs × 3 datasets × 2 tiers = 300 models.
  Tier 1 (original, IDs   0–149): shallow 2–3 layer CNNs, small FC
    IDs   0– 49  → MNIST
    IDs  50– 99  → FashionMNIST
    IDs 100–149  → GTSRB

  Tier 2 (extended, IDs 150–299): deeper 4–6 layer CNNs, large FC
    IDs 150–199  → MNIST
    IDs 200–249  → FashionMNIST
    IDs 250–299  → GTSRB


Configs are deterministic — config_id always maps to the same setting.
"""

_DATASETS = ["MNIST", "FashionMNIST", "GTSRB"]

# ── Tier 1: original shallow configs ────────────────────────────────────────

_CONV_CHANNELS_T1 = [
    [16, 32],
    [32, 64],
    [16, 32, 64],
    [32, 64, 128],
    [64, 128],
]

_FC_DIMS_T1 = [
    [64],
    [128],
    [256],
    [512],
    [128, 64],
    [256, 128],
    [64, 32],
    [512, 256],
    [128, 128],
    [256, 64],
]

# ── Tier 2: deeper/wider configs ────────────────────────────────────────────
# Covers 4–6 conv layers and large FC layers 

_CONV_CHANNELS_T2 = [
    [32, 32, 64, 64],               # 4-layer VGG-pair
    [32, 32, 64, 64, 128, 128],     # 6-layer VGG-pair (matches TrojanNet)
    [64, 64, 128, 128],             # 4-layer large
    [16, 16, 32, 32],               # 4-layer small
    [64, 64, 128, 256],             # 4-layer growing
    [32, 64, 128, 256],             # 4-layer wide growth
    [16, 32, 64, 128, 256],         # 5-layer progressive
    [32, 64, 64, 128, 128],         # 5-layer mixed
    [64, 128, 256],                 # 3-layer large
    [32, 32, 32, 64, 64, 128],      # 6-layer asymmetric
]

_FC_DIMS_T2 = [
    [512, 512],         # matches TrojanNet classifier
    [1024],             # large single
    [1024, 512],        # large tapering
    [256, 256],         # medium pair
    [512, 256, 128],    # three-layer tapering
    [128, 256],         # growing
    [1024, 256],        # large to small
    [512, 128],         # big jump
    [256, 128, 64],     # three-layer small
    [64, 64],           # small pair
]

_KERNEL_SIZES = [3, 4, 5]
_OPTIMIZERS   = ["adam", "sgd"]
_LRS          = [1e-3, 5e-4, 1e-2]
_EPOCHS       = [10, 20, 50]


def get_config(config_id: int) -> dict:
    """
    Deterministically maps config_id (0–299) to a training configuration.

      Tier 1 (0–149):   dataset = _DATASETS[config_id // 50],  base_id = config_id % 50
      Tier 2 (150–299): dataset = _DATASETS[(config_id - 150) // 50], base_id = (config_id - 150) % 50
    """
    assert 0 <= config_id < 300, "config_id must be 0–299"

    if config_id < 150:
        dataset  = _DATASETS[config_id // 50]
        base_id  = config_id % 50
        conv_channels = _CONV_CHANNELS_T1[base_id % len(_CONV_CHANNELS_T1)]
        fc_dims       = _FC_DIMS_T1[base_id % len(_FC_DIMS_T1)]
    else:
        ext_id   = config_id - 150
        dataset  = _DATASETS[ext_id // 50]
        base_id  = ext_id % 50
        conv_channels = _CONV_CHANNELS_T2[base_id % len(_CONV_CHANNELS_T2)]
        fc_dims       = _FC_DIMS_T2[base_id % len(_FC_DIMS_T2)]

    kernel_size = _KERNEL_SIZES[(base_id // max(len(_CONV_CHANNELS_T1), len(_CONV_CHANNELS_T2))) % len(_KERNEL_SIZES)]
    optimizer   = _OPTIMIZERS[base_id % len(_OPTIMIZERS)]
    lr          = _LRS[(base_id // 2) % len(_LRS)]
    epochs      = _EPOCHS[(base_id // 6) % len(_EPOCHS)]
    seed        = config_id

    # Dataset overrides
    if dataset == "GTSRB":
        optimizer = "adam"
        lr        = 1e-3
    if dataset == "FashionMNIST":
        optimizer = "adam"
        lr        = 1e-3
    if dataset == "MNIST" and base_id in {22, 28}:
        lr = 1e-3
    # Tier 2 deep CNNs (4–6 layers) don't converge reliably with SGD on MNIST
    if config_id >= 150 and dataset == "MNIST":
        optimizer = "adam"
        lr        = 1e-3

    # Cap conv depth to prevent spatial collapse from MaxPool(2) stacking:
    #   MNIST/FashionMNIST: 28×28 → max 4 pools before hitting 1×1
    #   GTSRB:              32×32 → max 5 pools before hitting 1×1
    if dataset in ("MNIST", "FashionMNIST"):
        conv_channels = conv_channels[:4]
    elif dataset == "GTSRB":
        conv_channels = conv_channels[:5]

    return {
        "config_id":     config_id,
        "dataset":       dataset,
        "conv_channels": conv_channels,
        "kernel_size":   kernel_size,
        "fc_dims":       fc_dims,
        "optimizer":     optimizer,
        "lr":            lr,
        "epochs":        epochs,
        "seed":          seed,
    }


if __name__ == "__main__":
    print(f"{'ID':>3}  {'Dataset':>12}  {'Kern':>4}  {'Conv':>30}  {'FC':>20}  {'Opt':>4}  {'LR':>6}  {'Ep':>3}")
    print("-" * 105)
    for i in range(300):
        c = get_config(i)
        print(
            f"{c['config_id']:>3}  {c['dataset']:>12}  {c['kernel_size']:>4}  "
            f"{str(c['conv_channels']):>30}  {str(c['fc_dims']):>20}  "
            f"{c['optimizer']:>4}  {c['lr']:>6.4f}  {c['epochs']:>3}"
        )
