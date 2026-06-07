#!/usr/bin/env python3
"""Backdoor Detection Suite — CLI entry point.

Usage:
    python -m Detection.run --config Detection/configs/default.yaml
    python -m Detection.run --config Detection/configs/default.yaml --attack dfba
    python -m Detection.run --config Detection/configs/default.yaml --detector weight_forensics
    python -m Detection.run --config Detection/configs/default.yaml --list-adapters
    python -m Detection.run --config Detection/configs/default.yaml --list-detectors
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

# Import adapters and detectors to trigger registration
import Detection.adapters  # noqa: F401
import Detection.detectors  # noqa: F401
from Detection.core.data_provider import DataProvider
from Detection.core.registry import get_adapter, get_detector, list_adapters, list_detectors
from Detection.core.results import ExperimentRun

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backdoor Detection Suite")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--attack", type=str, default=None, help="Run only this attack")
    parser.add_argument("--detector", type=str, default=None, help="Run only this detector")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--list-adapters", action="store_true", help="List available adapters")
    parser.add_argument("--list-detectors", action="store_true", help="List available detectors")
    args = parser.parse_args()

    if args.list_adapters:
        print("Available adapters:", list_adapters())
        return
    if args.list_detectors:
        print("Available detectors:", list_detectors())
        return

    if args.config is None:
        parser.error("--config is required when running detection")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif config["global"].get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["global"]["device"])
    logger.info(f"Using device: {device}")

    # Seed
    seed = config["global"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Output
    output_dir = Path(args.output_dir or config["global"].get("output_dir", "./results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data provider
    data_provider = DataProvider(config.get("data", {}))

    # Filter attacks and detectors
    attacks_to_run = config.get("attacks", [])
    if args.attack:
        attacks_to_run = [a for a in attacks_to_run if a["name"] == args.attack]

    detectors_to_run = config.get("detectors", [])
    if args.detector:
        detectors_to_run = [d for d in detectors_to_run if d["name"] == args.detector]

    experiment = ExperimentRun(
        timestamp=datetime.now().isoformat(),
        config_path=args.config,
    )

    for attack_cfg in attacks_to_run:
        adapter_name = attack_cfg["adapter"]
        AdapterClass = get_adapter(adapter_name)

        for model_cfg in attack_cfg.get("models", []):
            model_type = model_cfg.get("model_type", "default")
            dataset = model_cfg["dataset"]
            bd_path = model_cfg["backdoor_checkpoint"]

            logger.info(
                f"Loading {attack_cfg['name']}/{model_type}/{dataset}"
            )

            # Instantiate adapter with model-specific kwargs
            adapter_kwargs = {
                k: v for k, v in model_cfg.items()
                if k != "backdoor_checkpoint"
            }
            adapter = AdapterClass(**adapter_kwargs)

            # Load models
            bd_model = adapter.load_model(bd_path, device=device)
            model_info = adapter.get_model_info()
            model_info.extra["adapter"] = adapter
            model_info.extra["config_name"] = attack_cfg.get("config_name", attack_cfg["name"])

            # Get data loader
            # Adapters may store a required custom_transform in model_info.extra
            # (e.g. arch_backdoors expects Resize(70,70) + Normalize([-1,1])).
            # Pass it through so the model receives the correct input distribution.
            custom_transform = model_info.extra.get("custom_transform") if model_info.extra else None
            try:
                data_loader = data_provider.get_test_loader(dataset, custom_transform=custom_transform)
            except Exception as e:
                logger.warning(f"Could not load dataset '{dataset}': {e}")
                data_loader = None

            for det_cfg in detectors_to_run:
                DetectorClass = get_detector(det_cfg["name"])
                detector = DetectorClass(det_cfg.get("config", {}))

                if detector.requires_data() and data_loader is None:
                    logger.warning(
                        f"Skipping {det_cfg['name']} on {attack_cfg['name']}: "
                        f"no data available"
                    )
                    continue

                logger.info(
                    f"  Running {det_cfg['name']} on "
                    f"{attack_cfg['name']}/{model_type}/{dataset}"
                )

                result = detector.detect(
                    model=bd_model,
                    model_info=model_info,
                    data_loader=data_loader,
                    device=device,
                )
                experiment.add(result)

                status = "DETECTED" if result.is_backdoor_detected else "CLEAN"
                logger.info(
                    f"  -> [{status}] confidence={result.confidence_score:.3f}"
                )

    # Save results
    experiment.to_json(output_dir / "results.json")
    experiment.to_csv(output_dir / "results.csv")
    experiment.to_latex(output_dir / "results.tex")
    experiment.summary()

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
