"""Smoke test: clean accuracy across all configured adapters/datasets.

Runs each attack/model entry from a Detection config, loads the backdoored model
through its adapter, evaluates clean accuracy on the corresponding dataset, and
writes a compact summary report.

Usage:
    python -m Detection.tests.test_clean_accuracy_all_adapters
    python -m Detection.tests.test_clean_accuracy_all_adapters --config Detection/configs/default.yaml --max-batches 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

import Detection.adapters  # noqa: F401  (trigger adapter registration)
from Detection.core.data_provider import DataProvider
from Detection.core.registry import get_adapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test clean accuracy for all adapters/datasets")
    parser.add_argument("--config", type=str, default="Detection/configs/default.yaml")
    parser.add_argument("--attack", type=str, default=None, help="Only run this attack name")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | auto (default from config)")
    parser.add_argument("--max-batches", type=int, default=100, help="Max test batches per model")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./Detection/results/smoke_clean_accuracy",
        help="Directory to write JSON/CSV report",
    )
    return parser.parse_args()


def resolve_device(config: dict, cli_device: Optional[str]) -> torch.device:
    if cli_device:
        if cli_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(cli_device)

    cfg_device = config.get("global", {}).get("device", "auto")
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def extract_batch(batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, dict):
        images = batch.get("x", batch.get("image"))
        labels = batch.get("y", batch.get("label"))
    elif isinstance(batch, (tuple, list)):
        images = batch[0]
        labels = batch[1] if len(batch) > 1 else None
    else:
        images, labels = batch, None

    if images is None:
        raise ValueError("Could not extract images from batch")
    return images, labels


def evaluate_clean_accuracy(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            images, labels = extract_batch(batch)
            if labels is None:
                raise ValueError("Dataloader did not provide labels")

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            preds = torch.argmax(logits, dim=1)
            correct_top1 += (preds == labels).sum().item()

            k = min(5, logits.shape[1])
            topk = torch.topk(logits, k=k, dim=1).indices
            correct_top5 += topk.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    if total == 0:
        return {
            "num_samples": 0,
            "top1": 0.0,
            "top5": 0.0,
        }

    return {
        "num_samples": float(total),
        "top1": float(correct_top1 / total),
        "top5": float(correct_top5 / total),
    }


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = resolve_device(config, args.device)
    logger.info("Using device: %s", device)

    seed = config.get("global", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_cfg = dict(config.get("data", {}))
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers

    data_provider = DataProvider(data_cfg)

    attacks = config.get("attacks", [])
    if args.attack:
        attacks = [a for a in attacks if a.get("name") == args.attack]

    results: List[dict] = []

    for attack_cfg in attacks:
        attack_name = attack_cfg["name"]
        adapter_name = attack_cfg["adapter"]
        AdapterClass = get_adapter(adapter_name)

        for model_cfg in attack_cfg.get("models", []):
            model_type = model_cfg.get("model_type", "default")
            dataset = model_cfg["dataset"]
            checkpoint = model_cfg["backdoor_checkpoint"]
            variant = "clean" if model_cfg.get("is_backdoored") is False else "backdoor"

            row = {
                "attack": attack_name,
                "adapter": adapter_name,
                "model_type": model_type,
                "dataset": dataset,
                "variant": variant,
                "checkpoint": checkpoint,
                "status": "ok",
                "error": "",
                "num_samples": 0.0,
                "top1": 0.0,
                "top5": 0.0,
            }

            logger.info("Evaluating %s/%s/%s", attack_name, model_type, dataset)

            try:
                adapter_kwargs = {
                    k: v for k, v in model_cfg.items()
                    if k != "backdoor_checkpoint"
                }
                adapter = AdapterClass(**adapter_kwargs)
                model = adapter.load_model(checkpoint, device=device)
                model_info = adapter.get_model_info()

                custom_transform = model_info.extra.get("custom_transform") if model_info.extra else None
                loader = data_provider.get_test_loader(dataset, custom_transform=custom_transform)

                metrics = evaluate_clean_accuracy(
                    model=model,
                    loader=loader,
                    device=device,
                    max_batches=args.max_batches,
                )
                row.update(metrics)

                logger.info(
                    "  -> top1=%.4f top5=%.4f (%d samples)",
                    row["top1"],
                    row["top5"],
                    int(row["num_samples"]),
                )

            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
                logger.warning("  -> failed: %s", exc)

            results.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"clean_accuracy_smoke_{stamp}.json"
    csv_path = output_dir / f"clean_accuracy_smoke_{stamp}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "attack",
                "adapter",
                "model_type",
                "dataset",
                "variant",
                "checkpoint",
                "status",
                "error",
                "num_samples",
                "top1",
                "top5",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    ok = sum(1 for r in results if r["status"] == "ok")
    err = len(results) - ok
    logger.info("Done. %d succeeded, %d failed", ok, err)
    logger.info("JSON report: %s", json_path)
    logger.info("CSV report:  %s", csv_path)


if __name__ == "__main__":
    main()
