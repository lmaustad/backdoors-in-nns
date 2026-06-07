"""Smoke test: clean accuracy + Attack Success Rate (ASR) across all configured adapters.

For each attack/model entry in a Detection config, loads the backdoored model
through its adapter, evaluates clean top-1/top-5 accuracy on the test set,
and computes ASR using the adapter's own trigger-injection mechanism.

ASR modes (set via ``asr.mode`` in the YAML):
  - targeted:                    success = pred == asr.target_label
                                 (optionally skip samples already labelled as target).
  - untargeted:                  success = pred != ground_truth.
  - moving_offset:               success = pred == (ground_truth + asr.offset) % num_classes.
  - moving_offset_from_clean_pred:
                                 success = pred == (clean_pred + asr.offset) % num_classes
                                 where clean_pred is the model's prediction on the
                                 un-triggered sample (covers hiding-needles logit-roll).

Additional per-sample trigger injection:
  - asr.pass_label_as: <str>     inject the sample's true label as
                                 trigger_kwargs[<str>] before calling
                                 get_triggered_sample.  Needed by boone_bane,
                                 whose signed message encodes the sample's own
                                 label and the model then swaps its logits.

Usage:
    python -m Detection.tests.test_clean_accuracy_and_asr
    python -m Detection.tests.test_clean_accuracy_and_asr \
        --config Detection/configs/default.yaml --attack badnets_mnist \
        --max-batches 5 --asr-max-samples 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

import Detection.adapters  # noqa: F401  (trigger adapter registration)
from Detection.core.base_adapter import ModelAdapter
from Detection.core.data_provider import DataProvider
from Detection.core.registry import get_adapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean accuracy + ASR for all adapters")
    parser.add_argument("--config", type=str, default="Detection/configs/default.yaml")
    parser.add_argument("--attack", type=str, default=None, help="Only run this attack name")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | auto")
    parser.add_argument("--max-batches", type=int, default=100, help="Max batches for clean accuracy")
    parser.add_argument("--asr-max-samples", type=int, default=500, help="Max samples for ASR")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./Detection/results/smoke_clean_and_asr",
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
        return {"num_samples": 0.0, "top1": 0.0, "top5": 0.0}

    return {
        "num_samples": float(total),
        "top1": float(correct_top1 / total),
        "top5": float(correct_top5 / total),
    }


def evaluate_asr(
    adapter: ModelAdapter,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    asr_cfg: dict,
    num_classes: int,
    max_samples: int,
) -> Dict[str, Any]:
    mode = asr_cfg.get("mode", "targeted")
    trigger_kwargs = dict(asr_cfg.get("trigger_kwargs", {}) or {})
    pass_label_as = asr_cfg.get("pass_label_as")

    if mode == "targeted":
        if "target_label" not in asr_cfg:
            raise ValueError("asr.mode=targeted requires asr.target_label")
        target_label = int(asr_cfg["target_label"])
        exclude_target_class = bool(asr_cfg.get("exclude_target_class", True))
        offset = None
    elif mode == "untargeted":
        target_label = None
        exclude_target_class = False
        offset = None
    elif mode in ("moving_offset", "moving_offset_from_clean_pred"):
        if "offset" not in asr_cfg:
            raise ValueError(f"asr.mode={mode} requires asr.offset")
        offset = int(asr_cfg["offset"])
        target_label = None
        exclude_target_class = False
    else:
        raise ValueError(f"Unknown asr.mode={mode!r}")

    model.eval()

    evaluated = 0
    succeeded = 0
    skipped_no_trigger = 0
    skipped_already_target = 0

    with torch.no_grad():
        for batch in loader:
            if evaluated >= max_samples:
                break

            images, labels = extract_batch(batch)
            if labels is None:
                raise ValueError("Dataloader did not provide labels for ASR")

            for i in range(images.shape[0]):
                if evaluated >= max_samples:
                    break

                label = int(labels[i].item())
                if mode == "targeted" and exclude_target_class and label == target_label:
                    skipped_already_target += 1
                    continue

                sample_cpu = images[i].cpu()

                sample_trigger_kwargs = dict(trigger_kwargs)
                if pass_label_as:
                    sample_trigger_kwargs[pass_label_as] = label

                triggered = adapter.get_triggered_sample(
                    sample_cpu, **sample_trigger_kwargs
                )
                if triggered is None:
                    skipped_no_trigger += 1
                    continue

                if mode == "moving_offset_from_clean_pred":
                    clean_logits = model(sample_cpu.unsqueeze(0).to(device))
                    if isinstance(clean_logits, (tuple, list)):
                        clean_logits = clean_logits[0]
                    clean_pred = int(clean_logits.argmax(dim=1).item())

                with adapter.trigger_mode(model):
                    pred = adapter.predict_triggered(model, triggered, device)

                if mode == "targeted":
                    success = pred == target_label
                elif mode == "untargeted":
                    success = pred != label
                elif mode == "moving_offset":
                    success = pred == ((label + offset) % num_classes)
                else:  # moving_offset_from_clean_pred
                    success = pred == ((clean_pred + offset) % num_classes)

                succeeded += int(success)
                evaluated += 1

    result: Dict[str, Any] = {
        "asr_mode": mode,
        "asr_num_samples": evaluated,
        "asr_num_skipped_no_trigger": skipped_no_trigger,
        "asr_num_skipped_already_target": skipped_already_target,
    }
    if evaluated == 0:
        result["asr"] = "n/a"
        result["asr_status"] = (
            "no_triggered_samples" if skipped_no_trigger > 0 else "no_eligible_samples"
        )
    else:
        result["asr"] = float(succeeded / evaluated)
        result["asr_status"] = "ok"
    return result


def _serialize(value: Any) -> Union[str, float, int]:
    if isinstance(value, float) and (value != value):  # NaN guard
        return "n/a"
    return value


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
        asr_cfg = attack_cfg.get("asr")
        AdapterClass = get_adapter(adapter_name)

        for model_cfg in attack_cfg.get("models", []):
            model_type = model_cfg.get("model_type", "default")
            dataset = model_cfg["dataset"]
            checkpoint = model_cfg["backdoor_checkpoint"]

            row: Dict[str, Any] = {
                "attack": attack_name,
                "adapter": adapter_name,
                "model_type": model_type,
                "dataset": dataset,
                "checkpoint": checkpoint,
                "status": "ok",
                "error": "",
                "clean_num_samples": 0.0,
                "clean_top1": 0.0,
                "clean_top5": 0.0,
                "asr_mode": "",
                "asr_num_samples": 0,
                "asr_num_skipped_no_trigger": 0,
                "asr_num_skipped_already_target": 0,
                "asr": "n/a",
                "asr_status": "no_asr_config",
            }

            logger.info("Evaluating %s/%s/%s", attack_name, model_type, dataset)

            try:
                adapter_kwargs = {
                    k: v for k, v in model_cfg.items() if k != "backdoor_checkpoint"
                }
                adapter = AdapterClass(**adapter_kwargs)
                model = adapter.load_model(checkpoint, device=device)
                model_info = adapter.get_model_info()

                custom_transform = (
                    model_info.extra.get("custom_transform")
                    if model_info.extra
                    else None
                )
                loader = data_provider.get_test_loader(
                    dataset, custom_transform=custom_transform
                )

                clean = evaluate_clean_accuracy(
                    model=model,
                    loader=loader,
                    device=device,
                    max_batches=args.max_batches,
                )
                row["clean_num_samples"] = clean["num_samples"]
                row["clean_top1"] = clean["top1"]
                row["clean_top5"] = clean["top5"]

                if asr_cfg is not None:
                    asr_loader = data_provider.get_test_loader(
                        dataset, custom_transform=custom_transform
                    )
                    asr_res = evaluate_asr(
                        adapter=adapter,
                        model=model,
                        loader=asr_loader,
                        device=device,
                        asr_cfg=asr_cfg,
                        num_classes=model_info.num_classes,
                        max_samples=args.asr_max_samples,
                    )
                    row.update(asr_res)

                logger.info(
                    "  -> clean_top1=%.4f clean_top5=%.4f (%d samples)  asr=%s (%d samples, %s)",
                    row["clean_top1"],
                    row["clean_top5"],
                    int(row["clean_num_samples"]),
                    row["asr"] if isinstance(row["asr"], str) else f"{row['asr']:.4f}",
                    int(row["asr_num_samples"]),
                    row["asr_status"],
                )

            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
                logger.warning("  -> failed: %s", exc)

            results.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"clean_accuracy_and_asr_{stamp}.json"
    csv_path = output_dir / f"clean_accuracy_and_asr_{stamp}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "attack",
        "adapter",
        "model_type",
        "dataset",
        "checkpoint",
        "status",
        "error",
        "clean_num_samples",
        "clean_top1",
        "clean_top5",
        "asr_mode",
        "asr_num_samples",
        "asr_num_skipped_no_trigger",
        "asr_num_skipped_already_target",
        "asr",
        "asr_status",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: _serialize(row.get(k, "")) for k in fieldnames})

    ok = sum(1 for r in results if r["status"] == "ok")
    err = len(results) - ok
    logger.info("Done. %d succeeded, %d failed", ok, err)
    logger.info("JSON report: %s", json_path)
    logger.info("CSV report:  %s", csv_path)


if __name__ == "__main__":
    main()
