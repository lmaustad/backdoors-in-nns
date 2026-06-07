"""Export pretrained torchvision ResNet checkpoints.

Example:
	python BaselineModels/resnet.py
	python BaselineModels/resnet.py --models resnet50,resnet152

Outputs:
	/ResNet/checkpoints/<model>_imagenet1k_v1.pth
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import torch
import torchvision.models as tv_models


MODEL_SPECS = {
	"resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights.IMAGENET1K_V1),
	"resnet34": (tv_models.resnet34, tv_models.ResNet34_Weights.IMAGENET1K_V1),
	"resnet50": (tv_models.resnet50, tv_models.ResNet50_Weights.IMAGENET1K_V1),
	"resnet101": (tv_models.resnet101, tv_models.ResNet101_Weights.IMAGENET1K_V1),
	"resnet152": (tv_models.resnet152, tv_models.ResNet152_Weights.IMAGENET1K_V1),
}


def parse_models(raw: str) -> Iterable[str]:
	names = [name.strip().lower() for name in raw.split(",") if name.strip()]
	unknown = [name for name in names if name not in MODEL_SPECS]
	if unknown:
		raise ValueError(
			f"Unknown model(s): {unknown}. Available: {list(MODEL_SPECS.keys())}"
		)
	return names


def export_pretrained_resnets(models: Iterable[str], output_dir: Path) -> Dict[str, str]:
	output_dir.mkdir(parents=True, exist_ok=True)
	saved = {}

	for model_name in models:
		builder, weights = MODEL_SPECS[model_name]
		print(f"[INFO] Loading pretrained {model_name} ({weights})")
		model = builder(weights=weights)
		model.eval()

		checkpoint_path = output_dir / f"{model_name}_imagenet1k_v1.pth"
		torch.save(model.state_dict(), checkpoint_path)
		saved[model_name] = str(checkpoint_path)
		print(f"[OK] Saved {model_name} -> {checkpoint_path}")

	return saved


def main() -> None:
	parser = argparse.ArgumentParser(description="Export pretrained torchvision ResNets")
	parser.add_argument(
		"--models",
		type=str,
		default="resnet18,resnet34,resnet50,resnet101,resnet152",
		help="Comma-separated model list",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="./ResNet/checkpoints",
		help="Directory for saved .pth state_dict files",
	)
	args = parser.parse_args()

	model_names = parse_models(args.models)
	output_dir = Path(args.output_dir)

	saved = export_pretrained_resnets(model_names, output_dir)

	manifest = {
		"created_at": datetime.utcnow().isoformat() + "Z",
		"format": "torch_state_dict",
		"source": "torchvision pretrained IMAGENET1K_V1",
		"models": saved,
	}
	manifest_path = output_dir / "manifest.json"
	manifest_path.write_text(json.dumps(manifest, indent=2))
	print(f"[OK] Wrote manifest -> {manifest_path}")


if __name__ == "__main__":
	main()

