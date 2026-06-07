import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .base_detector import DetectionResult


@dataclass
class ExperimentRun:
    timestamp: str
    config_path: str
    results: List[DetectionResult] = field(default_factory=list)

    def add(self, result: DetectionResult):
        self.results.append(result)

    def to_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": self.timestamp,
            "config_path": self.config_path,
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def to_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "attack_name", "model_architecture", "dataset",
            "method_name", "is_backdoor_detected", "confidence_score",
            "flagged_labels",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow({
                    "attack_name": r.attack_name,
                    "model_architecture": r.model_architecture,
                    "dataset": r.dataset,
                    "method_name": r.method_name,
                    "is_backdoor_detected": r.is_backdoor_detected,
                    "confidence_score": f"{r.confidence_score:.4f}",
                    "flagged_labels": str(r.flagged_labels),
                })

    def to_latex(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append(r"\begin{tabular}{llllcc}")
        lines.append(r"\toprule")
        lines.append(
            r"Attack & Architecture & Dataset & Detector & Detected & Confidence \\"
        )
        lines.append(r"\midrule")
        for r in self.results:
            detected = r"\cmark" if r.is_backdoor_detected else r"\xmark"
            lines.append(
                f"{r.attack_name} & {r.model_architecture} & {r.dataset} & "
                f"{r.method_name} & {detected} & {r.confidence_score:.3f} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def summary(self) -> str:
        if not self.results:
            return "No results collected."
        lines = [
            f"\n{'='*70}",
            f"Detection Suite Results — {self.timestamp}",
            f"{'='*70}",
        ]
        for r in self.results:
            status = "DETECTED" if r.is_backdoor_detected else "CLEAN"
            lines.append(
                f"  [{status:>8}] {r.attack_name}/{r.model_architecture}/{r.dataset} "
                f"— {r.method_name} (conf={r.confidence_score:.3f})"
            )
            if r.flagged_labels:
                lines.append(f"           Flagged labels: {r.flagged_labels}")
        lines.append(f"{'='*70}\n")
        text = "\n".join(lines)
        print(text)
        return text
