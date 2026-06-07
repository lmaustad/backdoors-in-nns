"""Report generation utilities for the detection suite."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_full_report(experiment_run, output_dir: str):
    """Generate all output files from an experiment run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Structured outputs
    experiment_run.to_json(output_dir / "results.json")
    experiment_run.to_csv(output_dir / "results.csv")
    experiment_run.to_latex(output_dir / "results.tex")

    # Per-detector detailed results
    details_dir = output_dir / "details"
    details_dir.mkdir(exist_ok=True)

    for result in experiment_run.results:
        filename = f"{result.attack_name}_{result.model_architecture}_{result.dataset}_{result.method_name}.json"
        # Sanitize filename
        filename = filename.replace("/", "_").replace(" ", "_")
        with open(details_dir / filename, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    # Summary to stdout
    experiment_run.summary()

    logger.info(f"Full report generated in {output_dir}")
