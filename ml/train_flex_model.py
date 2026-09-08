"""Train the single-sample Flex ML bundle used by the eisyFIT GUI.

Flex is intentionally trained from the supplied 181 project.  It is a
deployment artifact for suggestions on related spectra, not a benchmark:
there is no independent physical sample available for validation here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .dataset import load_eisfit_projects
from .number_aware_pipeline import train_bundle


DEFAULT_SOURCE = Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\Ti overlayer\181.eisfit.json.gz")
DEFAULT_OUTPUT = Path("ml/analysis/number_aware_pipeline_flex_181")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=list), encoding="utf-8")
    temporary.replace(path)


def train_flex(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT, *, seed: int = 42) -> dict:
    source = Path(source).resolve()
    sample_id = source.name.split(".eisfit.json", 1)[0] or source.stem
    sample_ids = {str(source): sample_id, str(source.resolve()): sample_id}
    extraction = load_eisfit_projects(
        [source], sample_ids, require_fit=True, require_frequency_window=True
    )
    if not extraction.records:
        raise ValueError(f"no labelled training spectra were extracted from {source}")

    bundle, _ = train_bundle([source], sample_ids, seed, allow_single_sample=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "pipeline.joblib.tmp"
    joblib.dump(bundle, temporary)
    temporary.replace(output / "pipeline.joblib")
    report = {
        "model_name": "Flex",
        "training_projects": [str(source)],
        "training_samples": list(bundle.training_samples),
        "training_records": len(extraction.records),
        "training_exclusions": extraction.exclusion_counts,
        "circuit_classes": list(bundle.circuit_classes),
        "topology_classes": list(bundle.topology_classes),
        "parameter_models": len(bundle.parameter_models),
        "validation_policy": "single physical sample; no independent validation",
        "seed": seed,
        "artifact": str(output / "pipeline.joblib"),
    }
    _write_json(output / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(train_flex(args.source, args.output, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
