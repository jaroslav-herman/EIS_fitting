"""Split a legacy project containing embedded ML results into two files."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .results_schema import ML_RESULTS_FORMAT, ML_RESULTS_VERSION, spectrum_identifier


def split_project(input_path: Path, project_path: Path | None = None, results_path: Path | None = None):
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("format") != "eis-fitting-project":
        raise ValueError("input is not a normal EIS-fitting project")
    embedded = payload.get("ml_results")
    if not isinstance(embedded, dict) or not isinstance(embedded.get("spectra"), list):
        raise ValueError("project contains no embedded ML results")

    normal = copy.deepcopy(payload)
    normal.pop("ml_results", None)
    project_path = project_path or input_path.with_name(
        input_path.name.replace("_ML_initial_parameters", "")
    )
    results_path = results_path or input_path.with_name(
        input_path.name.replace(".eisfit.json", "_ml_results.json")
    )
    source_project = str(payload.get("source_path") or "")
    spectra = []
    for item in embedded["spectra"]:
        if not isinstance(item, dict):
            continue
        result = copy.deepcopy(item)
        result["spectrum_key"] = spectrum_identifier(
            result.get("frequency", []),
            result.get("z_real", []),
            result.get("z_imag", []),
            int(result.get("cycle", 0)),
            str(result.get("control") or "working"),
        )
        # The normal project remains the owner of the measured arrays.
        for key in ("frequency", "z_real", "z_imag"):
            result.pop(key, None)
        spectra.append(result)
    results = {
        "format": ML_RESULTS_FORMAT,
        "version": ML_RESULTS_VERSION,
        "source_project": str(project_path),
        "source_data": source_project,
        "source_project_format": payload.get("format"),
        "pipeline": {
            key: embedded.get(key)
            for key in (
                "schema_version", "model_version", "pipeline_version",
                "preprocessing_version", "training_dataset_version", "generated_at",
                "inference_sample", "training_samples", "frequency_model",
                "topology_model", "parameter_models", "drt_ridge",
            )
            if key in embedded
        },
        "spectra": spectra,
    }
    project_path.write_text(json.dumps(normal, indent=2), encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return project_path, results_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    project_path, results_path = split_project(args.input, args.project, args.results)
    print(f"project: {project_path}")
    print(f"ml results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
