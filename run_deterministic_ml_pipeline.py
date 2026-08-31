"""Run deterministic outlier selection, ML prediction, EEC fit, and refine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eis_gui import EISApplication
from ml.number_aware_pipeline import infer_bundle_records, load_pipeline_bundle
from ml.point_validity import detect_outliers_in_active_points
from ml.runtime_inference import make_runtime_spectrum
from ml.results_schema import write_ml_results

from apply_ml_results_and_fit import run as apply_results_and_fit


DEFAULT_MODEL = (
    Path(__file__).resolve().parent
    / "ml"
    / "analysis"
    / "number_aware_pipeline_455"
    / "pipeline.joblib"
)


def run(
    project: Path,
    *,
    model: Path = DEFAULT_MODEL,
    results_path: Path | None = None,
    report_path: Path = Path("ml_pipeline_fit_report.json"),
    outlier_threshold: float = 3.0,
    refine_z_threshold: float = 3.5,
    refine_iterations: int = 5,
) -> dict:
    """Run the complete ordered pipeline on every spectrum in a project."""
    if not model.exists():
        raise FileNotFoundError(f"ML model bundle not found: {model}")
    if outlier_threshold <= 0 or refine_z_threshold <= 0 or refine_iterations < 1:
        raise ValueError("thresholds must be positive and iterations must be at least 1")

    restored = EISApplication._load_saved_project(project)
    targets = []
    deterministic_removed = 0
    for dataset_id, loaded, state in restored:
        for spectrum in loaded.spectra:
            cycle = state.cycles[int(spectrum.cycle)]
            indices, _diagnostics = detect_outliers_in_active_points(
                cycle.frequency_hz,
                cycle.impedance,
                cycle.included,
                threshold=outlier_threshold,
            )
            cycle.apply_outliers(indices)
            deterministic_removed += int(len(indices))
            key = f"{dataset_id}::{state.control}::{cycle.cycle}"
            targets.append(make_runtime_spectrum(key, cycle, state.circuit))

    if not targets:
        raise ValueError("project contains no spectra")
    bundle = load_pipeline_bundle(model)
    predictions = infer_bundle_records(
        bundle,
        [target.record for target in targets],
        threshold=outlier_threshold,
    )
    if results_path is None:
        results_path = project.with_name(f"{project.stem}_ml_results.json")
    write_ml_results(
        results_path,
        predictions,
        pipeline={
            "name": "Sputtered cathode",
            "model": str(model),
            "actions": ["deterministic_outliers", "frequency", "model", "initial_parameters", "fit", "refine"],
            "outlier_threshold": outlier_threshold,
            "refine_z_threshold": refine_z_threshold,
            "refine_max_iterations": refine_iterations,
        },
    )
    report = apply_results_and_fit(
        project,
        results_path,
        report_path,
        preserve_existing_selection=True,
        refine_z_threshold=refine_z_threshold,
        refine_iterations=refine_iterations,
    )
    report["deterministic_outlier_threshold"] = outlier_threshold
    report["deterministic_points_removed_before_ml"] = deterministic_removed
    report["ml_model"] = str(model)
    # apply_results_and_fit records the standard refinement settings in each
    # spectrum; retain the CLI values in the top-level report as well.
    report["refine_z_threshold"] = refine_z_threshold
    report["refine_max_iterations"] = refine_iterations
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--results", type=Path, help="optional ML sidecar output")
    parser.add_argument("--report", type=Path, default=Path("ml_pipeline_fit_report.json"))
    parser.add_argument("--outlier-threshold", type=float, default=3.0)
    parser.add_argument("--refine-z-threshold", type=float, default=3.5)
    parser.add_argument("--refine-iterations", type=int, default=5)
    args = parser.parse_args()
    report = run(
        args.project,
        model=args.model,
        results_path=args.results,
        report_path=args.report,
        outlier_threshold=args.outlier_threshold,
        refine_z_threshold=args.refine_z_threshold,
        refine_iterations=args.refine_iterations,
    )
    print(
        f"processed={report['spectra']} fit={report['fit_count']} "
        f"refinement={report['refinement_count']} failures={report['failure_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
