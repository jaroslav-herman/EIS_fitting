from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .dataset import load_eisfit_projects
from .diagnostics import export_diagnostics
from .frequency_range import run_frequency_range_experiment
from .frequency_range_visualization import plot_frequency_range_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict manually selected EIS frequency limits")
    parser.add_argument("projects", nargs="+", type=Path)
    parser.add_argument("--sample", action="append", required=True, metavar="PROJECT=SAMPLE")
    parser.add_argument("--output", type=Path, default=Path("ml_frequency_range_results"))
    args = parser.parse_args(argv)
    mapping: dict[str, str] = {}
    for item in args.sample:
        if "=" not in item:
            parser.error("--sample must use PROJECT=SAMPLE")
        project, sample = item.split("=", 1)
        project_path = Path(project)
        mapping[project_path.name] = sample
        mapping[str(project_path)] = sample
        mapping[str(project_path.resolve())] = sample
    for project_path in args.projects:
        sample = mapping.get(str(project_path)) or mapping.get(project_path.name)
        if sample is not None:
            mapping[str(project_path)] = sample
            mapping[str(project_path.resolve())] = sample
    report = load_eisfit_projects(args.projects, mapping)
    if not report.records:
        parser.error(f"No usable spectra: {report.exclusion_counts}")
    args.output.mkdir(parents=True, exist_ok=True)
    export_diagnostics(report.records, args.output / "diagnostics")
    results = []
    for mode in ("voltage_only", "spectrum_only", "spectrum_plus_voltage"):
        experiment = run_frequency_range_experiment(report.records, feature_mode=mode)
        experiment.excluded = pd.DataFrame(report.exclusions)
        output = args.output / mode
        experiment.save(output)
        plot_frequency_range_results(report.records, experiment.predictions, output / "plots")
        results.append(experiment.overall_metrics.assign(feature_mode=mode))
        print(f"\n{mode}")
        print(experiment.overall_metrics.to_string(index=False))
    pd.concat(results, ignore_index=True).to_csv(args.output / "model_comparison.csv", index=False)
    print(f"\nExtracted {len(report.records)} spectra; excluded {len(report.exclusions)}")
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
