from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import load_eisfit_projects
from .diagnostics import export_diagnostics
from .topology_classifier import run_topology_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Leave-one-sample-out EIS topology classification")
    parser.add_argument("projects", nargs="+", type=Path, help=".eisfit JSON projects")
    parser.add_argument("--sample", action="append", required=True, metavar="PROJECT=SAMPLE", help="Explicit project-to-sample mapping; repeat once per project")
    parser.add_argument("--output", type=Path, default=Path("ml_results"))
    parser.add_argument("--metadata", action="store_true", help="Run spectrum plus voltage/current/time experiment")
    parser.add_argument("--both", action="store_true", help="Run raw and manually-cleaned spectrum experiments")
    parser.add_argument("--no-fit-requirement", action="store_true", help="Allow labelled cycles without saved fitted parameters")
    args = parser.parse_args(argv)
    mapping: dict[str, str] = {}
    for item in args.sample:
        if "=" not in item:
            parser.error("--sample must use PROJECT=SAMPLE")
        project, sample = item.split("=", 1)
        project_path = Path(project)
        mapping[str(project_path)] = sample
        mapping[str(project_path.resolve())] = sample
        mapping[project_path.name] = sample
    for project_path in args.projects:
        sample = mapping.get(str(project_path)) or mapping.get(project_path.name)
        if sample is not None:
            mapping[str(project_path)] = sample
            mapping[str(project_path.resolve())] = sample
    report = load_eisfit_projects(args.projects, mapping, require_fit=not args.no_fit_requirement)
    if not report.records:
        parser.error(f"No usable spectra were extracted; exclusions: {report.exclusion_counts}")
    export_diagnostics(report.records, args.output / "diagnostics")
    modes = ("raw", "cleaned") if args.both else ("cleaned" if args.metadata else "raw")
    for mode in modes:
        experiment = run_topology_experiment(report.records, use_metadata=False, spectrum_mode=mode)
        output = args.output / ("manually_cleaned" if mode == "cleaned" else "raw_spectrum")
        experiment.excluded = __import__("pandas").DataFrame(report.exclusions)
        experiment.save(output)
        print(f"\n{mode} spectrum results: {output}")
        print(experiment.fold_metrics.to_string(index=False))
    print(f"Extracted {len(report.records)} spectra; excluded {len(report.exclusions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
