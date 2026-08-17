from __future__ import annotations

import argparse
import os
from pathlib import Path

# Execution-only setting for joblib used by RandomForestClassifier. It avoids
# physical-core autodetection failures in constrained environments; model
# hyperparameters and random state remain unchanged.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

from .dataset import load_eisfit_projects
from .pipeline import run_frequency_topology_pipeline, save_pipeline_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LOSO ML frequency range -> existing outliers -> EEC topology")
    parser.add_argument("projects", nargs="+", type=Path)
    parser.add_argument("--sample", action="append", required=True, metavar="PROJECT=SAMPLE")
    parser.add_argument("--output", type=Path, default=Path("ml_frequency_topology_results"))
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outlier-threshold", type=float, default=1.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("ml_outlier_cache_181_159_140_129_150_157"))
    parser.add_argument("--workers", type=int, default=1, help="Bayes-DRT2 workers; default 1 for conservative reproducibility")
    args = parser.parse_args(argv)
    mapping: dict[str, str] = {}
    for item in args.sample:
        project, sample = item.split("=", 1) if "=" in item else ("", "")
        if not project or not sample:
            parser.error("--sample must use PROJECT=SAMPLE")
        p = Path(project)
        mapping.update({p.name: sample, str(p): sample, str(p.resolve()): sample})
    for project_path in args.projects:
        sample = mapping.get(project_path.name)
        if sample is not None:
            mapping.update({str(project_path): sample, str(project_path.resolve()): sample})
    report = load_eisfit_projects(args.projects, mapping)
    if not report.records:
        parser.error(f"No usable spectra: {report.exclusion_counts}")
    predictions, metrics, exclusions, cache_report = run_frequency_topology_pipeline(
        report.records, grid_size=args.grid_size, seed=args.seed, outlier_threshold=args.outlier_threshold,
        cache_dir=args.cache_dir, workers=args.workers,
    )
    save_pipeline_results(predictions, metrics, exclusions, args.output)
    (args.output / "cache_report.json").write_text(__import__("json").dumps(cache_report, indent=2), encoding="utf-8")
    report_frame = metrics[metrics["held_out_sample"].isna()] if "held_out_sample" in metrics else metrics
    print(report_frame.to_string(index=False))
    print(f"Extracted {len(report.records)} spectra; input exclusions {len(report.exclusions)}; preprocessing exclusions {len(exclusions)}")
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
