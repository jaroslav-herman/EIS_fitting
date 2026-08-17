from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import load_eisfit_projects
from .staged_topology import stage1_frequency_predictions, stage2_topology_datasets


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one resumable topology-experiment stage")
    parser.add_argument("--stage", choices=("1", "2"), required=True)
    parser.add_argument("--source", type=Path, default=Path("ml_frequency_range_181_159_140_129_150_157"))
    parser.add_argument("--output", type=Path, default=Path("ml/cache/frequency_predictions"))
    parser.add_argument("--frequency-cache", type=Path, default=Path("ml/cache/frequency_predictions"))
    parser.add_argument("--outlier-cache", type=Path, default=Path("ml_outlier_cache_181_159_140_129_150_157"))
    parser.add_argument("--dataset-output", type=Path, default=Path("ml/cache/topology_datasets"))
    parser.add_argument("--sample", action="append", default=[], metavar="PROJECT=SAMPLE")
    parser.add_argument("projects", nargs="*", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "1":
        report = stage1_frequency_predictions(args.source, args.output)
        print(report)
        return 0
    mapping = {}
    for item in args.sample:
        project, sample = item.split("=", 1); mapping[Path(project).name] = sample
    mapping.update({str(p): mapping[p.name] for p in args.projects})
    report = load_eisfit_projects(args.projects, mapping)
    result = stage2_topology_datasets(report.records, args.frequency_cache, args.outlier_cache, args.dataset_output)
    print(result)
    return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
