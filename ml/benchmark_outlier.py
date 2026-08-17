"""Benchmark the unchanged production Bayes-DRT2 outlier path."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

import numpy as np

from eis_model import CycleState
from eis_services import analyze_outliers, circuit_parameters
from .dataset import load_eisfit_projects


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("projects", nargs="+", type=Path)
    parser.add_argument("--sample", action="append", required=True, metavar="PROJECT=SAMPLE")
    parser.add_argument("--output", type=Path, default=Path("ml_outlier_benchmark.json"))
    args = parser.parse_args(argv)
    mapping = {}
    for value in args.sample:
        project, sample = value.split("=", 1)
        mapping[Path(project).name] = sample
    mapping.update({str(p): mapping[p.name] for p in args.projects})
    records = load_eisfit_projects(args.projects, mapping).records
    results = []
    for count in (10, 50, 100):
        selected = records[:min(count, len(records))]
        timings = []
        points = []
        widths = []
        started = time.perf_counter()
        for record in selected:
            state = CycleState(cycle=record.cycle, frequency_hz=record.frequency.copy(), impedance=record.impedance.copy(),
                               potential_v=float(record.voltage or 0.0), current_ma=float(record.current or 0.0),
                               time_s=record.time, frequency_window=(record.manual_f_min, record.manual_f_max),
                               circuit=record.original_eec_topology)
            call_started = time.perf_counter()
            analyze_outliers(state, 1.0, circuit_parameters(record.original_eec_topology))
            timings.append(time.perf_counter() - call_started)
            points.append(len(record.frequency))
            widths.append(np.log10(record.manual_f_max / record.manual_f_min))
        results.append({"n": len(selected), "total_s": time.perf_counter() - started,
                        "mean_s": float(np.mean(timings)), "median_s": float(np.median(timings)),
                        "min_s": float(np.min(timings)), "max_s": float(np.max(timings)),
                        "mean_points": float(np.mean(points)),
                        "correlation_points_s": float(np.corrcoef(points, timings)[0, 1]) if np.std(points) else None,
                        "correlation_log_range_width_s": float(np.corrcoef(widths, timings)[0, 1]) if np.std(widths) else None})
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
