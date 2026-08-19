"""Benchmark the local point-validity detector on synthetic spectra."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from .point_validity import detect_valid_points


def benchmark(spectra: int = 200, points: int = 121, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    frequency = np.logspace(5, -1, points)
    started = time.perf_counter()
    total_points = 0
    rejected = 0
    for index in range(spectra):
        x = np.log10(frequency)
        scale = 10.0 ** rng.uniform(-2.0, 2.0)
        impedance = scale * (2.0 + 0.5 / (1.0 + np.exp(x - rng.uniform(0.5, 3.5))))
        impedance = impedance - 1j * scale * (0.2 + 0.1 * np.sin(x))
        impedance += rng.normal(0, scale * 0.002, points) + 1j * rng.normal(0, scale * 0.002, points)
        mask, _, _ = detect_valid_points(frequency, impedance, threshold=4.0)
        total_points += points; rejected += int((~mask).sum())
    runtime = time.perf_counter() - started
    report = {"spectra_processed": spectra, "total_points": total_points, "total_runtime_s": runtime,
              "milliseconds_per_spectrum": 1000.0 * runtime / spectra, "points_per_second": total_points / runtime,
              "rejected_points": rejected}
    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra", type=int, default=200)
    parser.add_argument("--points", type=int, default=121)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(args.spectra, args.points, args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import json
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
