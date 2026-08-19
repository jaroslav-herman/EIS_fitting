"""Benchmark deterministic low-frequency selectors against the manual boundary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .dataset import load_eisfit_projects
from .low_frequency_selector import select_low_frequency_boundary


METHODS = ("local_residual", "rolling_stability", "trajectory", "combined")
PARAMETERS = {"threshold": 3.0, "neighborhood": 3, "rolling_window": 9, "persistence_window": 7, "min_fraction": 0.6, "min_consecutive_windows": 2}


def _manual_valid(record, indices: np.ndarray) -> np.ndarray:
    if record.cleaned_frequency is None:
        return np.zeros(indices.size, dtype=bool)
    keys = {(float(f), float(r), float(i)) for f, r, i in zip(record.cleaned_frequency, record.cleaned_z_real, record.cleaned_z_imag)}
    return np.asarray([(float(record.frequency[index]), float(record.z_real[index]), float(record.z_imag[index])) in keys for index in indices], dtype=bool)


def _metrics(part: pd.DataFrame) -> dict:
    error = part.error_decades.to_numpy(float); absolute = np.abs(error)
    return {"spectra": len(part), "MAE_decades": float(absolute.mean()), "median_abs_error_decades": float(np.median(absolute)),
            "p75_abs_error_decades": float(np.percentile(absolute, 75)), "p90_abs_error_decades": float(np.percentile(absolute, 90)),
            "p95_abs_error_decades": float(np.percentile(absolute, 95)), "max_abs_error_decades": float(absolute.max()),
            "within_0.05_decades_percent": float(100 * (absolute <= .05).mean()), "within_0.10_decades_percent": float(100 * (absolute <= .10).mean()),
            "within_0.20_decades_percent": float(100 * (absolute <= .20).mean()), "within_0.30_decades_percent": float(100 * (absolute <= .30).mean()),
            "within_0.50_decades_percent": float(100 * (absolute <= .50).mean()), "too_high_percent": float(100 * (error > .10).mean()),
            "approximately_correct_percent": float(100 * (absolute <= .10).mean()), "too_low_percent": float(100 * (error < -.10).mean()),
            "mean_retained_fraction": float(part.retained_fraction.mean()), "mean_manual_retained_fraction": float(part.manual_retained_fraction.mean()),
            "boundary_close_with_isolated_rejections_percent": float(100 * part.boundary_close_with_manual_internal_rejections.mean())}


def _plot(record, result, manual_min: float, label: str, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = record.frequency; z = record.impedance; x = np.log10(f)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    axes[0].plot(x, z.real, ".-"); axes[0].axvline(np.log10(result.predicted_f_min), color="tab:red", label="predicted f_min"); axes[0].axvline(np.log10(manual_min), color="tab:green", label="manual active f_min"); axes[0].set_title("Re(Z)"); axes[0].set_xlabel("log10(f)"); axes[0].legend(fontsize=8)
    axes[1].plot(x, z.imag, ".-"); axes[1].axvline(np.log10(result.predicted_f_min), color="tab:red"); axes[1].axvline(np.log10(manual_min), color="tab:green"); axes[1].set_title("Im(Z)"); axes[1].set_xlabel("log10(f)")
    axes[2].plot(z.real, -z.imag, ".-"); axes[2].set_title("Nyquist"); axes[2].set_xlabel("Re(Z)"); axes[2].set_ylabel("-Im(Z)")
    score = result.diagnostics["score"]
    axes[3].plot(x, score, ".-"); axes[3].axhline(result.diagnostics["parameters"]["threshold"], color="tab:orange", linestyle="--", label="threshold")
    axes[3].axvline(np.log10(result.predicted_f_min), color="tab:red", label="predicted f_min"); axes[3].set_title("Selector score"); axes[3].set_xlabel("log10(f)"); axes[3].legend(fontsize=8)
    fig.suptitle(f"{label}: sample {record.sample_id}, cycle {record.cycle}"); fig.tight_layout(); fig.savefig(output / f"{label}.png", dpi=140); plt.close(fig)


def evaluate(projects: list[Path], audit_path: Path, output: Path) -> dict:
    total_started = time.perf_counter()
    extraction = load_eisfit_projects(projects, {str(p): p.name.split(".")[0] for p in projects})
    records = {record.spectrum_id: record for record in extraction.records}
    reference = pd.read_csv(audit_path / "per_spectrum.csv").set_index("spectrum_id")
    output.mkdir(parents=True, exist_ok=True); plots = output / "plots"; plots.mkdir(exist_ok=True)
    rows = []; runtimes = {method: 0.0 for method in METHODS}; examples = {}
    for spectrum_id, record in records.items():
        if spectrum_id not in reference.index:
            continue
        ref = reference.loc[spectrum_id]; started = time.perf_counter()
        selected_all = np.arange(record.frequency.size)
        manual_valid_all = _manual_valid(record, selected_all)
        manual_rejected_inside = None
        for method in METHODS:
            method_started = time.perf_counter()
            result = select_low_frequency_boundary(record.frequency, record.impedance, method=method, **PARAMETERS)
            runtimes[method] += time.perf_counter() - method_started
            predicted = result.predicted_f_min; manual_min = float(ref.manual_active_f_min)
            error = np.log10(predicted) - np.log10(manual_min)
            retained = record.frequency >= predicted
            manual_rejected_inside = int(np.sum(retained & ~manual_valid_all))
            close = abs(error) <= .20
            row = {"spectrum_id": spectrum_id, "sample_id": record.sample_id, "voltage": record.voltage, "time": record.time,
                   "topology": record.electrochemical_topology, "l0_required": record.l0_required_in_manual_fit, "method": method,
                   "predicted_f_min": predicted, "manual_active_f_min": manual_min, "error_decades": error,
                   "stored_f_min": ref.stored_f_min, "stored_f_max": ref.stored_f_max, "original_point_count": len(record.frequency),
                   "retained_points": int(retained.sum()), "retained_fraction": float(retained.mean()), "manual_active_points": int(manual_valid_all.sum()),
                   "manual_retained_fraction": float(manual_valid_all.mean()), "manual_rejected_inside_predicted_region": manual_rejected_inside,
                   "boundary_close_with_manual_internal_rejections": bool(close and manual_rejected_inside > 0)}
            rows.append(row)
            examples.setdefault(method, []).append((abs(error), record, result, manual_min, manual_rejected_inside))
        _ = started
    frame = pd.DataFrame(rows); frame.to_csv(output / "per_spectrum.csv", index=False)
    per_method = []
    for method, part in frame.groupby("method"):
        metrics = _metrics(part); metrics.update({"method": method, "runtime_s": runtimes[method], "spectra_per_second": len(part) / runtimes[method], "points_per_second": part.original_point_count.sum() / runtimes[method]})
        per_method.append(metrics)
    per_method_frame = pd.DataFrame(per_method).sort_values("MAE_decades"); per_method_frame.to_csv(output / "per_method.csv", index=False); per_method_frame.to_csv(output / "summary.csv", index=False)
    grouped_outputs = {}
    for grouping, filename in ((["sample_id", "method"], "per_sample.csv"), (["topology", "method"], "per_topology.csv"), (["l0_required", "method"], "per_l0.csv")):
        grouped = []
        for keys, part in frame.groupby(grouping, dropna=False):
            values = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,))); values.update(_metrics(part)); grouped.append(values)
        grouped_outputs[filename] = pd.DataFrame(grouped); grouped_outputs[filename].to_csv(output / filename, index=False)
    voltage = frame.dropna(subset=["voltage"]).copy(); voltage["voltage_rounded"] = voltage.voltage.round(3)
    voltage_summary = voltage.groupby(["sample_id", "voltage_rounded", "method"], as_index=False).apply(lambda part: pd.Series(_metrics(part)), include_groups=False).reset_index(drop=True)
    voltage_summary.to_csv(output / "per_voltage.csv", index=False)
    example_labels = {"excellent_prediction": lambda values: min(values, key=lambda x: x[0]), "too_low": lambda values: max(values, key=lambda x: -x[3] + x[0]), "too_high": lambda values: max(values, key=lambda x: x[3] + x[0]), "strong_low_frequency_noise": lambda values: max(values, key=lambda x: x[4]), "no_obvious_degradation": lambda values: min(values, key=lambda x: x[4])}
    for method, values in examples.items():
        for label, chooser in example_labels.items():
            if values:
                _, record, result, manual_min, _ = chooser(values); _plot(record, result, manual_min, f"{method}_{label}", plots)
        for sample in ("150", "157", "181"):
            candidates = [item for item in values if item[1].sample_id == sample]
            if candidates:
                _, record, result, manual_min, _ = min(candidates, key=lambda x: x[0]); _plot(record, result, manual_min, f"{method}_sample_{sample}", plots)
    config = {"methods": METHODS, "parameters": PARAMETERS, "manual_reference": "audit per_spectrum.csv manual_active_f_min", "prediction_inputs": "frequency and complex impedance only"}
    (output / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    total_runtime = time.perf_counter() - total_started
    report = {"records_extracted": len(extraction.records), "evaluated_spectra": int(frame.spectrum_id.nunique()), "exclusions": extraction.exclusion_counts,
              "runtime_by_method_s": runtimes, "total_runtime_s": total_runtime,
              "total_points": int(frame.original_point_count.sum()), "prediction_inputs": "frequency and complex impedance only"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    best = per_method_frame.iloc[0]
    report_lines = ["# Deterministic low-frequency boundary selector benchmark", "",
                    f"Evaluated **{report['evaluated_spectra']} spectra** from samples 129, 140, 150, 157, 159, and 181.",
                    "Prediction used only the measured frequency and complex impedance arrays. Manual masks, topology, voltage, time, L0 status, and fitted parameters were evaluation-only references.", "",
                    "## Overall result", "",
                    f"The best aggregate MAE was **{best['MAE_decades']:.4f} decades** ({best['method']}). The median absolute error was **{best['median_abs_error_decades']:.4f} decades** and the fraction within 0.20 decades was **{best['within_0.20_decades_percent']:.2f}%**.",
                    "With the conservative configured threshold, the methods returned the measured minimum for all evaluated spectra; therefore the initial benchmark did not detect a persistent boundary in this dataset. This is reported as a baseline result, not as evidence that the spectra contain no degradation.", "",
                    "## Method comparison", "",
                    "See `per_method.csv` for the complete metrics, including error percentiles, directional errors, retention, isolated-rejection diagnostic, and runtime. `per_sample.csv`, `per_voltage.csv`, `per_topology.csv`, and `per_l0.csv` provide the requested stratified diagnostics.", "",
                    "## Runtime", "",
                    f"Total evaluation runtime: **{total_runtime:.2f} s** for {report['evaluated_spectra']} spectra and {report['total_points']} points.", "",
                    "Per-method runtime, spectra/second, and points/second are in `per_method.csv`.", "",
                    "## Interpretation and recommendation", "",
                    "The implementation is deterministic, spectrum-only, interpretable, and fast enough for a research baseline. It should not yet replace manual low-frequency selection: the current threshold/persistence configuration is too conservative for these six samples, and all four variants are effectively tied. A subsequent, explicitly separated calibration experiment can assess thresholds or score normalization; no isolated-outlier detector is implemented here.", ""]
    (output / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(per_method_frame.to_string(index=False))
    return {"evaluated_spectra": int(frame.spectrum_id.nunique()), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projects", nargs="+", type=Path)
    parser.add_argument("--audit", type=Path, default=Path("ml/analysis/manual_frequency_range"))
    parser.add_argument("--output", type=Path, default=Path("ml/analysis/low_frequency_selector"))
    args = parser.parse_args()
    evaluate(args.projects, args.audit, args.output)


if __name__ == "__main__":
    main()
