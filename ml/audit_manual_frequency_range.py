"""Audit saved frequency windows against the saved manual active-point masks."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eis_project import dataframe_from_payload
from eis_services import load_cycle

from .dataset import _payload_projects, load_eisfit_projects


def _percentile(values, percentile):
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, percentile)) if values.size else np.nan


def _interval_count(active: np.ndarray) -> tuple[int, list[int]]:
    if not active.size:
        return 0, []
    starts = active & ~np.r_[False, active[:-1]]
    ends = active & ~np.r_[active[1:], False]
    start_indices = np.flatnonzero(starts); end_indices = np.flatnonzero(ends)
    return int(len(start_indices)), [int(end - start + 1) for start, end in zip(start_indices, end_indices)]


def _record_payloads(projects: list[Path], records: dict[str, object]):
    for path in projects:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry_index, (state_payload, entry) in enumerate(_payload_projects(payload)):
            dataframe_payload = entry.get("dataframe")
            if not dataframe_payload:
                continue
            dataframe = dataframe_from_payload(dataframe_payload)
            control = str(state_payload.get("control", payload.get("control", "cell")))
            dataset_key = str(entry.get("dataset_id") or f"dataset_{entry_index}")
            for cycle_text, saved in (state_payload.get("cycles") or {}).items():
                spectrum_id = f"{path.resolve()}::{dataset_key}::{control}::{int(cycle_text)}"
                if spectrum_id in records:
                    yield records[spectrum_id], saved, dataframe, control


def _one_record(record, saved, dataframe, control):
    cycle = load_cycle(dataframe, record.cycle, control)
    frequency = np.asarray(cycle.frequency_hz, dtype=float)
    impedance = np.asarray(cycle.impedance, dtype=complex)
    valid = np.isfinite(frequency) & (frequency > 0) & np.isfinite(impedance.real) & np.isfinite(impedance.imag)
    manual_mask = np.asarray(saved.get("manually_included"), dtype=bool)
    if manual_mask.size != frequency.size:
        return None, "manual_mask_length_mismatch"
    stored_min, stored_max = sorted(map(float, saved["frequency_window"]))
    selected = valid & (frequency >= stored_min) & (frequency <= stored_max)
    if int(np.sum(selected)) < 1:
        return None, "no_valid_points_in_stored_window"
    order = np.argsort(frequency[selected], kind="mergesort")
    selected_indices = np.flatnonzero(selected)[order]
    active = manual_mask[selected_indices]
    if not active.any():
        return None, "no_manual_active_points_in_stored_window"
    selected_frequency = frequency[selected_indices]
    active_frequency = selected_frequency[active]
    active_min, active_max = float(np.min(active_frequency)), float(np.max(active_frequency))
    log_stored_min, log_stored_max = np.log10(stored_min), np.log10(stored_max)
    delta_min, delta_max = np.log10(active_min) - log_stored_min, np.log10(active_max) - log_stored_max
    grid_step = np.median(np.diff(np.log10(selected_frequency))) if selected_frequency.size > 1 else 0.0
    tolerance = max(1e-12, 0.5 * abs(float(grid_step)))
    changed_min = delta_min > tolerance
    changed_max = delta_max < -tolerance
    intervals, interval_lengths = _interval_count(active)
    if intervals > 1:
        classification = "unusual_non_contiguous"
    elif changed_min and changed_max:
        classification = "both_boundaries_changed"
    elif changed_min:
        classification = "low_frequency_truncation"
    elif changed_max:
        classification = "high_frequency_truncation"
    else:
        classification = "identical_range"
    first_active, last_active = int(np.flatnonzero(active)[0]), int(np.flatnonzero(active)[-1])
    internal_gaps = ~active[first_active:last_active + 1]
    gap_count = int(np.sum(internal_gaps))
    gap_lengths = []
    if gap_count:
        _, gap_lengths = _interval_count(internal_gaps)
    low_rejected = int(np.sum(~active & (selected_frequency < active_min)))
    high_rejected = int(np.sum(~active & (selected_frequency > active_max)))
    return {
        "spectrum_id": record.spectrum_id, "sample_id": record.sample_id, "cycle": record.cycle,
        "voltage": record.voltage, "time": record.time, "topology": record.electrochemical_topology,
        "stored_f_min": stored_min, "stored_f_max": stored_max,
        "manual_active_f_min": active_min, "manual_active_f_max": active_max,
        "delta_fmin_decades": delta_min, "delta_fmax_decades": delta_max,
        "ratio_fmin_active_to_stored": active_min / stored_min, "ratio_fmax_active_to_stored": active_max / stored_max,
        "stored_window_points": int(selected.size), "number_active_points": int(active.sum()),
        "number_rejected_points": int((~active).sum()), "number_active_intervals": intervals,
        "active_interval_lengths": json.dumps(interval_lengths), "internal_gap_count": gap_count,
        "internal_gap_lengths": json.dumps(gap_lengths), "contiguous_active_interval": intervals == 1,
        "low_frequency_rejected_points": low_rejected, "high_frequency_rejected_points": high_rejected,
        "low_rejection_fraction_stored_window": low_rejected / selected.size,
        "high_rejection_fraction_stored_window": high_rejected / selected.size,
        "changed_lower_boundary": bool(changed_min), "changed_upper_boundary": bool(changed_max),
        "classification": classification, "grid_tolerance_decades": tolerance,
        "_frequency": selected_frequency, "_active": active,
    }, None


def _write_plot(row: dict, output: Path, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frequency = row["_frequency"]; active = row["_active"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.log10(frequency[active]), np.zeros(np.sum(active)), "o", label="manually included")
    axes[0].plot(np.log10(frequency[~active]), np.zeros(np.sum(~active)), "x", label="manually rejected")
    axes[0].axvline(np.log10(row["stored_f_min"]), color="tab:orange", linestyle="--", label="stored limits")
    axes[0].axvline(np.log10(row["stored_f_max"]), color="tab:orange", linestyle="--")
    axes[0].axvline(np.log10(row["manual_active_f_min"]), color="tab:green", label="active limits")
    axes[0].axvline(np.log10(row["manual_active_f_max"]), color="tab:green")
    axes[0].set_xlabel("log10(frequency)"); axes[0].set_yticks([]); axes[0].legend(fontsize=8); axes[0].set_title(label)
    axes[1].plot(np.log10(frequency), active.astype(int), "o-")
    axes[1].set_xlabel("log10(frequency)"); axes[1].set_ylabel("manual active (1/0)"); axes[1].set_ylim(-0.1, 1.1)
    fig.suptitle(f"sample {row['sample_id']} / cycle {row['cycle']}"); fig.tight_layout()
    fig.savefig(output / f"{label}.png", dpi=140); plt.close(fig)


def audit(projects: list[Path], output: Path) -> dict:
    mapping = {str(path): path.name.split(".")[0] for path in projects}
    extraction = load_eisfit_projects(projects, mapping, require_fit=True)
    records = {record.spectrum_id: record for record in extraction.records}
    rows = []; failures = []
    for record, saved, dataframe, control in _record_payloads(projects, records):
        if saved.get("frequency_window") is None or saved.get("manually_included") is None:
            failures.append({"spectrum_id": record.spectrum_id, "reason": "missing_frequency_window_or_manual_mask"}); continue
        row, error = _one_record(record, saved, dataframe, control)
        if error: failures.append({"spectrum_id": record.spectrum_id, "reason": error}); continue
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    internal = [column for column in frame.columns if column.startswith("_")]
    plot_frame = frame.drop(columns=internal)
    plot_frame.to_csv(output / "per_spectrum.csv", index=False)
    frame["delta_fmin_decades_abs"] = frame.delta_fmin_decades.abs()
    summary = []
    for group_name, part in [("overall", frame), *[(str(s), g) for s, g in frame.groupby("sample_id")]]:
        summary.append({"group": group_name, "spectra": len(part),
                        "identical_count": int((part.classification == "identical_range").sum()),
                        "low_truncation_count": int((part.classification == "low_frequency_truncation").sum()),
                        "high_truncation_count": int((part.classification == "high_frequency_truncation").sum()),
                        "both_changed_count": int((part.classification == "both_boundaries_changed").sum()),
                        "non_contiguous_count": int((part.number_active_intervals > 1).sum()),
                        "contiguous_percent": 100 * part.contiguous_active_interval.mean(),
                        "lower_changed_percent": 100 * part.changed_lower_boundary.mean(),
                        "upper_changed_percent": 100 * part.changed_upper_boundary.mean(),
                        "mean_delta_fmin_decades": part.delta_fmin_decades.mean(), "median_delta_fmin_decades": part.delta_fmin_decades.median(),
                        "p25_delta_fmin_decades": part.delta_fmin_decades.quantile(.25), "p75_delta_fmin_decades": part.delta_fmin_decades.quantile(.75),
                        "p90_delta_fmin_decades": part.delta_fmin_decades.quantile(.90), "p95_delta_fmin_decades": part.delta_fmin_decades.quantile(.95),
                        "max_delta_fmin_decades": part.delta_fmin_decades.max(), "mean_delta_fmax_decades": part.delta_fmax_decades.mean(),
                        "mean_low_rejected_points": part.low_frequency_rejected_points.mean(), "mean_high_rejected_points": part.high_frequency_rejected_points.mean()})
    summary_frame = pd.DataFrame(summary); summary_frame.to_csv(output / "summary.csv", index=False)
    voltage = frame.dropna(subset=["voltage"]).copy(); voltage["voltage_rounded"] = voltage.voltage.round(3)
    per_voltage = voltage.groupby(["sample_id", "voltage_rounded"], as_index=False).agg(spectra=("spectrum_id", "count"), mean_stored_fmin=("stored_f_min", "mean"), mean_active_fmin=("manual_active_f_min", "mean"), mean_delta_fmin_decades=("delta_fmin_decades", "mean"), lower_changed_percent=("changed_lower_boundary", lambda x: 100 * x.mean()))
    per_voltage.to_csv(output / "per_voltage.csv", index=False)
    time_frame = frame.dropna(subset=["time"])
    if not time_frame.empty:
        time_frame[["spectrum_id", "sample_id", "time", "delta_fmin_decades", "delta_fmax_decades"]].sort_values(["sample_id", "time"]).to_csv(output / "per_time.csv", index=False)
    topology = frame.groupby("topology").agg(spectra=("spectrum_id", "count"), mean_delta_fmin_decades=("delta_fmin_decades", "mean"), mean_delta_fmax_decades=("delta_fmax_decades", "mean"), lower_changed_percent=("changed_lower_boundary", lambda x: 100*x.mean()), non_contiguous_percent=("number_active_intervals", lambda x: 100*(x>1).mean())).reset_index()
    topology.to_csv(output / "per_topology.csv", index=False)
    plots_dir = output / "plots"; plots_dir.mkdir(exist_ok=True)
    chosen = []
    for classification in ("identical_range", "low_frequency_truncation", "high_frequency_truncation", "both_boundaries_changed", "unusual_non_contiguous"):
        part = frame[frame.classification == classification]
        if not part.empty:
            chosen.append((classification, part.iloc[0]))
    for label, row in chosen:
        _write_plot(row.to_dict(), plots_dir, label)
    failed_frame = pd.DataFrame(failures); failed_frame.to_csv(output / "failures.csv", index=False)
    report = {"records_extracted": len(extraction.records), "audit_rows": len(frame), "extraction_exclusions": extraction.exclusion_counts, "audit_failures": len(failures), "output": str(output), "threshold_tolerance": "half the median log-frequency grid step per spectrum"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary_frame.to_string(index=False)); print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projects", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ml/analysis/manual_frequency_range"))
    args = parser.parse_args()
    audit(args.projects, args.output)


if __name__ == "__main__":
    main()
