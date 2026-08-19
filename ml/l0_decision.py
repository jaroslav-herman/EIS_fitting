"""Deterministic high-frequency decision for the optional L0 element.

The decision is deliberately independent of electrochemical process topology.
The stored impedance convention is ``Z = Re(Z) + j Im(Z)`` while the GUI
Nyquist ordinate is ``-Im(Z)``.  Consequently, the inductive signature used
here is a negative GUI/Nyquist ordinate (equivalently positive raw ``Im(Z)``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eis_services import circuit_parameters

from .dataset import _payload_projects, load_eisfit_projects
from .evaluate_stage4_parameters import load_automatic_masks


HF_FRACTION = 0.10
MIN_HF_POINTS = 5
MIN_CONSECUTIVE_POINTS = 2
NO_L0_QUANTILE = 0.95


def high_frequency_inductive_diagnostic(
    frequency,
    impedance,
    active_mask=None,
    *,
    fraction: float = HF_FRACTION,
    minimum_points: int = MIN_HF_POINTS,
) -> dict[str, float | int | bool | str | None]:
    """Measure a coherent inductive signature in the highest-frequency data."""
    f = np.asarray(frequency, dtype=float).reshape(-1)
    z = np.asarray(impedance, dtype=complex).reshape(-1)
    if f.size != z.size:
        raise ValueError("frequency and impedance must have equal lengths")
    mask = np.ones(f.size, dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool).reshape(-1)
    if mask.size != f.size:
        raise ValueError("active_mask must have the same length as frequency")
    valid = mask & np.isfinite(f) & (f > 0) & np.isfinite(z.real) & np.isfinite(z.imag)
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return {
            "valid_active_points": 0, "high_frequency_points": 0,
            "negative_imaginary_fraction": 0.0, "negative_imaginary_consecutive_points": 0,
            "minimum_nyquist_imaginary": None, "high_frequency_inductive_strength": 0.0,
            "negative_excursion_noise_ratio": 0.0, "highest_frequency_hz": None,
            "lowest_high_frequency_hz": None, "negative_signature_at_highest_frequency": False,
        }
    order = indices[np.argsort(f[indices], kind="mergesort")[::-1]]
    count = min(order.size, max(int(minimum_points), int(np.ceil(float(fraction) * order.size))))
    selected = order[:count]
    nyquist_imaginary = -z[selected].imag
    negative = nyquist_imaginary < 0.0
    consecutive = 0
    for is_negative in negative:
        if not is_negative:
            break
        consecutive += 1
    median_magnitude = max(float(np.median(np.abs(z[selected]))), np.finfo(float).eps)
    excursion = max(0.0, -float(np.min(nyquist_imaginary)))
    strength = excursion / median_magnitude
    differences = np.diff(nyquist_imaginary)
    if differences.size:
        median_difference = float(np.median(differences))
        mad = float(np.median(np.abs(differences - median_difference)))
        noise_scale = max(1.4826 * mad, np.finfo(float).eps)
    else:
        noise_scale = np.finfo(float).eps
    return {
        "valid_active_points": int(order.size),
        "high_frequency_points": int(count),
        "negative_imaginary_fraction": float(np.mean(negative)),
        "negative_imaginary_consecutive_points": int(consecutive),
        "minimum_nyquist_imaginary": float(np.min(nyquist_imaginary)),
        "high_frequency_inductive_strength": float(strength),
        "negative_excursion_noise_ratio": float(excursion / noise_scale),
        "highest_frequency_hz": float(f[selected[0]]),
        "lowest_high_frequency_hz": float(f[selected[-1]]),
        "negative_signature_at_highest_frequency": bool(consecutive > 0),
        "nyquist_convention": "negative Im(Z) means negative GUI ordinate -Im(Z); raw Im(Z) is positive",
    }


def decide_l0(diagnostic: dict, *, strength_threshold: float, minimum_consecutive: int = MIN_CONSECUTIVE_POINTS) -> dict[str, object]:
    """Apply the calibrated, persistent-signature rule to one diagnostic."""
    strength = float(diagnostic.get("high_frequency_inductive_strength") or 0.0)
    consecutive = int(diagnostic.get("negative_imaginary_consecutive_points") or 0)
    required = consecutive >= int(minimum_consecutive) and strength >= float(strength_threshold)
    if required:
        reason = "persistent_high_frequency_inductive_signature"
    elif consecutive < int(minimum_consecutive):
        reason = "no_persistent_high_frequency_inductive_signature"
    else:
        reason = "high_frequency_excursion_below_calibrated_strength_threshold"
    return {
        "l0_required": bool(required),
        "l0_reason": reason,
        "l0_strength_threshold": float(strength_threshold),
        "l0_minimum_consecutive_points": int(minimum_consecutive),
    }


def _reference_l0_values(projects, records) -> dict[str, float | None]:
    """Read reference L0 values from saved fits without using them as inputs."""
    wanted = {record.spectrum_id for record in records}
    values: dict[str, float | None] = {}
    for path in projects:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for index, (state, entry) in enumerate(_payload_projects(payload)):
            dataset = str(entry.get("dataset_id") or f"dataset_{index}")
            control = str(state.get("control", payload.get("control", "cell")))
            for cycle_text, saved in (state.get("cycles") or {}).items():
                spectrum_id = f"{Path(path).resolve()}::{dataset}::{control}::{int(cycle_text)}"
                if spectrum_id not in wanted:
                    continue
                circuit = str(saved.get("circuit") or state.get("circuit") or "")
                fit = saved.get("fit_parameters")
                names = [parameter.name for parameter in circuit_parameters(circuit)] if circuit else []
                if fit is None or len(names) != len(fit):
                    values[spectrum_id] = None
                    continue
                values[spectrum_id] = dict(zip(names, np.asarray(fit, dtype=float))).get("L0")
    return values


def _classification_metrics(predicted, reference) -> dict[str, float | int]:
    predicted = np.asarray(predicted, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    tp = int(np.sum(predicted & reference)); tn = int(np.sum(~predicted & ~reference))
    fp = int(np.sum(predicted & ~reference)); fn = int(np.sum(~predicted & reference))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def calibrate_l0_rule(records, automatic_masks, reference_values, *, output: Path | None = None) -> dict:
    """Calibrate and optionally export the L0 rule using six-sample labels."""
    diagnostics = []
    for record in records:
        measured = high_frequency_inductive_diagnostic(record.frequency, record.impedance, automatic_masks[record.spectrum_id])
        reference_l0 = reference_values.get(record.spectrum_id)
        diagnostics.append({
            "spectrum_id": record.spectrum_id, "sample_id": record.sample_id,
            "reference_l0_required": bool(record.l0_required_in_manual_fit),
            "reference_l0_value": reference_l0, **measured,
        })
    frame = pd.DataFrame(diagnostics)
    no_l0 = frame.loc[~frame.reference_l0_required, "high_frequency_inductive_strength"].to_numpy(float)
    finite_no_l0 = no_l0[np.isfinite(no_l0)]
    threshold = float(np.quantile(finite_no_l0, NO_L0_QUANTILE)) if finite_no_l0.size else 0.05
    predictions = [decide_l0(row, strength_threshold=threshold) for row in diagnostics]
    predicted = np.asarray([row["l0_required"] for row in predictions], dtype=bool)
    reference = frame.reference_l0_required.to_numpy(bool)
    frame["predicted_l0_required"] = predicted
    frame["l0_reason"] = [row["l0_reason"] for row in predictions]
    metrics = _classification_metrics(predicted, reference)
    required_l0 = frame.loc[frame.reference_l0_required & frame.reference_l0_value.notna(), "reference_l0_value"].to_numpy(float)
    artifact_floor = 1.0e-12
    physical_l0 = required_l0[np.isfinite(required_l0) & (required_l0 >= artifact_floor)]
    strength = frame.high_frequency_inductive_strength.to_numpy(float)
    report = {
        "records": int(len(frame)), "reference_l0_required": int(reference.sum()),
        "reference_l0_not_required": int((~reference).sum()), "predicted_l0_required": int(predicted.sum()),
        "threshold_selection": {
            "method": "95th percentile of high-frequency inductive strength in reference L0-absent spectra",
            "no_l0_quantile": NO_L0_QUANTILE, "strength_threshold": threshold,
            "high_frequency_fraction": HF_FRACTION, "minimum_high_frequency_points": MIN_HF_POINTS,
            "minimum_consecutive_points": MIN_CONSECUTIVE_POINTS,
            "calibration_samples": sorted({str(record.sample_id) for record in records}),
            "sample_178_used": False,
        },
        "confusion_matrix": {"labels": ["L0_absent", "L0_present"], "rows_reference": [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]},
        "metrics": metrics,
        "reference_l0_value_distribution": {
            "count": int(required_l0.size),
            "min": float(np.min(required_l0)) if required_l0.size else None,
            "median": float(np.median(required_l0)) if required_l0.size else None,
            "max": float(np.max(required_l0)) if required_l0.size else None,
            "artifact_floor_h": artifact_floor,
            "artifact_like_count_below_floor": int(required_l0.size - physical_l0.size),
            "physical_scale_count_above_floor": int(physical_l0.size),
            "physical_scale_median_above_floor": float(np.median(physical_l0)) if physical_l0.size else None,
        },
        "high_frequency_strength_distribution": {
            "all": {"min": float(np.min(strength)), "median": float(np.median(strength)), "max": float(np.max(strength))},
            "l0_absent": {"q50": float(np.quantile(finite_no_l0, .50)) if finite_no_l0.size else None, "q95": threshold, "max": float(np.max(finite_no_l0)) if finite_no_l0.size else None},
            "l0_present": {"q50": float(np.quantile(frame.loc[frame.reference_l0_required, "high_frequency_inductive_strength"], .50)), "q95": float(np.quantile(frame.loc[frame.reference_l0_required, "high_frequency_inductive_strength"], .95))},
        },
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output / "predictions.csv", index=False)
        pd.DataFrame([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]],], index=["L0_absent", "L0_present"], columns=["L0_absent", "L0_present"]).to_csv(output / "confusion_matrix.csv")
        pd.DataFrame([{"group": "L0_present_reference", "value": value} for value in required_l0] + [{"group": "L0_absent_reference", "value": value} for value in finite_no_l0]).to_csv(output / "distributions.csv", index=False)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"report": report, "frame": frame, "threshold": threshold}


def calibrate_l0_from_projects(projects, *, output: Path | None = None) -> dict:
    mapping = {str(path): Path(path).name.split(".")[0] for path in projects}
    extraction = load_eisfit_projects(list(projects), mapping, require_fit=True)
    records = extraction.records
    automatic_masks = load_automatic_masks(records)
    reference_values = _reference_l0_values(projects, records)
    result = calibrate_l0_rule(records, automatic_masks, reference_values, output=output)
    result["report"]["exclusions"] = extraction.exclusion_counts
    return result
