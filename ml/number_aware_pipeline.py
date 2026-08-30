"""Number-aware staged ML pipeline for EIS preprocessing and EEC initialization.

The pipeline deliberately treats EEC element numbers as physical labels.  It
uses the existing project loader and deterministic preprocessing, produces ML
suggestions in the normal sidecar contract, and keeps the final conventional
EEC fit separate from those suggestions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eis_model import CycleState, ParameterValue
from eis_services import FitOptions, circuit_parameters, fit_cycle

from .dataset import SpectrumRecord, load_eisfit_projects
from .frequency_range import _targets as frequency_targets
from .point_validity import detect_valid_points
from .preprocessing import SpectrumPreprocessor
from .results_schema import spectrum_identifier, write_ml_results


def _is_positive_parameter(name: str) -> bool:
    return str(name).startswith(("R", "L")) or (str(name).startswith("CPE") and not str(name).endswith("_1"))


def _is_alpha_parameter(name: str) -> bool:
    return str(name).startswith("CPE") and str(name).endswith("_1")


def _transform_parameter(values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if _is_positive_parameter(name):
        return np.log10(values)
    if _is_alpha_parameter(name):
        clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped))
    return values


def _inverse_parameter(value: float, name: str) -> float:
    if _is_positive_parameter(name):
        return float(10.0 ** value)
    if _is_alpha_parameter(name):
        return float(np.clip(1.0 / (1.0 + np.exp(-value)), 1e-4, 0.9999))
    return float(value)


def _physical_initial_cap(record: SpectrumRecord, name: str, value: float) -> float:
    """Prevent scale-model excursions from producing impossible EEC initials.

    For passive EECs, Re(Z) is bounded below by R0.  At the highest measured
    frequencies the dispersive branches are least influential, so a robust
    low quantile of Re(Z) is a conservative upper cap for R0.  L0 is similarly
    checked against the measured high-frequency |Im(Z)|/(2*pi*f) scale.  These
    are safeguards for initialization, not replacement fits.
    """
    frequency = np.asarray(record.frequency, dtype=float)
    impedance = np.asarray(record.impedance, dtype=complex)
    valid = np.isfinite(frequency) & (frequency > 0) & np.isfinite(impedance.real) & np.isfinite(impedance.imag)
    if valid.sum() < 3:
        return value
    frequency, impedance = frequency[valid], impedance[valid]
    high = np.argsort(frequency)[-max(3, frequency.size // 10):]
    if name == "R0":
        real = impedance.real[np.isfinite(impedance.real) & (impedance.real > 0)]
        if real.size:
            cap = float(np.nanpercentile(real, 10.0))
            if np.isfinite(cap) and cap > 0:
                return min(value, cap)
    elif name == "L0":
        apparent = np.abs(impedance.imag[high]) / (2.0 * np.pi * frequency[high])
        apparent = apparent[np.isfinite(apparent) & (apparent > 0)]
        if apparent.size:
            cap = 5.0 * float(np.nanmedian(apparent))
            if np.isfinite(cap) and cap > 0:
                return min(value, cap)
    return value


@dataclass
class PipelineBundle:
    frequency_preprocessor: SpectrumPreprocessor
    frequency_model: object
    topology_preprocessor: SpectrumPreprocessor
    topology_model: object
    topology_classes: tuple[str, ...]
    parameter_preprocessor: SpectrumPreprocessor
    parameter_models: dict[str, object]
    parameter_stats: dict[str, dict[str, float | int | str]]
    training_samples: tuple[str, ...]
    circuit_classes: tuple[str, ...]


def load_pipeline_bundle(path: Path) -> PipelineBundle:
    """Load bundles made either as a module or by a script entry point."""
    # Older runs serialized the dataclass as ``__main__.PipelineBundle``.
    # Register the current class there before unpickling those artifacts.
    setattr(sys.modules["__main__"], "PipelineBundle", PipelineBundle)
    return joblib.load(path)


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _records_with_mask(records: Iterable[SpectrumRecord], masks: dict[str, np.ndarray]) -> list[SpectrumRecord]:
    result = []
    for record in records:
        mask = np.asarray(masks[record.spectrum_id], dtype=bool)
        if mask.size != record.frequency.size:
            raise ValueError(f"mask length mismatch for {record.spectrum_id}")
        if int(mask.sum()) < 3:
            raise ValueError(f"fewer than three active points for {record.spectrum_id}")
        result.append(SpectrumRecord(
            spectrum_id=record.spectrum_id,
            source_project=record.source_project,
            sample_id=record.sample_id,
            cycle=record.cycle,
            voltage=record.voltage,
            current=record.current,
            time=record.time,
            frequency=record.frequency[mask],
            z_real=record.z_real[mask],
            z_imag=record.z_imag[mask],
            topology_label=record.topology_label,
            original_eec_topology=record.original_eec_topology,
            electrochemical_topology=record.electrochemical_topology,
            l0_required_in_manual_fit=record.l0_required_in_manual_fit,
            device_setup=record.device_setup,
            manual_f_min=record.manual_f_min,
            manual_f_max=record.manual_f_max,
            control=record.control,
        ))
    return result


def _manual_masks(records: Iterable[SpectrumRecord]) -> dict[str, np.ndarray]:
    masks = {}
    for record in records:
        mask = np.ones(record.frequency.size, dtype=bool)
        if record.manual_f_min is not None and record.manual_f_max is not None:
            mask &= (record.frequency >= record.manual_f_min) & (record.frequency <= record.manual_f_max)
        if record.cleaned_frequency is not None:
            # The loader has already applied manually_included and the window.
            # Match by values rather than assuming an ordering transformation.
            cleaned = {(float(f), float(r), float(i)) for f, r, i in zip(record.cleaned_frequency, record.cleaned_z_real, record.cleaned_z_imag)}
            mask = np.asarray([(float(f), float(r), float(i)) in cleaned for f, r, i in zip(record.frequency, record.z_real, record.z_imag)], dtype=bool)
        masks[record.spectrum_id] = mask
    return masks


def deterministic_masks(records: Iterable[SpectrumRecord], frequency_windows: dict[str, tuple[float, float]], threshold: float = 4.0) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    masks: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict] = {}
    for record in records:
        window = frequency_windows[record.spectrum_id]
        mask, _score, detector_diagnostics = detect_valid_points(
            record.frequency, record.impedance, threshold=threshold,
            frequency_range=window, max_iterations=2,
        )
        mask = np.asarray(mask, dtype=bool)
        if mask.size != record.frequency.size:
            raise ValueError(f"deterministic mask length mismatch for {record.spectrum_id}")
        if int(mask.sum()) < 3:
            raise ValueError(f"deterministic preprocessing left fewer than three points for {record.spectrum_id}")
        masks[record.spectrum_id] = mask
        diagnostics[record.spectrum_id] = {
            "threshold": float(threshold),
            "active_points": int(mask.sum()),
            "rejected_points": int((~mask).sum()),
            "rejection_reason": [str(x) for x in detector_diagnostics["rejection_reason"]],
        }
    return masks, diagnostics


def _frequency_model() -> object:
    return MultiOutputRegressor(Ridge(alpha=1.0))


def _topology_model(seed: int) -> object:
    return RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=seed, n_jobs=1, min_samples_leaf=2)


def _parameter_model() -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def _fit_frequency(records: list[SpectrumRecord], seed: int) -> tuple[SpectrumPreprocessor, object]:
    preprocessor = SpectrumPreprocessor(grid_size=64, use_metadata=True, spectrum_mode="raw", include_impedance_scale=True)
    x = preprocessor.fit_transform(records)
    model = _frequency_model()
    model.fit(x, frequency_targets(records))
    return preprocessor, model


def _fit_topology(records: list[SpectrumRecord], seed: int) -> tuple[SpectrumPreprocessor, object, tuple[str, ...]]:
    preprocessor = SpectrumPreprocessor(grid_size=64, use_metadata=True, spectrum_mode="raw", include_impedance_scale=True)
    x = preprocessor.fit_transform(records)
    labels = np.asarray([r.original_eec_topology for r in records], dtype=str)
    model = _topology_model(seed)
    model.fit(x, labels)
    return preprocessor, model, tuple(sorted(set(labels)))


def _fit_parameters(records: list[SpectrumRecord], projects: list[Path]) -> tuple[SpectrumPreprocessor, dict[str, object], dict[str, dict]]:
    preprocessor = SpectrumPreprocessor(grid_size=64, use_metadata=True, spectrum_mode="raw", include_impedance_scale=True)
    x = preprocessor.fit_transform(records)
    payloads = {str(Path(path).resolve()): json.loads(Path(path).read_text(encoding="utf-8")) for path in projects}
    targets: dict[str, list[float]] = {}
    target_rows: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        payload = payloads[str(Path(record.source_project).resolve())]
        found = None
        for entry in payload.get("datasets", []):
            state = entry.get("state", {})
            control = str(state.get("control", payload.get("control", "cell")))
            dataset_key = str(entry.get("dataset_id") or "")
            sid = f"{Path(record.source_project).resolve()}::{dataset_key}::{control}::{record.cycle}"
            if sid == record.spectrum_id:
                found = state.get("cycles", {}).get(str(record.cycle))
                break
        if not found or found.get("fit_parameters") is None:
            continue
        names = [p.name for p in circuit_parameters(str(found.get("circuit") or record.original_eec_topology))]
        values = np.asarray(found["fit_parameters"], dtype=float)
        if values.size != len(names) or not np.isfinite(values).all():
            continue
        for name, value in zip(names, values):
            if value > 0 and np.isfinite(value):
                targets.setdefault(name, []).append(float(value))
                target_rows.setdefault(name, []).append(index)
    models: dict[str, object] = {}
    stats: dict[str, dict] = {}
    for name, values in targets.items():
        indices = np.asarray(target_rows[name], dtype=int)
        model = _parameter_model()
        y = _transform_parameter(values, name)
        model.fit(x[indices], y)
        models[name] = model
        stats[name] = {"training_spectra": int(len(values)), "target_transform": "log10" if _is_positive_parameter(name) else "identity", "target_median": float(np.median(values))}
    return preprocessor, models, stats


def train_bundle(training_projects: list[Path], sample_ids: dict[str, str], seed: int = 42) -> tuple[PipelineBundle, object]:
    extraction = load_eisfit_projects(training_projects, sample_ids, require_fit=True, require_frequency_window=True)
    records = extraction.records
    if not records:
        raise ValueError("no labelled training spectra were extracted")
    samples = tuple(sorted({r.sample_id for r in records}))
    if len(samples) < 2:
        raise ValueError("at least two physical samples are required")
    circuit_classes = tuple(sorted({str(r.original_eec_topology) for r in records}))
    frequency_preprocessor, frequency_model = _fit_frequency(records, seed)
    manual = _manual_masks(records)
    topology_records = _records_with_mask(records, manual)
    topology_preprocessor, topology_model, topology_classes = _fit_topology(topology_records, seed)
    parameter_preprocessor, parameter_models, parameter_stats = _fit_parameters(records, training_projects)
    bundle = PipelineBundle(
        frequency_preprocessor, frequency_model, topology_preprocessor, topology_model,
        topology_classes, parameter_preprocessor, parameter_models, parameter_stats,
        samples, circuit_classes,
    )
    return bundle, extraction


def _predict_window(bundle: PipelineBundle, record: SpectrumRecord) -> tuple[float, float]:
    prediction = np.asarray(bundle.frequency_model.predict(bundle.frequency_preprocessor.transform([record]))[0], dtype=float)
    center, width = float(prediction[0]), float(np.exp(np.clip(prediction[1], -20.0, 20.0)))
    low, high = 10 ** (center - width / 2), 10 ** (center + width / 2)
    measured_low, measured_high = float(np.min(record.frequency)), float(np.max(record.frequency))
    return max(measured_low, min(low, measured_high)), min(measured_high, max(high, measured_low))


def _predict_topology(bundle: PipelineBundle, record: SpectrumRecord) -> tuple[str, dict[str, float], float, str | None]:
    x = bundle.topology_preprocessor.transform([record])
    predicted = str(bundle.topology_model.predict(x)[0])
    probabilities = bundle.topology_model.predict_proba(x)[0] if hasattr(bundle.topology_model, "predict_proba") else np.array([])
    classes = [str(x) for x in getattr(bundle.topology_model, "classes_", ())]
    result = {label: float(probabilities[classes.index(label)]) if label in classes else 0.0 for label in bundle.topology_classes}
    warning = "one_process_class_has_one_training_spectrum" if predicted.count("p(") == 1 and sum(1 for c in bundle.circuit_classes if c.count("p(") == 1) else None
    return predicted, result, float(max(result.values()) if result else 0.0), warning


def _predict_topologies(bundle: PipelineBundle, records: list[SpectrumRecord]) -> list[tuple[str, dict[str, float], float, str | None]]:
    """Predict all topologies in one estimator call."""
    x = bundle.topology_preprocessor.transform(records)
    labels = [str(value) for value in bundle.topology_model.predict(x)]
    probabilities = bundle.topology_model.predict_proba(x) if hasattr(bundle.topology_model, "predict_proba") else np.zeros((len(records), 0))
    classes = [str(value) for value in getattr(bundle.topology_model, "classes_", ())]
    one_process_available = sum(1 for c in bundle.circuit_classes if c.count("p(") == 1) > 0
    result = []
    for index, label in enumerate(labels):
        probs = {name: float(probabilities[index, classes.index(name)]) if name in classes else 0.0 for name in bundle.topology_classes}
        warning = "one_process_class_has_one_training_spectrum" if one_process_available and label.count("p(") == 1 else None
        result.append((label, probs, float(max(probs.values()) if probs else 0.0), warning))
    return result


def _predict_parameters_batch(bundle: PipelineBundle, records: list[SpectrumRecord], circuits: list[str]) -> list[tuple[list[ParameterValue], dict[str, dict]]]:
    x = bundle.parameter_preprocessor.transform(records)
    predictions = {name: np.asarray(model.predict(x), dtype=float) for name, model in bundle.parameter_models.items()}
    result = []
    for index, circuit in enumerate(circuits):
        params = []
        info = {}
        for parameter in circuit_parameters(circuit):
            model_available = parameter.name in predictions
            value = _inverse_parameter(float(predictions[parameter.name][index]), parameter.name) if model_available else float(parameter.initial)
            lower, upper = float(parameter.lower), float(parameter.upper)
            if _is_alpha_parameter(parameter.name):
                lower, upper = max(lower, 1e-4), min(upper, 0.9999)
            value = _physical_initial_cap(records[index], parameter.name, value)
            value = float(np.clip(value, lower, upper))
            params.append(ParameterValue(parameter.name, parameter.unit, value, lower, upper, fixed=parameter.fixed))
            info[parameter.name] = {"parameter_name": parameter.name, "initial_value": value, "lower_limit": lower, "upper_limit": upper, "source": "ml" if model_available else "repository_default", "available": model_available, "training_spectra": int(bundle.parameter_stats.get(parameter.name, {}).get("training_spectra", 0))}
        result.append((params, info))
    return result


def _predict_parameters(bundle: PipelineBundle, record: SpectrumRecord, circuit: str) -> tuple[list[ParameterValue], dict[str, dict]]:
    definitions = circuit_parameters(circuit)
    x = bundle.parameter_preprocessor.transform([record])
    params: list[ParameterValue] = []
    info: dict[str, dict] = {}
    for parameter in definitions:
        model = bundle.parameter_models.get(parameter.name)
        if model is None:
            value = float(parameter.initial)
            source, available = "repository_default", False
        else:
            predicted = float(model.predict(x)[0])
            value = _inverse_parameter(predicted, parameter.name)
            source, available = "ml", True
        lower, upper = float(parameter.lower), float(parameter.upper)
        if parameter.name.startswith("CPE") and parameter.name.endswith("_1"):
            lower, upper = max(lower, 1e-4), min(upper, 0.9999)
        value = _physical_initial_cap(record, parameter.name, value)
        value = float(np.clip(value, lower, upper))
        params.append(ParameterValue(parameter.name, parameter.unit, value, lower, upper, fixed=parameter.fixed))
        info[parameter.name] = {
            "parameter_name": parameter.name, "initial_value": value, "lower_limit": lower,
            "upper_limit": upper, "source": source, "available": available,
            "training_spectra": int(bundle.parameter_stats.get(parameter.name, {}).get("training_spectra", 0)),
        }
    return params, info


def infer_bundle_records(bundle: PipelineBundle, records: list[SpectrumRecord], *, threshold: float = 4.0) -> list[dict]:
    """Run the trained staged model on already-loaded spectra.

    This is the GUI inference entry point; it deliberately does not fit an
    EEC.  The caller can apply the returned suggestions as separate pipeline
    actions and decide when to start conventional fitting.
    """
    windows = {r.spectrum_id: _predict_window(bundle, r) for r in records}
    masks, diagnostics = deterministic_masks(records, windows, threshold)
    masked_records = _records_with_mask(records, masks)
    topology_predictions = _predict_topologies(bundle, masked_records)
    parameter_predictions = _predict_parameters_batch(bundle, masked_records, [item[0] for item in topology_predictions])
    results = []
    for record, topology_prediction, parameter_prediction in zip(records, topology_predictions, parameter_predictions):
        predicted_circuit, probabilities, confidence, warning = topology_prediction
        _parameters, parameter_info = parameter_prediction
        minimum, maximum = windows[record.spectrum_id]
        mask = masks[record.spectrum_id]
        results.append({
            "spectrum_id": record.spectrum_id,
            "spectrum_key": spectrum_identifier(record.frequency, record.z_real, record.z_imag, record.cycle, record.control),
            "source_project": record.source_project, "sample_id": record.sample_id, "cycle": record.cycle,
            "voltage": record.voltage, "current": record.current, "time": record.time,
            "predicted_f_min": minimum, "predicted_f_max": maximum,
            "deterministic_outlier_mask": (~mask).tolist(), "final_ml_active_mask": mask.tolist(),
            "deterministic_diagnostics": diagnostics[record.spectrum_id],
            "predicted_eec_model": predicted_circuit, "suggested_eec": predicted_circuit,
            "topology_probability": probabilities, "prediction_probability": confidence,
            "topology_prediction_warning": warning, "parameter_predictions": list(parameter_info.values()),
            "initial_guess_only": True, "automatic_eec_fit": False,
            "deterministic_fit_executed": False,
            "deterministic_fit": {"success": None, "skipped": True, "reason": "GUI pipeline controls fitting"},
        })
    return results


def _fit_validation(record: SpectrumRecord, circuit: str, parameters: list[ParameterValue]) -> dict:
    state = CycleState(record.cycle, record.frequency.copy(), record.impedance.copy(), record.voltage or 0.0, record.current or 0.0, record.time)
    state.frequency_window = (float(record.manual_f_min), float(record.manual_f_max)) if record.manual_f_min and record.manual_f_max else (float(np.min(record.frequency)), float(np.max(record.frequency)))
    state.manually_included = np.ones(record.frequency.size, dtype=bool)
    try:
        result = fit_cycle(state, circuit, parameters, FitOptions(method="least_squares"))
    except Exception as error:
        return {"success": False, "error": f"{type(error).__name__}: {error}"}
    return {"success": bool(result.converged), "converged": bool(result.converged), "objective": float(result.objective), "rmse": float(result.rmse), "fitted_parameters": result.fitted_parameters.tolist(), "errors_percent": result.errors_percent.tolist()}


def _loso_summary(records: list[SpectrumRecord], seed: int) -> list[dict]:
    """Evaluate frequency and numbered-topology stages by physical sample."""
    rows = []
    for held_out in sorted({r.sample_id for r in records}):
        train = [r for r in records if r.sample_id != held_out]
        test = [r for r in records if r.sample_id == held_out]
        frequency_preprocessor, frequency_model = _fit_frequency(train, seed)
        frequency_predictions = frequency_model.predict(frequency_preprocessor.transform(test))
        frequency_errors = []
        for record, prediction in zip(test, frequency_predictions):
            target = frequency_targets([record])[0]
            frequency_errors.append(float(np.mean(np.abs(np.asarray(prediction) - target))))
        manual = _manual_masks(train)
        topology_preprocessor, topology_model, _classes = _fit_topology(_records_with_mask(train, manual), seed)
        topology_test = _records_with_mask(test, _manual_masks(test))
        predicted = topology_model.predict(topology_preprocessor.transform(topology_test))
        truth = [r.original_eec_topology for r in test]
        rows.append({
            "held_out_sample": held_out,
            "training_samples": sorted({r.sample_id for r in train}),
            "test_spectra": len(test),
            "frequency_mean_log_endpoint_error": float(np.mean(frequency_errors)),
            "topology_accuracy": float(np.mean(np.asarray(predicted, dtype=str) == np.asarray(truth, dtype=str))),
            "topology_classes_seen_in_training": sorted({r.original_eec_topology for r in train}),
        })
    return rows


def run_pipeline(training_projects: list[Path], validation_project: Path, sample_ids: dict[str, str], output: Path, *, threshold: float = 4.0, seed: int = 42, save_models: bool = True, run_deterministic_fit: bool = False) -> dict:
    bundle, extraction = train_bundle(training_projects, sample_ids, seed)
    validation_sample = sample_ids.get(str(validation_project), sample_ids.get(str(validation_project.resolve()), validation_project.stem))
    validation = load_eisfit_projects([validation_project], {str(validation_project): validation_sample, str(validation_project.resolve()): validation_sample}, require_fit=False, require_frequency_window=False)
    if not validation.records:
        raise ValueError("no validation spectra were extracted")
    windows = {r.spectrum_id: _predict_window(bundle, r) for r in validation.records}
    masks, diagnostics = deterministic_masks(validation.records, windows, threshold)
    masked_records = _records_with_mask(validation.records, masks)
    topology_predictions = _predict_topologies(bundle, masked_records)
    parameter_predictions = _predict_parameters_batch(bundle, masked_records, [item[0] for item in topology_predictions])
    results = []
    for record, masked, topology_prediction, parameter_prediction in zip(validation.records, masked_records, topology_predictions, parameter_predictions):
        predicted_circuit, probabilities, confidence, warning = topology_prediction
        parameters, parameter_info = parameter_prediction
        fit = _fit_validation(masked, predicted_circuit, parameters) if run_deterministic_fit else {"success": None, "skipped": True, "reason": "deterministic EEC fitting disabled"}
        minimum, maximum = windows[record.spectrum_id]
        mask = masks[record.spectrum_id]
        item = {
            "spectrum_id": record.spectrum_id,
            "spectrum_key": spectrum_identifier(record.frequency, record.z_real, record.z_imag, record.cycle, record.control),
            "source_project": record.source_project, "sample_id": record.sample_id, "cycle": record.cycle,
            "voltage": record.voltage, "current": record.current, "time": record.time,
            "predicted_f_min": minimum, "predicted_f_max": maximum,
            "deterministic_outlier_mask": (~mask).tolist(), "final_ml_active_mask": mask.tolist(),
            "deterministic_diagnostics": diagnostics[record.spectrum_id],
            "predicted_eec_model": predicted_circuit, "suggested_eec": predicted_circuit,
            "topology_probability": probabilities, "prediction_probability": confidence,
            "topology_prediction_warning": warning, "parameter_predictions": list(parameter_info.values()),
            "initial_guess_only": True, "automatic_eec_fit": False,
            "deterministic_fit_executed": bool(run_deterministic_fit), "deterministic_fit": fit,
        }
        results.append(item)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / f"{validation_project.stem}_ml_results.json"
    write_ml_results(results_path, results, source_project=str(validation_project), pipeline={"name": "number_aware_staged_eis", "version": "1", "training_samples": list(bundle.training_samples), "frequency_features": "spectrum+voltage+current+time", "outlier_threshold": threshold, "circuit_classes": list(bundle.circuit_classes), "validation_ground_truth": False})
    report = {"training_samples": list(bundle.training_samples), "training_records": len(extraction.records), "validation_sample": validation_sample, "validation_records": len(results), "training_exclusions": extraction.exclusion_counts, "circuit_classes": list(bundle.circuit_classes), "parameter_training": bundle.parameter_stats, "output": str(results_path), "deterministic_fit_requested": bool(run_deterministic_fit), "warnings": ["raw-only validation has no ground-truth frequency, topology, or parameter accuracy metrics", "one-process class has only one training spectrum"]}
    if not run_deterministic_fit:
        report["warnings"].append("deterministic EEC fitting was disabled; ML initial guesses are available")
    report["loso_validation"] = _loso_summary(extraction.records, seed)
    _json_write(output / "report.json", report)
    if save_models:
        joblib.dump(bundle, output / "pipeline.joblib")
    return report


def _parse_mapping(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        path, sample = value.split("=", 1)
        result[str(Path(path))] = sample
        result[str(Path(path).resolve())] = sample
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="append", required=True, metavar="PROJECT=SAMPLE")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-sample", default=None)
    parser.add_argument("--output", type=Path, default=Path("ml/analysis/number_aware_pipeline"))
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--fit", action="store_true", help="also run conventional deterministic EEC fitting (disabled by default)")
    args = parser.parse_args(argv)
    mapping = _parse_mapping(args.train)
    if args.validation_sample:
        mapping[str(args.validation)] = args.validation_sample
        mapping[str(args.validation.resolve())] = args.validation_sample
    train_projects = [Path(value.split("=", 1)[0]) for value in args.train]
    report = run_pipeline(train_projects, args.validation, mapping, args.output, threshold=args.threshold, run_deterministic_fit=args.fit)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
