"""Leakage-controlled frequency-range -> outlier -> topology experiments."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from eis_model import CycleState
from eis_services import analyze_outliers, circuit_parameters

from .dataset import SpectrumRecord
from .frequency_range import _features, _fit_predict, _models as range_models, _targets
from .metrics import multiclass_brier, prediction_metrics
from .preprocessing import SpectrumPreprocessor
from .topology_classifier import _models as topology_models
from .outlier_cache import OutlierCache


def _selected_record(record: SpectrumRecord, frequency_window: tuple[float, float], *, outliers: bool) -> SpectrumRecord:
    """Apply a window and the existing outlier algorithm to a temporary state."""
    frequency = np.asarray(record.frequency, dtype=float)
    impedance = record.impedance
    minimum, maximum = sorted((float(frequency_window[0]), float(frequency_window[1])))
    minimum = max(minimum, float(np.min(frequency)))
    maximum = min(maximum, float(np.max(frequency)))
    if not maximum > minimum:
        raise ValueError("frequency window does not overlap measured data")
    state = CycleState(
        cycle=record.cycle,
        frequency_hz=frequency.copy(),
        impedance=impedance.copy(),
        potential_v=float(record.voltage or 0.0),
        current_ma=float(record.current or 0.0),
        time_s=record.time,
        frequency_window=(minimum, maximum),
        circuit=record.original_eec_topology,
    )
    if outliers:
        analysis = analyze_outliers(
            state,
            1.0,
            circuit_parameters(record.original_eec_topology),
        )
        state.apply_outliers(analysis.outlier_indices)
    active = state.included
    if int(np.count_nonzero(active)) < 3:
        raise ValueError("fewer than three active points after preprocessing")
    return replace(
        record,
        frequency=frequency[active],
        z_real=impedance.real[active],
        z_imag=impedance.imag[active],
        cleaned_frequency=None,
        cleaned_z_real=None,
        cleaned_z_imag=None,
    )


def _preprocess_records(records, windows, *, outliers, cache: OutlierCache, context=None):
    if not outliers:
        return list(records), []
    return cache.process(records, windows, context=context)


def _range_predictions(train, test, model_name: str, grid_size: int, seed: int):
    x_train, x_test = _features(train, test, "spectrum_only", grid_size)
    model = range_models(seed)[model_name]
    return _fit_predict(model, x_train, _targets(train), x_test)


def _nested_training_ranges(train, model_name: str, grid_size: int, seed: int):
    """Out-of-fold range predictions for outer-fold topology training spectra."""
    result = {}
    for inner_held_out in sorted({r.sample_id for r in train}):
        inner_train = [r for r in train if r.sample_id != inner_held_out]
        inner_test = [r for r in train if r.sample_id == inner_held_out]
        minimum, maximum = _range_predictions(inner_train, inner_test, model_name, grid_size, seed)
        result.update({r.spectrum_id: (10**float(lo), 10**float(hi)) for r, lo, hi in zip(inner_test, minimum, maximum)})
    return result


def _range_quality(record, predicted):
    manual_min, manual_max = np.log10(record.manual_f_min), np.log10(record.manual_f_max)
    predicted_min, predicted_max = np.log10(predicted[0]), np.log10(predicted[1])
    intersection = max(0.0, min(manual_max, predicted_max) - max(manual_min, predicted_min))
    union = max(manual_max, predicted_max) - min(manual_min, predicted_min)
    iou = intersection / union if union > 0 else 1.0
    return {
        "manual_f_min": float(record.manual_f_min), "manual_f_max": float(record.manual_f_max),
        "predicted_f_min": float(predicted[0]), "predicted_f_max": float(predicted[1]),
        "frequency_range_IoU": float(iou),
        "delta_log_fmin": float(predicted_min - manual_min), "delta_log_fmax": float(predicted_max - manual_max),
    }


def _topology_rows(train, test, pipeline, held_out, model_name, grid_size, seed, metadata):
    classes = sorted({r.electrochemical_topology for r in train + test})
    preprocessor = SpectrumPreprocessor(grid_size=grid_size, use_metadata=metadata, spectrum_mode="raw")
    x_train = preprocessor.fit_transform(train)
    x_test = preprocessor.transform(test)
    model = topology_models(seed)[model_name]
    model.fit(x_train, [r.topology_label for r in train])
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    model_classes = list(model.classes_)
    rows = []
    for record, predicted, probability in zip(test, predictions, probabilities):
        row = {"spectrum_id": record.spectrum_id, "sample_id": record.sample_id, "voltage": record.voltage,
               "current": record.current, "time": record.time, "device_setup": record.device_setup,
               "l0_required": record.l0_required_in_manual_fit, "original_eec_string": record.original_eec_topology,
               "canonical_topology": record.electrochemical_topology, "pipeline": pipeline,
               "validation_fold": held_out, "topology_model": model_name,
               "topology_prediction": str(predicted), "topology_correct": bool(str(predicted) == record.electrochemical_topology),
               "confidence": float(np.max(probability))}
        for label in classes:
            row[f"probability_{label}"] = float(probability[model_classes.index(label)]) if label in model_classes else 0.0
        rows.append(row)
    return rows


def _topology_rows_multi(train, test, pipeline, held_out, model_names, grid_size, seed, metadata):
    """Prepare fold-local features once, then fit the requested unchanged models."""
    classes = sorted({r.electrochemical_topology for r in train + test})
    preprocessor = SpectrumPreprocessor(grid_size=grid_size, use_metadata=metadata, spectrum_mode="raw")
    x_train = preprocessor.fit_transform(train)
    x_test = preprocessor.transform(test)
    labels = [r.topology_label for r in train]
    all_rows = []
    for model_name in model_names:
        model = topology_models(seed)[model_name]
        model.fit(x_train, labels)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)
        model_classes = list(model.classes_)
        for record, predicted, probability in zip(test, predictions, probabilities):
            row = {"spectrum_id": record.spectrum_id, "sample_id": record.sample_id, "voltage": record.voltage,
                   "current": record.current, "time": record.time, "device_setup": record.device_setup,
                   "l0_required": record.l0_required_in_manual_fit, "original_eec_string": record.original_eec_topology,
                   "canonical_topology": record.electrochemical_topology, "pipeline": pipeline,
                   "validation_fold": held_out, "topology_model": model_name,
                   "topology_prediction": str(predicted), "topology_correct": bool(str(predicted) == record.electrochemical_topology),
                   "confidence": float(np.max(probability))}
            for label in classes:
                row[f"probability_{label}"] = float(probability[model_classes.index(label)]) if label in model_classes else 0.0
            all_rows.append(row)
    return all_rows


def _metrics(frame, classes):
    values = prediction_metrics(frame.rename(columns={"topology_prediction": "predicted_topology", "canonical_topology": "true_topology"}), classes)
    return {key: value for key, value in values.items() if isinstance(value, (int, float, np.floating))}


def run_frequency_topology_pipeline(records: list[SpectrumRecord], *, grid_size=64, seed=42, outlier_threshold=1.0, cache_dir=Path("ml_outlier_cache"), workers=1):
    """Run A/B/C with six-fold LOSO and fold-local preprocessing.

    The existing outlier threshold is exposed for reproducibility; the default
    matches the application's command-line default. ``outlier_threshold`` is
    temporarily passed through the unchanged service function.
    """
    if len({r.sample_id for r in records}) < 2:
        raise ValueError("At least two samples are required")
    cache = OutlierCache(cache_dir, threshold=outlier_threshold, workers=workers)
    rows, exclusions = [], []
    samples = sorted({r.sample_id for r in records})
    topology_names = ("random_forest", "hist_gradient_boosting")
    for held_out in samples:
        train = [r for r in records if r.sample_id != held_out]
        test = [r for r in records if r.sample_id == held_out]
        # A: raw baseline
        rows.extend(_topology_rows_multi(train, test, "A_raw", held_out, topology_names, grid_size, seed, False))
        # B: manual window, then unchanged existing outlier detector
        manual_windows = {r.spectrum_id: (r.manual_f_min, r.manual_f_max) for r in train + test}
        b_train, failed = _preprocess_records(train, manual_windows, outliers=True, cache=cache, context={"pipeline": "B_manual_range", "held_out_sample": held_out}); exclusions.extend(failed)
        b_test, failed = _preprocess_records(test, manual_windows, outliers=True, cache=cache, context={"pipeline": "B_manual_range", "held_out_sample": held_out}); exclusions.extend(failed)
        if b_train and b_test:
            rows.extend(_topology_rows_multi(b_train, b_test, "B_manual_range", held_out, topology_names, grid_size, seed, False))
        # C: outer-test predictions and nested out-of-fold predictions for topology training.
        for range_model in ("random_forest", "hist_gradient_boosting"):
            train_windows = _nested_training_ranges(train, range_model, grid_size, seed)
            minimum, maximum = _range_predictions(train, test, range_model, grid_size, seed)
            test_windows = {r.spectrum_id: (10**float(lo), 10**float(hi)) for r, lo, hi in zip(test, minimum, maximum)}
            windows = {**train_windows, **test_windows}
            c_train, failed = _preprocess_records(train, windows, outliers=True, cache=cache, context={"pipeline": f"C_ml_range_{range_model}", "frequency_model": range_model, "held_out_sample": held_out, "range_source": "nested_training_or_outer_test"}); exclusions.extend(failed)
            c_test, failed = _preprocess_records(test, windows, outliers=True, cache=cache, context={"pipeline": f"C_ml_range_{range_model}", "frequency_model": range_model, "held_out_sample": held_out, "range_source": "nested_training_or_outer_test"}); exclusions.extend(failed)
            if c_train and c_test:
                pipeline = f"C_ml_range_{range_model}"
                new_rows = _topology_rows_multi(c_train, c_test, pipeline, held_out, topology_names, grid_size, seed, False)
                quality = {r.spectrum_id: _range_quality(r, test_windows[r.spectrum_id]) for r in test}
                for row in new_rows: row.update(quality[row["spectrum_id"]])
                rows.extend(new_rows)
    predictions = pd.DataFrame(rows)
    classes = sorted(predictions["canonical_topology"].dropna().unique())
    metric_rows = []
    for keys, frame in predictions.groupby(["pipeline", "topology_model"]):
        metric_rows.append({"pipeline": keys[0], "topology_model": keys[1], **_metrics(frame, classes)})
        for held_out, fold in frame.groupby("validation_fold"):
            metric_rows.append({"pipeline": keys[0], "topology_model": keys[1], "held_out_sample": held_out, **_metrics(fold, classes)})
    metrics = pd.DataFrame(metric_rows)
    cache_report = cache.write_report(Path(cache_dir) / "run_report.json")
    return predictions, metrics, pd.DataFrame(exclusions), cache_report


def save_pipeline_results(predictions: pd.DataFrame, metrics: pd.DataFrame, exclusions: pd.DataFrame, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(directory / "per_spectrum_results.csv", index=False)
    metrics.to_csv(directory / "metrics.csv", index=False)
    if not exclusions.empty: exclusions.to_csv(directory / "preprocessing_exclusions.csv", index=False)
    matrix_dir = directory / "confusion_matrices"
    for (pipeline, model), frame in predictions.groupby(["pipeline", "topology_model"]):
        target = matrix_dir / str(pipeline) / str(model); target.mkdir(parents=True, exist_ok=True)
        labels = sorted(set(frame.canonical_topology) | set(frame.topology_prediction))
        for fold, fold_frame in frame.groupby("validation_fold"):
            pd.DataFrame(confusion_matrix(fold_frame.canonical_topology, fold_frame.topology_prediction, labels=labels), index=labels, columns=labels).to_csv(target / f"held_out_{fold}.csv")
        pd.DataFrame(confusion_matrix(frame.canonical_topology, frame.topology_prediction, labels=labels), index=labels, columns=labels).to_csv(target / "aggregated.csv")
    # Diagnostics required by the prompt.
    c = predictions[predictions.pipeline.str.startswith("C_")].copy()
    if not c.empty:
        c["iou_group"] = pd.cut(c.frequency_range_IoU, [-np.inf, .5, .75, .9, np.inf], labels=["IoU <= 0.50", "0.50 < IoU <= 0.75", "0.75 < IoU <= 0.90", "IoU > 0.90"])
        group_rows = []
        for keys, frame in c.groupby(["pipeline", "topology_model", "iou_group"], observed=False):
            group_rows.append({"pipeline": keys[0], "topology_model": keys[1], "iou_group": str(keys[2]), **_metrics(frame, sorted(c.canonical_topology.unique()))})
        pd.DataFrame(group_rows).to_csv(directory / "topology_by_frequency_iou.csv", index=False)
    pd.DataFrame({"leakage_check": ["outer sample excluded from range and topology training; preprocessing grids/statistics fit on training records only; C training ranges are nested out-of-fold"]}).to_csv(directory / "leakage_check.csv", index=False)
