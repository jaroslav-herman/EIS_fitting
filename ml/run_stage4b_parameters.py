"""Lightweight Stage 4B parameter prediction with impedance-scale features.

This module deliberately treats the Stage 4A 192-dimensional cache as a fixed
spectral representation.  It adds only features calculated from the manually
active points, evaluates three small feature configurations with strict LOSO,
and runs HGB only for parameters for which Ridge remains weak.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import load_eisfit_projects, SpectrumRecord
from .evaluate_frequency_limit_ml import _clean_record, _manual_masks
from .evaluate_stage4_parameters import extract_parameter_targets
from .parameter_prediction import (
    ALPHA_PARAMETERS,
    inverse_target,
    transform_target,
)


TRAINING_SAMPLES = ("129", "140", "150", "157", "159", "181")
PARAMETERS = ("R1", "Q1", "R2", "Q2", "alpha1", "alpha2")
POSITIVE_PARAMETERS = ("R1", "Q1", "R2", "Q2")
CONFIGURATIONS = ("A", "B", "C")
FEATURE_CACHE = Path("ml/cache/stage4a_parameter_features.npz")
STAGE4A_METADATA = Path("ml/cache/stage4a_parameter_metadata.csv")
STAGE4B_CACHE = Path("ml/cache/stage4b_parameter_features.npz")
STAGE4B_METADATA = Path("ml/cache/stage4b_parameter_metadata.csv")
OUTPUT = Path("ml/analysis/stage4b_parameters")
RIDGE_ALPHA = 1.0
HGB_SETTINGS = {
    "max_iter": 12,
    "learning_rate": 0.06,
    "l2_regularization": 1.0,
    "random_state": 42,
}


def multiplicative_error(true_value: float, predicted_value: float) -> float:
    true_value, predicted_value = float(true_value), float(predicted_value)
    if true_value <= 0 or predicted_value <= 0:
        raise ValueError("multiplicative error requires positive values")
    return max(predicted_value / true_value, true_value / predicted_value)


def loso_splits(samples=TRAINING_SAMPLES):
    samples = tuple(str(s) for s in samples)
    if tuple(sorted(samples)) != tuple(sorted(TRAINING_SAMPLES)) or "178" in samples:
        raise ValueError("Stage 4B requires exactly the six samples and excludes 178")
    for held_out in samples:
        yield held_out, tuple(s for s in samples if s != held_out)


def _manual_records(projects, records):
    masks = _manual_masks(projects, {record.spectrum_id: record for record in records})
    cleaned = []
    for record in records:
        mask = masks.get(record.spectrum_id)
        if mask is None:
            raise RuntimeError(f"manual mask missing for {record.spectrum_id}")
        if record.manual_f_min is not None and record.manual_f_max is not None:
            mask &= (record.frequency >= record.manual_f_min) & (record.frequency <= record.manual_f_max)
        cleaned_record = _clean_record(record, mask)
        if cleaned_record.frequency.size < 3:
            raise RuntimeError(f"too few manually active points for {record.spectrum_id}")
        cleaned.append(cleaned_record)
    return cleaned


def _robust_endpoint(cleaned: SpectrumRecord, high: bool, count: int = 5) -> tuple[float, float, float]:
    """Return median Re, Im and |Z| among up to five nearest endpoint points."""
    frequency, impedance = cleaned.arrays("raw")
    order = np.argsort(frequency)
    indices = order[-count:] if high else order[:count]
    values = impedance[indices]
    return (float(np.median(values.real)), float(np.median(values.imag)), float(np.median(np.abs(values))))


def calculate_absolute_impedance_features(cleaned: SpectrumRecord) -> dict[str, float]:
    """Calculate Stage 4B scale features from one manually cleaned spectrum.

    The four distribution features are log10-transformed positive magnitudes.
    Endpoint values are medians of the five highest/lowest-frequency active
    points, making the representative values less sensitive to one point.
    """
    _, impedance = cleaned.arrays("raw")
    magnitude = np.abs(impedance)
    if magnitude.size < 3 or not np.all(np.isfinite(magnitude)) or np.any(magnitude <= 0):
        raise ValueError("manual impedance must contain at least three finite positive magnitudes")
    re_high, im_high, abs_high = _robust_endpoint(cleaned, high=True)
    re_low, im_low, abs_low = _robust_endpoint(cleaned, high=False)
    return {
        "log10_median_abs_Z": float(np.log10(np.median(magnitude))),
        "log10_mean_abs_Z": float(np.log10(np.mean(magnitude))),
        "log10_max_abs_Z": float(np.log10(np.max(magnitude))),
        "log10_min_abs_Z": float(np.log10(np.min(magnitude))),
        "Re_Z_high": re_high,
        "Im_Z_high": im_high,
        "Re_Z_low": re_low,
        "Im_Z_low": im_low,
        "log10_abs_Z_high": float(np.log10(abs_high)),
        "log10_abs_Z_low": float(np.log10(abs_low)),
    }


ABSOLUTE_FEATURE_NAMES = (
    "log10_median_abs_Z", "log10_mean_abs_Z", "log10_max_abs_Z", "log10_min_abs_Z",
    "Re_Z_high", "Im_Z_high", "Re_Z_low", "Im_Z_low",
    "log10_abs_Z_high", "log10_abs_Z_low",
)


def _load_stage4a_cache(cache_path: Path, spectrum_ids: list[str]) -> np.ndarray:
    if not cache_path.exists():
        raise FileNotFoundError(f"Stage 4A cache is required and was not found: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as cached:
        x = np.asarray(cached["X"], dtype=float)
        ids = np.asarray(cached["spectrum_ids"], dtype=str)
    expected = np.asarray(spectrum_ids, dtype=str)
    if x.shape != (len(expected), 192) or not np.array_equal(ids, expected):
        raise RuntimeError("Stage 4A cache is incompatible with the loaded six-sample dataset")
    return x


def _load_or_create_stage4b_cache(records, cleaned, targets, cache_path=STAGE4B_CACHE, metadata_path=STAGE4B_METADATA):
    ids = [r.spectrum_id for r in records]
    if cache_path.exists() and metadata_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                x = np.asarray(cached["X_additional"], dtype=float)
                cached_ids = np.asarray(cached["spectrum_ids"], dtype=str)
                names = tuple(np.asarray(cached["feature_names"], dtype=str).tolist())
            if x.shape == (len(records), len(ABSOLUTE_FEATURE_NAMES)) and np.array_equal(cached_ids, ids) and names == ABSOLUTE_FEATURE_NAMES:
                return x, True
        except (OSError, KeyError, ValueError):
            pass
    rows = [calculate_absolute_impedance_features(r) for r in cleaned]
    x = np.asarray([[row[name] for name in ABSOLUTE_FEATURE_NAMES] for row in rows], dtype=float)
    metadata = pd.DataFrame({
        "spectrum_id": ids,
        "sample_id": [r.sample_id for r in records],
        "voltage": [r.voltage for r in records],
        "current": [r.current for r in records],
        "topology": targets.topology.to_numpy(),
        **{name: x[:, i] for i, name in enumerate(ABSOLUTE_FEATURE_NAMES)},
    })
    for parameter in ("R0", "R1", "Q1", "alpha1", "R2", "Q2", "alpha2"):
        metadata[parameter] = targets[parameter]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_additional=x,
        spectrum_ids=np.asarray(ids, dtype=str),
        sample_ids=np.asarray([r.sample_id for r in records], dtype=str),
        voltage=np.asarray([r.voltage for r in records], dtype=float),
        current=np.asarray([r.current for r in records], dtype=float),
        topology=np.asarray(targets.topology.to_numpy(), dtype=str),
        feature_names=np.asarray(ABSOLUTE_FEATURE_NAMES, dtype=str),
        **{parameter: targets[parameter].to_numpy(float) for parameter in ("R0", "R1", "Q1", "alpha1", "R2", "Q2", "alpha2")},
    )
    metadata.to_csv(metadata_path, index=False)
    return x, False


def _fold_append(train_extra, test_extra, train_metadata, test_metadata, names):
    """Fill and standardize appended features using training rows only."""
    train_extra = np.asarray(train_extra, dtype=float).copy()
    test_extra = np.asarray(test_extra, dtype=float).copy()
    train_metadata = np.asarray(train_metadata, dtype=float).copy()
    test_metadata = np.asarray(test_metadata, dtype=float).copy()
    train = np.hstack([train_metadata, train_extra])
    test = np.hstack([test_metadata, test_extra])
    fill = np.nanmedian(train, axis=0)
    fill[~np.isfinite(fill)] = 0.0
    train = np.where(np.isfinite(train), train, fill)
    test = np.where(np.isfinite(test), test, fill)
    return train, test, tuple(names)


def _features(base_x, extra_x, metadata, train_indices, test_indices, configuration):
    if configuration == "A":
        names = ("voltage",)
        train_meta = metadata.iloc[train_indices].loc[:, ["voltage"]].to_numpy(float)
        test_meta = metadata.iloc[test_indices].loc[:, ["voltage"]].to_numpy(float)
        return _fold_append(np.empty((len(train_indices), 0)), np.empty((len(test_indices), 0)), train_meta, test_meta, names)[:2], names
    if configuration == "B":
        names = ("voltage", "current")
        train_meta = metadata.iloc[train_indices].loc[:, list(names)].to_numpy(float)
        test_meta = metadata.iloc[test_indices].loc[:, list(names)].to_numpy(float)
        return _fold_append(np.empty((len(train_indices), 0)), np.empty((len(test_indices), 0)), train_meta, test_meta, names)[:2], names
    names = ("voltage", "current") + ABSOLUTE_FEATURE_NAMES
    train_meta = metadata.iloc[train_indices].loc[:, ["voltage", "current"]].to_numpy(float)
    test_meta = metadata.iloc[test_indices].loc[:, ["voltage", "current"]].to_numpy(float)
    train, test, _ = _fold_append(extra_x[train_indices], extra_x[test_indices], train_meta, test_meta, names)
    return (np.hstack([base_x[train_indices], train]), np.hstack([base_x[test_indices], test])), names


def _metric_row(true, predicted, parameter, model, feature_configuration, held_out_sample=None):
    true, predicted = np.asarray(true, float), np.asarray(predicted, float)
    error = predicted - true
    row = {
        "model": model, "parameter": parameter, "feature_configuration": feature_configuration,
        "held_out_sample": held_out_sample, "spectra": int(true.size),
        "correlation": float(np.corrcoef(true, predicted)[0, 1]) if true.size > 1 and np.std(true) and np.std(predicted) else np.nan,
    }
    if parameter in ALPHA_PARAMETERS:
        absolute = np.abs(error)
        row.update({"mae": float(np.mean(absolute)), "rmse": float(np.sqrt(np.mean(error ** 2))), "median_absolute_error": float(np.median(absolute)), "within_0.02": float(np.mean(absolute <= .02)), "within_0.05": float(np.mean(absolute <= .05)), "within_0.10": float(np.mean(absolute <= .10))})
    else:
        log_error = np.log10(predicted) - np.log10(true)
        absolute = np.abs(log_error)
        row.update({"mae_log10": float(np.mean(absolute)), "rmse_log10": float(np.sqrt(np.mean(log_error ** 2))), "median_absolute_log10": float(np.median(absolute)), "within_x1.25": float(np.mean(absolute <= np.log10(1.25))), "within_x2": float(np.mean(absolute <= np.log10(2))), "within_x5": float(np.mean(absolute <= np.log10(5))), "within_x10": float(np.mean(absolute <= 1.0))})
    return row


def _prediction_rows(metadata, indices, parameter, truth, predicted, model, configuration, held_out):
    rows = []
    positive = parameter in POSITIVE_PARAMETERS
    for index, true, pred in zip(indices, truth, predicted):
        row = metadata.iloc[index]
        transformed_error = float(transform_target([pred], parameter)[0] - transform_target([true], parameter)[0])
        rows.append({
            "sample_id": row.sample_id, "spectrum_id": row.spectrum_id, "voltage": row.voltage, "current": row.current,
            "topology": row.topology, "parameter": parameter, "true_value": float(true), "predicted_value": float(pred),
            "log_error": transformed_error, "absolute_error": float(abs(pred - true)),
            "multiplicative_error": multiplicative_error(true, pred) if positive else np.nan,
            "feature_configuration": configuration, "model": model, "held_out_sample": held_out,
        })
    return rows


def _fit_model(model_name, x_train, y_train, parameter):
    if model_name == "RIDGE":
        model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    elif model_name == "HGB":
        model = HistGradientBoostingRegressor(**HGB_SETTINGS)
    else:
        raise ValueError(model_name)
    model.fit(x_train, transform_target(y_train, parameter))
    return model


def _primary_metric(row, parameter):
    return float(row["mae"] if parameter in ALPHA_PARAMETERS else row["mae_log10"])


def _plot_best(frame, output, parameter):
    if frame.empty:
        return
    output.mkdir(parents=True, exist_ok=True)
    (output.parent / "error_vs_voltage").mkdir(parents=True, exist_ok=True)
    true = frame.true_value.to_numpy(float); pred = frame.predicted_value.to_numpy(float)
    low, high = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
    fig, ax = plt.subplots(figsize=(5, 4)); ax.scatter(true, pred, s=6, alpha=.4); ax.plot([low, high], [low, high], "k--"); ax.set_xlabel("manual fitted value"); ax.set_ylabel("predicted value"); ax.set_title(parameter); fig.tight_layout(); fig.savefig(output / f"{parameter}.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); ax.scatter(frame.voltage, frame.log_error, s=6, alpha=.4); ax.axhline(0, color="k", linestyle="--"); ax.set_xlabel("voltage"); ax.set_ylabel("transformed prediction error"); ax.set_title(parameter); fig.tight_layout(); fig.savefig(output.parent / "error_vs_voltage" / f"{parameter}.png", dpi=130); plt.close(fig)


def run(projects: list[Path], output: Path = OUTPUT, stage4a_cache: Path = FEATURE_CACHE, stage4b_cache: Path = STAGE4B_CACHE, stage4b_metadata: Path = STAGE4B_METADATA):
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=True)
    mapping = {str(path): path.name.split(".")[0] for path in projects}
    extraction = load_eisfit_projects(projects, mapping, require_fit=True)
    records = extraction.records
    if sorted({r.sample_id for r in records}) != sorted(TRAINING_SAMPLES) or "178" in {r.sample_id for r in records}:
        raise RuntimeError("Stage 4B requires exactly samples 129, 140, 150, 157, 159, 181")
    cleaned = _manual_records(projects, records)
    base_x = _load_stage4a_cache(stage4a_cache, [r.spectrum_id for r in records])
    targets = extract_parameter_targets(projects, records)
    extra_x, stage4b_cache_hit = _load_or_create_stage4b_cache(records, cleaned, targets, stage4b_cache, stage4b_metadata)
    metadata = pd.DataFrame({
        "spectrum_id": [r.spectrum_id for r in records], "sample_id": [r.sample_id for r in records],
        "voltage": [r.voltage for r in records], "current": [r.current for r in records],
        "topology": targets.topology.to_numpy(),
        **{name: extra_x[:, i] for i, name in enumerate(ABSOLUTE_FEATURE_NAMES)},
        **{parameter: targets[parameter].to_numpy() for parameter in ("R0", "R1", "Q1", "alpha1", "R2", "Q2", "alpha2")},
    })
    metadata.to_csv(stage4b_metadata, index=False)

    prediction_rows = []
    metric_rows = []
    # Baseline is the same training-fold median used by Stage 4A.
    for held_out, _ in loso_splits():
        train_mask = metadata.sample_id != held_out; test_mask = metadata.sample_id == held_out
        for parameter in PARAMETERS:
            train_indices = np.flatnonzero(train_mask & metadata[parameter].notna().to_numpy())
            test_indices = np.flatnonzero(test_mask & metadata[parameter].notna().to_numpy())
            if not train_indices.size or not test_indices.size:
                continue
            y_train = metadata.iloc[train_indices][parameter].to_numpy(float)
            y_test = metadata.iloc[test_indices][parameter].to_numpy(float)
            baseline = np.full(y_test.size, np.median(y_train))
            prediction_rows.extend(_prediction_rows(metadata, test_indices, parameter, y_test, baseline, "MEDIAN_BASELINE", "STAGE4A_BASELINE", held_out))
            metric_rows.append(_metric_row(y_test, baseline, parameter, "MEDIAN_BASELINE", "STAGE4A_BASELINE", held_out))
            for configuration in CONFIGURATIONS:
                (x_train_extra, x_test_extra), _ = _features(base_x, extra_x, metadata, train_indices, test_indices, configuration)
                if configuration == "A":
                    x_train, x_test = np.hstack([base_x[train_indices], x_train_extra]), np.hstack([base_x[test_indices], x_test_extra])
                elif configuration == "B":
                    x_train, x_test = np.hstack([base_x[train_indices], x_train_extra]), np.hstack([base_x[test_indices], x_test_extra])
                else:
                    x_train, x_test = x_train_extra, x_test_extra
                model = _fit_model("RIDGE", x_train, y_train, parameter)
                prediction = inverse_target(model.predict(x_test), parameter)
                prediction_rows.extend(_prediction_rows(metadata, test_indices, parameter, y_test, prediction, "RIDGE", configuration, held_out))
                metric_rows.append(_metric_row(y_test, prediction, parameter, "RIDGE", configuration, held_out))

    predictions = pd.DataFrame(prediction_rows)
    ridge_metrics = pd.DataFrame(metric_rows)
    ridge_overall = pd.DataFrame([
        _metric_row(frame.true_value, frame.predicted_value, key[1], key[0], key[2])
        for key, frame in predictions[predictions.model == "RIDGE"].groupby(["model", "parameter", "feature_configuration"])
    ])
    # Select the best Ridge feature configuration using held-out primary error.
    best_rows = []
    for parameter in PARAMETERS:
        candidates = ridge_overall[(ridge_overall.model == "RIDGE") & (ridge_overall.parameter == parameter)]
        best = candidates.loc[candidates.apply(lambda row: _primary_metric(row, parameter), axis=1).idxmin()]
        best_rows.append({"parameter": parameter, "feature_configuration": best.feature_configuration, "model": "RIDGE", "selection_metric": "mae" if parameter in ALPHA_PARAMETERS else "mae_log10", "selection_value": _primary_metric(best, parameter)})

    # A compact fixed HGB pass for Ridge-weak parameters only.
    weak = []
    for row in best_rows:
        metric = ridge_overall[(ridge_overall.model == "RIDGE") & (ridge_overall.parameter == row["parameter"]) & (ridge_overall.feature_configuration == row["feature_configuration"])].iloc[0]
        practical = metric.get("within_0.05", np.nan) if row["parameter"] in ALPHA_PARAMETERS else metric.get("within_x2", np.nan)
        if not np.isfinite(practical) or practical < 0.5:
            weak.append(row["parameter"])
    hgb_rows = []
    for parameter in weak:
        configuration = next(row["feature_configuration"] for row in best_rows if row["parameter"] == parameter)
        for held_out, _ in loso_splits():
            train_indices = np.flatnonzero((metadata.sample_id != held_out) & metadata[parameter].notna().to_numpy())
            test_indices = np.flatnonzero((metadata.sample_id == held_out) & metadata[parameter].notna().to_numpy())
            y_train = metadata.iloc[train_indices][parameter].to_numpy(float); y_test = metadata.iloc[test_indices][parameter].to_numpy(float)
            (x_train_extra, x_test_extra), _ = _features(base_x, extra_x, metadata, train_indices, test_indices, configuration)
            if configuration in ("A", "B"):
                x_train = np.hstack([base_x[train_indices], x_train_extra]); x_test = np.hstack([base_x[test_indices], x_test_extra])
            else:
                x_train, x_test = x_train_extra, x_test_extra
            model = _fit_model("HGB", x_train, y_train, parameter)
            prediction = inverse_target(model.predict(x_test), parameter)
            hgb_rows.extend(_prediction_rows(metadata, test_indices, parameter, y_test, prediction, "HGB", configuration, held_out))
            metric_rows.append(_metric_row(y_test, prediction, parameter, "HGB", configuration, held_out))
    predictions = pd.DataFrame(prediction_rows + hgb_rows)
    all_metrics = pd.DataFrame(metric_rows)
    overall = pd.DataFrame([_metric_row(frame.true_value, frame.predicted_value, key[1], key[0], key[2]) for key, frame in predictions.groupby(["model", "parameter", "feature_configuration"])])
    per_sample = all_metrics.copy()
    predictions.to_csv(output / "predictions.csv", index=False)
    overall.to_csv(output / "overall_metrics.csv", index=False)
    per_sample.to_csv(output / "per_sample_metrics.csv", index=False)

    # Stage 4A is read from its completed artifact, not recomputed.
    stage4a_overall = pd.read_csv(Path("ml/analysis/stage4a_parameters/overall_metrics.csv"))
    comparison = []
    for parameter in PARAMETERS:
        for _, row in stage4a_overall[(stage4a_overall.parameter == parameter) & (stage4a_overall.model == "RIDGE")].iterrows():
            comparison.append({"stage": "Stage4A", "parameter": parameter, "model": "RIDGE", "feature_configuration": "A", **row.to_dict()})
        for _, row in overall[(overall.parameter == parameter) & (overall.model == "RIDGE")].iterrows():
            comparison.append({"stage": "Stage4B", **row.to_dict()})
    pd.DataFrame(comparison).to_csv(output / "feature_comparison.csv", index=False)
    model_comparison = overall.copy(); model_comparison.to_csv(output / "model_comparison.csv", index=False)

    # Best final selection: lowest held-out primary error, then require an
    # improvement over the identical Stage 4A median baseline on that metric.
    final_models = []
    for row in best_rows:
        candidates = overall[(overall.parameter == row["parameter"]) & (overall.model.isin(["RIDGE", "HGB"]))]
        candidates = candidates.sort_values("mae" if row["parameter"] in ALPHA_PARAMETERS else "mae_log10")
        selected = candidates.iloc[0]
        baseline = overall[(overall.parameter == row["parameter"]) & (overall.model == "MEDIAN_BASELINE")].iloc[0]
        metric = "mae" if row["parameter"] in ALPHA_PARAMETERS else "mae_log10"
        practical = "within_0.05" if row["parameter"] in ALPHA_PARAMETERS else "within_x2"
        meaningful = float(selected[metric]) < float(baseline[metric]) and float(selected[practical]) > float(baseline[practical])
        row.update({"selected_model": selected.model, "selected_feature_configuration": selected.feature_configuration, "meaningful_improvement_over_stage4a_baseline": bool(meaningful)})
        if meaningful:
            final_models.append(row)
    pd.DataFrame(best_rows).to_csv(output / "parameter_summary.csv", index=False)

    model_dir = output / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    saved_models = []
    for selected in final_models:
        parameter = selected["parameter"]; configuration = selected["selected_feature_configuration"]; model_name = selected["selected_model"]
        indices = np.flatnonzero(metadata[parameter].notna().to_numpy()); y = metadata.iloc[indices][parameter].to_numpy(float)
        (x_extra, _), _ = _features(base_x, extra_x, metadata, indices, indices, configuration)
        if configuration in ("A", "B"):
            x = np.hstack([base_x[indices], x_extra])
        else:
            x = x_extra
        model = _fit_model(model_name, x, y, parameter)
        path = model_dir / f"{parameter}_{model_name.lower()}.joblib"
        feature_names = [f"stage4a_{i:03d}" for i in range(192)] + (["voltage"] if configuration == "A" else ["voltage", "current"] if configuration == "B" else ["voltage", "current", *ABSOLUTE_FEATURE_NAMES])
        target_transformation = "log10" if parameter in POSITIVE_PARAMETERS else "logit_alpha"
        model_parameters = {"alpha": RIDGE_ALPHA} if model_name == "RIDGE" else dict(HGB_SETTINGS)
        joblib.dump({"model": model, "parameter": parameter, "model_type": model_name, "feature_configuration": configuration, "feature_names": feature_names, "target_transformation": target_transformation, "model_parameters": model_parameters, "training_samples": list(TRAINING_SAMPLES), "training_spectra": int(len(indices))}, path)
        saved_models.append({"parameter": parameter, "path": str(path), "model": model_name, "feature_configuration": configuration, "target_transformation": target_transformation, "model_parameters": model_parameters, "feature_names": feature_names})

    best_frame_rows = []
    for selected in best_rows:
        frame = predictions[(predictions.parameter == selected["parameter"]) & (predictions.model == selected["selected_model"]) & (predictions.feature_configuration == selected["selected_feature_configuration"])]
        _plot_best(frame, output / "predicted_vs_true", selected["parameter"])
        metric = "mae" if selected["parameter"] in ALPHA_PARAMETERS else "mae_log10"
        for bin_name, group in frame.groupby(pd.cut(frame.voltage, bins=4, duplicates="drop"), observed=True):
            best_frame_rows.append({"parameter": selected["parameter"], "model": selected["selected_model"], "feature_configuration": selected["selected_feature_configuration"], "voltage_bin": str(bin_name), "spectra": len(group), "mae": float(np.mean(group.absolute_error)), "median_absolute_error": float(np.median(group.absolute_error)), "metric": metric})
    pd.DataFrame(best_frame_rows).to_csv(output / "voltage_error_summary.csv", index=False)
    config = {
        "training_samples": list(TRAINING_SAMPLES), "parameters": list(PARAMETERS), "sample_178_used": False,
        "stage4a_cache": str(stage4a_cache), "stage4b_cache": str(stage4b_cache), "stage4b_cache_hit": stage4b_cache_hit,
        "base_feature_definition": "fixed Stage 4A 192 features: 64 common log-frequency + 64 normalized Re(Z) + 64 normalized Im(Z)",
        "absolute_feature_definition": {"endpoint": "median of up to five manually active points nearest the minimum/maximum frequency", "distribution": "log10 of median, mean, maximum, and minimum active |Z|", "features": list(ABSOLUTE_FEATURE_NAMES)},
        "feature_configurations": {"A": "Stage4A 192 + voltage", "B": "Stage4A 192 + voltage + current", "C": "Stage4A 192 + voltage + current + absolute impedance features"},
        "models": {"Ridge": {"alpha": RIDGE_ALPHA}, "HGB": HGB_SETTINGS}, "hgb_parameters": weak,
        "target_transformations": {parameter: ("log10" if parameter in POSITIVE_PARAMETERS else "logit_alpha") for parameter in PARAMETERS},
        "manual_cleaned_only": True, "automatic_outlier_detection": False, "conventional_eec_fitting": False,
        "final_models": saved_models, "runtime_s": time.perf_counter() - started, "exclusions": extraction.exclusion_counts,
    }
    (model_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps({**config, "best_models": best_rows}, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "stage4b_cache_hit": stage4b_cache_hit, "runtime_s": config["runtime_s"], "best_models": best_rows, "final_models": saved_models}, indent=2))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projects", nargs=6, type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(list(args.projects), args.output)


if __name__ == "__main__":
    main()
