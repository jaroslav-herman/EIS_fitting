"""Fast, strict Stage 4A Ridge prediction of fitted EEC parameters."""
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
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import load_eisfit_projects
from .evaluate_frequency_limit_ml import _clean_record, _manual_masks
from .evaluate_stage4_parameters import extract_parameter_targets
from .parameter_prediction import (
    ALPHA_PARAMETERS, PARAMETERS, ONE_PROCESS_PARAMETERS, TWO_PROCESS_PARAMETERS,
    inverse_target, parameter_mapping, topology_for_circuit, transform_target,
)
from .preprocessing import SpectrumPreprocessor


TRAINING_SAMPLES = ("129", "140", "150", "157", "159", "181")
FEATURE_CACHE = Path("ml/cache/stage4a_parameter_features.npz")
METADATA_CACHE = Path("ml/cache/stage4a_parameter_metadata.csv")
OUTPUT = Path("ml/analysis/stage4a_parameters")
RIDGE_ALPHA = 1.0


def multiplicative_error(true_value: float, predicted_value: float) -> float:
    true_value, predicted_value = float(true_value), float(predicted_value)
    if true_value <= 0 or predicted_value <= 0:
        raise ValueError("multiplicative error requires positive values")
    return max(predicted_value / true_value, true_value / predicted_value)


def loso_splits(samples=TRAINING_SAMPLES):
    samples = tuple(samples)
    if len(set(samples)) != len(samples) or "178" in samples:
        raise ValueError("LOSO samples must be unique and exclude sample 178")
    for held_out in samples:
        yield held_out, tuple(sample for sample in samples if sample != held_out)


def _manual_records(projects, records):
    masks = _manual_masks(projects, {record.spectrum_id: record for record in records})
    cleaned = []
    for record in records:
        if record.spectrum_id not in masks:
            raise RuntimeError(f"manual mask missing for {record.spectrum_id}")
        mask = masks[record.spectrum_id]
        if record.manual_f_min is not None and record.manual_f_max is not None:
            mask &= (record.frequency >= record.manual_f_min) & (record.frequency <= record.manual_f_max)
        cleaned_record = _clean_record(record, mask)
        if cleaned_record.frequency.size < 3:
            raise RuntimeError(f"too few manually active points for {record.spectrum_id}")
        cleaned.append(cleaned_record)
    return cleaned


def _load_or_create_features(records, cleaned, cache_path=FEATURE_CACHE, metadata_path=METADATA_CACHE):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected_ids = np.asarray([record.spectrum_id for record in records], dtype=str)
    if cache_path.exists() and metadata_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            cached_ids = np.asarray(cached["spectrum_ids"], dtype=str)
            x = np.asarray(cached["X"], dtype=float)
            if x.shape == (len(records), 192) and np.array_equal(cached_ids, expected_ids):
                return x, True
        except (OSError, KeyError, ValueError):
            pass
    preprocessor = SpectrumPreprocessor(grid_size=64, use_metadata=False, spectrum_mode="raw")
    x = preprocessor.fit_transform(cleaned)
    if x.shape != (len(records), 192):
        raise RuntimeError(f"unexpected cached feature shape: {x.shape}")
    np.savez_compressed(cache_path, X=x, spectrum_ids=expected_ids, feature_dimension=np.asarray([192]), feature_definition=np.asarray(["64 common log-frequency + 64 median-|Z|-normalized Re(Z) + 64 median-|Z|-normalized Im(Z)"]))
    return x, False


def _metric_row(true_values, predicted_values, parameter, model):
    true = np.asarray(true_values, dtype=float); predicted = np.asarray(predicted_values, dtype=float)
    raw_error = predicted - true; absolute = np.abs(raw_error)
    row = {"model": model, "parameter": parameter, "spectra": int(true.size), "correlation": float(np.corrcoef(true, predicted)[0, 1]) if true.size > 1 and np.std(true) > 0 and np.std(predicted) > 0 else np.nan}
    if parameter in ALPHA_PARAMETERS:
        row.update({"mae": float(np.mean(absolute)), "rmse": float(np.sqrt(np.mean(raw_error ** 2))), "median_absolute_error": float(np.median(absolute)), "within_0.02": float(np.mean(absolute <= .02)), "within_0.05": float(np.mean(absolute <= .05)), "within_0.10": float(np.mean(absolute <= .10))})
    else:
        log_error = np.log10(predicted) - np.log10(true); abs_log = np.abs(log_error)
        row.update({"mae_log10": float(np.mean(abs_log)), "rmse_log10": float(np.sqrt(np.mean(log_error ** 2))), "median_absolute_log10": float(np.median(abs_log)), "within_x1.25": float(np.mean(abs_log <= np.log10(1.25))), "within_x2": float(np.mean(abs_log <= np.log10(2))), "within_x5": float(np.mean(abs_log <= np.log10(5))), "within_x10": float(np.mean(abs_log <= 1.0))})
    return row


def _make_prediction_row(meta, parameter, true_value, predicted_value, model):
    positive = parameter not in ALPHA_PARAMETERS
    result = {"sample_id": meta["sample_id"], "spectrum_id": meta["spectrum_id"], "voltage": meta["voltage"], "topology": meta["topology"], "parameter": parameter, "true_value": float(true_value), "predicted_value": float(predicted_value), "log_true_value": float(np.log10(true_value)) if positive else np.nan, "log_predicted_value": float(np.log10(predicted_value)) if positive else np.nan, "absolute_error": float(abs(predicted_value - true_value)), "log_error": float(np.log10(predicted_value) - np.log10(true_value)) if positive else float(predicted_value - true_value), "multiplicative_error": multiplicative_error(true_value, predicted_value) if positive else np.nan, "model": model, "held_out_sample": meta["sample_id"]}
    return result


def _plot_parameter(frame, output, parameter):
    if frame.empty:
        return
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ridge = frame[frame.model == "RIDGE"]; baseline = frame[frame.model == "MEDIAN_BASELINE"]
    ax.scatter(baseline.true_value, baseline.predicted_value, s=5, alpha=.2, label="median baseline")
    ax.scatter(ridge.true_value, ridge.predicted_value, s=6, alpha=.45, label="Ridge")
    low = float(min(frame.true_value.min(), frame.predicted_value.min())); high = float(max(frame.true_value.max(), frame.predicted_value.max()))
    ax.plot([low, high], [low, high], "k--"); ax.set_xlabel("manual fitted value"); ax.set_ylabel("predicted value"); ax.set_title(parameter); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output / f"{parameter}.png", dpi=130); plt.close(fig)


def run(projects: list[Path], output: Path = OUTPUT, cache_path: Path = FEATURE_CACHE, metadata_path: Path = METADATA_CACHE):
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=True)
    mapping = {str(path): path.name.split(".")[0] for path in projects}
    extraction = load_eisfit_projects(projects, mapping, require_fit=True)
    records = extraction.records
    if sorted({record.sample_id for record in records}) != sorted(TRAINING_SAMPLES):
        raise RuntimeError("Stage 4A requires exactly the six training samples")
    if "178" in {record.sample_id for record in records}:
        raise RuntimeError("sample 178 entered Stage 4A")
    cleaned = _manual_records(projects, records)
    x, cache_hit = _load_or_create_features(records, cleaned, cache_path, metadata_path)
    targets = extract_parameter_targets(projects, records)
    record_metadata = pd.DataFrame([{"spectrum_id": record.spectrum_id, "sample_id": record.sample_id, "voltage": record.voltage, "current": record.current, "time": record.time} for record in records])
    metadata = record_metadata.merge(targets, on="spectrum_id", validate="one_to_one")
    metadata.to_csv(metadata_path, index=False)
    id_to_index = {spectrum_id: index for index, spectrum_id in enumerate(metadata.spectrum_id)}
    prediction_rows, metric_rows, sample_metric_rows = [], [], []
    for held_out, _training_samples in loso_splits():
        train_mask = metadata.sample_id != held_out; test_mask = metadata.sample_id == held_out
        for parameter in PARAMETERS:
            train_indices = np.flatnonzero(train_mask & metadata[parameter].notna().to_numpy())
            test_indices = np.flatnonzero(test_mask & metadata[parameter].notna().to_numpy())
            if not train_indices.size or not test_indices.size:
                continue
            y_train = metadata.iloc[train_indices][parameter].to_numpy(float); y_test = metadata.iloc[test_indices][parameter].to_numpy(float)
            x_train = np.column_stack([x[train_indices], metadata.iloc[train_indices].voltage.to_numpy(float)]); x_test = np.column_stack([x[test_indices], metadata.iloc[test_indices].voltage.to_numpy(float)])
            target_train = transform_target(y_train, parameter)
            model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)); model.fit(x_train, target_train)
            prediction = inverse_target(model.predict(x_test), parameter)
            if parameter in ALPHA_PARAMETERS: prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
            baseline = np.full(y_test.size, np.median(y_train))
            ridge_rows = [_make_prediction_row(metadata.iloc[index], parameter, truth, pred, "RIDGE") for index, truth, pred in zip(test_indices, y_test, prediction)]
            baseline_rows = [_make_prediction_row(metadata.iloc[index], parameter, truth, pred, "MEDIAN_BASELINE") for index, truth, pred in zip(test_indices, y_test, baseline)]
            prediction_rows.extend(ridge_rows + baseline_rows)
            metric_rows.extend([{"held_out_sample": held_out, **_metric_row(y_test, prediction, parameter, "RIDGE")}, {"held_out_sample": held_out, **_metric_row(y_test, baseline, parameter, "MEDIAN_BASELINE")}])
            sample_metric_rows.extend(metric_rows[-2:])
    predictions = pd.DataFrame(prediction_rows); predictions.to_csv(output / "predictions.csv", index=False)
    metrics = pd.DataFrame(metric_rows); metrics.to_csv(output / "per_sample_metrics.csv", index=False)
    overall_rows = []
    for keys, frame in predictions.groupby(["model", "parameter"]): overall_rows.append(_metric_row(frame.true_value, frame.predicted_value, keys[1], keys[0]))
    overall = pd.DataFrame(overall_rows); overall.to_csv(output / "overall_metrics.csv", index=False)
    summary = []
    for parameter in PARAMETERS:
        ridge = overall[(overall.parameter == parameter) & (overall.model == "RIDGE")]
        baseline = overall[(overall.parameter == parameter) & (overall.model == "MEDIAN_BASELINE")]
        if ridge.empty: continue
        row = {"parameter": parameter, "ridge_model": "Ridge(alpha=1.0)", "ridge_spectra": int(ridge.spectra.iloc[0]), "baseline_spectra": int(baseline.spectra.iloc[0]) if not baseline.empty else 0}
        score = "mae" if parameter in ALPHA_PARAMETERS else "mae_log10"
        row.update({f"ridge_{score}": float(ridge[score].iloc[0]), f"baseline_{score}": float(baseline[score].iloc[0]) if not baseline.empty else np.nan, "ridge_correlation": float(ridge.correlation.iloc[0]), "baseline_correlation": float(baseline.correlation.iloc[0]) if not baseline.empty else np.nan})
        summary.append(row)
    pd.DataFrame(summary).to_csv(output / "parameter_summary.csv", index=False)
    plots = output / "predicted_vs_true"
    for parameter in PARAMETERS: _plot_parameter(predictions[predictions.parameter == parameter], plots, parameter)
    model_dir = output / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    final_manifest = []
    for parameter in PARAMETERS:
        indices = np.flatnonzero(metadata[parameter].notna().to_numpy())
        if not indices.size: continue
        y = metadata.iloc[indices][parameter].to_numpy(float); xx = np.column_stack([x[indices], metadata.iloc[indices].voltage.to_numpy(float)])
        final_model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)); final_model.fit(xx, transform_target(y, parameter))
        model_path = model_dir / f"{parameter}_ridge.joblib"; joblib.dump({"model": final_model, "parameter": parameter, "target_transformation": "log10" if parameter not in ALPHA_PARAMETERS else "direct_alpha_clipped", "feature_dimension": 193, "feature_cache": str(cache_path), "training_samples": list(TRAINING_SAMPLES), "training_spectra": int(indices.size)}, model_path); final_manifest.append({"parameter": parameter, "path": str(model_path), "training_spectra": int(indices.size)})
    config = {"training_samples": list(TRAINING_SAMPLES), "training_spectra": len(records), "one_process_spectra": int((metadata.topology == "ONE_PROCESS").sum()), "two_process_spectra": int((metadata.topology == "TWO_PROCESS").sum()), "parameters": list(PARAMETERS), "topology_mapping": {"ONE_PROCESS": list(ONE_PROCESS_PARAMETERS), "TWO_PROCESS": list(TWO_PROCESS_PARAMETERS)}, "feature_definition": "192 cached topology spectral features + voltage", "feature_dimension": 193, "voltage_used": True, "ridge_alpha": RIDGE_ALPHA, "manual_cleaned_only": True, "automatic_outlier_detection": False, "target_transformations": {parameter: ("log10" if parameter not in ALPHA_PARAMETERS else "direct alpha clipped to (0,1)") for parameter in PARAMETERS}, "feature_cache": str(cache_path), "metadata_cache": str(metadata_path), "cache_hit": cache_hit, "sample_178_used": False, "conventional_eec_fitting": False, "runtime_s": time.perf_counter() - started, "final_models": final_manifest}
    (model_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps({**config, "exclusions": extraction.exclusion_counts}, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2)); return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projects", nargs=6, type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--feature-cache", type=Path, default=FEATURE_CACHE)
    parser.add_argument("--metadata-cache", type=Path, default=METADATA_CACHE)
    args = parser.parse_args(); run(list(args.projects), args.output, args.feature_cache, args.metadata_cache)


if __name__ == "__main__": main()
