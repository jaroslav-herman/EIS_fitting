from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import SpectrumRecord
from .metrics import regression_range_metrics
from .preprocessing import SpectrumPreprocessor


def _models(seed: int) -> dict[str, object]:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=250, random_state=seed, n_jobs=-1, min_samples_leaf=2),
        "hist_gradient_boosting": MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=150, random_state=seed)),
    }


def _targets(records: list[SpectrumRecord]) -> np.ndarray:
    lower = np.log10([r.manual_f_min for r in records])
    upper = np.log10([r.manual_f_max for r in records])
    center = (lower + upper) / 2.0
    log_width = np.log(np.maximum(upper - lower, np.finfo(float).eps))
    return np.column_stack([center, log_width])


def _features(train: list[SpectrumRecord], test: list[SpectrumRecord], mode: str, grid_size: int):
    if mode == "voltage_only":
        train_voltage = np.asarray([[r.voltage] for r in train], dtype=float)
        test_voltage = np.asarray([[r.voltage] for r in test], dtype=float)
        fill = np.nanmedian(train_voltage, axis=0)
        fill[~np.isfinite(fill)] = 0.0
        train_voltage[~np.isfinite(train_voltage)] = np.take(fill, np.where(~np.isfinite(train_voltage))[1])
        test_voltage[~np.isfinite(test_voltage)] = np.take(fill, np.where(~np.isfinite(test_voltage))[1])
        return train_voltage, test_voltage
    preprocessor = SpectrumPreprocessor(grid_size=grid_size, spectrum_mode="raw")
    x_train = preprocessor.fit_transform(train)
    x_test = preprocessor.transform(test)
    if mode == "spectrum_plus_voltage":
        train_voltage = np.asarray([[r.voltage] for r in train], dtype=float)
        test_voltage = np.asarray([[r.voltage] for r in test], dtype=float)
        mean = np.nanmean(train_voltage, axis=0)
        scale = np.nanstd(train_voltage, axis=0)
        mean[~np.isfinite(mean)] = 0.0
        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
        for values in (train_voltage, test_voltage):
            missing = ~np.isfinite(values)
            values[missing] = np.take(mean, np.where(missing)[1])
            values -= mean
            values /= scale
        x_train = np.hstack([x_train, train_voltage])
        x_test = np.hstack([x_test, test_voltage])
    return x_train, x_test


def _fit_predict(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)
    center, log_width = model.predict(x_test).T
    width = np.exp(np.clip(log_width, -20.0, 20.0))
    return center - width / 2.0, center + width / 2.0


@dataclass
class FrequencyRangeExperiment:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    overall_metrics: pd.DataFrame
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.predictions.to_csv(directory / "predictions.csv", index=False)
        self.fold_metrics.to_csv(directory / "fold_metrics.csv", index=False)
        self.overall_metrics.to_csv(directory / "overall_metrics.csv", index=False)
        if not self.excluded.empty:
            self.excluded.to_csv(directory / "exclusions.csv", index=False)


def run_frequency_range_experiment(
    records: list[SpectrumRecord],
    *,
    feature_mode: str,
    model_names: tuple[str, ...] = ("ridge", "random_forest", "hist_gradient_boosting"),
    grid_size: int = 64,
    seed: int = 42,
) -> FrequencyRangeExperiment:
    if feature_mode not in {"voltage_only", "spectrum_only", "spectrum_plus_voltage"}:
        raise ValueError("unknown feature_mode")
    records = [r for r in records if r.manual_f_min is not None and r.manual_f_max is not None]
    samples = sorted({r.sample_id for r in records})
    if len(samples) < 2:
        raise ValueError("At least two samples are required")
    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for held_out in samples:
        train = [r for r in records if r.sample_id != held_out]
        test = [r for r in records if r.sample_id == held_out]
        x_train, x_test = _features(train, test, feature_mode, grid_size)
        y_train = _targets(train)
        for model_name in model_names:
            models = _models(seed)
            if model_name not in models:
                raise ValueError(f"unknown model: {model_name}")
            predicted_min, predicted_max = _fit_predict(models[model_name], x_train, y_train, x_test)
            for record, minimum, maximum in zip(test, predicted_min, predicted_max):
                manual_min = float(np.log10(record.manual_f_min))
                manual_max = float(np.log10(record.manual_f_max))
                row = {
                    "spectrum_id": record.spectrum_id,
                    "source_project": record.source_project,
                    "sample_id": record.sample_id,
                    "voltage": record.voltage,
                    "current": record.current,
                    "time": record.time,
                    "device_setup": record.device_setup,
                    "manual_f_min": record.manual_f_min,
                    "manual_f_max": record.manual_f_max,
                    "manual_log_f_min": manual_min,
                    "manual_log_f_max": manual_max,
                    "predicted_log_f_min": float(minimum),
                    "predicted_log_f_max": float(maximum),
                    "predicted_f_min": float(10**minimum),
                    "predicted_f_max": float(10**maximum),
                    "measured_f_min": float(np.min(record.frequency)),
                    "measured_f_max": float(np.max(record.frequency)),
                    "model_name": model_name,
                    "feature_mode": feature_mode,
                    "validation_fold": held_out,
                }
                row.update(regression_range_metrics(row))
                rows.append(row)
            fold_frame = pd.DataFrame([r for r in rows if r["model_name"] == model_name and r["validation_fold"] == held_out])
            fold_rows.append({"model_name": model_name, "held_out_sample": held_out, **regression_range_metrics(fold_frame, summary=True)})
    predictions = pd.DataFrame(rows)
    overall_rows = []
    for model_name in model_names:
        frame = predictions[predictions["model_name"] == model_name]
        overall_rows.append({"model_name": model_name, **regression_range_metrics(frame, summary=True)})
    return FrequencyRangeExperiment(predictions, pd.DataFrame(fold_rows), pd.DataFrame(overall_rows))
