"""Reusable Stage 4 parameter-target, transformation, and bound utilities."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


POSITIVE_PARAMETERS = ("R0", "R1", "Q1", "R2", "Q2")
ALPHA_PARAMETERS = ("alpha1", "alpha2")
PARAMETERS = POSITIVE_PARAMETERS + ALPHA_PARAMETERS
ONE_PROCESS_PARAMETERS = ("R0", "R1", "Q1", "alpha1")
TWO_PROCESS_PARAMETERS = ONE_PROCESS_PARAMETERS + ("R2", "Q2", "alpha2")
TOPOLOGY_PARAMETERS = {"ONE_PROCESS": ONE_PROCESS_PARAMETERS, "TWO_PROCESS": TWO_PROCESS_PARAMETERS}


def topology_for_circuit(circuit: str) -> str:
    return "TWO_PROCESS" if "p(R2,CPE2)" in str(circuit) else "ONE_PROCESS"


def parameter_mapping(circuit: str, values, names) -> dict[str, float]:
    """Map the application's fitted names to Stage 4 names, excluding L0."""
    if len(values) != len(names):
        raise ValueError("fit parameter/name length mismatch")
    by_name = {str(name): float(value) for name, value in zip(names, values)}
    result = {"R0": by_name["R0"], "R1": by_name["R1"], "Q1": by_name["CPE1_0"], "alpha1": by_name["CPE1_1"]}
    if topology_for_circuit(circuit) == "TWO_PROCESS":
        result.update({"R2": by_name["R2"], "Q2": by_name["CPE2_0"], "alpha2": by_name["CPE2_1"]})
    return result


def transform_target(values, parameter: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if parameter in POSITIVE_PARAMETERS:
        if np.any(~np.isfinite(values) | (values <= 0)):
            raise ValueError(f"{parameter} requires strictly positive finite targets")
        return np.log10(values)
    if parameter in ALPHA_PARAMETERS:
        clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped))
    raise KeyError(parameter)


def inverse_target(values, parameter: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if parameter in POSITIVE_PARAMETERS:
        return 10.0 ** values
    if parameter in ALPHA_PARAMETERS:
        return np.clip(1.0 / (1.0 + np.exp(-values)), 1e-6, 1.0 - 1e-6)
    raise KeyError(parameter)


def residual_values(true_values, predicted_values, parameter: str) -> np.ndarray:
    """Residual in the model/bound space: log10 for positive, raw for alpha."""
    return transform_target(true_values, parameter) - transform_target(predicted_values, parameter)


def residual_quantiles(residuals, level: float) -> tuple[float, float]:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        raise ValueError("cannot calculate residual quantiles from empty data")
    tail = (1.0 - float(level)) / 2.0
    return float(np.quantile(residuals, tail)), float(np.quantile(residuals, 1.0 - tail))


def enforce_bounds(predicted, lower, upper, parameter: str) -> tuple[np.ndarray, np.ndarray, int]:
    predicted = np.asarray(predicted, dtype=float)
    lower = np.asarray(lower, dtype=float).copy()
    upper = np.asarray(upper, dtype=float).copy()
    clipped = 0
    if parameter in POSITIVE_PARAMETERS:
        before = (lower <= 0) | ~np.isfinite(lower) | (upper <= lower) | ~np.isfinite(upper)
        lower = np.maximum(np.where(np.isfinite(lower), lower, np.finfo(float).tiny), np.finfo(float).tiny)
        upper = np.where(np.isfinite(upper), upper, np.maximum(predicted, lower * 10.0))
        upper = np.maximum(upper, np.maximum(predicted, lower * (1.0 + 1e-9)))
        clipped = int(np.count_nonzero(before))
    elif parameter in ALPHA_PARAMETERS:
        before = (lower <= 0) | (upper >= 1) | (upper <= lower) | ~np.isfinite(lower) | ~np.isfinite(upper)
        lower = np.clip(np.where(np.isfinite(lower), lower, 1e-6), 1e-6, 1.0 - 1e-6)
        upper = np.clip(np.where(np.isfinite(upper), upper, 1.0 - 1e-6), 1e-6, 1.0 - 1e-6)
        upper = np.maximum(upper, np.minimum(1.0 - 1e-6, np.maximum(predicted, lower + 1e-6)))
        lower = np.minimum(lower, upper - 1e-6)
        clipped = int(np.count_nonzero(before))
    else:
        raise KeyError(parameter)
    return lower, upper, clipped


def bounds_from_residuals(predicted, residuals, parameter: str, levels=(0.90, 0.95, 0.99)) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    predicted = np.asarray(predicted, dtype=float)
    transformed = transform_target(predicted, parameter)
    result = {}
    for level in levels:
        low_residual, high_residual = residual_quantiles(residuals, level)
        lower = inverse_target(transformed + low_residual, parameter)
        upper = inverse_target(transformed + high_residual, parameter)
        lower, upper, clipped = enforce_bounds(predicted, lower, upper, parameter)
        result[str(int(level * 100))] = (lower, upper, clipped)
    return result


def bound_metrics(true_values, lower, upper) -> dict[str, float]:
    true_values = np.asarray(true_values, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    inside = (true_values >= lower) & (true_values <= upper)
    width = upper - lower
    return {"coverage": float(np.mean(inside)), "median_width": float(np.median(width)), "mean_width": float(np.mean(width)), "inside_count": int(np.sum(inside)), "count": int(true_values.size)}


@dataclass
class FoldFeatureBuilder:
    feature_set: str
    spectrum_preprocessor: object | None = None
    metadata_fill: np.ndarray | None = None
    metadata_mean: np.ndarray | None = None
    metadata_scale: np.ndarray | None = None

    def fit(self, records):
        from sklearn.preprocessing import StandardScaler
        from .preprocessing import SpectrumPreprocessor
        if self.feature_set != "VOLTAGE_ONLY":
            self.spectrum_preprocessor = SpectrumPreprocessor(grid_size=64, use_metadata=False, spectrum_mode="raw").fit(records)
        columns = self._columns(records)
        if columns.size:
            self.metadata_fill = np.nanmedian(columns, axis=0)
            self.metadata_fill[~np.isfinite(self.metadata_fill)] = 0.0
            filled = np.where(np.isfinite(columns), columns, self.metadata_fill)
            scaler = StandardScaler().fit(filled)
            self.metadata_mean, self.metadata_scale = scaler.mean_, scaler.scale_
            self.metadata_scale = np.where(self.metadata_scale <= 1e-12, 1.0, self.metadata_scale)
        return self

    def _columns(self, records):
        names = {"VOLTAGE_ONLY": ("voltage",), "SPECTRUM_VOLTAGE": ("voltage",), "SPECTRUM_VOLTAGE_CURRENT": ("voltage", "current"), "SPECTRUM_VOLTAGE_CURRENT_TIME": ("voltage", "current", "time")}.get(self.feature_set)
        if names is None:
            return np.empty((len(records), 0), dtype=float)
        return np.asarray([[getattr(record, name) for name in names] for record in records], dtype=float)

    def transform(self, records):
        if self.feature_set == "VOLTAGE_ONLY":
            spectrum = np.empty((len(records), 0), dtype=float)
        else:
            spectrum = self.spectrum_preprocessor.transform(records)
        columns = self._columns(records)
        if columns.size:
            columns = np.where(np.isfinite(columns), columns, self.metadata_fill)
            columns = (columns - self.metadata_mean) / self.metadata_scale
        return np.hstack([spectrum, columns])


def model_factories(seed: int = 42):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        # The Stage 4 benchmark uses the compact, reproducible budget already
        # used by the frequency-limit study.  The model families and seeds are
        # unchanged; this keeps the complete feature/parameter/LOSO grid
        # computationally tractable.
        "random_forest": lambda: RandomForestRegressor(n_estimators=12, min_samples_leaf=3, random_state=seed, n_jobs=1),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(max_iter=12, learning_rate=0.06, l2_regularization=1.0, random_state=seed),
    }
