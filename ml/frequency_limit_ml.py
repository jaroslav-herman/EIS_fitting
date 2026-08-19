"""Leakage-controlled regression of active-mask frequency limits."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODEL_NAMES = ("ridge", "random_forest", "hist_gradient_boosting")


def models(seed: int = 42) -> dict[str, object]:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(n_estimators=12, min_samples_leaf=3, random_state=seed, n_jobs=1),
        "hist_gradient_boosting": MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=12, learning_rate=0.06, l2_regularization=1.0, random_state=seed)),
    }


def _interp(grid, x, values):
    return np.interp(grid, x, values, left=values[0], right=values[-1])


def _point_features(frequency, impedance):
    order = np.argsort(frequency, kind="mergesort"); f = np.asarray(frequency, dtype=float)[order]; z = np.asarray(impedance, dtype=complex)[order]
    x = np.log10(f); scale = max(float(np.nanmedian(np.abs(z))), np.finfo(float).eps)
    zr, zi = z.real / scale, z.imag / scale; mag = np.log10(np.maximum(np.abs(z) / scale, np.finfo(float).eps)); phase = np.unwrap(np.angle(z))
    def derivative(values): return np.gradient(values, x, edge_order=1) if x.size > 1 else np.zeros_like(values)
    slope_r, slope_i, slope_m, slope_p = map(derivative, (zr, zi, mag, phase))
    curvature = derivative(slope_r) + derivative(slope_i)
    local_step = np.abs(np.diff(z, prepend=z[0])) / scale
    residual = np.zeros_like(local_step)
    if z.size >= 3:
        prediction = np.interp(x[1:-1], x[[0, -1]], [z[0].real, z[-1].real]) + 1j * np.interp(x[1:-1], x[[0, -1]], [z[0].imag, z[-1].imag])
        residual[1:-1] = np.abs(z[1:-1] - prediction) / scale
    values = np.vstack((x, zr, zi, mag, phase, slope_r, slope_i, slope_m, slope_p, curvature, residual, local_step))
    return x, values


@dataclass
class SpectrumFeatureExtractor:
    grid_size: int = 64
    grid_: np.ndarray | None = None
    fill_: np.ndarray | None = None

    def fit(self, records):
        low = min(np.min(np.log10(r.frequency[r.frequency > 0])) for r in records)
        high = max(np.max(np.log10(r.frequency[r.frequency > 0])) for r in records)
        self.grid_ = np.linspace(low, high, self.grid_size)
        raw = np.vstack([self._one(r.frequency, r.impedance) for r in records])
        self.fill_ = np.nanmedian(raw, axis=0); self.fill_[~np.isfinite(self.fill_)] = 0.0
        return self

    def _one(self, frequency, impedance):
        if self.grid_ is None: raise RuntimeError("feature extractor is not fitted")
        x, values = _point_features(frequency, impedance)
        return np.concatenate([_interp(self.grid_, x, row) for row in values])

    def transform(self, records):
        if self.grid_ is None or self.fill_ is None: raise RuntimeError("feature extractor is not fitted")
        x = np.vstack([self._one(r.frequency, r.impedance) for r in records])
        missing = ~np.isfinite(x); rows, cols = np.where(missing); x[rows, cols] = self.fill_[cols]
        return x

    def fit_transform(self, records): return self.fit(records).transform(records)


def target_values(targets):
    return np.asarray([[np.log10(t["f_min"]), np.log10(t["f_max"])] for t in targets], dtype=float)


def regression_metrics(frame, prefix=""):
    output = {}
    for name in ("f_min", "f_max"):
        error = frame[f"error_{name}_decades"].to_numpy(float); absolute = np.abs(error)
        output.update({f"{prefix}{name}_MAE_decades": float(absolute.mean()), f"{prefix}{name}_median_abs_decades": float(np.median(absolute)),
                       f"{prefix}{name}_p75_abs_decades": float(np.percentile(absolute, 75)), f"{prefix}{name}_p90_abs_decades": float(np.percentile(absolute, 90)), f"{prefix}{name}_p95_abs_decades": float(np.percentile(absolute, 95)),
                       f"{prefix}{name}_within_0.05_percent": float(100*np.mean(absolute <= .05)), f"{prefix}{name}_within_0.10_percent": float(100*np.mean(absolute <= .10)),
                       f"{prefix}{name}_within_0.20_percent": float(100*np.mean(absolute <= .20)), f"{prefix}{name}_within_0.30_percent": float(100*np.mean(absolute <= .30)), f"{prefix}{name}_within_0.50_percent": float(100*np.mean(absolute <= .50)),
                       f"{prefix}{name}_too_low_percent": float(100*np.mean(error < -.10)), f"{prefix}{name}_correct_percent": float(100*np.mean(absolute <= .10)), f"{prefix}{name}_too_high_percent": float(100*np.mean(error > .10))})
    return output
