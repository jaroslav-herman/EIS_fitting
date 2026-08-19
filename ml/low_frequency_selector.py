"""Deterministic spectrum-only low-frequency boundary selectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np


@dataclass(frozen=True)
class BoundaryResult:
    predicted_f_min: float
    score: np.ndarray
    diagnostics: dict[str, Any]


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if values.size < 2 or window <= 1:
        return values.astype(float, copy=True)
    pad_left = window // 2; pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=-1)


def _rolling_mad(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if values.size < 2 or window <= 1:
        return np.zeros(values.size, dtype=float)
    pad_left = window // 2; pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    medians = np.median(view, axis=-1)
    return np.median(np.abs(view - medians[:, None]), axis=-1)


def _robust_normalize(values: np.ndarray, window: int) -> np.ndarray:
    median = _rolling_median(values, window)
    mad = 1.4826 * _rolling_mad(values, window)
    scale = np.maximum(mad, 0.05 * np.maximum(np.abs(median), np.finfo(float).eps))
    return np.abs(values - median) / scale


def _local_residual(frequency: np.ndarray, impedance: np.ndarray, neighborhood: int) -> tuple[np.ndarray, dict]:
    x = np.log10(frequency); n = x.size; neighborhood = max(1, int(neighborhood))
    left_values = []; right_values = []
    for offset in range(1, neighborhood + 1):
        left_values.append(np.r_[np.full(offset, np.nan + 1j*np.nan), impedance[:-offset]])
        right_values.append(np.r_[impedance[offset:], np.full(offset, np.nan + 1j*np.nan)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        left = np.nanmedian(np.vstack(left_values), axis=0)
        right = np.nanmedian(np.vstack(right_values), axis=0)
        left_x = np.nanmedian(np.vstack([np.r_[np.full(offset, np.nan), x[:-offset]] for offset in range(1, neighborhood + 1)]), axis=0)
        right_x = np.nanmedian(np.vstack([np.r_[x[offset:], np.full(offset, np.nan)] for offset in range(1, neighborhood + 1)]), axis=0)
    fraction = np.divide(x - left_x, right_x - left_x, out=np.full(n, 0.5), where=np.abs(right_x-left_x) > np.finfo(float).eps)
    prediction = left + fraction * (right - left)
    step = np.abs(np.diff(impedance, prepend=impedance[0]))
    local_scale = np.maximum(1.4826 * _rolling_mad(step, 2 * neighborhood + 1), 0.05 * _rolling_median(np.abs(impedance), 2 * neighborhood + 1))
    residual = np.abs(impedance - prediction)
    score = residual / np.maximum(local_scale, np.finfo(float).eps)
    return score, {"predicted_impedance": prediction, "local_residual": residual, "local_scale": local_scale}


def _stability_scores(frequency: np.ndarray, impedance: np.ndarray, window: int) -> tuple[np.ndarray, dict]:
    x = np.log10(frequency)
    dx = np.diff(x, prepend=x[0]); dz = np.diff(impedance, prepend=impedance[0])
    slope = np.divide(dz, dx, out=np.zeros_like(dz), where=np.abs(dx) > np.finfo(float).eps)
    step_norm = np.abs(dz) / np.maximum(_rolling_median(np.abs(impedance), window), np.finfo(float).eps)
    slope_norm = _robust_normalize(np.abs(slope), window)
    step_score = _robust_normalize(step_norm, window)
    score = _rolling_median(step_score + 0.5 * slope_norm, window)
    return score, {"step_norm": step_norm, "slope_score": slope_norm, "rolling_step_score": _rolling_median(step_score, window)}


def _trajectory_scores(frequency: np.ndarray, impedance: np.ndarray, window: int) -> tuple[np.ndarray, dict]:
    x = np.log10(frequency); dx = np.diff(x, prepend=x[0]); dz = np.diff(impedance, prepend=impedance[0])
    slope = np.divide(dz, dx, out=np.zeros_like(dz), where=np.abs(dx) > np.finfo(float).eps)
    angle = np.unwrap(np.angle(slope + np.finfo(float).eps))
    angle_change = np.abs(np.diff(angle, prepend=angle[0]))
    curvature = _robust_normalize(angle_change, window)
    relative_magnitude = _robust_normalize(np.abs(impedance), window)
    score = _rolling_median(curvature + 0.5 * relative_magnitude, window)
    return score, {"angle_change": angle_change, "curvature_score": curvature, "relative_magnitude_score": relative_magnitude}


def _persistent_cutoff(frequency: np.ndarray, score: np.ndarray, threshold: float, persistence_window: int, min_fraction: float, min_consecutive_windows: int) -> tuple[int, np.ndarray]:
    n = frequency.size; window = min(max(2, int(persistence_window)), n)
    bad = np.isfinite(score) & (score >= float(threshold))
    if n < window:
        return 0, bad
    fractions = np.convolve(bad.astype(float), np.ones(window) / window, mode="valid")
    persistent = fractions >= float(min_fraction)
    run = 0
    for value in persistent:
        if value:
            run += 1
        else:
            break
    if run < int(min_consecutive_windows):
        return 0, bad
    cutoff_index = min(n - 1, run + window - 1)
    return int(cutoff_index), bad


def select_low_frequency_boundary(
    frequency,
    impedance,
    *,
    method: str = "combined",
    threshold: float = 3.0,
    neighborhood: int = 3,
    rolling_window: int = 9,
    persistence_window: int = 7,
    min_fraction: float = 0.6,
    min_consecutive_windows: int = 2,
) -> BoundaryResult:
    """Select a persistent low-frequency cutoff from one spectrum.

    Frequencies are sorted internally from low to high and the returned
    ``predicted_f_min`` is the first retained frequency after a persistent
    low-frequency degradation.  The input arrays are never modified and all
    diagnostic score arrays are returned in original input order.
    """
    methods = {"local_residual", "rolling_stability", "trajectory", "combined"}
    if method not in methods:
        raise ValueError(f"method must be one of {sorted(methods)}")
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be positive and finite")
    f = np.asarray(frequency, dtype=float).reshape(-1); z = np.asarray(impedance, dtype=complex).reshape(-1)
    if f.size != z.size or f.size < 3:
        raise ValueError("frequency and impedance must have equal lengths and at least three points")
    valid = np.isfinite(f) & (f > 0) & np.isfinite(z.real) & np.isfinite(z.imag)
    if valid.sum() < 3:
        raise ValueError("at least three finite positive-frequency points are required")
    original_valid_indices = np.flatnonzero(valid); order = np.argsort(f[valid], kind="mergesort")
    indices = original_valid_indices[order]; fs = f[indices]; zs = z[indices]
    residual_score, residual_diag = _local_residual(fs, zs, neighborhood)
    stability_score, stability_diag = _stability_scores(fs, zs, rolling_window)
    trajectory_score, trajectory_diag = _trajectory_scores(fs, zs, rolling_window)
    components = {"local_residual": residual_score, "rolling_stability": stability_score, "trajectory": trajectory_score}
    if method == "combined":
        score_sorted = (residual_score + stability_score + trajectory_score) / 3.0
    else:
        score_sorted = components[method]
    cutoff_index, bad = _persistent_cutoff(fs, score_sorted, threshold, persistence_window, min_fraction, min_consecutive_windows)
    predicted = float(fs[cutoff_index])
    score = np.full(f.size, np.nan); score[indices] = score_sorted
    diagnostics = {"frequency": f.copy(), "log_frequency": np.where(f > 0, np.log10(f), np.nan), "score": score.copy(),
                   "bad_point": np.isin(np.arange(f.size), indices[bad]), "cutoff_index_sorted": cutoff_index,
                   "cutoff_frequency": predicted, "sorted_indices": indices, "valid_input": valid,
                   "components": {name: np.where(np.isfinite(f), np.nan, np.nan) for name in ()},
                   "parameters": {"method": method, "threshold": threshold, "neighborhood": neighborhood, "rolling_window": rolling_window,
                                  "persistence_window": persistence_window, "min_fraction": min_fraction,
                                  "min_consecutive_windows": min_consecutive_windows}}
    for name, values in {**residual_diag, **stability_diag, **trajectory_diag}.items():
        if isinstance(values, np.ndarray) and values.size == fs.size:
            dtype = complex if np.iscomplexobj(values) else float
            fill = np.nan + 1j * np.nan if dtype is complex else np.nan
            original = np.full(f.size, fill, dtype=dtype); original[indices] = values; diagnostics[name] = original
    for name, values in components.items():
        original = np.full(f.size, np.nan); original[indices] = values; diagnostics["components"][name] = original
    return BoundaryResult(predicted, score, diagnostics)
