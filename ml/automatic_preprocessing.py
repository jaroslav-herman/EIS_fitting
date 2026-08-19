"""Deterministic preprocessing primitives for the frequency-limit ML study."""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np



@dataclass(frozen=True)
class MaskResult:
    mask: np.ndarray
    score: np.ndarray
    diagnostics: dict


def conservative_mask(frequency, impedance, *, threshold: float = 10.0) -> MaskResult:
    """Remove only very strong local anomalies before feature extraction."""
    return _vectorized_local_mask(frequency, impedance, threshold, None)


def sensitive_mask(frequency, impedance, frequency_range, *, threshold: float = 4.0) -> MaskResult:
    """Remove isolated local anomalies inside an already fixed ML envelope."""
    return _vectorized_local_mask(frequency, impedance, threshold, frequency_range)


def _vectorized_local_mask(frequency, impedance, threshold, frequency_range):
    """Vectorized robust local interpolation used only by this experiment."""
    f = np.asarray(frequency, dtype=float).reshape(-1); z = np.asarray(impedance, dtype=complex).reshape(-1)
    valid = np.isfinite(f) & (f > 0) & np.isfinite(z.real) & np.isfinite(z.imag)
    in_range = np.ones(f.size, dtype=bool) if frequency_range is None else ((f >= min(frequency_range)) & (f <= max(frequency_range)))
    candidate = valid & in_range; score = np.full(f.size, np.nan); keep = valid & in_range
    indices = np.flatnonzero(candidate)[np.argsort(f[candidate], kind="mergesort")]
    if indices.size < 7:
        return MaskResult(keep, score, {"input_valid": valid, "in_selected_range": in_range, "rejection_reason": np.full(f.size, "insufficient_context", dtype=object), "normalized_score": score})
    x, zs = np.log10(f[indices]), z[indices]; n = indices.size; neighborhood = 3
    left = np.vstack([np.r_[np.full(k, np.nan + 1j*np.nan), zs[:-k]] for k in range(1, neighborhood + 1)])
    right = np.vstack([np.r_[zs[k:], np.full(k, np.nan + 1j*np.nan)] for k in range(1, neighborhood + 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lx = np.nanmedian(np.vstack([np.r_[np.full(k, np.nan), x[:-k]] for k in range(1, neighborhood + 1)]), axis=0)
        rx = np.nanmedian(np.vstack([np.r_[x[k:], np.full(k, np.nan)] for k in range(1, neighborhood + 1)]), axis=0)
        lz, rz = np.nanmedian(left.real, axis=0) + 1j*np.nanmedian(left.imag, axis=0), np.nanmedian(right.real, axis=0) + 1j*np.nanmedian(right.imag, axis=0)
    fraction = np.divide(x-lx, rx-lx, out=np.full(n, .5), where=np.abs(rx-lx) > np.finfo(float).eps); residual = np.abs(zs - (lz + fraction*(rz-lz)))
    step = np.abs(np.diff(zs, prepend=zs[0])); padded = np.pad(step, (3, 3), mode="edge"); view = np.lib.stride_tricks.sliding_window_view(padded, 7); med = np.median(view, axis=1); mad = np.median(np.abs(view-med[:, None]), axis=1)
    scale = np.maximum(1.4826*mad, .05*np.maximum(np.abs(np.median(np.vstack([np.abs(zs)]), axis=0)), np.finfo(float).eps)); scores = residual/np.maximum(scale, np.finfo(float).eps); bad = np.isfinite(scores) & (scores > threshold)
    score[indices] = scores; keep[indices[bad]] = False
    reasons = np.full(f.size, "input_invalid", dtype=object); reasons[valid & ~in_range] = "outside_range"; reasons[indices] = "valid"; reasons[indices[bad]] = "local_anomaly"
    return MaskResult(keep, score, {"input_valid": valid, "in_selected_range": in_range, "rejection_reason": reasons, "normalized_score": score, "sorted_indices": indices})


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.size:
        return []
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def active_boundary_targets(frequency, manual_active, *, persistence: int = 3) -> dict[str, float | int | bool]:
    """Return literal and persistence-aware active-mask boundary targets.

    Arrays are sorted low-to-high. ``robust`` requires an active run of
    ``persistence`` points; ``ignore_isolated`` uses a longer run and is the
    intentionally more conservative candidate definition.
    """
    f = np.asarray(frequency, dtype=float); active = np.asarray(manual_active, dtype=bool)
    valid = np.isfinite(f) & (f > 0) & (active | ~active)
    order = np.argsort(f[valid], kind="mergesort"); fs = f[valid][order]; a = active[valid][order]
    if not a.any():
        raise ValueError("manual mask contains no active points")
    active_runs = _runs(a); inactive_runs = _runs(~a)
    literal_low = float(fs[np.flatnonzero(a)[0]]); literal_high = float(fs[np.flatnonzero(a)[-1]])

    def persistent_edge(reverse: bool, length: int) -> tuple[float, bool]:
        runs = list(reversed(active_runs)) if reverse else active_runs
        for start, end in runs:
            if end - start + 1 >= length:
                index = end if reverse else start
                return float(fs[index]), True
        return (literal_high if reverse else literal_low), False

    robust_low, robust_low_found = persistent_edge(False, max(2, int(persistence)))
    robust_high, robust_high_found = persistent_edge(True, max(2, int(persistence)))
    isolated_low, isolated_low_found = persistent_edge(False, max(4, int(persistence) + 1))
    isolated_high, isolated_high_found = persistent_edge(True, max(4, int(persistence) + 1))
    return {
        "measured_f_min": float(fs[0]), "measured_f_max": float(fs[-1]),
        "literal_f_min": literal_low, "literal_f_max": literal_high,
        "robust_f_min": robust_low, "robust_f_max": robust_high,
        "ignore_isolated_f_min": isolated_low, "ignore_isolated_f_max": isolated_high,
        "robust_low_found": robust_low_found, "robust_high_found": robust_high_found,
        "ignore_isolated_low_found": isolated_low_found, "ignore_isolated_high_found": isolated_high_found,
        "active_points": int(a.sum()), "inactive_points": int((~a).sum()),
        "active_run_lengths": [b - s + 1 for s, b in active_runs],
        "inactive_run_lengths": [b - s + 1 for s, b in inactive_runs],
        "active_runs": len(active_runs), "inactive_runs": len(inactive_runs),
        "lowest_active_run_length": next((b - s + 1 for s, b in active_runs if s == int(np.flatnonzero(a)[0])), 0),
        "highest_active_run_length": next((b - s + 1 for s, b in reversed(active_runs) if b == int(np.flatnonzero(a)[-1])), 0),
    }


def binary_metrics(predicted, reference) -> dict[str, float]:
    p = np.asarray(predicted, dtype=bool); y = np.asarray(reference, dtype=bool)
    tp = int(np.sum(p & y)); tn = int(np.sum(~p & ~y)); fp = int(np.sum(p & ~y)); fn = int(np.sum(~p & y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "specificity": specificity, "balanced_accuracy": (recall + specificity) / 2.0}
