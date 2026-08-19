"""Fast local-consistency validity detector for individual EIS points.

The detector is intentionally independent of Bayes-DRT2, DRT, EEC fitting,
and the application's manual point mask.  It predicts each point from nearby
points in log-frequency space and scores the complex residual.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _local_prediction(x: np.ndarray, z: np.ndarray, index: int, neighborhood: int, degree: int) -> tuple[complex, float] | None:
    """Leave-one-out robust local interpolation prediction and scale."""
    left = max(0, index - neighborhood)
    right = min(x.size, index + neighborhood + 1)
    neighbours = np.concatenate((np.arange(left, index), np.arange(index + 1, right)))
    if neighbours.size < 2:
        return None
    left = np.arange(max(0, index - neighborhood), index)
    right = np.arange(index + 1, min(x.size, index + neighborhood + 1))
    # Robust side medians prevent one contaminated neighbour from propagating
    # an anomaly to the surrounding points.  Between the two robust anchors
    # we use log-frequency interpolation; at an edge we use a one-sided local
    # level rather than treating the edge as anomalous.
    if left.size and right.size:
        left_x, right_x = float(np.median(x[left])), float(np.median(x[right]))
        left_z, right_z = np.median(z[left].real) + 1j * np.median(z[left].imag), np.median(z[right].real) + 1j * np.median(z[right].imag)
        denominator = right_x - left_x
        fraction = (float(x[index]) - left_x) / denominator if abs(denominator) > np.finfo(float).eps else 0.5
        prediction = complex(left_z + fraction * (right_z - left_z))
    elif left.size:
        prediction = complex(float(np.median(z[left].real)), float(np.median(z[left].imag)))
    elif right.size:
        prediction = complex(float(np.median(z[right].real)), float(np.median(z[right].imag)))
    else:
        return None

    neighbour_z = z[neighbours]
    differences = np.abs(np.diff(neighbour_z))
    variation = 1.4826 * _mad(differences)
    magnitude_floor = 0.05 * float(np.median(np.abs(neighbour_z)))
    scale = max(variation, magnitude_floor, np.finfo(float).eps)
    return prediction, scale


def _diagnostic_derivatives(x: np.ndarray, z: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope = np.full(x.size, np.nan, dtype=float)
    curvature = np.full(x.size, np.nan, dtype=float)
    if x.size < 3:
        return slope, curvature
    dx_left = x[1:-1] - x[:-2]
    dx_right = x[2:] - x[1:-1]
    valid = (np.abs(dx_left) > np.finfo(float).eps) & (np.abs(dx_right) > np.finfo(float).eps)
    differences = np.diff(z, axis=0)
    left_slope = np.divide(differences[:-1], dx_left, out=np.full(dx_left.shape, np.nan + 1j * np.nan), where=np.abs(dx_left) > np.finfo(float).eps)
    right_slope = np.divide(differences[1:], dx_right, out=np.full(dx_right.shape, np.nan + 1j * np.nan), where=np.abs(dx_right) > np.finfo(float).eps)
    slope_change = np.abs(right_slope - left_slope)
    slope_middle = slope[1:-1]
    curvature_middle = curvature[1:-1]
    slope_middle[valid] = slope_change[valid]
    curvature_middle[valid] = slope_change[valid] / np.maximum(np.abs(left_slope[valid]) + np.abs(right_slope[valid]), np.finfo(float).eps)
    slope[1:-1] = slope_middle
    curvature[1:-1] = curvature_middle
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size:
        scale = max(1.4826 * _mad(finite_scores), 0.1 * float(np.median(finite_scores)), np.finfo(float).eps)
        slope = slope / scale
        curvature = curvature / scale
    return slope, curvature


def detect_valid_points(
    frequency,
    impedance,
    *,
    threshold: float = 4.0,
    neighborhood: int = 3,
    min_points: int = 4,
    degree: int = 1,
    frequency_range: tuple[float, float] | None = None,
    max_iterations: int = 2,
    return_diagnostics: bool = True,
):
    """Detect isolated locally inconsistent EIS points.

    Each candidate is predicted by interpolation between robust medians of
    neighbouring points on either side, excluding the candidate itself.  The score is

    ``abs(Z_i - Z_pred_i) / max(1.4826*MAD(|diff(Z_neighbours)|),
    0.05*median(|Z_neighbours|), eps)``.

    The returned mask has the original input ordering.  Invalid input points,
    points outside ``frequency_range``, and points lacking enough context are
    never silently discarded: their reason is recorded in diagnostics.
    ``max_iterations`` removes only scores above ``max(threshold * 2, 8)``
    between passes, then applies the requested threshold on the final pass.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    if int(neighborhood) < 1 or int(min_points) < 2:
        raise ValueError("neighborhood must be >= 1 and min_points >= 2")
    if int(degree) not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be >= 1")

    f = np.asarray(frequency, dtype=float).reshape(-1)
    z = np.asarray(impedance, dtype=complex).reshape(-1)
    if f.size != z.size:
        raise ValueError("frequency and impedance must have equal lengths")
    n = f.size
    valid_input = np.isfinite(f) & (f > 0) & np.isfinite(z.real) & np.isfinite(z.imag)
    in_range = np.ones(n, dtype=bool)
    if frequency_range is not None:
        minimum, maximum = sorted(map(float, frequency_range))
        if not np.isfinite(minimum) or minimum <= 0 or not np.isfinite(maximum) or maximum <= minimum:
            raise ValueError("frequency_range must contain two positive, distinct finite values")
        in_range = (f >= minimum) & (f <= maximum)
    candidate = valid_input & in_range
    mask = np.zeros(n, dtype=bool)
    score = np.full(n, np.nan, dtype=float)
    scale_output = np.full(n, np.nan, dtype=float)
    predicted = np.full(n, np.nan + 1j * np.nan, dtype=complex)
    reason = np.full(n, "input_invalid", dtype=object)
    reason[valid_input & ~in_range] = "outside_range"
    reason[candidate] = "insufficient_context"

    if np.count_nonzero(candidate) < min_points + 1:
        diagnostics = _diagnostics(f, z, np.full(n, np.nan), score, scale_output, predicted, mask, reason, valid_input, in_range)
        return (mask, score, diagnostics) if return_diagnostics else (mask, score)

    order = np.argsort(f[candidate], kind="mergesort")
    original_indices = np.flatnonzero(candidate)[order]
    x = np.log10(f[original_indices])
    z_sorted = z[original_indices]
    active = np.ones(x.size, dtype=bool)
    strong_threshold = max(float(threshold) * 2.0, 8.0)
    passes = max(1, int(max_iterations))
    last_scores = np.full(x.size, np.nan, dtype=float)
    last_scales = np.full(x.size, np.nan, dtype=float)
    last_predictions = np.full(x.size, np.nan + 1j * np.nan, dtype=complex)
    first_scores = np.full(x.size, np.nan, dtype=float)
    first_scales = np.full(x.size, np.nan, dtype=float)
    first_predictions = np.full(x.size, np.nan + 1j * np.nan, dtype=complex)
    removed_strong = np.zeros(x.size, dtype=bool)
    for pass_index in range(passes):
        local_x = x[active]
        local_z = z_sorted[active]
        local_to_sorted = np.flatnonzero(active)
        current_scores = np.full(x.size, np.nan, dtype=float)
        current_scales = np.full(x.size, np.nan, dtype=float)
        current_predictions = np.full(x.size, np.nan + 1j * np.nan, dtype=complex)
        current_reason = np.full(x.size, "insufficient_context", dtype=object)
        for local_index, sorted_index in enumerate(local_to_sorted):
            if local_x.size - 1 < min_points:
                continue
            result = _local_prediction(local_x, local_z, local_index, neighborhood, degree)
            if result is None:
                continue
            prediction, local_scale = result
            current_predictions[sorted_index] = prediction
            current_scales[sorted_index] = local_scale
            current_scores[sorted_index] = abs(local_z[local_index] - prediction) / local_scale
            current_reason[sorted_index] = "valid"
        last_scores, last_scales, last_predictions = current_scores, current_scales, current_predictions
        if pass_index == 0:
            first_scores, first_scales, first_predictions = current_scores.copy(), current_scales.copy(), current_predictions.copy()
        if pass_index < passes - 1:
            remove = active & np.isfinite(current_scores) & (current_scores > strong_threshold)
            active[remove] = False
            removed_strong |= remove
    final_outliers = active & np.isfinite(last_scores) & (last_scores > float(threshold))
    final_valid = active & np.isfinite(last_scores) & ~final_outliers
    reported_scores = np.where(np.isfinite(last_scores), last_scores, first_scores)
    reported_scales = np.where(np.isfinite(last_scales), last_scales, first_scales)
    reported_predictions = np.where(np.isfinite(last_predictions), last_predictions, first_predictions)
    for sorted_index, original_index in enumerate(original_indices):
        score[original_index] = reported_scores[sorted_index]
        scale_output[original_index] = reported_scales[sorted_index]
        predicted[original_index] = reported_predictions[sorted_index]
        if removed_strong[sorted_index]:
            reason[original_index] = "local_anomaly"
        elif not np.isfinite(last_scores[sorted_index]):
            reason[original_index] = "insufficient_context"
        elif final_outliers[sorted_index]:
            reason[original_index] = "local_anomaly"
        else:
            reason[original_index] = "valid"
            mask[original_index] = bool(final_valid[sorted_index])
    slope_score, curvature_score = _diagnostic_derivatives(x, z_sorted, reported_scores)
    slope_original = np.full(n, np.nan); curvature_original = np.full(n, np.nan)
    slope_original[original_indices] = slope_score; curvature_original[original_indices] = curvature_score
    diagnostics = _diagnostics(f, z, last_scores, score, scale_output, predicted, mask, reason, valid_input, in_range)
    diagnostics["slope_score"] = slope_original
    diagnostics["curvature_score"] = curvature_original
    diagnostics["sorted_indices"] = original_indices
    return (mask, score, diagnostics) if return_diagnostics else (mask, score)


def detect_outliers_in_active_points(
    frequency,
    impedance,
    active_mask,
    *,
    threshold: float = 4.0,
    **detector_options,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return original indices of deterministic outliers among active points.

    The detector receives only currently active points, so inactive points can
    neither influence local statistics nor become active as a side effect.
    """
    frequency = np.asarray(frequency, dtype=float).reshape(-1)
    impedance = np.asarray(impedance, dtype=complex).reshape(-1)
    active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    if frequency.size != impedance.size or active_mask.size != frequency.size:
        raise ValueError("frequency, impedance, and active_mask must have equal lengths")
    active_indices = np.flatnonzero(active_mask)
    if active_indices.size == 0:
        return np.empty(0, dtype=int), {"active_indices": active_indices}
    valid_mask, _scores, diagnostics = detect_valid_points(
        frequency[active_indices],
        impedance[active_indices],
        threshold=threshold,
        **detector_options,
    )
    reasons = np.asarray(diagnostics["rejection_reason"], dtype=object)
    local_outliers = reasons == "local_anomaly"
    return active_indices[local_outliers], diagnostics


def _diagnostics(f, z, sorted_scores, scores, scales, predicted, mask, reason, valid_input, in_range) -> dict[str, Any]:
    residual = np.abs(z - predicted)
    return {
        "frequency": f.copy(), "log_frequency": np.where(f > 0, np.log10(f), np.nan),
        "complex_residual": residual, "normalized_score": scores.copy(),
        "local_scale": scales.copy(), "predicted_real": predicted.real.copy(),
        "predicted_imag": predicted.imag.copy(), "valid": mask.copy(),
        "rejection_reason": reason.copy(), "input_valid": valid_input.copy(),
        "in_selected_range": in_range.copy(),
    }
