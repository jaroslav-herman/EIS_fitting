"""Transparent, data-informed limits for ML EEC initial parameters."""
from __future__ import annotations

import numpy as np


ALPHA_LIMITS = (1.0e-4, 0.9999)

RELIABILITY = {
    "R0": ("high", 1.0, 0.90),
    "L0": ("high_if_signature", 1.0, 0.85),
    "R1": ("high", 1.0, 0.85),
    "Q1": ("medium", 1.25, 0.70),
    "R2": ("medium_low", 2.0, 0.55),
    "Q2": ("low", 2.5, 0.40),
    "alpha1": ("medium", 1.25, 0.70),
    "alpha2": ("low", 1.75, 0.50),
}


def training_parameter_statistics(training_metadata, parameter: str) -> dict:
    values = np.asarray(training_metadata.get(parameter, []), dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        return {"count": 0, "median": None, "log10_std": None, "q05": None, "q95": None}
    logs = np.log10(values)
    return {
        "count": int(values.size), "median": float(np.median(values)),
        "log10_std": float(np.std(logs)), "q05": float(np.quantile(values, .05)),
        "q95": float(np.quantile(values, .95)),
    }


def build_limit_strategy(training_metadata) -> dict[str, dict]:
    strategy = {}
    for parameter in ("R0", "L0", "R1", "Q1", "R2", "Q2", "alpha1", "alpha2"):
        stats = training_parameter_statistics(training_metadata, parameter)
        reliability, multiplier, confidence = RELIABILITY[parameter]
        if parameter in {"alpha1", "alpha2"}:
            factor = None
            method = "physical_CPE_range_with_safety_margin"
        elif parameter == "R0":
            factor = 3.0
            method = "DRT_R0_reference_times_three_intersected_with_application_limit"
        elif parameter == "L0":
            factor = 10.0
            method = "DRT_L0_reference_times_ten_intersected_with_application_limit"
        else:
            log_std = stats["log10_std"]
            distribution_factor = 10.0 ** (2.0 * log_std) if log_std is not None else 10.0
            factor = float(min(100.0, max(3.0, distribution_factor) * multiplier))
            method = "training_log10_std_two_sigma_times_reliability_multiplier"
        strategy[parameter] = {
            "reliability": reliability, "confidence": confidence, "limit_factor": factor,
            "method": method, "training_distribution": stats,
        }
    return strategy


def make_parameter_limit(
    parameter: str,
    initial_value: float,
    application_lower: float,
    application_upper: float,
    strategy: dict,
) -> dict:
    value = float(initial_value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite initial value for {parameter}")
    if parameter in {"alpha1", "alpha2"}:
        lower, upper = ALPHA_LIMITS
        initial = float(np.clip(value, lower, upper))
        return {
            "initial_value": initial, "lower": lower, "upper": upper,
            "lower_limit": lower, "upper_limit": upper, "limit_factor": None,
            "source": "ML_STAGE4B_INITIAL_GUESS", "reliability": strategy["reliability"],
            "confidence": strategy["confidence"], "limit_strategy": strategy["method"],
        }
    factor = float(strategy["limit_factor"])
    lower = max(float(application_lower), value / factor)
    upper = min(float(application_upper), value * factor)
    lower = max(lower, np.finfo(float).tiny)
    if upper <= lower:
        upper = max(lower * (1.0 + 1.0e-9), value)
    initial = float(np.clip(value, lower, upper))
    return {
        "initial_value": initial, "lower": float(lower), "upper": float(upper),
        "lower_limit": float(lower), "upper_limit": float(upper), "limit_factor": factor,
        "source": "DRT_RIDGE_HIGH_FREQUENCY" if parameter in {"R0", "L0"} else "ML_STAGE4B_INITIAL_GUESS",
        "reliability": strategy["reliability"], "confidence": strategy["confidence"],
        "limit_strategy": strategy["method"], "training_distribution": strategy["training_distribution"],
    }
