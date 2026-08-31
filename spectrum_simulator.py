"""Pure numerical helpers for the Spectra Simulator GUI mode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulatedSpectrum:
    frequency_hz: np.ndarray
    ideal_impedance: np.ndarray
    impedance: np.ndarray
    noise_enabled: bool = False
    noise_level_percent: float = 0.0
    seed: int | None = None


def logarithmic_frequencies(
    minimum_hz: float, maximum_hz: float, points_per_decade: int
) -> np.ndarray:
    minimum_hz = float(minimum_hz)
    maximum_hz = float(maximum_hz)
    points_per_decade = int(points_per_decade)
    if not np.isfinite(minimum_hz) or minimum_hz <= 0:
        raise ValueError("Minimum frequency must be a positive number")
    if not np.isfinite(maximum_hz) or maximum_hz <= minimum_hz:
        raise ValueError("Maximum frequency must be greater than minimum frequency")
    if points_per_decade < 1:
        raise ValueError("Points per decade must be at least 1")
    count = max(2, int(round(np.log10(maximum_hz / minimum_hz) * points_per_decade)) + 1)
    return np.logspace(np.log10(maximum_hz), np.log10(minimum_hz), count)


def evaluate_circuit(circuit: str, frequency_hz, parameters) -> np.ndarray:
    """Evaluate an impedance circuit without fitting or changing parameters."""
    from impedance.models.circuits import CustomCircuit

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    values = [float(value) for value in parameters]
    model = CustomCircuit(circuit, initial_guess=values)
    # These are explicit simulation parameters, not merely an initial guess.
    # Mark them as active so impedance.py does not emit its misleading warning.
    model.parameters_ = np.asarray(values, dtype=float)
    return np.asarray(model.predict(frequency_hz), dtype=complex)


def simulate_spectrum(
    circuit: str,
    frequency_hz,
    parameters,
    *,
    noise_enabled: bool = False,
    noise_level_percent: float = 0.0,
    seed: int | None = None,
) -> SimulatedSpectrum:
    if float(noise_level_percent) < 0:
        raise ValueError("Noise level cannot be negative")
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    ideal = evaluate_circuit(circuit, frequency_hz, parameters)
    impedance = ideal.copy()
    if noise_enabled and float(noise_level_percent) > 0:
        rng = np.random.default_rng(seed)
        scale = np.maximum(np.abs(ideal), np.finfo(float).eps)
        relative = float(noise_level_percent) / 100.0
        impedance = ideal + scale * relative * (
            rng.standard_normal(ideal.size) + 1j * rng.standard_normal(ideal.size)
        )
    return SimulatedSpectrum(
        frequency_hz=frequency_hz,
        ideal_impedance=ideal,
        impedance=impedance,
        noise_enabled=bool(noise_enabled),
        noise_level_percent=float(noise_level_percent),
        seed=seed,
    )
