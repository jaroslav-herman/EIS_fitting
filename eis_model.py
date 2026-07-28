from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def as_1d_array(values) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def copy_parameter_values(parameters: list["ParameterValue"]) -> list["ParameterValue"]:
    return [
        ParameterValue(
            parameter.name,
            parameter.unit,
            parameter.initial,
            parameter.lower,
            parameter.upper,
            parameter.error_percent,
            parameter.fixed,
        )
        for parameter in parameters
    ]


def sort_spectrum(
    frequency_hz: np.ndarray,
    impedance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(frequency_hz)[::-1]
    return frequency_hz[order], impedance[order]


@dataclass
class ParameterValue:
    name: str
    unit: str
    initial: float
    lower: float
    upper: float
    error_percent: float | None = None
    fixed: bool = False


@dataclass
class CycleState:
    cycle: int
    frequency_hz: np.ndarray
    impedance: np.ndarray
    potential_v: float = 0.0
    current_ma: float = 0.0
    manually_included: np.ndarray | None = None
    outliers: np.ndarray | None = None
    frequency_window: tuple[float, float] | None = None
    parameters: list[ParameterValue] = field(default_factory=list)
    fit_parameters: np.ndarray | None = None
    fit_frequency_hz: np.ndarray | None = None
    fit_impedance: np.ndarray | None = None
    fit_at_data_impedance: np.ndarray | None = None
    ridge_tau_s: np.ndarray | None = None
    ridge_gamma_ohm: np.ndarray | None = None
    drt_label: str | None = None
    saved_ridge_tau_s: np.ndarray | None = None
    saved_ridge_gamma_ohm: np.ndarray | None = None
    saved_ridge_included_mask: np.ndarray | None = None
    saved_ridge_outlier_indices: np.ndarray | None = None
    saved_ridge_parameters: list[ParameterValue] = field(default_factory=list)
    saved_ridge_threshold: float | None = None
    saved_ridge_peak_count: int | None = None
    saved_ridge_ohmic_resistance: float | None = None
    saved_ridge_inductance: float | None = None
    saved_hybrid_tau_s: np.ndarray | None = None
    saved_hybrid_gamma_ohm: np.ndarray | None = None
    saved_hybrid_included_mask: np.ndarray | None = None
    saved_hybrid_ohmic_resistance: float | None = None
    kk_fit_impedance: np.ndarray | None = None
    kk_residual_real: np.ndarray | None = None
    kk_residual_imag: np.ndarray | None = None
    kk_included_mask: np.ndarray | None = None
    custom_metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frequency_hz = as_1d_array(self.frequency_hz)
        self.impedance = as_1d_array(self.impedance)
        if self.frequency_hz.size != self.impedance.size:
            raise ValueError("Frequency and impedance arrays must have equal lengths")
        if self.manually_included is None:
            self.manually_included = np.ones(self.frequency_hz.size, dtype=bool)
        else:
            self.manually_included = as_1d_array(self.manually_included).astype(bool)
        if self.outliers is None:
            self.outliers = np.zeros(self.frequency_hz.size, dtype=bool)
        else:
            self.outliers = as_1d_array(self.outliers).astype(bool)
        if self.saved_ridge_included_mask is not None:
            self.saved_ridge_included_mask = (
                as_1d_array(self.saved_ridge_included_mask).astype(bool)
            )
        if self.saved_ridge_outlier_indices is not None:
            self.saved_ridge_outlier_indices = (
                as_1d_array(self.saved_ridge_outlier_indices).astype(int)
            )
        if self.saved_hybrid_included_mask is not None:
            self.saved_hybrid_included_mask = (
                as_1d_array(self.saved_hybrid_included_mask).astype(bool)
            )
        if self.kk_included_mask is not None:
            self.kk_included_mask = as_1d_array(self.kk_included_mask).astype(bool)
        if self.frequency_window is None and self.frequency_hz.size:
            self.frequency_window = (
                float(np.nanmin(self.frequency_hz)),
                float(np.nanmax(self.frequency_hz)),
            )

    @property
    def included(self) -> np.ndarray:
        mask = self.manually_included.copy()
        if self.frequency_window is None:
            return mask
        minimum, maximum = sorted(self.frequency_window)
        return mask & (self.frequency_hz >= minimum) & (self.frequency_hz <= maximum)

    def toggle_point(self, index: int) -> None:
        if not 0 <= index < self.frequency_hz.size:
            return
        if self.frequency_window is not None:
            minimum, maximum = sorted(self.frequency_window)
            if not minimum <= self.frequency_hz[index] <= maximum:
                return
        self.manually_included[index] = not self.manually_included[index]
        if self.manually_included[index]:
            self.outliers[index] = False
        self.invalidate_drt_cache()
        self.clear_fit()

    def apply_outliers(self, indices: np.ndarray) -> None:
        valid = as_1d_array(indices).astype(int)
        valid = valid[(valid >= 0) & (valid < self.frequency_hz.size)]
        self.outliers[valid] = True
        self.manually_included[valid] = False
        self.invalidate_drt_cache()
        self.clear_fit()

    def reset_selection(self) -> None:
        self.manually_included[:] = True
        self.outliers[:] = False
        self.invalidate_drt_cache()
        self.clear_fit()

    def clear_fit(self) -> None:
        self.fit_parameters = None
        self.fit_frequency_hz = None
        self.fit_impedance = None
        self.fit_at_data_impedance = None
        for parameter in self.parameters:
            parameter.error_percent = None

    def invalidate_drt_cache(self) -> None:
        self.ridge_tau_s = None
        self.ridge_gamma_ohm = None
        self.drt_label = None
        self.saved_ridge_tau_s = None
        self.saved_ridge_gamma_ohm = None
        self.saved_ridge_included_mask = None
        self.saved_ridge_outlier_indices = None
        self.saved_ridge_parameters = []
        self.saved_ridge_threshold = None
        self.saved_ridge_peak_count = None
        self.saved_ridge_ohmic_resistance = None
        self.saved_ridge_inductance = None
        self.saved_hybrid_tau_s = None
        self.saved_hybrid_gamma_ohm = None
        self.saved_hybrid_included_mask = None
        self.saved_hybrid_ohmic_resistance = None
        self.kk_fit_impedance = None
        self.kk_residual_real = None
        self.kk_residual_imag = None
        self.kk_included_mask = None

    def ridge_cache_matches(
        self,
        threshold: float,
        parameter_names: list[str],
    ) -> bool:
        if (
            self.saved_ridge_tau_s is None
            or self.saved_ridge_gamma_ohm is None
            or self.saved_ridge_included_mask is None
            or self.saved_ridge_outlier_indices is None
            or self.saved_ridge_threshold is None
            or not self.saved_ridge_parameters
        ):
            return False
        return (
            self.saved_ridge_included_mask.size == self.frequency_hz.size
            and np.array_equal(self.saved_ridge_included_mask, self.included)
            and np.isclose(self.saved_ridge_threshold, threshold)
            and [parameter.name for parameter in self.saved_ridge_parameters]
            == parameter_names
        )

    def hybrid_cache_matches(self) -> bool:
        return (
            self.saved_hybrid_tau_s is not None
            and self.saved_hybrid_gamma_ohm is not None
            and self.saved_hybrid_included_mask is not None
            and self.saved_hybrid_included_mask.size == self.frequency_hz.size
            and np.array_equal(self.saved_hybrid_included_mask, self.included)
        )

    def store_ridge_analysis(
        self,
        threshold: float,
        outlier_indices: np.ndarray,
        parameters: list[ParameterValue],
        peak_count: int,
        ohmic_resistance: float,
        inductance: float,
        tau_s: np.ndarray,
        gamma_ohm: np.ndarray,
    ) -> None:
        self.saved_ridge_threshold = float(threshold)
        self.saved_ridge_included_mask = self.included.copy()
        self.saved_ridge_outlier_indices = as_1d_array(outlier_indices).astype(int)
        self.saved_ridge_parameters = copy_parameter_values(parameters)
        self.saved_ridge_peak_count = int(peak_count)
        self.saved_ridge_ohmic_resistance = float(ohmic_resistance)
        self.saved_ridge_inductance = float(inductance)
        self.saved_ridge_tau_s = as_1d_array(tau_s).astype(float)
        self.saved_ridge_gamma_ohm = as_1d_array(gamma_ohm).astype(float)
        self.show_ridge_drt()

    def store_hybrid_drt(
        self,
        tau_s: np.ndarray,
        gamma_ohm: np.ndarray,
        ohmic_resistance: float | None,
    ) -> None:
        self.saved_hybrid_included_mask = self.included.copy()
        self.saved_hybrid_tau_s = as_1d_array(tau_s).astype(float)
        self.saved_hybrid_gamma_ohm = as_1d_array(gamma_ohm).astype(float)
        self.saved_hybrid_ohmic_resistance = (
            float(ohmic_resistance) if ohmic_resistance is not None else None
        )
        self.show_hybrid_drt()

    def show_ridge_drt(self) -> None:
        self.ridge_tau_s = (
            self.saved_ridge_tau_s.copy() if self.saved_ridge_tau_s is not None else None
        )
        self.ridge_gamma_ohm = (
            self.saved_ridge_gamma_ohm.copy()
            if self.saved_ridge_gamma_ohm is not None
            else None
        )
        self.drt_label = "Ridge DRT" if self.ridge_tau_s is not None else None

    def show_hybrid_drt(self) -> None:
        self.ridge_tau_s = (
            self.saved_hybrid_tau_s.copy()
            if self.saved_hybrid_tau_s is not None
            else None
        )
        self.ridge_gamma_ohm = (
            self.saved_hybrid_gamma_ohm.copy()
            if self.saved_hybrid_gamma_ohm is not None
            else None
        )
        self.drt_label = "Hybrid DRT" if self.ridge_tau_s is not None else None

    def kk_cache_matches(self) -> bool:
        return (
            self.kk_fit_impedance is not None
            and self.kk_residual_real is not None
            and self.kk_residual_imag is not None
            and self.kk_included_mask is not None
            and self.kk_included_mask.size == self.frequency_hz.size
            and np.array_equal(self.kk_included_mask, self.included)
        )

    def store_kk_result(
        self,
        fit_impedance: np.ndarray,
        residual_real: np.ndarray,
        residual_imag: np.ndarray,
    ) -> None:
        self.kk_fit_impedance = as_1d_array(fit_impedance)
        self.kk_residual_real = as_1d_array(residual_real).astype(float)
        self.kk_residual_imag = as_1d_array(residual_imag).astype(float)
        self.kk_included_mask = self.included.copy()


@dataclass
class ProjectState:
    source_path: Path
    circuit: str
    control: str
    available_cycles: list[int]
    active_cycle: int
    default_parameters: list[ParameterValue]
    cycles: dict[int, CycleState] = field(default_factory=dict)
    all_frequency_window: tuple[float, float] | None = None

    @property
    def active(self) -> CycleState:
        return self.cycles[self.active_cycle]

    def remember_parameters(self, parameters: list[ParameterValue]) -> None:
        self.active.parameters = [
            ParameterValue(
                p.name, p.unit, p.initial, p.lower, p.upper, p.error_percent, p.fixed
            )
            for p in parameters
        ]

    def parameters_for(self, cycle: int) -> list[ParameterValue]:
        state = self.cycles.get(cycle)
        source = state.parameters if state and state.parameters else self.default_parameters
        return [
            ParameterValue(
                p.name, p.unit, p.initial, p.lower, p.upper, p.error_percent, p.fixed
            )
            for p in source
        ]

    def apply_frequency_window_to_all(self, window: tuple[float, float]) -> None:
        self.all_frequency_window = window
        for cycle in self.cycles.values():
            cycle.frequency_window = window
            cycle.clear_fit()

    def replace_circuit(
        self,
        circuit: str,
        parameters: list[ParameterValue],
    ) -> None:
        self.circuit = circuit
        self.default_parameters = [
            ParameterValue(
                p.name, p.unit, p.initial, p.lower, p.upper, p.error_percent, p.fixed
            )
            for p in parameters
        ]
        for cycle in self.cycles.values():
            cycle.parameters = [
                ParameterValue(
                    p.name, p.unit, p.initial, p.lower, p.upper, p.error_percent, p.fixed
                )
                for p in parameters
            ]
            cycle.clear_fit()
