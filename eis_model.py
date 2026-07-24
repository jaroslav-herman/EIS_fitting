from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def as_1d_array(values) -> np.ndarray:
    return np.asarray(values).reshape(-1)


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
        self.clear_fit()

    def apply_outliers(self, indices: np.ndarray) -> None:
        outliers = np.zeros(self.frequency_hz.size, dtype=bool)
        valid = as_1d_array(indices).astype(int)
        valid = valid[(valid >= 0) & (valid < self.frequency_hz.size)]
        outliers[valid] = True
        self.outliers = outliers
        self.manually_included = ~outliers
        self.clear_fit()

    def reset_selection(self) -> None:
        self.manually_included[:] = True
        self.outliers[:] = False
        self.clear_fit()

    def clear_fit(self) -> None:
        self.fit_parameters = None
        self.fit_frequency_hz = None
        self.fit_impedance = None
        self.fit_at_data_impedance = None


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
            ParameterValue(p.name, p.unit, p.initial, p.lower, p.upper)
            for p in parameters
        ]

    def parameters_for(self, cycle: int) -> list[ParameterValue]:
        state = self.cycles.get(cycle)
        source = state.parameters if state and state.parameters else self.default_parameters
        return [
            ParameterValue(p.name, p.unit, p.initial, p.lower, p.upper)
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
            ParameterValue(p.name, p.unit, p.initial, p.lower, p.upper)
            for p in parameters
        ]
        for cycle in self.cycles.values():
            cycle.parameters = [
                ParameterValue(p.name, p.unit, p.initial, p.lower, p.upper)
                for p in parameters
            ]
            cycle.clear_fit()
