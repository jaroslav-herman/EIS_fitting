from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np

from eis_model import CycleState, ParameterValue, ProjectState, as_1d_array, sort_spectrum


@dataclass
class LoadedProject:
    dataframe: object
    state: ProjectState
    technique: str


@dataclass
class BatchCycleFit:
    cycle: CycleState
    parameters: list[ParameterValue]
    fitted_parameters: np.ndarray
    fit_frequency_hz: np.ndarray
    fit_impedance: np.ndarray
    fit_at_data_impedance: np.ndarray


@dataclass
class BatchFitReport:
    fits: list[BatchCycleFit]
    failed_cycle: int | None = None
    error: str | None = None


def _safe_unique_ints(values: Iterable[object]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            integer = int(value)
        except (TypeError, ValueError):
            continue
        if integer not in seen:
            result.append(integer)
            seen.add(integer)
    return sorted(result)


def infer_initial(name: str) -> float:
    normalized = name.lower()
    if normalized.startswith("r"):
        return 0.1
    if normalized.startswith("l"):
        return 1e-8
    if normalized.startswith("cpe") and normalized.endswith("_0"):
        return 1e-6
    if normalized.startswith("cpe") and normalized.endswith("_1"):
        return 0.9
    if normalized.startswith("c"):
        return 1e-6
    return 0.1


def infer_bounds(name: str) -> tuple[float, float]:
    normalized = name.lower()
    if normalized.startswith("cpe") and normalized.endswith("_1"):
        return 0.0, 1.0
    if normalized.startswith("r"):
        return 0.0, 1e6
    if normalized.startswith("l"):
        return 0.0, 1.0
    if normalized.startswith("cpe") and normalized.endswith("_0"):
        return 1e-12, 1e3
    if normalized.startswith("c"):
        return 1e-12, 1e3
    return 0.0, 1e6


def circuit_parameters(circuit: str) -> list[ParameterValue]:
    from impedance.models.circuits import CustomCircuit
    from impedance.models.circuits.circuits import calculateCircuitLength

    length = int(calculateCircuitLength(circuit))
    model = CustomCircuit(circuit, initial_guess=[1.0] * length)
    names, units = model.get_param_names()
    parameters = []
    for name, unit in zip(names, units):
        lower, upper = infer_bounds(name)
        parameters.append(
            ParameterValue(name, unit, infer_initial(name), lower, upper)
        )
    return parameters


def load_cycle(dataframe, cycle: int, control: str) -> CycleState:
    if "freq_hz" not in dataframe.columns:
        raise KeyError("The file has no 'freq_hz' column and is not a PEIS spectrum")
    rows = dataframe["freq_hz"] != 0
    if "cycle_number" in dataframe.columns:
        rows &= dataframe["cycle_number"] == cycle

    if control == "Ewe":
        real_column, imaginary_column, potential_column = (
            "re_z_ohm",
            "minus_im_z_ohm",
            "ewe_v",
        )
    else:
        real_column, imaginary_column, potential_column = (
            "re_zwe_ce_ohm",
            "minus_im_zwe_ce_ohm",
            "ewe_ece_v",
        )
    missing = [
        column
        for column in (real_column, imaginary_column)
        if column not in dataframe.columns
    ]
    if missing:
        raise KeyError(f"Missing impedance columns {missing}; check the control setting")

    frequency = dataframe.loc[rows, "freq_hz"].to_numpy()
    impedance = (
        dataframe.loc[rows, real_column].to_numpy()
        - 1j * dataframe.loc[rows, imaginary_column].to_numpy()
    )
    frequency, impedance = sort_spectrum(as_1d_array(frequency), as_1d_array(impedance))
    if frequency.size == 0:
        raise ValueError(f"Cycle {cycle} contains no impedance points")
    potential = (
        float(np.nanmean(dataframe.loc[rows, potential_column].to_numpy()))
        if potential_column in dataframe.columns
        else 0.0
    )
    current = (
        float(np.nanmean(dataframe.loc[rows, "i_ma"].to_numpy()))
        if "i_ma" in dataframe.columns
        else 0.0
    )
    return CycleState(cycle, frequency, impedance, potential, current)


def load_project(
    path: Path,
    cycle: int,
    control: str,
    circuit: str,
) -> LoadedProject:
    from wepy import read_mpt_dataframe

    dataframe, _metadata, technique = read_mpt_dataframe(path)
    cycles = (
        _safe_unique_ints(dataframe["cycle_number"].values)
        if "cycle_number" in dataframe.columns
        else [cycle]
    )
    if not cycles:
        raise ValueError("No cycles were found in the file")
    active_cycle = cycle if cycle in cycles else cycles[0]
    parameters = circuit_parameters(circuit)
    active = load_cycle(dataframe, active_cycle, control)
    active.parameters = [
        ParameterValue(p.name, p.unit, p.initial, p.lower, p.upper)
        for p in parameters
    ]
    state = ProjectState(
        source_path=path,
        circuit=circuit,
        control=control,
        available_cycles=cycles,
        active_cycle=active_cycle,
        default_parameters=parameters,
        cycles={active_cycle: active},
    )
    return LoadedProject(dataframe, state, technique or "Unknown")


def find_outlier_indices(state: CycleState, threshold: float) -> np.ndarray:
    from wepy.eis import find_outliers

    return as_1d_array(
        find_outliers(state.frequency_hz.copy(), state.impedance.copy(), threshold)
    ).astype(int)


def find_outliers_for_all_cycles(
    dataframe,
    cycles: list[int],
    control: str,
    threshold: float,
) -> dict[int, tuple[CycleState, np.ndarray]]:
    from wepy.eis import find_outliers

    results: dict[int, tuple[CycleState, np.ndarray]] = {}
    for cycle_number in cycles:
        cycle = load_cycle(dataframe, cycle_number, control)
        indices = as_1d_array(
            find_outliers(
                cycle.frequency_hz.copy(),
                cycle.impedance.copy(),
                threshold,
            )
        ).astype(int)
        results[cycle_number] = (cycle, indices)
    return results


def fit_cycle(
    state: CycleState,
    circuit: str,
    parameters: list[ParameterValue],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from impedance.models.circuits import CustomCircuit
    from wepy.eis import fit_spectrum, show_fit

    included = state.included
    frequency = state.frequency_hz[included]
    impedance = state.impedance[included]
    if frequency.size < 3:
        raise ValueError("At least three included points are required for fitting")
    frequency, impedance = sort_spectrum(frequency, impedance)
    initial = [parameter.initial for parameter in parameters]
    bounds = (
        [parameter.lower for parameter in parameters],
        [parameter.upper for parameter in parameters],
    )
    fitted, _errors = fit_spectrum(
        frequency,
        impedance,
        cir=circuit,
        init=initial,
        bounds=bounds,
        outliers=False,
        E=state.potential_v,
        I=state.current_ma,
    )
    circuit_parameters_only = as_1d_array(fitted)[2:]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Simulating circuit based on initial parameters",
            category=UserWarning,
        )
        fit_frequency, fit_impedance = show_fit(
            frequency,
            circuit,
            circuit_parameters_only,
            points=200,
        )
    fitted_model = CustomCircuit(circuit, initial_guess=circuit_parameters_only)
    fitted_model.parameters_ = circuit_parameters_only
    fit_at_data = fitted_model.predict(state.frequency_hz)
    return (
        circuit_parameters_only,
        as_1d_array(fit_frequency),
        as_1d_array(fit_impedance),
        as_1d_array(fit_at_data),
    )


def batch_fit_from_cycle(
    dataframe,
    project: ProjectState,
    start_cycle: int,
    initial_parameters: list[ParameterValue],
) -> BatchFitReport:
    start_index = project.available_cycles.index(start_cycle)
    cycle_numbers = project.available_cycles[start_index:]
    next_parameters = [
        ParameterValue(p.name, p.unit, p.initial, p.lower, p.upper)
        for p in initial_parameters
    ]
    completed: list[BatchCycleFit] = []
    for cycle_number in cycle_numbers:
        cycle = project.cycles.get(cycle_number)
        if cycle is None:
            cycle = load_cycle(dataframe, cycle_number, project.control)
            if project.all_frequency_window is not None:
                cycle.frequency_window = project.all_frequency_window
        try:
            fitted, fit_frequency, fit_impedance, fit_at_data = fit_cycle(
                cycle,
                project.circuit,
                next_parameters,
            )
        except Exception as error:
            return BatchFitReport(
                fits=completed,
                failed_cycle=cycle_number,
                error=f"{type(error).__name__}: {error}",
            )
        fitted_parameters = [
            ParameterValue(
                parameter.name,
                parameter.unit,
                float(value),
                parameter.lower,
                parameter.upper,
            )
            for parameter, value in zip(next_parameters, fitted)
        ]
        completed.append(
            BatchCycleFit(
                cycle=cycle,
                parameters=fitted_parameters,
                fitted_parameters=fitted,
                fit_frequency_hz=fit_frequency,
                fit_impedance=fit_impedance,
                fit_at_data_impedance=fit_at_data,
            )
        )
        next_parameters = fitted_parameters
    return BatchFitReport(fits=completed)
