from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from eis_model import CycleState, ParameterValue, ProjectState, as_1d_array


PROJECT_FORMAT = "eis-fitting-project"
PROJECT_VERSION = 1


def _parameter_to_dict(parameter: ParameterValue) -> dict[str, object]:
    return {
        "name": parameter.name,
        "unit": parameter.unit,
        "initial": parameter.initial,
        "lower": parameter.lower,
        "upper": parameter.upper,
        "error_percent": parameter.error_percent,
    }


def _parameter_from_dict(data: dict[str, object]) -> ParameterValue:
    return ParameterValue(
        name=str(data["name"]),
        unit=str(data.get("unit", "")),
        initial=float(data["initial"]),
        lower=float(data["lower"]),
        upper=float(data["upper"]),
        error_percent=(
            float(data["error_percent"])
            if data.get("error_percent") is not None
            else None
        ),
    )


def _optional_array(values) -> np.ndarray | None:
    if values is None:
        return None
    return as_1d_array(values)


def _cycle_to_dict(cycle: CycleState) -> dict[str, object]:
    fit_impedance = None
    if cycle.fit_impedance is not None:
        fit_impedance = {
            "real": cycle.fit_impedance.real.tolist(),
            "imaginary": cycle.fit_impedance.imag.tolist(),
        }
    fit_at_data = None
    if cycle.fit_at_data_impedance is not None:
        fit_at_data = {
            "real": cycle.fit_at_data_impedance.real.tolist(),
            "imaginary": cycle.fit_at_data_impedance.imag.tolist(),
        }
    return {
        "frequency_window": (
            list(cycle.frequency_window) if cycle.frequency_window is not None else None
        ),
        "manually_included": cycle.manually_included.astype(bool).tolist(),
        "outliers": cycle.outliers.astype(bool).tolist(),
        "parameters": [_parameter_to_dict(value) for value in cycle.parameters],
        "fit_parameters": (
            cycle.fit_parameters.tolist() if cycle.fit_parameters is not None else None
        ),
        "fit_frequency_hz": (
            cycle.fit_frequency_hz.tolist()
            if cycle.fit_frequency_hz is not None
            else None
        ),
        "fit_impedance": fit_impedance,
        "fit_at_data_impedance": fit_at_data,
    }


def save_project_file(state: ProjectState, path: Path) -> None:
    payload = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "source_path": str(state.source_path),
        "circuit": state.circuit,
        "control": state.control,
        "active_cycle": state.active_cycle,
        "all_frequency_window": (
            list(state.all_frequency_window)
            if state.all_frequency_window is not None
            else None
        ),
        "default_parameters": [
            _parameter_to_dict(value) for value in state.default_parameters
        ],
        "cycles": {
            str(cycle_number): _cycle_to_dict(cycle)
            for cycle_number, cycle in sorted(state.cycles.items())
        },
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_project_file(
    current: ProjectState,
    dataframe,
    path: Path,
) -> ProjectState:
    from eis_services import load_cycle

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != PROJECT_FORMAT:
        raise ValueError("This is not an EIS fitting project file")
    if int(payload.get("version", 0)) != PROJECT_VERSION:
        raise ValueError(f"Unsupported project version: {payload.get('version')}")

    circuit = str(payload["circuit"])
    control = str(payload.get("control", current.control))
    defaults = [
        _parameter_from_dict(value) for value in payload["default_parameters"]
    ]
    restored_cycles: dict[int, CycleState] = {}
    available = set(current.available_cycles)
    for cycle_text, saved in payload.get("cycles", {}).items():
        cycle_number = int(cycle_text)
        if cycle_number not in available:
            continue
        cycle = load_cycle(dataframe, cycle_number, control)
        included = as_1d_array(saved["manually_included"]).astype(bool)
        outliers = as_1d_array(saved.get("outliers", [])).astype(bool)
        if included.size != cycle.frequency_hz.size:
            raise ValueError(
                f"Cycle {cycle_number} has {cycle.frequency_hz.size} points, "
                f"but the project mask has {included.size}"
            )
        if outliers.size == 0:
            outliers = np.zeros(cycle.frequency_hz.size, dtype=bool)
        if outliers.size != cycle.frequency_hz.size:
            raise ValueError(f"Cycle {cycle_number} has an incompatible outlier mask")
        cycle.manually_included = included
        cycle.outliers = outliers
        window = saved.get("frequency_window")
        cycle.frequency_window = (
            (float(window[0]), float(window[1])) if window is not None else None
        )
        cycle.parameters = [
            _parameter_from_dict(value) for value in saved.get("parameters", [])
        ] or [
            ParameterValue(
                p.name, p.unit, p.initial, p.lower, p.upper, p.error_percent
            )
            for p in defaults
        ]
        cycle.fit_parameters = _optional_array(saved.get("fit_parameters"))
        cycle.fit_frequency_hz = _optional_array(saved.get("fit_frequency_hz"))
        saved_impedance = saved.get("fit_impedance")
        if saved_impedance is not None:
            real = as_1d_array(saved_impedance["real"]).astype(float)
            imaginary = as_1d_array(saved_impedance["imaginary"]).astype(float)
            if real.size != imaginary.size:
                raise ValueError(f"Cycle {cycle_number} has an invalid saved fit curve")
            cycle.fit_impedance = real + 1j * imaginary
        saved_at_data = saved.get("fit_at_data_impedance")
        if saved_at_data is not None:
            real = as_1d_array(saved_at_data["real"]).astype(float)
            imaginary = as_1d_array(saved_at_data["imaginary"]).astype(float)
            if real.size != cycle.frequency_hz.size or imaginary.size != real.size:
                raise ValueError(
                    f"Cycle {cycle_number} has invalid fitted data-point values"
                )
            cycle.fit_at_data_impedance = real + 1j * imaginary
        restored_cycles[cycle_number] = cycle

    requested_active = int(payload.get("active_cycle", current.active_cycle))
    active_cycle = (
        requested_active
        if requested_active in current.available_cycles
        else current.active_cycle
    )
    if active_cycle not in restored_cycles:
        active = load_cycle(dataframe, active_cycle, control)
        active.parameters = [
            ParameterValue(
                p.name, p.unit, p.initial, p.lower, p.upper, p.error_percent
            )
            for p in defaults
        ]
        restored_cycles[active_cycle] = active
    all_window = payload.get("all_frequency_window")
    return ProjectState(
        source_path=current.source_path,
        circuit=circuit,
        control=control,
        available_cycles=current.available_cycles.copy(),
        active_cycle=active_cycle,
        default_parameters=defaults,
        cycles=restored_cycles,
        all_frequency_window=(
            (float(all_window[0]), float(all_window[1]))
            if all_window is not None
            else None
        ),
    )


def export_fit_parameters(state: ProjectState, path: Path) -> int:
    fitted_cycles = [
        cycle
        for _, cycle in sorted(state.cycles.items())
        if cycle.fit_parameters is not None
    ]
    if not fitted_cycles:
        raise ValueError("No cycles have fitted parameters to export")
    parameter_names = [parameter.name for parameter in state.default_parameters]
    parameter_columns = []
    for name in parameter_names:
        parameter_columns.extend((name, f"{name}_error_percent"))
    fieldnames = [
        "source_file",
        "cycle",
        "circuit",
        "potential_V",
        "current_mA",
        "included_points",
        *parameter_columns,
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for cycle in fitted_cycles:
            values = as_1d_array(cycle.fit_parameters)
            if values.size != len(parameter_names):
                raise ValueError(f"Cycle {cycle.cycle} has incompatible fit parameters")
            row = {
                "source_file": state.source_path.name,
                "cycle": cycle.cycle,
                "circuit": state.circuit,
                "potential_V": cycle.potential_v,
                "current_mA": cycle.current_ma,
                "included_points": int(np.count_nonzero(cycle.included)),
            }
            row.update(dict(zip(parameter_names, values.tolist())))
            errors_by_name = {
                parameter.name: parameter.error_percent
                for parameter in cycle.parameters
            }
            row.update(
                {
                    f"{name}_error_percent": errors_by_name.get(name)
                    for name in parameter_names
                }
            )
            writer.writerow(row)
    return len(fitted_cycles)
