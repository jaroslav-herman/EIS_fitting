from __future__ import annotations

import csv
import gzip
from io import StringIO
import json
import re
from pathlib import Path

import numpy as np

from eis_model import CycleState, ParameterValue, ProjectState, as_1d_array
from wepy.eis import capacitance as cpe_capacitance
from wepy.eis import tau as cpe_tau

PROJECT_FORMAT = "eis-fitting-project"
PROJECT_VERSION = 4


def load_json_payload(path: Path) -> dict[str, object]:
    """Load a project-like JSON payload, transparently handling gzip."""
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The JSON project payload must be an object")
    return payload


def _is_gzip_path(path: Path) -> bool:
    return path.suffix.lower() == ".gz"


def _json_bytes(payload: dict[str, object], *, compressed: bool) -> bytes:
    if compressed:
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        return gzip.compress(raw, compresslevel=9, mtime=0)
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def _external_parameter_name(name: str) -> str:
    match = re.fullmatch(r"CPE(\d+)_(0|1)", name)
    if match:
        suffix = "Q" if match.group(2) == "0" else "a"
        return f"{suffix}{match.group(1)}"
    return name


def _external_column_name(name: str) -> str:
    direct = {
        "potential_V": "Ecell_V",
        "current_mA": "I_mA",
        "time_s": "time/s",
        "minimum_frequency_Hz": "fmin_Hz",
        "maximum_frequency_Hz": "fmax_Hz",
        "active_minimum_frequency_Hz": "fmin_act_Hz",
        "active_maximum_frequency_Hz": "fmax_act_Hz",
        "Cell voltage (V)": "Ecell_V",
    }
    if name in direct:
        return direct[name]
    if name.endswith("_error_percent"):
        return f"{_external_parameter_name(name[:-14])}_e"
    match = re.fullmatch(r"p_R(\d+)_CPE\1_(tau_s|capacitance_F)", name)
    if match:
        return f"{('tau' if match.group(2) == 'tau_s' else 'C')}{match.group(1)}"
    return _external_parameter_name(name)


def _externalize_record(record: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in record.items():
        result[_external_column_name(name)] = value
    return result


def _migrate_metadata(metadata: dict[str, object]) -> dict[str, object]:
    migrated: dict[str, object] = {}
    for name, value in metadata.items():
        external_name = _external_column_name(str(name))
        if external_name not in migrated or str(name) == external_name:
            migrated[external_name] = value
    return migrated


def _parameter_to_dict(parameter: ParameterValue) -> dict[str, object]:
    return {
        "name": parameter.name,
        "unit": parameter.unit,
        "initial": parameter.initial,
        "lower": parameter.lower,
        "upper": parameter.upper,
        "error_percent": parameter.error_percent,
        "fixed": parameter.fixed,
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
        fixed=bool(data.get("fixed", False)),
    )


def _optional_array(values) -> np.ndarray | None:
    if values is None:
        return None
    return as_1d_array(values)


def _parallel_rcpe_block_ids(circuit: str) -> list[str]:
    block_ids = re.findall(r"p\(\s*R(\d+)\s*,\s*CPE\1\s*\)", circuit)
    return list(dict.fromkeys(block_ids))


def _derived_block_columns(circuit: str, parameter_names: list[str]) -> list[str]:
    columns: list[str] = []
    for block_id in _parallel_rcpe_block_ids(circuit):
        if f"R{block_id}" not in parameter_names:
            continue
        if f"CPE{block_id}_0" not in parameter_names:
            continue
        if f"CPE{block_id}_1" not in parameter_names:
            continue
        columns.extend(
            (
                f"p_R{block_id}_CPE{block_id}_capacitance_F",
                f"p_R{block_id}_CPE{block_id}_tau_s",
            )
        )
    return columns


def _derived_block_values(
    circuit: str,
    parameter_names: list[str],
    parameter_values: np.ndarray,
) -> dict[str, float]:
    values_by_name = dict(zip(parameter_names, parameter_values.tolist()))
    derived: dict[str, float] = {}
    for block_id in _parallel_rcpe_block_ids(circuit):
        r_name = f"R{block_id}"
        q_name = f"CPE{block_id}_0"
        alpha_name = f"CPE{block_id}_1"
        if (
            r_name not in values_by_name
            or q_name not in values_by_name
            or alpha_name not in values_by_name
        ):
            continue
        resistance = float(values_by_name[r_name])
        q_value = float(values_by_name[q_name])
        alpha_value = float(values_by_name[alpha_name])
        derived[f"p_R{block_id}_CPE{block_id}_capacitance_F"] = float(
            cpe_capacitance(resistance, q_value, alpha_value)
        )
        derived[f"p_R{block_id}_CPE{block_id}_tau_s"] = float(
            cpe_tau(resistance, q_value, alpha_value)
        )
    return derived


def _custom_metadata_columns(cycles: list[CycleState]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    reserved = {
        "Ecell_V",
        "I_mA",
        "time_s",
        "time/s",
        "fmin_Hz",
        "fmax_Hz",
        "fmin_act_Hz",
        "fmax_act_Hz",
    }
    for cycle in cycles:
        for name in cycle.custom_metadata:
            if name in reserved:
                continue
            if name not in seen:
                seen.add(name)
                columns.append(name)
    return columns


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
        "circuit": cycle.circuit,
        "potential_v": cycle.potential_v,
        "current_ma": cycle.current_ma,
        "time_s": cycle.time_s,
        "frequency_window": (
            list(cycle.frequency_window) if cycle.frequency_window is not None else None
        ),
        "auto_max_frequency": bool(cycle.auto_max_frequency),
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
        "fit_provenance": dict(cycle.fit_provenance),
        "ridge_tau_s": (
            cycle.ridge_tau_s.tolist() if cycle.ridge_tau_s is not None else None
        ),
        "ridge_gamma_ohm": (
            cycle.ridge_gamma_ohm.tolist()
            if cycle.ridge_gamma_ohm is not None
            else None
        ),
        "drt_label": cycle.drt_label,
        "saved_ridge_tau_s": (
            cycle.saved_ridge_tau_s.tolist()
            if cycle.saved_ridge_tau_s is not None
            else None
        ),
        "saved_ridge_gamma_ohm": (
            cycle.saved_ridge_gamma_ohm.tolist()
            if cycle.saved_ridge_gamma_ohm is not None
            else None
        ),
        "saved_ridge_included_mask": (
            cycle.saved_ridge_included_mask.astype(bool).tolist()
            if cycle.saved_ridge_included_mask is not None
            else None
        ),
        "saved_ridge_outlier_indices": (
            cycle.saved_ridge_outlier_indices.tolist()
            if cycle.saved_ridge_outlier_indices is not None
            else None
        ),
        "saved_ridge_parameters": [
            _parameter_to_dict(value) for value in cycle.saved_ridge_parameters
        ],
        "saved_ridge_threshold": cycle.saved_ridge_threshold,
        "saved_ridge_peak_count": cycle.saved_ridge_peak_count,
        "saved_ridge_ohmic_resistance": cycle.saved_ridge_ohmic_resistance,
        "saved_ridge_inductance": cycle.saved_ridge_inductance,
        "saved_ridge_peak_parameters": [
            dict(peak) for peak in cycle.saved_ridge_peak_parameters
        ],
        "saved_hybrid_tau_s": (
            cycle.saved_hybrid_tau_s.tolist()
            if cycle.saved_hybrid_tau_s is not None
            else None
        ),
        "saved_hybrid_gamma_ohm": (
            cycle.saved_hybrid_gamma_ohm.tolist()
            if cycle.saved_hybrid_gamma_ohm is not None
            else None
        ),
        "saved_hybrid_included_mask": (
            cycle.saved_hybrid_included_mask.astype(bool).tolist()
            if cycle.saved_hybrid_included_mask is not None
            else None
        ),
        "saved_hybrid_ohmic_resistance": cycle.saved_hybrid_ohmic_resistance,
        "saved_hybrid_inductance": cycle.saved_hybrid_inductance,
        "saved_hybrid_peak_parameters": [
            dict(peak) for peak in cycle.saved_hybrid_peak_parameters
        ],
        "custom_metadata": _migrate_metadata(dict(cycle.custom_metadata)),
    }


def _state_to_payload(state: ProjectState) -> dict[str, object]:
    return {
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


def _dataframe_to_payload(dataframe) -> str:
    return dataframe.to_json(orient="split", date_format="iso")


def dataframe_from_payload(payload: str):
    import pandas as pd

    return pd.read_json(StringIO(payload), orient="split")


def save_project_file(
    state: ProjectState,
    path: Path,
    datasets: list[tuple[str, ProjectState, object]] | None = None,
    procedure_blocks: dict[str, list[dict[str, str]]] | None = None,
    procedures: dict[str, list[dict[str, object]]] | None = None,
) -> None:
    payload = _state_to_payload(state)
    if procedure_blocks is not None:
        payload["procedure_blocks"] = procedure_blocks
    if procedures is not None:
        payload["procedures"] = procedures
    if datasets:
        payload["datasets"] = [
            {
                "dataset_id": dataset_id,
                "state": _state_to_payload(dataset_state),
                "dataframe": _dataframe_to_payload(dataframe),
            }
            for dataset_id, dataset_state, dataframe in datasets
        ]
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(_json_bytes(payload, compressed=_is_gzip_path(path)))
    temporary.replace(path)


def load_project_file(
    current: ProjectState,
    dataframe,
    path: Path,
    payload: dict[str, object] | None = None,
) -> ProjectState:
    from eis_services import load_cycle

    if payload is None:
        payload = load_json_payload(path)
    if payload.get("format") != PROJECT_FORMAT:
        raise ValueError("This is not an EIS fitting project file")
    version = int(payload.get("version", 0))
    if version not in {1, 2, 3, PROJECT_VERSION}:
        raise ValueError(f"Unsupported project version: {payload.get('version')}")

    circuit = str(payload["circuit"])
    control = str(payload.get("control", current.control))
    defaults = [_parameter_from_dict(value) for value in payload["default_parameters"]]
    restored_cycles: dict[int, CycleState] = {}
    available = set(current.available_cycles)
    for cycle_text, saved in payload.get("cycles", {}).items():
        cycle_number = int(cycle_text)
        if cycle_number not in available:
            continue
        cycle = load_cycle(dataframe, cycle_number, control)
        cycle.time_s = (
            float(saved["time_s"])
            if saved.get("time_s") is not None
            else None
        )
        cycle.circuit = str(saved.get("circuit") or circuit)
        if saved.get("potential_v") is not None:
            cycle.potential_v = float(saved["potential_v"])
        if saved.get("current_ma") is not None:
            cycle.current_ma = float(saved["current_ma"])
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
        cycle.auto_max_frequency = bool(saved.get("auto_max_frequency", False))
        cycle.parameters = [
            _parameter_from_dict(value) for value in saved.get("parameters", [])
        ] or [
            ParameterValue(
                p.name,
                p.unit,
                p.initial,
                p.lower,
                p.upper,
                p.error_percent,
                p.fixed,
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
        saved_provenance = saved.get("fit_provenance", {})
        cycle.fit_provenance = dict(saved_provenance) if isinstance(saved_provenance, dict) else {}
        cycle.ridge_tau_s = _optional_array(saved.get("ridge_tau_s"))
        cycle.ridge_gamma_ohm = _optional_array(saved.get("ridge_gamma_ohm"))
        if (
            cycle.ridge_tau_s is not None
            and cycle.ridge_gamma_ohm is not None
            and cycle.ridge_tau_s.size != cycle.ridge_gamma_ohm.size
        ):
            raise ValueError(f"Cycle {cycle_number} has invalid saved ridge DRT data")
        cycle.drt_label = (
            str(saved["drt_label"]) if saved.get("drt_label") is not None else None
        )
        cycle.saved_ridge_tau_s = _optional_array(saved.get("saved_ridge_tau_s"))
        cycle.saved_ridge_gamma_ohm = _optional_array(
            saved.get("saved_ridge_gamma_ohm")
        )
        cycle.saved_ridge_included_mask = _optional_array(
            saved.get("saved_ridge_included_mask")
        )
        cycle.saved_ridge_outlier_indices = _optional_array(
            saved.get("saved_ridge_outlier_indices")
        )
        cycle.saved_ridge_parameters = [
            _parameter_from_dict(value)
            for value in saved.get("saved_ridge_parameters", [])
        ]
        cycle.saved_ridge_threshold = (
            float(saved["saved_ridge_threshold"])
            if saved.get("saved_ridge_threshold") is not None
            else None
        )
        cycle.saved_ridge_peak_count = (
            int(saved["saved_ridge_peak_count"])
            if saved.get("saved_ridge_peak_count") is not None
            else None
        )
        cycle.saved_ridge_ohmic_resistance = (
            float(saved["saved_ridge_ohmic_resistance"])
            if saved.get("saved_ridge_ohmic_resistance") is not None
            else None
        )
        cycle.saved_ridge_inductance = (
            float(saved["saved_ridge_inductance"])
            if saved.get("saved_ridge_inductance") is not None
            else None
        )
        cycle.saved_ridge_peak_parameters = [
            {
                key: str(value) if key == "shape" else float(value)
                for key, value in peak.items()
            }
            for peak in saved.get("saved_ridge_peak_parameters", [])
        ]
        cycle.saved_hybrid_tau_s = _optional_array(saved.get("saved_hybrid_tau_s"))
        cycle.saved_hybrid_gamma_ohm = _optional_array(
            saved.get("saved_hybrid_gamma_ohm")
        )
        cycle.saved_hybrid_included_mask = _optional_array(
            saved.get("saved_hybrid_included_mask")
        )
        cycle.saved_hybrid_ohmic_resistance = (
            float(saved["saved_hybrid_ohmic_resistance"])
            if saved.get("saved_hybrid_ohmic_resistance") is not None
            else None
        )
        cycle.saved_hybrid_inductance = (
            float(saved["saved_hybrid_inductance"])
            if saved.get("saved_hybrid_inductance") is not None
            else None
        )
        cycle.saved_hybrid_peak_parameters = [
            {
                key: str(value) if key == "shape" else float(value)
                for key, value in peak.items()
            }
            for peak in saved.get("saved_hybrid_peak_parameters", [])
        ]
        if cycle.drt_label == "Hybrid DRT" and cycle.saved_hybrid_tau_s is not None:
            cycle.show_hybrid_drt()
        elif cycle.drt_label == "Ridge DRT" and cycle.saved_ridge_tau_s is not None:
            cycle.show_ridge_drt()
        cycle.custom_metadata = _migrate_metadata(
            dict(saved.get("custom_metadata", {}))
        )
        restored_cycles[cycle_number] = cycle

    requested_active = int(payload.get("active_cycle", current.active_cycle))
    active_cycle = (
        requested_active
        if requested_active in current.available_cycles
        else current.active_cycle
    )
    if active_cycle not in restored_cycles:
        active = load_cycle(dataframe, active_cycle, control)
        active.time_s = None
        active.circuit = circuit
        active.parameters = [
            ParameterValue(
                p.name,
                p.unit,
                p.initial,
                p.lower,
                p.upper,
                p.error_percent,
                p.fixed,
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
    return export_fit_parameters_for_states([state], path)


def export_fit_parameters_for_states(
    states: list[ProjectState],
    path: Path,
) -> int:
    fitted = [
        (state, cycle)
        for state in states
        for _, cycle in sorted(state.cycles.items())
        if cycle.fit_parameters is not None
    ]
    if not fitted:
        raise ValueError("No spectra have fitted parameters to export")
    parameter_names = list(
        dict.fromkeys(
            parameter.name for state, cycle in fitted for parameter in cycle.parameters
        )
    )
    custom_metadata_columns = list(
        dict.fromkeys(
            name for _state, cycle in fitted for name in cycle.custom_metadata.keys()
        )
    )
    parameter_columns = []
    for name in parameter_names:
        parameter_columns.extend((name, f"{name}_error_percent"))
    derived_columns = list(
        dict.fromkeys(
            column
            for state, cycle in fitted
            for column in _derived_block_columns(
                cycle.model(state.circuit), [parameter.name for parameter in cycle.parameters]
            )
        )
    )
    internal_fieldnames = [
        "source_file",
        "source_path",
        "cycle",
        "circuit",
        "potential_V",
        "current_mA",
        "time_s",
        "total_points",
        "active_points",
        "minimum_frequency_Hz",
        "maximum_frequency_Hz",
        "active_minimum_frequency_Hz",
        "active_maximum_frequency_Hz",
        *custom_metadata_columns,
        *parameter_columns,
        *derived_columns,
    ]
    fieldnames = [_external_column_name(name) for name in internal_fieldnames]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for state, cycle in fitted:
            values = as_1d_array(cycle.fit_parameters)
            cycle_names = [parameter.name for parameter in cycle.parameters]
            if values.size != len(cycle_names):
                raise ValueError(
                    f"Cycle {cycle.cycle} in {state.source_path.name} has "
                    "incompatible fit parameters"
                )
            row = {
                "source_file": state.source_path.name,
                "source_path": str(state.source_path),
                "cycle": cycle.cycle,
                "circuit": cycle.model(state.circuit),
                "potential_V": cycle.potential_v,
                "current_mA": cycle.current_ma,
                "time_s": cycle.time_s,
                "total_points": int(cycle.frequency_hz.size),
                "active_points": int(np.count_nonzero(cycle.included)),
                "minimum_frequency_Hz": float(np.min(cycle.frequency_hz)),
                "maximum_frequency_Hz": float(np.max(cycle.frequency_hz)),
                "active_minimum_frequency_Hz": (
                    float(np.min(cycle.frequency_hz[cycle.included]))
                    if np.any(cycle.included)
                    else None
                ),
                "active_maximum_frequency_Hz": (
                    float(np.max(cycle.frequency_hz[cycle.included]))
                    if np.any(cycle.included)
                    else None
                ),
            }
            row.update({name: cycle.custom_metadata.get(name) for name in custom_metadata_columns})
            row.update(dict(zip(cycle_names, values.tolist())))
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
            row.update(_derived_block_values(cycle.model(state.circuit), cycle_names, values))
            writer.writerow(_externalize_record(row))
    return len(fitted)


def export_python_workspace(
    states: list[ProjectState],
    path: Path,
) -> tuple[int, Path]:
    fitted = [
        (state, cycle)
        for state in states
        for _, cycle in sorted(state.cycles.items())
        if cycle.fit_parameters is not None
    ]
    if not fitted:
        raise ValueError("No spectra have fitted parameters to export")

    parameter_names = list(
        dict.fromkeys(
            parameter.name for _state, cycle in fitted for parameter in cycle.parameters
        )
    )
    custom_metadata_columns = list(
        dict.fromkeys(
            name for _state, cycle in fitted for name in cycle.custom_metadata.keys()
        )
    )
    derived_columns = list(
        dict.fromkeys(
            column
            for state, cycle in fitted
            for column in _derived_block_columns(
                cycle.model(state.circuit), [parameter.name for parameter in cycle.parameters]
            )
        )
    )
    metadata_columns = [
        "source_file",
        "source_path",
        "cycle",
        "circuit",
        "potential_V",
        "current_mA",
        "time_s",
        "total_points",
        "active_points",
        "minimum_frequency_Hz",
        "maximum_frequency_Hz",
        "active_minimum_frequency_Hz",
        "active_maximum_frequency_Hz",
        *custom_metadata_columns,
    ]
    parameter_columns = [
        column for name in parameter_names for column in (name, f"{name}_error_percent")
    ]
    external_metadata_columns = [
        _external_column_name(column) for column in metadata_columns
    ]
    external_parameter_columns = [
        _external_column_name(column) for column in parameter_columns
    ]
    external_derived_columns = [
        _external_column_name(column) for column in derived_columns
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                *external_metadata_columns,
                *external_parameter_columns,
                *external_derived_columns,
            ],
        )
        writer.writeheader()
        for state, cycle in fitted:
            values = as_1d_array(cycle.fit_parameters)
            cycle_names = [parameter.name for parameter in cycle.parameters]
            if values.size != len(cycle_names):
                raise ValueError(
                    f"Cycle {cycle.cycle} in {state.source_path.name} has "
                    "incompatible fit parameters"
                )
            active = cycle.included
            active_frequency = cycle.frequency_hz[active]
            row = {
                "source_file": state.source_path.name,
                "source_path": str(state.source_path),
                "cycle": cycle.cycle,
                "circuit": cycle.model(state.circuit),
                "potential_V": cycle.potential_v,
                "current_mA": cycle.current_ma,
                "time_s": cycle.time_s,
                "total_points": int(cycle.frequency_hz.size),
                "active_points": int(np.count_nonzero(active)),
                "minimum_frequency_Hz": float(np.min(cycle.frequency_hz)),
                "maximum_frequency_Hz": float(np.max(cycle.frequency_hz)),
                "active_minimum_frequency_Hz": (
                    float(np.min(active_frequency)) if active_frequency.size else None
                ),
                "active_maximum_frequency_Hz": (
                    float(np.max(active_frequency)) if active_frequency.size else None
                ),
            }
            row.update({name: cycle.custom_metadata.get(name) for name in custom_metadata_columns})
            row.update(dict(zip(cycle_names, values.tolist())))
            row.update(
                {
                    f"{parameter.name}_error_percent": parameter.error_percent
                    for parameter in cycle.parameters
                }
            )
            row.update(
                _derived_block_values(
                    cycle.model(state.circuit),
                    cycle_names,
                    values,
                )
            )
            writer.writerow(_externalize_record(row))

    script_path = path.with_suffix(".py")
    script = f"""
# %%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
DATA_FILE = Path(__file__).with_name({path.name!r})

# Complete export: one spectrum per row.
fit_data = pd.read_csv(DATA_FILE)

# Metadata describing each fitted spectrum.
metadata_columns = {external_metadata_columns!r}
metadata = fit_data.loc[:, metadata_columns].copy()

# Custom metadata imported from the clipboard.
custom_metadata_columns = {custom_metadata_columns!r}
custom_metadata = (
    fit_data.loc[:, custom_metadata_columns].copy()
    if custom_metadata_columns
    else pd.DataFrame(index=fit_data.index)
)

# Fitted values and their percentage errors.
parameter_columns = {external_parameter_columns!r}
fit_parameters = fit_data.loc[:, parameter_columns].copy()

# Convenient spectrum-indexed tables for analysis and plotting.
index_columns = ["source_path", "cycle"]
if "Spectrum" in fit_data.columns:
    index_columns.append("Spectrum")
indexed_fit_data = fit_data.set_index(index_columns).sort_index()
parameter_values = indexed_fit_data.loc[:, [
    column for column in parameter_columns
    if not column.endswith("_e")
]]
parameter_errors_percent = indexed_fit_data.loc[:, [
    column for column in parameter_columns
    if column.endswith("_e")
]]
derived_columns = {external_derived_columns!r}
derived_values = indexed_fit_data.loc[:, derived_columns].copy()

print(f"Loaded {{len(fit_data)}} fitted spectra from {{DATA_FILE.name}}")
print(fit_data.head())
# %%

"""
    script_path.write_text(script, encoding="utf-8")
    return len(fitted), script_path


def export_drts_for_states(
    states: list[ProjectState],
    path: Path,
) -> int:
    internal_fieldnames = [
        "source_file",
        "source_path",
        "cycle",
        "circuit",
        "potential_V",
        "current_mA",
        "time_s",
        "drt_method",
        "point_index",
        "tau_s",
        "gamma_ohm",
        "ohmic_resistance_ohm",
        "ridge_threshold",
    ]
    fieldnames = [_external_column_name(name) for name in internal_fieldnames]
    rows: list[dict[str, object]] = []
    for state in states:
        for cycle_number, cycle in sorted(state.cycles.items()):
            for method, tau_values, gamma_values, ohmic_resistance, threshold in (
                (
                    "ridge",
                    cycle.saved_ridge_tau_s,
                    cycle.saved_ridge_gamma_ohm,
                    cycle.saved_ridge_ohmic_resistance,
                    cycle.saved_ridge_threshold,
                ),
                (
                    "hybrid",
                    cycle.saved_hybrid_tau_s,
                    cycle.saved_hybrid_gamma_ohm,
                    cycle.saved_hybrid_ohmic_resistance,
                    None,
                ),
            ):
                if tau_values is None or gamma_values is None:
                    continue
                tau_values = as_1d_array(tau_values).astype(float)
                gamma_values = as_1d_array(gamma_values).astype(float)
                if tau_values.size != gamma_values.size:
                    raise ValueError(
                        f"Cycle {cycle_number} in {state.source_path.name} has "
                        f"an invalid saved {method} DRT"
                    )
                for point_index, (tau_s, gamma_ohm) in enumerate(
                    zip(tau_values, gamma_values),
                    start=1,
                ):
                    rows.append(
                        {
                            "source_file": state.source_path.name,
                            "source_path": str(state.source_path),
                            "cycle": cycle_number,
                            "circuit": cycle.model(state.circuit),
                            "potential_V": cycle.potential_v,
                            "current_mA": cycle.current_ma,
                            "time_s": cycle.time_s,
                            "drt_method": method,
                            "point_index": point_index,
                            "tau_s": float(tau_s),
                            "gamma_ohm": float(gamma_ohm),
                            "ohmic_resistance_ohm": ohmic_resistance,
                            "ridge_threshold": threshold,
                        }
                    )
    if not rows:
        raise ValueError("No saved DRTs are available to export")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_externalize_record(row) for row in rows)
    return len(rows)
