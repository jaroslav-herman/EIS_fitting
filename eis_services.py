from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
from pathlib import Path
import re
import threading
import time
from typing import Iterable
import warnings

import numpy as np

from eis_model import (
    CycleState,
    ParameterValue,
    ProjectState,
    as_1d_array,
    copy_parameter_values,
    sort_spectrum,
)
from circuit_structure import circuits_equivalent, map_parameter_name, parameter_name_mapping

SPECTRUM_KIND_COLUMN_MAP = {
    "working": ("re_z_ohm", "minus_im_z_ohm", "ewe_v"),
    "cell": ("re_zwe_ce_ohm", "minus_im_zwe_ce_ohm", "ewe_ece_v"),
    "counter": ("re_zce_ohm", "minus_im_zce_ohm", "ece_v"),
    "ewe": ("re_z_ohm", "minus_im_z_ohm", "ewe_v"),
    "ece": ("re_zwe_ce_ohm", "minus_im_zwe_ce_ohm", "ewe_ece_v"),
}
SPECTRUM_KIND_LABELS = {
    "working": "WE",
    "cell": "Cell",
    "counter": "CE",
}
SPECTRUM_METADATA_COLUMN = "Spectrum"
WORKING_POTENTIAL_COLUMN = "Working electrode potential (V)"
COUNTER_POTENTIAL_COLUMN = "Counter electrode potential (V)"
CELL_POTENTIAL_COLUMN = "Ecell_V"


@dataclass(frozen=True)
class SpectrumMetadata:
    cycle: int
    potential_v: float
    current_ma: float
    time_s: float | None
    point_count: int
    minimum_frequency_hz: float
    maximum_frequency_hz: float
    custom_metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class LoadedProject:
    dataframe: object
    state: ProjectState
    technique: str
    spectra: list[SpectrumMetadata]
    dataset_id: str
    dataset_label: str
    skipped_cycles: tuple[int, ...] = ()


@dataclass
class ProjectImportReport:
    loaded: list[tuple[str, LoadedProject]]
    errors: list[tuple[Path, str]]


@dataclass
class RidgeInitialization:
    outlier_indices: np.ndarray
    parameters: list[ParameterValue]
    peak_count: int
    ohmic_resistance: float
    inductance: float
    ridge_tau_s: np.ndarray
    ridge_gamma_ohm: np.ndarray


@dataclass
class DRTComputation:
    tau_s: np.ndarray
    gamma_ohm: np.ndarray
    ohmic_resistance: float | None = None


@dataclass
class AutomaticEECModel:
    circuit: str
    initials: dict[str, float]
    peak_count: int
    criterion: str
    ohmic_resistance: float
    inductance: float


@dataclass
class KKResiduals:
    fit_impedance: np.ndarray
    residual_real: np.ndarray
    residual_imag: np.ndarray


@dataclass(frozen=True)
class FitOptions:
    """Explicit, serializable controls for equivalent-circuit fitting."""

    method: str = "least_squares"
    pipeline: tuple[str, ...] = ()
    seed: int | None = None
    population_size: int = 30
    iterations: int = 200
    weight_by_modulus: bool = False
    use_analytical_jacobian: bool = False
    runtime_checks: bool = True

    def stages(self) -> tuple[str, ...]:
        stages = tuple(str(value).strip().casefold() for value in self.pipeline if str(value).strip())
        if stages:
            return stages
        return (str(self.method).strip().casefold() or "least_squares",)

    def validated(self) -> "FitOptions":
        stages = self.stages()
        allowed = {"least_squares", "basinhopping", "pso", "ga"}
        if any(stage not in allowed for stage in stages):
            raise ValueError("Unknown EEC optimizer; use least_squares, basinhopping, pso, or ga")
        if self.seed is not None:
            int(self.seed)
        if int(self.population_size) < 4 or int(self.iterations) < 1:
            raise ValueError("Optimizer population and iteration limits must be positive")
        return self


@dataclass
class FitResult:
    fitted_parameters: np.ndarray
    errors_percent: np.ndarray
    fit_frequency_hz: np.ndarray
    fit_impedance: np.ndarray
    fit_at_data_impedance: np.ndarray
    objective: float
    rmse: float
    converged: bool
    stages: list[dict[str, object]] = field(default_factory=list)
    options: FitOptions = field(default_factory=FitOptions)
    elapsed_seconds: float = 0.0

    def __iter__(self):
        """Backward-compatible unpacking for existing batch/refinement callers."""
        yield self.fitted_parameters
        yield self.errors_percent
        yield self.fit_frequency_hz
        yield self.fit_impedance
        yield self.fit_at_data_impedance


def _fit_provenance(result: FitResult) -> dict[str, object]:
    if not isinstance(result, FitResult):
        return {}
    return {
        "pipeline": list(result.options.stages()),
        "seed": result.options.seed,
        "objective": result.objective,
        "rmse": result.rmse,
        "converged": result.converged,
        "elapsed_seconds": result.elapsed_seconds,
        "stages": result.stages,
    }


@dataclass
class BatchCycleFit:
    cycle: CycleState
    parameters: list[ParameterValue]
    fitted_parameters: np.ndarray
    fitted_errors_percent: np.ndarray
    fit_frequency_hz: np.ndarray
    fit_impedance: np.ndarray
    fit_at_data_impedance: np.ndarray
    fit_provenance: dict[str, object] = field(default_factory=dict)


@dataclass
class BatchFitReport:
    fits: list[BatchCycleFit]
    failed_cycle: int | None = None
    error: str | None = None
    stopped: bool = False
    skipped_cycles: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class SpectrumFitTarget:
    loaded: LoadedProject
    cycle: int
    label: str


@dataclass
class SpectrumBatchFit:
    loaded: LoadedProject
    fit: BatchCycleFit


@dataclass
class SpectrumBatchReport:
    fits: list[SpectrumBatchFit]
    failed_label: str | None = None
    error: str | None = None
    stopped: bool = False
    skipped_labels: list[str] = field(default_factory=list)


class FitTimeoutError(TimeoutError):
    """Raised when an impedance EEC fit exceeds its configured time limit."""


_FIT_WORKER_PROCESS = None
_FIT_WORKER_CONNECTION = None
_FIT_WORKER_LOCK = threading.Lock()


def _fit_cycle_process_entry(connection) -> None:
    try:
        while True:
            task = connection.recv()
            if task is None:
                return
            state, circuit, parameters, options = task
            try:
                connection.send((True, fit_cycle(state, circuit, parameters, options)))
            except BaseException as error:
                connection.send((False, f"{type(error).__name__}: {error}"))
    except (EOFError, OSError):
        return
    finally:
        connection.close()


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


def _cycles_with_impedance(
    dataframe,
    cycles: list[int],
    control: str,
) -> tuple[list[int], list[int]]:
    spectrum_kind = _normalize_spectrum_kind(control)
    real_column, imaginary_column, _potential_column = SPECTRUM_KIND_COLUMN_MAP[
        spectrum_kind
    ]
    valid_cycles: list[int] = []
    skipped_cycles: list[int] = []
    for cycle in cycles:
        rows = dataframe["cycle_number"] == cycle if "cycle_number" in dataframe.columns else np.ones(len(dataframe), dtype=bool)
        try:
            frequency = np.asarray(dataframe.loc[rows, "freq_hz"], dtype=float)
            real = np.asarray(dataframe.loc[rows, real_column], dtype=float)
            imaginary = np.asarray(dataframe.loc[rows, imaginary_column], dtype=float)
            valid = (
                np.isfinite(frequency)
                & (frequency != 0)
                & np.isfinite(real)
                & np.isfinite(imaginary)
            )
        except (KeyError, TypeError, ValueError):
            valid = np.empty(0, dtype=bool)
        if np.any(valid):
            valid_cycles.append(cycle)
        else:
            skipped_cycles.append(cycle)
    return valid_cycles, skipped_cycles


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
        return 0.5, 1.0
    if normalized.startswith("r"):
        return 0.0, 1e6
    if normalized.startswith("l"):
        return 0.0, 1.0
    if normalized.startswith("cpe") and normalized.endswith("_0"):
        return 1e-6, 1e3
    if normalized.startswith("c"):
        return 1e-12, 1e3
    return 0.0, 1e6


def _normalize_spectrum_kind(control: str) -> str:
    normalized = control.strip().lower()
    if normalized not in SPECTRUM_KIND_COLUMN_MAP:
        raise ValueError(f"Unsupported spectrum kind '{control}'")
    if normalized == "ewe":
        return "working"
    if normalized == "ece":
        return "cell"
    return normalized


def _available_spectrum_kinds(dataframe, header_meta: dict[str, str]) -> list[str]:
    potential_control = header_meta.get("Potential control", "").strip()
    if potential_control == "Ewe-Ece":
        roles = []
        for kind in ("cell", "counter", "working"):
            real_column, imaginary_column, _potential_column = SPECTRUM_KIND_COLUMN_MAP[kind]
            if real_column in dataframe.columns and imaginary_column in dataframe.columns:
                roles.append(kind)
        if roles:
            return roles
    return ["working"]


def _order_spectrum_kinds(
    kinds: list[str],
    requested_control: str,
) -> list[str]:
    preferred = _normalize_spectrum_kind(requested_control)
    ordered = [kind for kind in kinds if kind == preferred]
    ordered.extend(kind for kind in kinds if kind != preferred)
    return ordered


def _mean_if_present(dataframe, rows, column: str) -> float | None:
    if column not in dataframe.columns:
        return None
    values = dataframe.loc[rows, column].to_numpy()
    if values.size == 0:
        return None
    return float(np.nanmean(values))


def _cycle_custom_metadata(dataframe, cycle: int, spectrum_kind: str) -> dict[str, object]:
    rows = dataframe["freq_hz"] != 0
    if "cycle_number" in dataframe.columns:
        rows &= dataframe["cycle_number"] == cycle
    metadata = {
        SPECTRUM_METADATA_COLUMN: SPECTRUM_KIND_LABELS.get(
            spectrum_kind, spectrum_kind.title()
        ),
    }
    if spectrum_kind == "working" and "ewe_ece_v" not in dataframe.columns:
        metadata[CELL_POTENTIAL_COLUMN] = _mean_if_present(dataframe, rows, "ewe_v")
        return metadata
    metadata.update(
        {
            WORKING_POTENTIAL_COLUMN: _mean_if_present(dataframe, rows, "ewe_v"),
            COUNTER_POTENTIAL_COLUMN: _mean_if_present(dataframe, rows, "ece_v"),
            CELL_POTENTIAL_COLUMN: _mean_if_present(dataframe, rows, "ewe_ece_v"),
        }
    )
    return metadata


def _dataset_id(path: Path, spectrum_kind: str) -> str:
    return f"{path.resolve()}::{spectrum_kind}"


def circuit_parameters(
    circuit: str,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> list[ParameterValue]:
    from impedance.models.circuits import CustomCircuit
    from impedance.models.circuits.circuits import calculateCircuitLength

    length = int(calculateCircuitLength(circuit))
    model = CustomCircuit(circuit, initial_guess=[1.0] * length)
    names, units = model.get_param_names()
    parameters = []
    for name, unit in zip(names, units):
        lower, upper = infer_bounds(name)
        if bounds:
            normalized = name.lower()
            category = (
                "cpe_q"
                if normalized.startswith("cpe") and normalized.endswith("_0")
                else "cpe_alpha"
                if normalized.startswith("cpe") and normalized.endswith("_1")
                else "r"
                if normalized.startswith("r")
                else "l"
                if normalized.startswith("l")
                else None
            )
            if category in bounds:
                lower, upper = bounds[category]
        parameters.append(
            ParameterValue(
                name,
                unit,
                float(np.clip(infer_initial(name), lower, upper)),
                lower,
                upper,
            )
        )
    return parameters


def load_cycle(dataframe, cycle: int, control: str) -> CycleState:
    if "freq_hz" not in dataframe.columns:
        raise KeyError("The file has no 'freq_hz' column and is not a PEIS spectrum")
    rows = dataframe["freq_hz"] != 0
    if "cycle_number" in dataframe.columns:
        rows &= dataframe["cycle_number"] == cycle

    spectrum_kind = _normalize_spectrum_kind(control)
    real_column, imaginary_column, potential_column = SPECTRUM_KIND_COLUMN_MAP[
        spectrum_kind
    ]
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
    time_s = _mean_if_present(dataframe, rows, "time_s")
    return CycleState(
        cycle,
        frequency,
        impedance,
        potential,
        current,
        time_s,
        custom_metadata=_cycle_custom_metadata(dataframe, cycle, spectrum_kind),
    )


def catalog_spectra(
    dataframe,
    cycles: list[int],
    control: str,
    cycle_metadata: dict[int, dict[str, object]] | None = None,
) -> list[SpectrumMetadata]:
    spectra = []
    cycle_metadata = cycle_metadata or {}
    spectrum_kind = _normalize_spectrum_kind(control)
    for cycle_number in cycles:
        cycle = load_cycle(dataframe, cycle_number, control)
        metadata = _cycle_custom_metadata(dataframe, cycle_number, spectrum_kind)
        overrides = dict(cycle_metadata.get(cycle_number, {}))
        if "time_s" in overrides:
            cycle.time_s = overrides.pop("time_s")
        metadata.update(overrides)
        spectra.append(
            SpectrumMetadata(
                cycle=cycle_number,
                potential_v=cycle.potential_v,
                current_ma=cycle.current_ma,
                time_s=cycle.time_s,
                point_count=int(cycle.frequency_hz.size),
                minimum_frequency_hz=float(np.nanmin(cycle.frequency_hz)),
                maximum_frequency_hz=float(np.nanmax(cycle.frequency_hz)),
                custom_metadata=metadata,
            )
        )
    return spectra


def load_project(
    path: Path,
    cycle: int,
    control: str,
    circuit: str,
) -> LoadedProject:
    from wepy import read_mpt_dataframe

    projects = load_projects_for_file(path, cycle, control, circuit)
    if not projects:
        raise ValueError(f"No impedance spectra could be loaded from {path.name}")
    return projects[0]


def _read_eis_dataframe(path: Path):
    if path.suffix.casefold() == ".mpr":
        import pandas as pd
        from galvani.BioLogic import MPRfile, MPR_MAGIC

        file_size = path.stat().st_size
        prefix = path.read_bytes()[: len(MPR_MAGIC)]
        if prefix != MPR_MAGIC:
            raise ValueError(
                f"File is not a recognized BioLogic .mpr file: {path.name} "
                f"(size={file_size} bytes, signature={prefix[:16].hex()})"
            )
        try:
            mpr = MPRfile(str(path), error_on_unknown_column=False)
        except Exception as error:
            raise ValueError(
                f"Could not parse the BioLogic .mpr binary header in {path.name} "
                f"(size={file_size} bytes, signature={prefix[:16].hex()}); "
                f"the file may use an unsupported MPR format or be incomplete"
            ) from error
        dataframe = pd.DataFrame(mpr.data)
        rename = {
            "freq/Hz": "freq_hz",
            "cycle number": "cycle_number",
            "z cycle": "z_cycle",
            "time/s": "time_s",
            "<time>/s": "time_s",
            "I/mA": "i_ma",
            "<I>/mA": "i_ma",
            "Ewe/V": "ewe_v",
            "<Ewe>/V": "ewe_v",
            "Ece/V": "ece_v",
            "<Ece>/V": "ece_v",
            "Ewe-Ece/V": "ewe_ece_v",
            "Re(Z)/Ohm": "re_z_ohm",
            "-Im(Z)/Ohm": "minus_im_z_ohm",
            "Re(Zce)/Ohm": "re_zce_ohm",
            "-Im(Zce)/Ohm": "minus_im_zce_ohm",
            "Re(Zwe-ce)/Ohm": "re_zwe_ce_ohm",
            "-Im(Zwe-ce)/Ohm": "minus_im_zwe_ce_ohm",
        }
        dataframe = dataframe.rename(
            columns={name: value for name, value in rename.items() if name in dataframe}
        )
        if "z_cycle" in dataframe and "freq_hz" in dataframe:
            nonzero_frequency = np.isfinite(dataframe["freq_hz"]) & (
                dataframe["freq_hz"] != 0
            )
            z_cycles = _safe_unique_ints(dataframe.loc[nonzero_frequency, "z_cycle"])
            if len(z_cycles) > 1:
                dataframe["cycle_number"] = dataframe["z_cycle"]
        if {"ewe_v", "ece_v"}.issubset(dataframe.columns):
            dataframe["ewe_ece_v"] = dataframe["ewe_v"] - dataframe["ece_v"]
        three_electrode = "ece_v" in dataframe or {
            "re_zce_ohm",
            "minus_im_zce_ohm",
        }.issubset(dataframe.columns)
        if not three_electrode:
            if "re_zwe_ce_ohm" not in dataframe and "re_z_ohm" in dataframe:
                dataframe["re_zwe_ce_ohm"] = dataframe["re_z_ohm"]
            if (
                "minus_im_zwe_ce_ohm" not in dataframe
                and "minus_im_z_ohm" in dataframe
            ):
                dataframe["minus_im_zwe_ce_ohm"] = dataframe["minus_im_z_ohm"]
            dataframe = dataframe.drop(
                columns=["re_z_ohm", "minus_im_z_ohm"], errors="ignore"
            )
            if "ewe_ece_v" not in dataframe and "ewe_v" in dataframe:
                dataframe["ewe_ece_v"] = dataframe["ewe_v"]
        header_meta = {"Potential control": "Ewe-Ece"}
        return dataframe, header_meta, "PEIS"

    from wepy import read_mpt_dataframe

    return read_mpt_dataframe(path)


def load_projects_for_file(
    path: Path,
    cycle: int,
    control: str,
    circuit: str,
    spectrum_kinds: list[str] | None = None,
) -> list[LoadedProject]:
    dataframe, header_meta, technique = _read_eis_dataframe(path)
    cycles = (
        _safe_unique_ints(dataframe["cycle_number"].values)
        if "cycle_number" in dataframe.columns
        else [cycle]
    )
    if not cycles:
        raise ValueError("No cycles were found in the file")
    parameters = circuit_parameters(circuit)
    projects: list[LoadedProject] = []
    available_kinds = _available_spectrum_kinds(dataframe, header_meta)
    ordered_kinds = _order_spectrum_kinds(
        available_kinds,
        control,
    )
    if spectrum_kinds is not None:
        requested = set(spectrum_kinds)
        ordered_kinds = [kind for kind in ordered_kinds if kind in requested]
    for spectrum_kind in ordered_kinds:
        valid_cycles, skipped_cycles = _cycles_with_impedance(
            dataframe, cycles, spectrum_kind
        )
        if not valid_cycles:
            continue
        active_cycle = cycle if cycle in valid_cycles else valid_cycles[0]
        active = load_cycle(dataframe, active_cycle, spectrum_kind)
        active.parameters = [
        ParameterValue(p.name, p.unit, p.initial, p.lower, p.upper, None, p.fixed)
        for p in parameters
    ]
        state = ProjectState(
            source_path=path,
            circuit=circuit,
            control=spectrum_kind,
            available_cycles=valid_cycles.copy(),
            active_cycle=active_cycle,
            default_parameters=parameters,
            cycles={active_cycle: active},
        )
        active.circuit = circuit
        spectra = catalog_spectra(dataframe, valid_cycles, spectrum_kind)
        label = f"{path.name} [{SPECTRUM_KIND_LABELS.get(spectrum_kind, spectrum_kind.title())}]"
        projects.append(
            LoadedProject(
                dataframe=dataframe,
                state=state,
                technique=technique or "Unknown",
                spectra=spectra,
                dataset_id=_dataset_id(path, spectrum_kind),
                dataset_label=label,
                skipped_cycles=tuple(skipped_cycles),
            )
        )
    return projects


def load_project_from_dataframe(
    dataframe,
    source_path: Path,
    cycle: int,
    control: str,
    circuit: str,
    technique: str = "Saved project",
) -> LoadedProject:
    spectrum_kind = _normalize_spectrum_kind(control)
    if "freq_hz" not in dataframe.columns:
        raise KeyError("The saved data has no 'freq_hz' column")
    cycles = (
        _safe_unique_ints(dataframe["cycle_number"].values)
        if "cycle_number" in dataframe.columns
        else [cycle]
    )
    if not cycles:
        raise ValueError("No cycles were found in the saved data")
    parameters = circuit_parameters(circuit)
    cycles, skipped_cycles = _cycles_with_impedance(dataframe, cycles, spectrum_kind)
    if not cycles:
        raise ValueError("No cycles with impedance data were found in the saved data")
    active_cycle = cycle if cycle in cycles else cycles[0]
    active = load_cycle(dataframe, active_cycle, spectrum_kind)
    active.parameters = [
        ParameterValue(p.name, p.unit, p.initial, p.lower, p.upper, None, p.fixed)
        for p in parameters
    ]
    state = ProjectState(
        source_path=source_path,
        circuit=circuit,
        control=spectrum_kind,
        available_cycles=cycles.copy(),
        active_cycle=active_cycle,
        default_parameters=parameters,
        cycles={active_cycle: active},
    )
    active.circuit = circuit
    dataset_id = _dataset_id(source_path, spectrum_kind)
    return LoadedProject(
        dataframe=dataframe,
        state=state,
        technique=technique,
        spectra=catalog_spectra(dataframe, cycles, spectrum_kind),
        dataset_id=dataset_id,
        dataset_label=f"{source_path.name} [{SPECTRUM_KIND_LABELS.get(spectrum_kind, spectrum_kind.title())}]",
        skipped_cycles=tuple(skipped_cycles),
    )


def load_projects(
    paths: list[Path],
    control: str,
    circuit: str,
    cycle: int = 1,
    spectrum_kinds_by_path: dict[Path, list[str]] | None = None,
) -> ProjectImportReport:
    loaded: list[tuple[str, LoadedProject]] = []
    errors: list[tuple[Path, str]] = []
    for path in paths:
        try:
            selected_kinds = (
                spectrum_kinds_by_path.get(path.resolve())
                if spectrum_kinds_by_path is not None
                else None
            )
            projects = load_projects_for_file(
                path, cycle, control, circuit, selected_kinds
            )
        except Exception as error:
            errors.append((path, f"{type(error).__name__}: {error}"))
        else:
            loaded.extend((project.dataset_id, project) for project in projects)
    return ProjectImportReport(loaded, errors)


def inspect_eis_file_spectrum_kinds(path: Path) -> list[str]:
    """Return the electrode-pair spectra available in an EIS data file."""
    _dataframe, header_meta, _technique = _read_eis_dataframe(path)
    return _available_spectrum_kinds(_dataframe, header_meta)


def _clamp_initial(value: float, parameter: ParameterValue) -> float:
    if not np.isfinite(value):
        return parameter.initial
    return float(np.clip(value, parameter.lower, parameter.upper))


def _map_ridge_to_parameters(
    parameters: list[ParameterValue],
    ohmic_resistance: float,
    inductance: float,
    peak_resistance: np.ndarray,
    peak_tau: np.ndarray,
    peak_alpha: np.ndarray,
    peak_beta: np.ndarray,
) -> list[ParameterValue]:
    mapped = [
        ParameterValue(p.name, p.unit, p.initial, p.lower, p.upper, p.error_percent, p.fixed)
        for p in parameters
    ]
    by_name = {parameter.name: parameter for parameter in mapped}
    if "R0" in by_name:
        parameter = by_name["R0"]
        parameter.initial = _clamp_initial(max(ohmic_resistance, 0.0), parameter)
    for parameter in mapped:
        if re.fullmatch(r"L\d+", parameter.name):
            parameter.initial = _clamp_initial(max(inductance, 0.0), parameter)

    branch_ids = sorted(
        int(match.group(1))
        for parameter in mapped
        if (match := re.fullmatch(r"R(\d+)", parameter.name))
        and int(match.group(1)) != 0
    )
    branch_count = min(len(branch_ids), peak_resistance.size)
    if branch_count == 0:
        return mapped
    strongest = np.argsort(peak_resistance)[-branch_count:]
    strongest = strongest[np.argsort(peak_tau[strongest])]
    for branch_id, peak_index in zip(branch_ids, strongest):
        resistance = max(float(peak_resistance[peak_index]), np.finfo(float).eps)
        tau = max(float(peak_tau[peak_index]), np.finfo(float).eps)
        exponent = float(
            np.clip(peak_alpha[peak_index] * peak_beta[peak_index], 0.05, 1.0)
        )
        resistance_parameter = by_name.get(f"R{branch_id}")
        if resistance_parameter is not None:
            resistance_parameter.initial = _clamp_initial(
                resistance, resistance_parameter
            )
        q_parameter = by_name.get(f"CPE{branch_id}_0")
        exponent_parameter = by_name.get(f"CPE{branch_id}_1")
        capacitance_parameter = by_name.get(f"C{branch_id}")
        if q_parameter is not None:
            q_parameter.initial = _clamp_initial(
                tau**exponent / resistance,
                q_parameter,
            )
        if exponent_parameter is not None:
            exponent_parameter.initial = _clamp_initial(
                exponent,
                exponent_parameter,
            )
        if capacitance_parameter is not None:
            capacitance_parameter.initial = _clamp_initial(
                tau / resistance,
                capacitance_parameter,
            )
    return mapped


def analyze_outliers(
    state: CycleState,
    threshold: float,
    parameters: list[ParameterValue],
) -> RidgeInitialization:
    from bayes_drt2 import peak_fit
    from bayes_drt2.inversion import Inverter

    active_mask = state.included
    active_indices = np.flatnonzero(active_mask)
    if active_indices.size < 3:
        raise ValueError("At least three active points are required for outlier search")

    inverter = Inverter()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Hyperparametric solution did not converge.*",
            category=UserWarning,
        )
        outliers = inverter.check_outliers(
            state.frequency_hz[active_mask].copy(),
            state.impedance[active_mask].copy(),
            threshold=threshold,
            use_existing_fit=False,
        )

    peak_resistance = np.array([], dtype=float)
    peak_tau = np.array([], dtype=float)
    peak_alpha = np.array([], dtype=float)
    peak_beta = np.array([], dtype=float)
    ridge_tau = np.array([], dtype=float)
    ridge_gamma = np.array([], dtype=float)
    try:
        distribution = next(iter(inverter.distributions))
        basis_tau = inverter.distributions[distribution]["tau"]
        minimum_log_tau = np.log10(np.min(basis_tau)) - 1
        maximum_log_tau = np.log10(np.max(basis_tau)) + 1
        evaluation_tau = np.logspace(
            minimum_log_tau,
            maximum_log_tau,
            int(10 * (maximum_log_tau - minimum_log_tau) + 1),
        )
        distribution_values = inverter.predict_distribution(
            name=distribution,
            tau=evaluation_tau,
        )
        ridge_tau = as_1d_array(evaluation_tau).astype(float)
        ridge_gamma = as_1d_array(distribution_values).astype(float)
        trapz_missing = not hasattr(np, "trapz")
        if trapz_missing:
            np.trapz = np.trapezoid
        try:
            peak_parameters = peak_fit.fit_peaks(
                evaluation_tau,
                distribution_values,
                inverter.predict_Rp(),
                nonneg=bool(np.min(distribution_values) >= 0),
                check_shoulders=False,
                prom_rthresh=0.001,
                R_rthresh=0.005,
                l1_penalty=0,
                l2_penalty=0.01,
            )
        finally:
            if trapz_missing:
                delattr(np, "trapz")
        peak_resistance = as_1d_array(peak_parameters[::4]).astype(float)
        peak_tau = np.exp(as_1d_array(peak_parameters[1::4]).astype(float))
        peak_alpha = as_1d_array(peak_parameters[2::4]).astype(float)
        peak_beta = as_1d_array(peak_parameters[3::4]).astype(float)
    except Exception:
        pass

    mapped_parameters = _map_ridge_to_parameters(
        parameters,
        float(inverter.R_inf),
        float(inverter.inductance),
        peak_resistance,
        peak_tau,
        peak_alpha,
        peak_beta,
    )
    return RidgeInitialization(
        outlier_indices=active_indices[as_1d_array(outliers).astype(int)],
        parameters=mapped_parameters,
        peak_count=int(peak_resistance.size),
        ohmic_resistance=float(inverter.R_inf),
        inductance=float(inverter.inductance),
        ridge_tau_s=ridge_tau,
        ridge_gamma_ohm=ridge_gamma,
    )


def find_outliers_for_all_cycles(
    dataframe,
    project: ProjectState,
    threshold: float,
    stop_event=None,
) -> dict[int, tuple[CycleState, RidgeInitialization]]:
    results: dict[int, tuple[CycleState, RidgeInitialization]] = {}
    cycle_numbers = list(project.available_cycles)
    for index, cycle_number in enumerate(cycle_numbers):
        if stop_event is not None and stop_event.is_set():
            break
        cycle = project.cycles.get(cycle_number)
        if cycle is None:
            cycle = load_cycle(dataframe, cycle_number, project.control)
            if project.all_frequency_window is not None:
                cycle.frequency_window = project.all_frequency_window
            cycle.circuit = project.circuit
        analysis = analyze_outliers(
            cycle,
            threshold,
            project.parameters_for(cycle_number),
        )
        results[cycle_number] = (cycle, analysis)
    return results


def calculate_hybrid_drt(state: CycleState) -> DRTComputation:
    from wepy.eis import get_drt

    active_mask = state.included
    if int(np.count_nonzero(active_mask)) < 3:
        raise ValueError("At least three active points are required for DRT")
    frequency = state.frequency_hz[active_mask]
    impedance = state.impedance[active_mask]
    frequency, impedance = sort_spectrum(
        as_1d_array(frequency),
        as_1d_array(impedance),
    )
    tau_s, gamma_ohm, ohmic_resistance = get_drt(
        frequency,
        impedance,
        method="hybrid",
    )
    return DRTComputation(
        tau_s=as_1d_array(tau_s).astype(float),
        gamma_ohm=as_1d_array(gamma_ohm).astype(float),
        ohmic_resistance=float(ohmic_resistance),
    )


def select_eec_model_from_hybrid_drt(
    state: CycleState,
    settings: dict[str, object] | None = None,
) -> AutomaticEECModel:
    from copy import deepcopy

    from hybdrt.models import DRT

    active_mask = state.included
    if int(np.count_nonzero(active_mask)) < 3:
        raise ValueError("At least three active points are required for model selection")
    frequency = state.frequency_hz[active_mask]
    impedance = state.impedance[active_mask]
    frequency, impedance = sort_spectrum(
        as_1d_array(frequency),
        as_1d_array(impedance),
    )
    drt = DRT()
    drt.fit_eis(frequency, impedance)
    dual = deepcopy(drt)
    settings = settings or {}
    criterion = str(settings.get("criterion", "lml-bic"))
    if criterion not in {"bic", "lml", "lml-bic"}:
        criterion = "lml-bic"
    max_num_peaks = max(int(settings.get("max_num_peaks", 10)), 1)
    prior = bool(settings.get("prior", True))
    prior_strength = settings.get("prior_strength")
    generate_kw = {}
    find_peaks_kw = {}
    if settings.get("peak_prominence") is not None:
        find_peaks_kw["prominence"] = float(settings["peak_prominence"])
    if settings.get("peak_height") is not None:
        find_peaks_kw["height"] = float(settings["peak_height"])
    if find_peaks_kw:
        generate_kw["find_peaks_kw"] = find_peaks_kw
    dual.dual_fit_eis(
        frequency,
        impedance,
        generate_kw=generate_kw or None,
        discrete_kw=dict(
            max_num_peaks=max_num_peaks,
            prior=prior,
            prior_strength=prior_strength,
            model_init_kw=dict(drt_element="RQ"),
        ),
    )
    candidate_id = dual.get_best_candidate_id("discrete", criterion=criterion)
    candidate = dual.get_candidate(candidate_id, "discrete")
    model = candidate["model"]
    values = model.parameter_dict

    def _value(*names: str) -> float | None:
        for name in names:
            if name in values:
                try:
                    value = float(values[name])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    return value
        return None

    ohmic_resistance = _value("R_R0", "R_R_inf", "R_inf")
    if ohmic_resistance is None:
        raise ValueError("Hybrid DRT did not return an ohmic resistance")
    log_inductance = _value("lnL_L0", "lnL_inductance")
    direct_inductance = _value("L_L0", "L_inductance", "inductance")
    if log_inductance is not None:
        inductance = float(np.exp(log_inductance))
    elif direct_inductance is not None:
        inductance = direct_inductance
    else:
        inductance = 0.0

    minimum_r0 = settings.get("min_r0")
    minimum_l0 = settings.get("min_l0")
    include_r0 = minimum_r0 is None or ohmic_resistance >= float(minimum_r0)
    include_l0 = minimum_l0 is None or inductance >= float(minimum_l0)
    initials = {}
    elements = []
    if include_r0:
        elements.append("R0")
        initials["R0"] = max(ohmic_resistance, 0.0)
    if include_l0:
        elements.append("L0")
        initials["L0"] = max(inductance, 0.0)
    branch_count = 0
    for element_name, element_type in zip(
        getattr(model, "element_names", ()),
        getattr(model, "element_types", ()),
    ):
        if str(element_type).upper() != "RQ":
            continue
        resistance = _value(f"R_{element_name}")
        log_tau = _value(f"lntau_{element_name}")
        exponent = _value(f"beta_{element_name}")
        if resistance is None or log_tau is None or exponent is None:
            continue
        resistance = max(resistance, np.finfo(float).eps)
        tau_s = float(np.exp(log_tau))
        exponent = float(np.clip(exponent, 1e-3, 1.0))
        q_value = tau_s**exponent / resistance
        branch_count += 1
        initials.update(
            {
                f"R{branch_count}": resistance,
                f"CPE{branch_count}_0": q_value,
                f"CPE{branch_count}_1": exponent,
            }
        )

    elements.extend(
        f"p(R{index},CPE{index})" for index in range(1, branch_count + 1)
    )
    return AutomaticEECModel(
        circuit="-".join(elements),
        initials=initials,
        peak_count=branch_count,
        criterion=criterion,
        ohmic_resistance=float(ohmic_resistance),
        inductance=float(inductance),
    )


def calculate_lin_kk_residuals(state: CycleState) -> KKResiduals:
    import impedance.validation as validation

    validation.np = np
    validation.eval_linKK.__globals__["np"] = np
    validation.circuit_elements["np"] = np

    included = state.included
    if int(np.count_nonzero(included)) < 3:
        raise ValueError("At least three active points are required for Lin-KK")
    frequency = state.frequency_hz[included]
    impedance = state.impedance[included]
    frequency, impedance = sort_spectrum(
        as_1d_array(frequency),
        as_1d_array(impedance),
    )
    _M, _mu, fit_impedance, residual_real, residual_imag = validation.linKK(
        frequency,
        impedance,
        fit_type="complex",
    )
    return KKResiduals(
        fit_impedance=as_1d_array(fit_impedance),
        residual_real=as_1d_array(residual_real).astype(float),
        residual_imag=as_1d_array(residual_imag).astype(float),
    )


def fit_cycle(
    state: CycleState,
    circuit: str,
    parameters: list[ParameterValue],
    options: FitOptions | None = None,
    stop_event=None,
) -> FitResult:
    from impedance.models.circuits import CustomCircuit
    from scipy.optimize import least_squares
    from wepy.eis import show_fit

    options = (options or FitOptions()).validated()
    included = state.included
    frequency = state.frequency_hz[included]
    impedance = state.impedance[included]
    if frequency.size < 3:
        raise ValueError("At least three included points are required for fitting")
    frequency, impedance = sort_spectrum(frequency, impedance)
    initial_parameters = np.array(
        [parameter.initial for parameter in parameters],
        dtype=float,
    )
    circuit_parameters_only = initial_parameters.copy()
    fixed_constants = {
        parameter.name: float(parameter.initial)
        for parameter in parameters
        if parameter.fixed
    }
    free_parameters = [parameter for parameter in parameters if not parameter.fixed]
    started = time.perf_counter()
    free_indices = [index for index, parameter in enumerate(parameters) if not parameter.fixed]

    # Keep the default path identical to the pre-extension impedance.py path.
    # This preserves curve_fit's optimizer, weighting, and uncertainty semantics.
    if options.stages() == ("least_squares",):
        from impedance.models.circuits.fitting import circuit_fit

        fitted = initial_parameters.copy()
        absolute_errors = np.zeros(len(parameters), dtype=float)
        if free_parameters:
            fitted_free, errors_free = circuit_fit(
                frequency,
                impedance,
                circuit,
                [parameter.initial for parameter in free_parameters],
                constants=fixed_constants,
                bounds=(
                    [parameter.lower for parameter in free_parameters],
                    [parameter.upper for parameter in free_parameters],
                ),
                weight_by_modulus=options.weight_by_modulus,
            )
            fitted_free = as_1d_array(fitted_free).astype(float)
            errors_free = np.abs(as_1d_array(errors_free).astype(float))
            fitted[free_indices] = fitted_free
            absolute_errors[free_indices] = errors_free
        magnitudes = np.abs(fitted)
        errors_percent = np.zeros(fitted.size, dtype=float)
        nonfixed = np.array([not parameter.fixed for parameter in parameters], dtype=bool)
        finite = nonfixed & (magnitudes > np.finfo(float).eps)
        errors_percent[finite] = absolute_errors[finite] / magnitudes[finite] * 100.0
        near_zero = nonfixed & ~finite & (absolute_errors > np.finfo(float).eps)
        errors_percent[near_zero] = np.inf
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Simulating circuit based on initial parameters", category=UserWarning)
            fit_frequency, fit_impedance = show_fit(frequency, circuit, fitted, points=200)
        fitted_model = CustomCircuit(circuit, initial_guess=fitted)
        fitted_model.parameters_ = fitted
        fit_at_data = as_1d_array(fitted_model.predict(state.frequency_hz))
        difference = fit_at_data[included] - state.impedance[included]
        if options.weight_by_modulus:
            difference = difference / np.maximum(np.abs(state.impedance[included]), np.finfo(float).eps)
        objective_value = float(np.sum(np.concatenate((difference.real, difference.imag)) ** 2))
        return FitResult(
            fitted_parameters=fitted,
            errors_percent=errors_percent,
            fit_frequency_hz=as_1d_array(fit_frequency),
            fit_impedance=as_1d_array(fit_impedance),
            fit_at_data_impedance=fit_at_data,
            objective=objective_value,
            rmse=float(np.sqrt(np.mean(np.concatenate((difference.real, difference.imag)) ** 2))),
            converged=True,
            stages=[{"method": "least_squares", "objective": objective_value, "converged": True}],
            options=options,
            elapsed_seconds=time.perf_counter() - started,
        )

    free_initial = initial_parameters[free_indices]
    lower = np.array([parameter.lower for parameter in free_parameters], dtype=float)
    upper = np.array([parameter.upper for parameter in free_parameters], dtype=float)

    def predict(values: np.ndarray, frequencies=frequency) -> np.ndarray:
        full = initial_parameters.copy()
        full[free_indices] = values
        model = CustomCircuit(circuit, initial_guess=full)
        model.parameters_ = full
        return as_1d_array(model.predict(frequencies)).astype(complex)

    def residual(values: np.ndarray) -> np.ndarray:
        if stop_event is not None and stop_event.is_set():
            raise FitTimeoutError("EEC fit cancelled")
        difference = predict(values) - impedance
        if options.weight_by_modulus:
            difference = difference / np.maximum(np.abs(impedance), np.finfo(float).eps)
        return np.concatenate((difference.real, difference.imag))

    def objective(values: np.ndarray) -> float:
        values = np.clip(np.asarray(values, dtype=float), lower, upper)
        return float(np.sum(residual(values) ** 2))

    def from_unit(values: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("PSO/GA fitting requires finite parameter bounds")
        return lower + np.clip(np.asarray(values, dtype=float), 0.0, 1.0) * (upper - lower)

    stages = []
    current = np.clip(free_initial, lower, upper)
    if free_parameters:
        for stage in options.stages():
            stage_started = time.perf_counter()
            if stage == "least_squares":
                result = least_squares(
                    residual, current, bounds=(lower, upper),
                    max_nfev=max(int(options.iterations) * 10, 100),
                )
                current = result.x
                converged = bool(result.success)
                detail = {"status": int(result.status), "message": str(result.message)}
            elif stage == "basinhopping":
                from scipy.optimize import basinhopping
                result = basinhopping(
                    objective, current, niter=int(options.iterations),
                    seed=options.seed,
                    minimizer_kwargs={"method": "L-BFGS-B", "bounds": list(zip(lower, upper))},
                )
                current = np.clip(result.x, lower, upper)
                converged = bool(result.success)
                detail = {"message": str(result.message)}
            elif stage == "pso":
                try:
                    import pyswarms as ps
                except ImportError as error:
                    raise ImportError("PSO fitting requires the optional 'pyswarms' package") from error
                rng = np.random.default_rng(options.seed)
                def swarm_objective(points):
                    return np.asarray([objective(from_unit(point)) for point in points])
                optimizer = ps.single.GlobalBestPSO(
                    n_particles=int(options.population_size), dimensions=len(current),
                    options={"c1": 0.5, "c2": 0.3, "w": 0.9},
                    bounds=(np.zeros_like(lower), np.ones_like(upper)),
                    init_pos=rng.uniform(0.0, 1.0, (int(options.population_size), len(current))),
                )
                cost, position = optimizer.optimize(swarm_objective, iters=int(options.iterations), verbose=False)
                current = from_unit(position)
                converged = np.isfinite(cost)
                detail = {"best_cost": float(cost)}
            elif stage == "ga":
                try:
                    import pygad
                except ImportError as error:
                    raise ImportError("GA fitting requires the optional 'pygad' package") from error
                def fitness(_ga, solution, _index):
                    return -objective(np.asarray(solution, dtype=float))
                ga = pygad.GA(
                    num_generations=int(options.iterations),
                    sol_per_pop=int(options.population_size),
                    num_parents_mating=max(2, int(options.population_size) // 3),
                    num_genes=len(current), fitness_func=fitness,
                    gene_space=[{"low": 0.0, "high": 1.0} for _ in current],
                    random_seed=options.seed,
                    suppress_warnings=True,
                )
                ga.run()
                _solution, fitness_value, _ = ga.best_solution()
                current = from_unit(_solution)
                converged = np.isfinite(fitness_value)
                detail = {"best_cost": float(-fitness_value)}
            stages.append({"method": stage, "objective": objective(current), "converged": converged,
                           "elapsed_seconds": time.perf_counter() - stage_started, **detail})
    absolute_errors = np.zeros(len(parameters), dtype=float)
    if free_parameters:
        circuit_parameters_only[free_indices] = current
        # Estimate covariance from the local Jacobian when available.
        try:
            from scipy.optimize._numdiff import approx_derivative
            jacobian = approx_derivative(residual, current, method="2-point", bounds=(lower, upper))
            covariance = np.linalg.pinv(jacobian.T @ jacobian)
            # Match scipy curve_fit's default absolute_sigma=False behavior:
            # covariance is scaled by the residual variance (reduced chi-square).
            degrees_of_freedom = max(len(residual(current)) - len(free_indices), 1)
            covariance *= objective(current) / degrees_of_freedom
            absolute_errors[free_indices] = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        except Exception:
            pass
    errors_percent = np.zeros(circuit_parameters_only.size, dtype=float)
    nonfixed = np.array([not parameter.fixed for parameter in parameters], dtype=bool)
    magnitudes = np.abs(circuit_parameters_only)
    finite = nonfixed & (magnitudes > np.finfo(float).eps)
    errors_percent[finite] = absolute_errors[finite] / magnitudes[finite] * 100.0
    near_zero = nonfixed & ~finite & (absolute_errors > np.finfo(float).eps)
    errors_percent[near_zero] = np.inf
    if free_parameters and not any(stage.get("method") == "least_squares" for stage in stages):
        # A global search has no statistically calibrated local covariance.
        errors_percent[nonfixed] = np.nan
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
    final_residual = residual(current) if free_parameters else np.concatenate(((predict(np.array([])) - impedance).real, (predict(np.array([])) - impedance).imag))
    objective_value = float(np.sum(final_residual ** 2))
    return FitResult(
        fitted_parameters=circuit_parameters_only,
        errors_percent=errors_percent,
        fit_frequency_hz=as_1d_array(fit_frequency),
        fit_impedance=as_1d_array(fit_impedance),
        fit_at_data_impedance=as_1d_array(fit_at_data),
        objective=objective_value,
        rmse=float(np.sqrt(np.mean(final_residual ** 2))),
        converged=not stages or bool(stages[-1]["converged"]),
        stages=stages,
        options=options,
        elapsed_seconds=time.perf_counter() - started,
    )


def fit_cycle_with_timeout(
    state: CycleState,
    circuit: str,
    parameters: list[ParameterValue],
    timeout_seconds: float,
    options: FitOptions | None = None,
) -> FitResult:
    """Run one EEC fit in a reusable, terminable process with a hard limit."""
    global _FIT_WORKER_PROCESS, _FIT_WORKER_CONNECTION
    timeout_seconds = float(timeout_seconds)
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("The EEC fit time limit must be a positive finite number")

    with _FIT_WORKER_LOCK:
        context = mp.get_context("spawn")
        if (
            _FIT_WORKER_PROCESS is None
            or not _FIT_WORKER_PROCESS.is_alive()
            or _FIT_WORKER_CONNECTION is None
        ):
            if _FIT_WORKER_CONNECTION is not None:
                _FIT_WORKER_CONNECTION.close()
            receiver, sender = context.Pipe(duplex=True)
            process = context.Process(target=_fit_cycle_process_entry, args=(sender,))
            process.daemon = True
            process.start()
            sender.close()
            _FIT_WORKER_PROCESS = process
            _FIT_WORKER_CONNECTION = receiver

        connection = _FIT_WORKER_CONNECTION
        process = _FIT_WORKER_PROCESS
        try:
            connection.send((state, circuit, parameters, options or FitOptions()))
            if not connection.poll(timeout_seconds):
                process.terminate()
                process.join()
                connection.close()
                _FIT_WORKER_PROCESS = None
                _FIT_WORKER_CONNECTION = None
                raise FitTimeoutError(
                    f"The EEC fit exceeded the {timeout_seconds:g} s time limit"
                )
            succeeded, payload = connection.recv()
        except (EOFError, OSError) as error:
            if process.is_alive():
                process.terminate()
                process.join()
            connection.close()
            _FIT_WORKER_PROCESS = None
            _FIT_WORKER_CONNECTION = None
            raise RuntimeError("The EEC fit worker stopped unexpectedly") from error
    if not succeeded:
        raise RuntimeError(str(payload))
    return payload


def refine_fit_cycle(
    state: CycleState,
    circuit: str,
    parameters: list[ParameterValue],
    z_threshold: float,
    max_iterations: int,
    fit_timeout_seconds: float | None = None,
    fit_options: FitOptions | None = None,
):
    from copy import deepcopy

    if state.fit_parameters is None or state.fit_at_data_impedance is None:
        raise ValueError("Refine fit requires an existing fit")
    active_indices = np.flatnonzero(state.included)
    if state.fit_at_data_impedance.size != state.frequency_hz.size:
        raise ValueError("The existing fit does not match the spectrum points")

    working_state = deepcopy(state)
    working_parameters = copy_parameter_values(parameters)
    current_result = (
        as_1d_array(state.fit_parameters).astype(float).copy(),
        np.asarray(
            [parameter.error_percent or 0.0 for parameter in parameters],
            dtype=float,
        ),
        as_1d_array(state.fit_frequency_hz).astype(float).copy(),
        as_1d_array(state.fit_impedance).astype(complex).copy(),
        as_1d_array(state.fit_at_data_impedance).astype(complex).copy(),
    )
    removed_indices: list[int] = []
    iterations = 0

    def robust_scale(values: np.ndarray) -> float:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad
        if scale <= np.finfo(float).eps:
            scale = float(np.std(values))
        return max(scale, np.finfo(float).eps)

    while iterations < max_iterations:
        active_indices = np.flatnonzero(working_state.included)
        if active_indices.size <= 3:
            break
        order = np.argsort(working_state.frequency_hz[active_indices])[::-1]
        measured = working_state.impedance[active_indices][order]
        calculated = current_result[4][active_indices][order]
        if calculated.size != measured.size:
            raise ValueError("The fit and active points have different lengths")
        residual = measured - calculated
        real_scale = robust_scale(residual.real)
        imaginary_scale = robust_scale(residual.imag)
        normalized = np.hypot(
            residual.real / real_scale,
            residual.imag / imaginary_scale,
        )
        score_center = float(np.median(normalized))
        score_scale = robust_scale(normalized)
        robust_z = 0.6745 * (normalized - score_center) / score_scale
        candidates = np.flatnonzero(
            np.isfinite(robust_z) & (robust_z > z_threshold)
        )
        if candidates.size == 0:
            break
        candidates = candidates[np.argsort(robust_z[candidates])[::-1]]
        candidates = candidates[: max(active_indices.size - 3, 0)]
        original_indices = active_indices[order[candidates]]
        working_state.manually_included[original_indices] = False
        working_state.outliers[original_indices] = True
        removed_indices.extend(int(index) for index in original_indices)
        for parameter, value in zip(working_parameters, current_result[0]):
            parameter.initial = float(value)
        fit_function = (
            fit_cycle_with_timeout
            if fit_timeout_seconds is not None
            else fit_cycle
        )
        current_result = fit_function(
            working_state,
            circuit,
            working_parameters,
            **({"timeout_seconds": fit_timeout_seconds} if fit_timeout_seconds is not None else {}),
            **({"options": fit_options} if fit_options is not None else {}),
        )
        iterations += 1

    return current_result, np.asarray(removed_indices, dtype=int), iterations


def _batch_parameters_with_initials(
    target_parameters: list[ParameterValue],
    source_parameters: list[ParameterValue],
    source_circuit: str | None = None,
    target_circuit: str | None = None,
) -> list[ParameterValue]:
    element_mapping = (
        parameter_name_mapping(source_circuit, target_circuit)
        if source_circuit and target_circuit
        else None
    )
    source_by_name = {
        map_parameter_name(parameter.name, element_mapping) if element_mapping else parameter.name: parameter
        for parameter in source_parameters
    }
    copied = []
    for target in target_parameters:
        source = source_by_name.get(target.name)
        initial = target.initial if source is None else source.initial
        copied.append(
            ParameterValue(
                target.name,
                target.unit,
                _clamp_initial(initial, target),
                target.lower,
                target.upper,
                target.error_percent,
                target.fixed,
            )
        )
    return copied


def batch_fit_from_cycle(
    dataframe,
    project: ProjectState,
    start_cycle: int,
    initial_parameters: list[ParameterValue],
    stop_event=None,
    initial_circuit: str | None = None,
    fit_timeout_seconds: float | None = None,
    fit_options: FitOptions | None = None,
) -> BatchFitReport:
    start_index = project.available_cycles.index(start_cycle)
    cycle_numbers = project.available_cycles[start_index:]
    next_parameters = [
        ParameterValue(
            p.name, p.unit, p.initial, p.lower, p.upper, p.error_percent, p.fixed
        )
        for p in initial_parameters
    ]
    next_circuit = initial_circuit or project.circuit
    completed: list[BatchCycleFit] = []
    for index, cycle_number in enumerate(cycle_numbers):
        if stop_event is not None and stop_event.is_set():
            return BatchFitReport(
                fits=completed,
                stopped=True,
                skipped_cycles=cycle_numbers[index:],
            )
        cycle = project.cycles.get(cycle_number)
        if cycle is None:
            cycle = load_cycle(dataframe, cycle_number, project.control)
            if project.all_frequency_window is not None:
                cycle.frequency_window = project.all_frequency_window
            cycle.circuit = project.circuit
            cycle.circuit = project.circuit
            cycle.circuit = project.circuit
        cycle_circuit = cycle.model(project.circuit)
        cycle_parameters = project.parameters_for(cycle_number)
        if not circuits_equivalent(next_circuit, cycle_circuit):
            return BatchFitReport(
                fits=completed,
                failed_cycle=cycle_number,
                error="The spectrum uses a different fitting model",
                skipped_cycles=cycle_numbers[index + 1 :],
            )
        fit_parameters = _batch_parameters_with_initials(
            cycle_parameters,
            next_parameters,
            next_circuit,
            cycle_circuit,
        )
        try:
            fit_function = (
                fit_cycle_with_timeout
                if fit_timeout_seconds is not None
                else fit_cycle
            )
            fit_kwargs = (
                {"timeout_seconds": fit_timeout_seconds}
                if fit_timeout_seconds is not None
                else {}
            )
            if fit_options is not None:
                fit_kwargs["options"] = fit_options
            fit_result = fit_function(
                cycle, cycle_circuit, fit_parameters, **fit_kwargs
            )
            fitted, errors_percent, fit_frequency, fit_impedance, fit_at_data = fit_result
        except Exception as error:
            return BatchFitReport(
                fits=completed,
                failed_cycle=cycle_number,
                error=f"{type(error).__name__}: {error}",
                skipped_cycles=cycle_numbers[index + 1 :],
            )
        fitted_parameters = [
            ParameterValue(
                parameter.name,
                parameter.unit,
                float(value),
                parameter.lower,
                parameter.upper,
                float(error_percent),
                parameter.fixed,
            )
              for parameter, value, error_percent in zip(
                  fit_parameters, fitted, errors_percent
            )
        ]
        completed.append(
            BatchCycleFit(
                cycle=cycle,
                parameters=fitted_parameters,
                fitted_parameters=fitted,
                fitted_errors_percent=errors_percent,
                fit_frequency_hz=fit_frequency,
                fit_impedance=fit_impedance,
                fit_at_data_impedance=fit_at_data,
                fit_provenance=_fit_provenance(fit_result),
            )
        )
        next_parameters = fitted_parameters
        next_circuit = cycle_circuit
        if stop_event is not None and stop_event.is_set():
            return BatchFitReport(
                fits=completed,
                stopped=True,
                skipped_cycles=cycle_numbers[index + 1 :],
            )
    return BatchFitReport(fits=completed)


def batch_fit_spectra(
    targets: list[SpectrumFitTarget],
    initial_parameters: list[ParameterValue],
    *,
    use_target_initial_parameters: bool = False,
    stop_event=None,
    initial_circuit: str | None = None,
    fit_timeout_seconds: float | None = None,
    fit_options: FitOptions | None = None,
) -> SpectrumBatchReport:
    if not targets:
        return SpectrumBatchReport([])
    first_project = targets[0].loaded.state
    first_cycle = first_project.cycles.get(targets[0].cycle)
    expected_circuit = (
        first_cycle.model(first_project.circuit)
        if first_cycle is not None
        else first_project.circuit
    )
    next_parameters = [
        ParameterValue(
            p.name, p.unit, p.initial, p.lower, p.upper, p.error_percent, p.fixed
        )
        for p in initial_parameters
    ]
    next_circuit = initial_circuit or expected_circuit
    completed: list[SpectrumBatchFit] = []
    for index, target in enumerate(targets):
        if stop_event is not None and stop_event.is_set():
            return SpectrumBatchReport(
                completed,
                stopped=True,
                skipped_labels=[item.label for item in targets[index:]],
            )
        project = target.loaded.state
        cycle = project.cycles.get(target.cycle)
        if cycle is None:
            cycle = load_cycle(target.loaded.dataframe, target.cycle, project.control)
            if project.all_frequency_window is not None:
                cycle.frequency_window = project.all_frequency_window
            cycle.circuit = project.circuit
        target_circuit = cycle.model(project.circuit)
        target_parameters = project.parameters_for(target.cycle)
        if use_target_initial_parameters:
            fit_circuit = target_circuit
            fit_parameters = target_parameters
        else:
            if not circuits_equivalent(next_circuit, target_circuit):
                return SpectrumBatchReport(
                    completed,
                    target.label,
                    "The spectrum uses a different fitting model",
                    skipped_labels=[item.label for item in targets[index + 1 :]],
                )
            element_mapping = parameter_name_mapping(
                next_circuit, target_circuit
            )
            mapped_names = {
                map_parameter_name(parameter.name, element_mapping)
                for parameter in initial_parameters
            }
            if mapped_names != {parameter.name for parameter in target_parameters}:
                return SpectrumBatchReport(
                    completed,
                    target.label,
                    "The spectrum has incompatible fitting parameters",
                    skipped_labels=[item.label for item in targets[index + 1 :]],
                )
            fit_circuit = target_circuit
            fit_parameters = _batch_parameters_with_initials(
                target_parameters,
                next_parameters,
                next_circuit,
                target_circuit,
            )
        try:
            fit_function = (
                fit_cycle_with_timeout
                if fit_timeout_seconds is not None
                else fit_cycle
            )
            fit_kwargs = (
                {"timeout_seconds": fit_timeout_seconds}
                if fit_timeout_seconds is not None
                else {}
            )
            if fit_options is not None:
                fit_kwargs["options"] = fit_options
            fit_result = fit_function(
                cycle, fit_circuit, fit_parameters, **fit_kwargs
            )
            fitted, errors_percent, fit_frequency, fit_impedance, fit_at_data = fit_result
        except Exception as error:
            return SpectrumBatchReport(
                completed,
                target.label,
                f"{type(error).__name__}: {error}",
                skipped_labels=[item.label for item in targets[index + 1 :]],
            )
        fitted_parameters = [
            ParameterValue(
                parameter.name,
                parameter.unit,
                float(value),
                parameter.lower,
                parameter.upper,
                float(error_percent),
                parameter.fixed,
            )
            for parameter, value, error_percent in zip(
                fit_parameters, fitted, errors_percent
            )
        ]
        completed.append(
            SpectrumBatchFit(
                loaded=target.loaded,
                fit=BatchCycleFit(
                    cycle=cycle,
                    parameters=fitted_parameters,
                    fitted_parameters=fitted,
                    fitted_errors_percent=errors_percent,
                    fit_frequency_hz=fit_frequency,
                    fit_impedance=fit_impedance,
                fit_at_data_impedance=fit_at_data,
                fit_provenance=_fit_provenance(fit_result),
                ),
            )
        )
        if not use_target_initial_parameters:
            next_parameters = fitted_parameters
            next_circuit = target_circuit
        if stop_event is not None and stop_event.is_set():
            return SpectrumBatchReport(
                completed,
                stopped=True,
                skipped_labels=[item.label for item in targets[index + 1 :]],
            )
    return SpectrumBatchReport(completed)
