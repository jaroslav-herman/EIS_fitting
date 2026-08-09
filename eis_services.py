from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable
import warnings

import numpy as np

from eis_model import CycleState, ParameterValue, ProjectState, as_1d_array, sort_spectrum

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


@dataclass
class BatchCycleFit:
    cycle: CycleState
    parameters: list[ParameterValue]
    fitted_parameters: np.ndarray
    fitted_errors_percent: np.ndarray
    fit_frequency_hz: np.ndarray
    fit_impedance: np.ndarray
    fit_at_data_impedance: np.ndarray


@dataclass
class BatchFitReport:
    fits: list[BatchCycleFit]
    failed_cycle: int | None = None
    error: str | None = None


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


def load_projects_for_file(
    path: Path,
    cycle: int,
    control: str,
    circuit: str,
) -> list[LoadedProject]:
    from wepy import read_mpt_dataframe

    dataframe, header_meta, technique = read_mpt_dataframe(path)
    cycles = (
        _safe_unique_ints(dataframe["cycle_number"].values)
        if "cycle_number" in dataframe.columns
        else [cycle]
    )
    if not cycles:
        raise ValueError("No cycles were found in the file")
    parameters = circuit_parameters(circuit)
    projects: list[LoadedProject] = []
    spectrum_kinds = _order_spectrum_kinds(
        _available_spectrum_kinds(dataframe, header_meta),
        control,
    )
    for spectrum_kind in spectrum_kinds:
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
) -> ProjectImportReport:
    loaded: list[tuple[str, LoadedProject]] = []
    errors: list[tuple[Path, str]] = []
    for path in paths:
        try:
            projects = load_projects_for_file(path, cycle, control, circuit)
        except Exception as error:
            errors.append((path, f"{type(error).__name__}: {error}"))
        else:
            loaded.extend((project.dataset_id, project) for project in projects)
    return ProjectImportReport(loaded, errors)


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
) -> dict[int, tuple[CycleState, RidgeInitialization]]:
    results: dict[int, tuple[CycleState, RidgeInitialization]] = {}
    for cycle_number in project.available_cycles:
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


def select_eec_model_from_hybrid_drt(state: CycleState) -> AutomaticEECModel:
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
    dual.dual_fit_eis(
        frequency,
        impedance,
        discrete_kw=dict(
            prior=True,
            prior_strength=None,
            model_init_kw=dict(drt_element="RQ"),
        ),
    )
    criterion = "lml-bic"
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

    initials = {
        "R0": max(ohmic_resistance, 0.0),
        "L0": max(inductance, 0.0),
    }
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

    elements = ["R0", "L0"]
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from impedance.models.circuits import CustomCircuit
    from impedance.models.circuits.fitting import circuit_fit
    from wepy.eis import show_fit

    included = state.included
    frequency = state.frequency_hz[included]
    impedance = state.impedance[included]
    if frequency.size < 3:
        raise ValueError("At least three included points are required for fitting")
    frequency, impedance = sort_spectrum(frequency, impedance)
    circuit_parameters_only = np.array(
        [parameter.initial for parameter in parameters],
        dtype=float,
    )
    absolute_errors = np.zeros(len(parameters), dtype=float)
    fixed_constants = {
        parameter.name: float(parameter.initial)
        for parameter in parameters
        if parameter.fixed
    }
    free_parameters = [parameter for parameter in parameters if not parameter.fixed]
    if free_parameters:
        free_initial = [parameter.initial for parameter in free_parameters]
        free_bounds = (
            [parameter.lower for parameter in free_parameters],
            [parameter.upper for parameter in free_parameters],
        )
        fitted_free, errors_free = circuit_fit(
            frequency,
            impedance,
            circuit,
            free_initial,
            constants=fixed_constants,
            bounds=free_bounds,
        )
        fitted_free = as_1d_array(fitted_free).astype(float)
        errors_free = np.abs(as_1d_array(errors_free).astype(float))
        free_index = 0
        for index, parameter in enumerate(parameters):
            if parameter.fixed:
                continue
            circuit_parameters_only[index] = fitted_free[free_index]
            absolute_errors[index] = errors_free[free_index]
            free_index += 1
    errors_percent = np.zeros(circuit_parameters_only.size, dtype=float)
    nonfixed = np.array([not parameter.fixed for parameter in parameters], dtype=bool)
    magnitudes = np.abs(circuit_parameters_only)
    finite = nonfixed & (magnitudes > np.finfo(float).eps)
    errors_percent[finite] = absolute_errors[finite] / magnitudes[finite] * 100.0
    near_zero = nonfixed & ~finite & (absolute_errors > np.finfo(float).eps)
    errors_percent[near_zero] = np.inf
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
        errors_percent,
        as_1d_array(fit_frequency),
        as_1d_array(fit_impedance),
        as_1d_array(fit_at_data),
    )


def _batch_parameters_with_initials(
    target_parameters: list[ParameterValue],
    source_parameters: list[ParameterValue],
) -> list[ParameterValue]:
    source_by_name = {parameter.name: parameter for parameter in source_parameters}
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
) -> BatchFitReport:
    start_index = project.available_cycles.index(start_cycle)
    cycle_numbers = project.available_cycles[start_index:]
    next_parameters = [
        ParameterValue(
            p.name, p.unit, p.initial, p.lower, p.upper, p.error_percent, p.fixed
        )
        for p in initial_parameters
    ]
    completed: list[BatchCycleFit] = []
    for cycle_number in cycle_numbers:
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
        if [parameter.name for parameter in cycle_parameters] != [
            parameter.name for parameter in next_parameters
        ]:
            return BatchFitReport(
                fits=completed,
                failed_cycle=cycle_number,
                error="The spectrum uses a different fitting model",
            )
        fit_parameters = _batch_parameters_with_initials(
            cycle_parameters,
            next_parameters,
        )
        try:
            fitted, errors_percent, fit_frequency, fit_impedance, fit_at_data = fit_cycle(
                cycle,
                cycle_circuit,
                fit_parameters,
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
            )
        )
        next_parameters = fitted_parameters
    return BatchFitReport(fits=completed)


def batch_fit_spectra(
    targets: list[SpectrumFitTarget],
    initial_parameters: list[ParameterValue],
    *,
    use_target_initial_parameters: bool = False,
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
    expected_names = [parameter.name for parameter in initial_parameters]
    next_parameters = [
        ParameterValue(
            p.name, p.unit, p.initial, p.lower, p.upper, p.error_percent, p.fixed
        )
        for p in initial_parameters
    ]
    completed: list[SpectrumBatchFit] = []
    for target in targets:
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
            if target_circuit != expected_circuit:
                return SpectrumBatchReport(
                    completed,
                    target.label,
                    "The spectrum uses a different fitting model",
                )
            if [parameter.name for parameter in target_parameters] != expected_names:
                return SpectrumBatchReport(
                    completed,
                    target.label,
                    "The spectrum has incompatible fitting parameters",
                )
            fit_circuit = expected_circuit
            fit_parameters = _batch_parameters_with_initials(
                target_parameters,
                next_parameters,
            )
        try:
            fitted, errors_percent, fit_frequency, fit_impedance, fit_at_data = fit_cycle(
                cycle,
                fit_circuit,
                fit_parameters,
            )
        except Exception as error:
            return SpectrumBatchReport(
                completed,
                target.label,
                f"{type(error).__name__}: {error}",
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
                ),
            )
        )
        if not use_target_initial_parameters:
            next_parameters = fitted_parameters
    return SpectrumBatchReport(completed)
