"""Apply saved Sputtered cathode ML results, fit, and refine an EIS project."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from eis_gui import EISApplication
from eis_project import load_json_payload, save_project_file
from eis_services import (
    FitOptions,
    circuit_parameters,
    fit_cycle_with_timeout,
    refine_fit_cycle,
    _fit_provenance,
)
from circuit_structure import circuits_equivalent, map_parameter_name, parameter_name_mapping
from ml.gui_results import load_ml_results, suggested_eec
from ml.results_schema import spectrum_identifier


def _set_initial_parameters(cycle, result) -> None:
    circuit = suggested_eec(result) or result.model_circuit
    if not circuit:
        raise ValueError("ML EEC model is unavailable")
    parameters = circuit_parameters(circuit)
    current_model = cycle.model(circuit)
    mapping = parameter_name_mapping(circuit, current_model) if circuits_equivalent(circuit, current_model) else {}
    by_name = {parameter.name: parameter for parameter in parameters}
    for name, value in result.model_parameters.items():
        target_name = map_parameter_name(name, mapping) or name
        parameter = by_name.get(target_name)
        if parameter is None:
            continue
        limits = result.parameter_limits.get(name)
        if limits is not None:
            parameter.lower, parameter.upper = limits
        parameter.initial = float(np.clip(value, parameter.lower, parameter.upper))
    cycle.circuit = circuit
    cycle.parameters = parameters
    cycle.clear_fit()
    cycle.invalidate_drt_cache()


def _apply_fit(cycle, fit_result, *, refinement: dict | None = None) -> None:
    cycle.fit_parameters = np.asarray(fit_result.fitted_parameters, dtype=float)
    cycle.fit_frequency_hz = np.asarray(fit_result.fit_frequency_hz, dtype=float)
    cycle.fit_impedance = np.asarray(fit_result.fit_impedance, dtype=complex)
    cycle.fit_at_data_impedance = np.asarray(fit_result.fit_at_data_impedance, dtype=complex)
    for parameter, value, error in zip(cycle.parameters, fit_result.fitted_parameters, fit_result.errors_percent):
        parameter.initial = float(value)
        parameter.error_percent = float(error)
    cycle.fit_provenance = _fit_provenance(fit_result)
    if refinement is not None:
        cycle.fit_provenance["refinement"] = refinement


def _copy_nearby_initials(target, source) -> None:
    """Copy fitted initials from a nearby spectrum, mapping equivalent EEC names."""
    source_model = source.circuit
    target_model = target.circuit
    mapping = parameter_name_mapping(source_model, target_model) if circuits_equivalent(source_model, target_model) else {}
    source_by_target = {
        map_parameter_name(parameter.name, mapping) or parameter.name: parameter
        for parameter in source.parameters
    }
    for parameter in target.parameters:
        source_parameter = source_by_target.get(parameter.name)
        if source_parameter is None:
            continue
        parameter.initial = float(np.clip(source_parameter.initial, parameter.lower, parameter.upper))


def _voltage_time_distance(target, candidate) -> float:
    voltage = abs(float(target.potential_v or 0.0) - float(candidate.potential_v or 0.0)) / 0.01
    time = abs(float(target.time_s or 0.0) - float(candidate.time_s or 0.0)) / 120.0
    return voltage + time


def run(project: Path, results_path: Path, report_path: Path) -> dict:
    original_payload = load_json_payload(project)
    restored = EISApplication._load_saved_project(project)
    results = load_ml_results(results_path)
    result_by_key = {
        result.spectrum_key: result
        for result in results.values()
        if result.spectrum_key
    }
    expected = []
    raw_arrays = {}
    operations = []
    for dataset_id, loaded, state in restored:
        for spectrum in loaded.spectra:
            cycle = state.cycles[int(spectrum.cycle)]
            key = spectrum_identifier(
                cycle.frequency_hz,
                cycle.impedance.real,
                cycle.impedance.imag,
                cycle.cycle,
                state.control,
            )
            expected.append(key)
            raw_arrays[key] = (cycle.frequency_hz.copy(), cycle.impedance.copy())
            result = result_by_key.get(key)
            if result is None:
                raise ValueError(f"missing ML result for cycle {cycle.cycle} in {loaded.state.source_path.name}")
            operations.append((dataset_id, loaded, state, cycle, result))
    if len(operations) != 288 or len(result_by_key) != 288 or len(set(expected)) != 288:
        raise ValueError(
            f"expected 288 unique matched spectra, got operations={len(operations)}, "
            f"results={len(result_by_key)}, unique_keys={len(set(expected))}"
        )

    fit_options = FitOptions(
        pipeline=("least_squares",),
        population_size=30,
        iterations=200,
        weight_by_modulus=False,
        jacobian_mode="numerical",
    ).validated()
    report = {"spectra": len(operations), "fit": [], "refinement": [], "failures": []}
    successful_cycles = []
    failed_operations = []
    for dataset_id, loaded, state, cycle, result in operations:
        label = f"{loaded.state.source_path.name}, cycle {cycle.cycle}"
        try:
            if not result.frequency_ranges:
                raise ValueError("ML frequency range is unavailable")
            if result.active_mask is None or result.active_mask.size != cycle.frequency_hz.size:
                raise ValueError("ML active-point mask is unavailable or misaligned")
            cycle.frequency_window = tuple(result.frequency_ranges[0])
            cycle.manually_included = result.active_mask.copy()
            cycle.outliers = (
                result.outlier_mask.copy()
                if result.outlier_mask is not None and result.outlier_mask.size == cycle.frequency_hz.size
                else ~cycle.manually_included
            )
            cycle.clear_fit()
            cycle.invalidate_drt_cache()
            _set_initial_parameters(cycle, result)
            fit = fit_cycle_with_timeout(
                cycle,
                cycle.model(state.circuit),
                cycle.parameters,
                10.0,
                fit_options,
            )
            _apply_fit(cycle, fit)
            report["fit"].append({"spectrum": label, "converged": bool(fit.converged), "rmse": float(fit.rmse)})
            refined, removed, iterations = refine_fit_cycle(
                cycle,
                cycle.model(state.circuit),
                copy.deepcopy(cycle.parameters),
                3.5,
                5,
                10.0,
                fit_options,
            )
            valid = removed[(removed >= 0) & (removed < cycle.frequency_hz.size)]
            cycle.manually_included[valid] = False
            cycle.outliers[valid] = True
            cycle.invalidate_drt_cache()
            _apply_fit(
                cycle,
                refined,
                refinement={"z_threshold": 3.5, "max_iterations": 5, "iterations": int(iterations), "removed_points": int(valid.size)},
            )
            report["refinement"].append({"spectrum": label, "converged": bool(refined.converged), "removed_points": int(valid.size), "iterations": int(iterations)})
            successful_cycles.append(cycle)
        except Exception as error:
            failed_operations.append((label, cycle, state, result, error))

    for label, cycle, state, result, first_error in failed_operations:
        if not successful_cycles:
            report["failures"].append({"spectrum": label, "stage": "fit_or_refine", "error": f"{type(first_error).__name__}: {first_error}"})
            continue
        source = min(successful_cycles, key=lambda candidate: _voltage_time_distance(cycle, candidate))
        try:
            _copy_nearby_initials(cycle, source)
            fit = fit_cycle_with_timeout(cycle, cycle.model(state.circuit), cycle.parameters, 10.0, fit_options)
            _apply_fit(cycle, fit)
            report["fit"].append({"spectrum": label, "converged": bool(fit.converged), "rmse": float(fit.rmse), "retry": True, "initial_source_cycle": int(source.cycle), "initial_source_voltage": float(source.potential_v), "initial_source_time": float(source.time_s)})
            refined, removed, iterations = refine_fit_cycle(cycle, cycle.model(state.circuit), copy.deepcopy(cycle.parameters), 3.5, 5, 10.0, fit_options)
            valid = removed[(removed >= 0) & (removed < cycle.frequency_hz.size)]
            cycle.manually_included[valid] = False
            cycle.outliers[valid] = True
            cycle.invalidate_drt_cache()
            _apply_fit(cycle, refined, refinement={"z_threshold": 3.5, "max_iterations": 5, "iterations": int(iterations), "removed_points": int(valid.size), "retry": True, "initial_source_cycle": int(source.cycle)})
            report["refinement"].append({"spectrum": label, "converged": bool(refined.converged), "removed_points": int(valid.size), "iterations": int(iterations), "retry": True})
        except Exception as error:
            report["failures"].append({"spectrum": label, "stage": "retry_fit_or_refine", "error": f"{type(error).__name__}: {error}", "initial_source_cycle": int(source.cycle)})

    for _dataset_id, _loaded, _state, cycle, _result in operations:
        frequency, impedance = raw_arrays[spectrum_identifier(cycle.frequency_hz, cycle.impedance.real, cycle.impedance.imag, cycle.cycle, _state.control)]
        if not np.array_equal(frequency, cycle.frequency_hz) or not np.array_equal(impedance, cycle.impedance):
            raise ValueError(f"raw data changed for cycle {cycle.cycle}")
    save_project_file(
        restored[0][2],
        project,
        datasets=[(dataset_id, state, loaded.dataframe) for dataset_id, loaded, state in restored],
        procedure_blocks=original_payload.get("procedure_blocks"),
        procedures=original_payload.get("procedures"),
    )
    report["fit_count"] = len(report["fit"])
    report["refinement_count"] = len(report["refinement"])
    report["failure_count"] = len(report["failures"])
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--report", type=Path, default=Path("467_III_cathode_etching_series_20min_Cell_ml_fit_report.json"))
    args = parser.parse_args()
    report = run(args.project, args.results, args.report)
    print(json.dumps({key: report[key] for key in ("spectra", "fit_count", "refinement_count", "failure_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
