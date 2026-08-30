from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .results_schema import spectrum_identifier

ML_EEC_MODELS = {
    "R0-p(R1,CPE1)",
    "R0-L0-p(R1,CPE1)",
    "R0-p(R1,CPE1)-p(R2,CPE2)",
    "R0-L0-p(R1,CPE1)-p(R2,CPE2)",
    "R0-L0-p(R1,CPE1)-p(R3,CPE3)",
    "R0-p(R1,CPE1)-p(R3,CPE3)",
}


@dataclass
class MLResult:
    spectrum_id: str
    spectrum_key: str | None = None
    source_name: str | None = None
    cycle: int | None = None
    source_project: str | None = None
    control: str | None = None
    frequency_ranges: list[tuple[float, float]] = field(default_factory=list)
    active_mask: np.ndarray | None = None
    outlier_mask: np.ndarray | None = None
    model_circuit: str | None = None
    model_parameters: dict[str, float] = field(default_factory=dict)
    residual_real: np.ndarray | None = None
    residual_imag: np.ndarray | None = None
    topology_prediction: str | None = None
    confidence: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    suggested_eec: str | None = None
    predicted_process_count: int | None = None
    predicted_l0_required: bool | None = None
    initial_sources: dict[str, str] = field(default_factory=dict)
    parameter_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    parameter_reliability: dict[str, str] = field(default_factory=dict)

    @property
    def has_eec_model(self) -> bool:
        return bool(self.model_circuit and self.model_parameters)


def _number(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def _boolean(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    normalized = str(value).strip().casefold()
    if normalized in {"true", "yes", "required", "1", "l0_required"}:
        return True
    if normalized in {"false", "no", "not_required", "0", "l0_not_required"}:
        return False
    return None


def suggested_eec(result: MLResult) -> str | None:
    """Return the validated ML circuit suggestion without fitting anything."""
    explicit = (result.suggested_eec or "").strip()
    if explicit in ML_EEC_MODELS:
        return explicit
    if result.predicted_process_count not in (1, 2):
        return None
    if result.predicted_l0_required is None:
        return None
    prefix = "R0-L0" if result.predicted_l0_required else "R0"
    branches = "p(R1,CPE1)"
    if result.predicted_process_count == 2:
        branches += "-p(R2,CPE2)"
    return f"{prefix}-{branches}"


def _result_for(results: dict[str, MLResult], spectrum_id: str) -> MLResult:
    result = results.get(spectrum_id)
    if result is None:
        result = MLResult(spectrum_id=spectrum_id)
        results[spectrum_id] = result
    return result


def _add_row(results: dict[str, MLResult], row) -> None:
    spectrum_id = _text(row.get("spectrum_id"))
    if not spectrum_id:
        return
    result = _result_for(results, spectrum_id)
    result.cycle = int(_number(row.get("cycle"))) if _number(row.get("cycle")) is not None else result.cycle
    result.source_name = _text(row.get("source_name")) or result.source_name
    result.predicted_process_count = (
        int(_number(row.get("predicted_process_count")))
        if _number(row.get("predicted_process_count")) is not None
        else result.predicted_process_count
    )
    l0 = _boolean(row.get("predicted_l0_required"))
    if l0 is not None:
        result.predicted_l0_required = l0
    result.suggested_eec = (
        _text(row.get("suggested_EEC"))
        or _text(row.get("suggested_eec"))
        or _text(row.get("predicted_eec_model"))
        or result.suggested_eec
    )
    result.source_project = _text(row.get("source_project")) or result.source_project
    result.control = _text(row.get("control")) or result.control
    minimum = _number(row.get("predicted_fmin"))
    maximum = _number(row.get("predicted_fmax"))
    if minimum is not None and maximum is not None and minimum > 0 and maximum > 0:
        frequency_range = tuple(sorted((minimum, maximum)))
        if frequency_range not in result.frequency_ranges:
            result.frequency_ranges.append(frequency_range)
    result.topology_prediction = (
        _text(row.get("topology_prediction"))
        or _text(row.get("predicted_topology"))
        or result.topology_prediction
    )
    confidence = _number(row.get("confidence"))
    if confidence is not None:
        result.confidence = confidence
    result.model_circuit = (
        _text(row.get("ml_eec_model"))
        or _text(row.get("predicted_eec_model"))
        or _text(row.get("eec_model"))
        or result.model_circuit
        or result.topology_prediction
    )
    if result.model_circuit:
        parameter_names = set()
        for name in result.model_circuit.replace("-", "(").replace(")", "(").split("("):
            if name.startswith(("R", "L", "CPE", "W")) and name.isidentifier():
                parameter_names.add(name)
        for name in parameter_names:
            for column in (name, f"parameter_{name}", f"ml_{name}"):
                value = _number(row.get(column))
                if value is not None:
                    result.model_parameters[name] = value
                    break


def _add_json_spectrum(
    results: dict[str, MLResult], spectrum: dict, source_project: str | None
) -> None:
    spectrum_id = _text(spectrum.get("spectrum_id"))
    if not spectrum_id:
        return
    source_name = _text(spectrum.get("source_name"))
    if not source_name:
        parts = spectrum_id.split("::")
        source_name = parts[1] if len(parts) >= 4 else source_project
    row = {
        "spectrum_id": spectrum_id,
        "source_name": source_name,
        "cycle": spectrum.get("cycle"),
        "source_project": source_project,
        "control": _text(spectrum.get("control"))
        or (source_name.rsplit("::", 1)[1] if source_name and "::" in source_name else None),
        "predicted_fmin": spectrum.get("predicted_f_min"),
        "predicted_fmax": spectrum.get("predicted_f_max"),
        "topology_prediction": spectrum.get("hgb_topology")
        or spectrum.get("rf_topology")
        or spectrum.get("predicted_topology"),
        "confidence": spectrum.get("hgb_confidence")
        or spectrum.get("rf_confidence"),
        "predicted_process_count": spectrum.get("predicted_process_count")
        or spectrum.get("process_count"),
        "predicted_l0_required": spectrum.get("predicted_l0_required")
        if "predicted_l0_required" in spectrum
        else spectrum.get("l0_required"),
        "suggested_EEC": spectrum.get("suggested_EEC")
        or spectrum.get("suggested_eec")
        or spectrum.get("predicted_eec_model"),
    }
    _add_row(results, row)
    result = results[spectrum_id]
    result.spectrum_key = (
        _text(spectrum.get("spectrum_key"))
        or _text(spectrum.get("canonical_spectrum_id"))
    )
    if result.spectrum_key is None:
        frequency = spectrum.get("frequency")
        real = spectrum.get("z_real")
        imaginary = spectrum.get("z_imag")
        if frequency is not None and real is not None and imaginary is not None:
            try:
                result.spectrum_key = spectrum_identifier(
                    frequency,
                    real,
                    imaginary,
                    int(spectrum.get("cycle")),
                    str(result.control or "working"),
                )
            except (TypeError, ValueError):
                pass
    if result.predicted_process_count is None:
        topology = _text(row.get("topology_prediction"))
        if topology:
            if topology.casefold().startswith("one_process"):
                result.predicted_process_count = 1
            elif topology.casefold().startswith("two_process"):
                result.predicted_process_count = 2
    metadata = spectrum.get("metadata")
    if isinstance(metadata, dict):
        result.metadata.update(metadata)
    for key in (
        "final_ml_active_mask",
        "stage2_active_mask",
        "ml_frequency_active_mask",
        "ml_envelope_mask",
        "stage1_active_mask",
    ):
        mask = spectrum.get(key)
        if mask is not None:
            result.active_mask = np.asarray(mask, dtype=bool)
            break
    outlier_mask = spectrum.get("deterministic_outlier_mask")
    if outlier_mask is not None:
        result.outlier_mask = np.asarray(outlier_mask, dtype=bool)
    model_parameters = spectrum.get("ml_eec_parameters")
    if isinstance(model_parameters, dict):
        for name, value in model_parameters.items():
            number = _number(value)
            if number is not None:
                result.model_parameters[str(name)] = number
    parameter_predictions = spectrum.get("parameter_predictions")
    if isinstance(parameter_predictions, list):
        for prediction in parameter_predictions:
            if not isinstance(prediction, dict):
                continue
            name = _text(prediction.get("eec_parameter_name")) or _text(prediction.get("parameter_name"))
            lower = _number(prediction.get("lower_limit", prediction.get("lower")))
            upper = _number(prediction.get("upper_limit", prediction.get("upper")))
            if name and lower is not None and upper is not None and lower < upper:
                result.parameter_limits[name] = (lower, upper)
            source = _text(prediction.get("source"))
            if name and source:
                result.initial_sources[name] = source
            initial = _number(prediction.get("initial_value", prediction.get("initial")))
            if name and initial is not None:
                result.model_parameters[name] = initial
            reliability = _text(prediction.get("reliability"))
            if name and reliability:
                result.parameter_reliability[name] = reliability
    model = (
        _text(spectrum.get("predicted_eec_model"))
        or _text(spectrum.get("suggested_EEC"))
    )
    if model:
        result.model_circuit = model


def load_ml_results_payload(payload: dict) -> dict[str, MLResult]:
    """Decode an ML sidecar payload without requiring a file on disk."""
    results: dict[str, MLResult] = {}
    root = payload.get("ml_results", payload)
    source_project = _text(root.get("source_file")) or _text(root.get("source_project"))
    for spectrum in root.get("spectra", []):
        if isinstance(spectrum, dict):
            _add_json_spectrum(results, spectrum, source_project)
    return results


def load_ml_results(directory: Path) -> dict[str, MLResult]:
    """Load optional CSV predictions and cached masks from an ML output folder."""
    directory = Path(directory)
    results: dict[str, MLResult] = {}
    if directory.is_file():
        try:
            payload = json.loads(directory.read_text(encoding="utf-8"))
            if payload.get("format") == "eis-fitting-project":
                embedded = payload.get("ml_results")
                if isinstance(embedded, dict) and isinstance(embedded.get("spectra"), list):
                    source_project = _text(payload.get("source_path"))
                    results = {}
                    for spectrum in embedded["spectra"]:
                        if isinstance(spectrum, dict):
                            _add_json_spectrum(results, spectrum, source_project)
                    return results
                return _load_project_initial_results(payload)
            results = load_ml_results_payload(payload)
        except (OSError, ValueError, TypeError, AttributeError):
            return {}
        return results
    for path in directory.rglob("*.csv"):
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError, UnicodeError):
            continue
        if "spectrum_id" not in frame.columns:
            continue
        for row in frame.to_dict("records"):
            _add_row(results, row)

    for metadata_path in directory.rglob("entries/*.json"):
        try:
            metadata = pd.read_json(metadata_path, typ="series").to_dict()
            mask_path = metadata_path.with_suffix(".npz")
            if not mask_path.exists() or not metadata.get("spectrum_id"):
                continue
            result = _result_for(results, str(metadata["spectrum_id"]))
            result.active_mask = np.asarray(
                np.load(mask_path)["active_mask"], dtype=bool
            )
            minimum = _number(metadata.get("frequency_min"))
            maximum = _number(metadata.get("frequency_max"))
            if minimum is not None and maximum is not None:
                frequency_range = tuple(sorted((minimum, maximum)))
                if frequency_range not in result.frequency_ranges:
                    result.frequency_ranges.append(frequency_range)
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return results


def _load_project_initial_results(payload: dict) -> dict[str, MLResult]:
    """Adapt an ML-initialization project without changing project loading."""
    source_path = _text(payload.get("source_path"))
    control = _text(payload.get("control"))
    default_circuit = _text(payload.get("circuit"))
    results: dict[str, MLResult] = {}
    cycles = payload.get("cycles", {})
    if not isinstance(cycles, dict):
        return results
    for cycle_key, cycle in cycles.items():
        if not isinstance(cycle, dict):
            continue
        try:
            cycle_number = int(cycle_key)
        except (TypeError, ValueError):
            continue
        circuit = _text(cycle.get("circuit")) or default_circuit
        if not circuit:
            continue
        spectrum_id = "::".join(
            part for part in (source_path, control, str(cycle_number)) if part
        )
        result = MLResult(
            spectrum_id=spectrum_id,
            source_name=source_path,
            cycle=cycle_number,
            source_project=source_path,
            control=control,
            model_circuit=circuit,
            suggested_eec=circuit,
            predicted_l0_required="L0" in circuit,
        )
        window = cycle.get("frequency_window")
        if isinstance(window, (list, tuple)) and len(window) == 2:
            minimum, maximum = _number(window[0]), _number(window[1])
            if minimum is not None and maximum is not None:
                result.frequency_ranges.append(tuple(sorted((minimum, maximum))))
        included = cycle.get("manually_included")
        outliers = cycle.get("outliers")
        if isinstance(included, list):
            mask = np.asarray(included, dtype=bool)
            if isinstance(outliers, list) and len(outliers) == len(mask):
                mask &= ~np.asarray(outliers, dtype=bool)
            result.active_mask = mask
        parameters = cycle.get("parameters")
        if not isinstance(parameters, list):
            parameters = payload.get("default_parameters", [])
        for parameter in parameters:
            if not isinstance(parameter, dict):
                continue
            name = _text(parameter.get("name"))
            value = _number(parameter.get("initial"))
            if name and value is not None:
                result.model_parameters[name] = value
                result.initial_sources[name] = "DRT" if name in {"R0", "L0"} else "ML"
        metadata = cycle.get("custom_metadata")
        if isinstance(metadata, dict):
            result.metadata.update(metadata)
        results[spectrum_id] = result
    return results
