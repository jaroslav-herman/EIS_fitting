"""Create sample-178 ML EEC initial guesses without performing EEC fitting.

Frequency limits and deterministic active masks are consumed from the already
generated sample-178 preprocessing artifact.  Saved topology and Stage 4B
models are used for inference.  Bayes-DRT2 is called only through ``ridge_fit``
to estimate R0/L0; its outlier checker is never called here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from eis_services import circuit_parameters

from .dataset import SpectrumRecord, load_eisfit_projects
from .evaluate_stage4_parameters import extract_parameter_targets
from .evaluate_stage4_parameters import load_automatic_masks
from .l0_decision import (
    _reference_l0_values,
    calibrate_l0_rule,
    decide_l0,
    high_frequency_inductive_diagnostic,
)
from .parameter_prediction import ALPHA_PARAMETERS, inverse_target, transform_target
from .parameter_limits import build_limit_strategy, make_parameter_limit
from .preprocessing import SpectrumPreprocessor
from .run_stage4b_parameters import (
    ABSOLUTE_FEATURE_NAMES,
    TRAINING_SAMPLES,
    _features,
    _load_or_create_stage4b_cache,
    _load_stage4a_cache,
    _manual_records,
    calculate_absolute_impedance_features,
)
from .results_schema import ML_RESULTS_FORMAT, ML_RESULTS_VERSION, spectrum_identifier


INFERENCE_SAMPLE = "178"
PREPROCESSED_INPUT = Path("ml/analysis/unseen_178_preprocessed/178_ML_preprocessed_results.eisfit.json")
OUTPUT_DIR = Path("ml/analysis/unseen_178_initial_parameters")
STAGE4B_MODEL_DIR = Path("ml/analysis/stage4b_parameters/models")
STAGE4A_CACHE = Path("ml/cache/stage4a_parameter_features.npz")
STAGE4B_CACHE = Path("ml/cache/stage4b_parameter_features.npz")
STAGE4B_METADATA = Path("ml/cache/stage4b_parameter_metadata.csv")
TOPOLOGY_DIR = Path("ml/analysis/unseen_178_new_topology")
FREQUENCY_MODEL = Path("ml/analysis/unseen_178_new_frequency/final_frequency_model.joblib")
TOPOLOGY_MODEL = TOPOLOGY_DIR / "final_topology_hgb.joblib"
TOPOLOGY_PREPROCESSOR = TOPOLOGY_DIR / "topology_preprocessor.joblib"

PARAMETERS = ("R1", "Q1", "R2", "Q2", "alpha1", "alpha2")
SELECTED_MODELS = {
    "R1": ("RIDGE", "C"),
    "Q1": ("RIDGE", "B"),
    "R2": ("HGB", "A"),
    "Q2": ("RIDGE", "C"),
    "alpha1": ("RIDGE", "A"),
    "alpha2": ("HGB", "B"),
}


def build_eec_model(process_count: int, l0_required: bool) -> str:
    if process_count not in (1, 2):
        raise ValueError("process_count must be 1 or 2")
    prefix = "R0-L0" if l0_required else "R0"
    branches = "p(R1,CPE1)"
    if process_count == 2:
        branches += "-p(R2,CPE2)"
    return f"{prefix}-{branches}"


def l0_status_from_ridge(inductance) -> tuple[str, float | None, bool]:
    """Interpret the existing Ridge-DRT inductance without inventing L0."""
    try:
        value = float(inductance)
    except (TypeError, ValueError):
        return "unavailable", None, False
    if not np.isfinite(value):
        return "unavailable", None, False
    if value <= 0:
        return "not_required", None, False
    return "required", value, True


def inverse_stage4b_prediction(raw_prediction, parameter: str) -> float:
    value = float(inverse_target(np.asarray([raw_prediction], dtype=float), parameter)[0])
    if not np.isfinite(value):
        raise ValueError(f"non-finite prediction for {parameter}")
    if parameter in ALPHA_PARAMETERS and not 0.0 < value < 1.0:
        raise ValueError(f"invalid alpha prediction for {parameter}: {value}")
    if parameter not in ALPHA_PARAMETERS and value <= 0:
        raise ValueError(f"invalid positive prediction for {parameter}: {value}")
    return value


def _record_from_item(item: dict, source_file: str) -> SpectrumRecord:
    frequency = np.asarray(item["frequency"], dtype=float)
    real = np.asarray(item["z_real"], dtype=float)
    imag = np.asarray(item["z_imag"], dtype=float)
    if not (frequency.size == real.size == imag.size):
        raise ValueError(f"raw arrays have incompatible lengths for {item.get('spectrum_id')}")
    return SpectrumRecord(
        spectrum_id=str(item["spectrum_id"]), source_project=source_file,
        sample_id=INFERENCE_SAMPLE, cycle=int(item["cycle"]),
        voltage=item.get("voltage"), current=item.get("current"), time=item.get("time"),
        frequency=frequency, z_real=real, z_imag=imag,
        topology_label="R0-p(R1,CPE1)", source_name=str(item.get("source_name") or ""),
        metadata=dict(item.get("metadata") or {}),
    )


def _feature_names(configuration: str) -> list[str]:
    if configuration == "A":
        extras = ["voltage"]
    elif configuration == "B":
        extras = ["voltage", "current"]
    elif configuration == "C":
        extras = ["voltage", "current", *ABSOLUTE_FEATURE_NAMES]
    else:
        raise ValueError(configuration)
    return [f"stage4a_{i:03d}" for i in range(192)] + extras


def _model_parameters(model_name: str) -> dict[str, float | int]:
    if model_name == "RIDGE":
        return {"alpha": 1.0}
    if model_name == "HGB":
        return {"max_iter": 12, "learning_rate": 0.06, "l2_regularization": 1.0, "random_state": 42}
    raise ValueError(model_name)


def _make_model(model_name: str):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if model_name == "RIDGE":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_name == "HGB":
        return HistGradientBoostingRegressor(**_model_parameters(model_name))
    raise ValueError(model_name)


def _training_context(training_projects: list[Path]):
    mapping = {str(path): path.name.split(".")[0] for path in training_projects}
    extraction = load_eisfit_projects(training_projects, mapping, require_fit=True)
    records = extraction.records
    if tuple(sorted({r.sample_id for r in records})) != tuple(sorted(TRAINING_SAMPLES)):
        raise RuntimeError("Stage 5A training context does not contain exactly the six approved samples")
    if INFERENCE_SAMPLE in {r.sample_id for r in records}:
        raise RuntimeError("sample 178 entered the Stage 5A model materialization context")
    cleaned = _manual_records(training_projects, records)
    base_x = _load_stage4a_cache(STAGE4A_CACHE, [r.spectrum_id for r in records])
    targets = extract_parameter_targets(training_projects, records)
    extra_x, _ = _load_or_create_stage4b_cache(records, cleaned, targets, STAGE4B_CACHE, STAGE4B_METADATA)
    metadata = pd.DataFrame({
        "spectrum_id": [r.spectrum_id for r in records], "sample_id": [r.sample_id for r in records],
        "voltage": [r.voltage for r in records], "current": [r.current for r in records],
        **{name: extra_x[:, i] for i, name in enumerate(ABSOLUTE_FEATURE_NAMES)},
        **{parameter: targets[parameter].to_numpy() for parameter in ("R0", "R1", "Q1", "alpha1", "R2", "Q2", "alpha2")},
    })
    spectrum_preprocessor = SpectrumPreprocessor(grid_size=64, use_metadata=False, spectrum_mode="raw")
    reconstructed = spectrum_preprocessor.fit_transform(cleaned)
    if not np.allclose(reconstructed, base_x, equal_nan=True, rtol=1e-10, atol=1e-12):
        raise RuntimeError("Stage 4A cache cannot be reproduced from the approved manual training context")
    return records, cleaned, base_x, extra_x, metadata, spectrum_preprocessor


def materialize_stage4b_models(training_projects: list[Path], model_dir: Path = STAGE4B_MODEL_DIR) -> dict[str, dict]:
    """Materialize only the missing selected Stage 4B artifacts from cached data."""
    records, cleaned, base_x, extra_x, metadata, _ = _training_context(training_projects)
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for parameter, (model_name, configuration) in SELECTED_MODELS.items():
        path = model_dir / f"{parameter}_{model_name.lower()}.joblib"
        if path.exists():
            bundle = joblib.load(path)
            if bundle.get("parameter") == parameter and bundle.get("feature_configuration") == configuration and bundle.get("model_type") == model_name:
                manifest[parameter] = {"path": str(path), "model": model_name, "feature_configuration": configuration, "materialized": False, "training_spectra": bundle.get("training_spectra")}
                continue
        indices = np.flatnonzero(metadata[parameter].notna().to_numpy())
        selected_metadata = metadata.iloc[indices].reset_index(drop=True)
        selected_base = base_x[indices]
        selected_extra = extra_x[indices]
        selected_indices = np.arange(len(indices))
        (x_selected, _), _ = _features(selected_base, selected_extra, selected_metadata, selected_indices, selected_indices, configuration)
        y = selected_metadata[parameter].to_numpy(float)
        model = _make_model(model_name)
        model.fit(x_selected, transform_target(y, parameter))
        bundle = {
            "model": model, "parameter": parameter, "model_type": model_name,
            "feature_configuration": configuration, "feature_names": _feature_names(configuration),
            "target_transformation": "log10" if parameter not in ALPHA_PARAMETERS else "logit_alpha",
            "model_parameters": _model_parameters(model_name), "training_samples": list(TRAINING_SAMPLES),
            "training_spectra": int(len(indices)), "materialized_from_completed_stage4b": True,
        }
        joblib.dump(bundle, path)
        manifest[parameter] = {"path": str(path), "model": model_name, "feature_configuration": configuration, "materialized": True, "training_spectra": int(len(indices))}
    return {"manifest": manifest, "context": (records, cleaned, base_x, extra_x, metadata)}


def _inference_features(context, active_records: list[SpectrumRecord], training_metadata: pd.DataFrame, training_base: np.ndarray, training_extra: np.ndarray, spectrum_preprocessor: SpectrumPreprocessor):
    test_base = spectrum_preprocessor.transform(active_records)
    test_extra_rows = [calculate_absolute_impedance_features(record) for record in active_records]
    test_extra = np.asarray([[row[name] for name in ABSOLUTE_FEATURE_NAMES] for row in test_extra_rows], dtype=float)
    test_metadata = pd.DataFrame({
        "voltage": [record.voltage for record in active_records], "current": [record.current for record in active_records],
        **{name: test_extra[:, i] for i, name in enumerate(ABSOLUTE_FEATURE_NAMES)},
    })
    combined_metadata = pd.concat([training_metadata.reset_index(drop=True), test_metadata], ignore_index=True)
    combined_base = np.vstack([training_base, test_base])
    combined_extra = np.vstack([training_extra, test_extra])
    train_indices = np.arange(len(training_metadata))
    test_indices = np.arange(len(training_metadata), len(combined_metadata))
    result = {}
    for configuration in ("A", "B", "C"):
        (train_features, test_features), _ = _features(combined_base, combined_extra, combined_metadata, train_indices, test_indices, configuration)
        # The model pipelines perform their own train-only StandardScaler fit;
        # these arrays only reproduce the Stage 4B feature ordering/fill rules.
        result[configuration] = (train_features, test_features)
    return result


def _predict_topology(active_records: list[SpectrumRecord]):
    preprocessor = joblib.load(TOPOLOGY_PREPROCESSOR)
    model = joblib.load(TOPOLOGY_MODEL)
    x = preprocessor.transform(active_records)
    labels = [str(value) for value in model.predict(x)]
    probabilities = model.predict_proba(x) if hasattr(model, "predict_proba") else None
    classes = [str(value) for value in getattr(model, "classes_", ("ONE_PROCESS", "TWO_PROCESS"))]
    diagnostics = []
    for index, label in enumerate(labels):
        probs = {name: float(probabilities[index, classes.index(name)]) if probabilities is not None and name in classes else 0.0 for name in ("ONE_PROCESS", "TWO_PROCESS")}
        diagnostics.append({"topology": label, "confidence": max(probs.values()), "probabilities": probs})
    return diagnostics


def _ridge_r0_l0(active_record: SpectrumRecord):
    from bayes_drt2.inversion import Inverter
    frequency, impedance = active_record.arrays("raw")
    if frequency.size < 3:
        return {"R0": None, "L0": None, "L0_status": "unavailable", "error": "fewer than three active points"}
    try:
        inverter = Inverter()
        # This is the application's Ridge-DRT fit path.  Crucially, it does
        # not call check_outliers and receives only the already active points.
        inverter.ridge_fit(frequency, impedance)
        r0 = float(inverter.R_inf)
        if not np.isfinite(r0) or r0 <= 0:
            r0 = None
        status, l0, required = l0_status_from_ridge(inverter.inductance)
        return {"R0": r0, "L0": l0, "L0_status": status, "L0_required": required, "ridge_fit": True, "error": None}
    except Exception as error:
        return {"R0": None, "L0": None, "L0_status": "unavailable", "L0_required": False, "ridge_fit": False, "error": f"{type(error).__name__}: {error}"}


def _parameter_records(model_bundles, feature_arrays, metadata_rows: list[dict], topology_labels: list[str]):
    predictions = {parameter: [] for parameter in PARAMETERS}
    for parameter, (model_name, configuration) in SELECTED_MODELS.items():
        bundle = model_bundles[parameter]
        model = bundle["model"]
        x = feature_arrays[configuration][1]
        transformed = model.predict(x)
        predictions[parameter] = [inverse_stage4b_prediction(value, parameter) for value in transformed]
    return predictions


def _parameter_specs(circuit: str, values: dict[str, float], sources: dict[str, str], model_info: dict[str, dict], l0_status: str, limit_strategy: dict[str, dict]):
    definitions = {parameter.name: parameter for parameter in circuit_parameters(circuit)}
    canonical_to_actual = {"R0": "R0", "L0": "L0", "R1": "R1", "Q1": "CPE1_0", "alpha1": "CPE1_1", "R2": "R2", "Q2": "CPE2_0", "alpha2": "CPE2_1"}
    result = []
    for canonical, actual in canonical_to_actual.items():
        if actual not in definitions or canonical not in values:
            continue
        raw = float(values[canonical])
        limits = make_parameter_limit(canonical, raw, definitions[actual].lower, definitions[actual].upper, limit_strategy[canonical])
        limits["source"] = sources[canonical]
        info = {"parameter_name": canonical, "eec_parameter_name": actual, "raw_prediction": raw, **limits}
        if canonical == "L0":
            info["L0_status"] = l0_status
        if canonical in model_info:
            info.update(model_info[canonical])
        result.append(info)
    return result


def _dataset_key_from_spectrum_id(spectrum_id: str):
    parts = str(spectrum_id).split("::")
    if len(parts) < 4:
        raise ValueError(f"invalid spectrum ID: {spectrum_id}")
    return "::".join(parts[1:-2]), parts[-2], int(parts[-1])


def _validate_output(path: Path, results_path: Path, results_by_id: dict[str, dict]):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "eis-fitting-project":
        raise RuntimeError("generated file is not an EIS-fitting project")
    for dataset in payload.get("datasets", []):
        state = dataset["state"]
        for cycle_text, saved in state.get("cycles", {}).items():
            cycle = int(cycle_text)
            if not saved.get("parameters"):
                continue
            names = {str(value["name"]) for value in saved["parameters"]}
            expected = {"R0", "R1", "CPE1_0", "CPE1_1"}
            if "p(R2,CPE2)" in str(saved.get("circuit")):
                expected |= {"R2", "CPE2_0", "CPE2_1"}
            if "-L0-" in str(saved.get("circuit")):
                expected.add("L0")
            if not expected.issubset(names):
                raise RuntimeError(f"cycle {cycle} lacks required serialized parameters")
            if len(saved["manually_included"]) != len(saved["outliers"]):
                raise RuntimeError(f"cycle {cycle} masks have different lengths")
            circuit_parameters(str(saved["circuit"]))
    ml_results = json.loads(results_path.read_text(encoding="utf-8"))
    if ml_results.get("format") != ML_RESULTS_FORMAT:
        raise RuntimeError("generated ML file has an invalid format")
    if len(ml_results.get("spectra", [])) != len(results_by_id):
        raise RuntimeError("ML result spectrum count does not match predictions")
    loaded = load_eisfit_projects([path], {str(path): INFERENCE_SAMPLE}, require_fit=False)
    if len(loaded.records) != len(results_by_id):
        raise RuntimeError("generated project did not deserialize all spectra")
    return len(loaded.records)


def run(training_projects: list[Path], input_path: Path = PREPROCESSED_INPUT, output: Path = OUTPUT_DIR):
    started = time.perf_counter()
    materialized = materialize_stage4b_models(training_projects)
    records, cleaned, base_x, extra_x, training_metadata = materialized["context"]
    limit_strategy = build_limit_strategy(training_metadata)
    automatic_masks = load_automatic_masks(records)
    reference_l0_values = _reference_l0_values(training_projects, records)
    l0_calibration = calibrate_l0_rule(records, automatic_masks, reference_l0_values, output=output / "l0_decision_validation")
    _, _, _, _, _, spectrum_preprocessor = _training_context(training_projects)
    model_bundles = {parameter: joblib.load(STAGE4B_MODEL_DIR / f"{parameter}_{model_name.lower()}.joblib") for parameter, (model_name, _configuration) in SELECTED_MODELS.items()}
    frequency_bundle = joblib.load(FREQUENCY_MODEL)
    topology_model = joblib.load(TOPOLOGY_MODEL)
    topology_preprocessor = joblib.load(TOPOLOGY_PREPROCESSOR)
    if not isinstance(frequency_bundle, dict) or not frequency_bundle.get("model"):
        raise RuntimeError("saved frequency model artifact is invalid")
    input_payload = json.loads(input_path.read_text(encoding="utf-8"))
    input_root = input_payload.get("ml_results", input_payload)
    items = input_root.get("spectra", [])
    if not items:
        raise RuntimeError("sample-178 preprocessing artifact contains no spectra")
    records_178 = [_record_from_item(item, str(input_root.get("source_file") or input_path)) for item in items]
    active_records = []
    for item, record in zip(items, records_178):
        final = np.asarray(item.get("final_ml_active_mask"), dtype=bool)
        if final.size != record.frequency.size or final.sum() < 3:
            raise RuntimeError(f"invalid final active mask for {record.spectrum_id}")
        active_records.append(SpectrumRecord(
            spectrum_id=record.spectrum_id, source_project=record.source_project, sample_id=record.sample_id,
            cycle=record.cycle, voltage=record.voltage, current=record.current, time=record.time,
            frequency=record.frequency[final], z_real=record.z_real[final], z_imag=record.z_imag[final],
            topology_label="R0-p(R1,CPE1)", source_name=record.source_name, metadata=record.metadata,
        ))
    topology_x = topology_preprocessor.transform(active_records)
    topology_labels = [str(value) for value in topology_model.predict(topology_x)]
    topology_probabilities = topology_model.predict_proba(topology_x) if hasattr(topology_model, "predict_proba") else None
    topology_classes = [str(value) for value in getattr(topology_model, "classes_", ("ONE_PROCESS", "TWO_PROCESS"))]
    feature_arrays = _inference_features(None, active_records, training_metadata, base_x, extra_x, spectrum_preprocessor)
    parameter_predictions = _parameter_records(model_bundles, feature_arrays, [], topology_labels)
    results = []
    raw_by_id = {}
    for index, (item, record, active_record, topology) in enumerate(zip(items, records_178, active_records, topology_labels)):
        final_mask = np.asarray(item["final_ml_active_mask"], dtype=bool)
        deterministic_mask = np.asarray(item.get("deterministic_outlier_mask", np.zeros(final_mask.size, dtype=bool)), dtype=bool)
        ridge = _ridge_r0_l0(active_record)
        hf_diagnostic = high_frequency_inductive_diagnostic(
            record.frequency, record.impedance,
            final_mask & ~deterministic_mask,
        )
        l0_decision = decide_l0(
            hf_diagnostic,
            strength_threshold=l0_calibration["threshold"],
        )
        l0_required = bool(l0_decision["l0_required"])
        if ridge["L0_status"] == "unavailable":
            l0_status = "unavailable"
            l0_required = False
            l0_decision["l0_reason"] = "DRT_RIDGE_unavailable_after_deterministic_decision"
        else:
            l0_status = "required" if l0_required else "not_required"
        process_count = 2 if topology == "TWO_PROCESS" else 1
        circuit = build_eec_model(process_count, l0_required)
        values = {"R0": ridge["R0"]} if ridge["R0"] is not None else {}
        sources = {"R0": "DRT_RIDGE"}
        model_info = {}
        for parameter in ("R1", "Q1", "alpha1") + (("R2", "Q2", "alpha2") if process_count == 2 else ()):
            values[parameter] = parameter_predictions[parameter][index]
            sources[parameter] = "ML_STAGE4B_INITIAL_GUESS"
            model_name, configuration = SELECTED_MODELS[parameter]
            model_info[parameter] = {"model_name": model_name, "model_file": str(STAGE4B_MODEL_DIR / f"{parameter}_{model_name.lower()}.joblib"), "feature_configuration": configuration, "preprocessing_version": "stage5a_from_existing_stage4b_cache"}
        if l0_required and ridge.get("L0") is not None:
            values["L0"] = ridge["L0"]; sources["L0"] = "DRT_RIDGE"
        parameter_specs = _parameter_specs(circuit, values, sources, model_info, l0_status, limit_strategy)
        if any(spec["parameter_name"] in {"R0", "R1", "Q1", "alpha1"} for spec in parameter_specs) is False:
            raise RuntimeError(f"required one-process parameters missing for {record.spectrum_id}")
        if process_count == 2 and not {"R2", "Q2", "alpha2"}.issubset({spec["parameter_name"] for spec in parameter_specs}):
            raise RuntimeError(f"required two-process parameters missing for {record.spectrum_id}")
        if ridge["R0"] is None:
            raise RuntimeError(f"R0 unavailable for {record.spectrum_id}: {ridge.get('error')}")
        probabilities = {name: float(topology_probabilities[index, topology_classes.index(name)]) if topology_probabilities is not None and name in topology_classes else 0.0 for name in ("ONE_PROCESS", "TWO_PROCESS")}
        actual_parameters = [{"name": spec["eec_parameter_name"], "unit": next(p.unit for p in circuit_parameters(circuit) if p.name == spec["eec_parameter_name"]), "initial": spec["initial_value"], "lower": spec["lower"], "upper": spec["upper"], "error_percent": None, "fixed": False} for spec in parameter_specs]
        result = dict(item)
        result.update({
            "predicted_topology": topology, "recommended_topology": topology, "predicted_process_count": process_count,
            "topology_model": "hist_gradient_boosting", "topology_source": "ML_TOPOLOGY_SAVED", "topology_prediction_source": "ML_TOPOLOGY_SAVED", "topology_probability": probabilities,
            "prediction_probability": max(probabilities.values()), "predicted_L0_required": l0_required, "L0_prediction_status": "DETERMINISTIC_HIGH_FREQUENCY_SIGNATURE",
            "L0_status": l0_status, "l0_required": l0_required, "l0_reason": l0_decision["l0_reason"],
            "l0_diagnostics": {**hf_diagnostic, **l0_decision},
            "predicted_eec_model": circuit, "suggested_EEC": circuit,
            "ml_eec_parameters": {spec["eec_parameter_name"]: spec["initial_value"] for spec in parameter_specs},
            "parameter_predictions": parameter_specs, "eec_parameters": actual_parameters,
            "r0_l0_drt_ridge": {"R0": ridge["R0"], "L0": ridge.get("L0"), "L0_status": ridge["L0_status"], "used_as_l0": l0_required, "error": ridge.get("error")},
            "final_ml_active_mask": final_mask.tolist(), "deterministic_outlier_mask": deterministic_mask.tolist(),
            "initial_guess_only": True, "automatic_eec_fit": False,
        })
        results.append(result); raw_by_id[record.spectrum_id] = result
    output.mkdir(parents=True, exist_ok=True)
    raw_source_path = Path(input_root.get("source_file") or "")
    if not raw_source_path.exists():
        raise FileNotFoundError(f"original sample-178 source file not found: {raw_source_path}")
    raw_payload = json.loads(raw_source_path.read_text(encoding="utf-8"))
    original_metadata = {}
    for dataset_index, dataset in enumerate(raw_payload.get("datasets", [])):
        dataset_id = str(dataset.get("dataset_id") or f"dataset_{dataset_index}")
        state = dataset.get("state", {})
        control = str(state.get("control", "working"))
        for cycle_text, saved in state.get("cycles", {}).items():
            original_metadata[(dataset_id, control, int(cycle_text))] = dict(saved.get("custom_metadata") or {})
    for result in results:
        key = _dataset_key_from_spectrum_id(result["spectrum_id"])
        result["metadata"] = {**original_metadata.get(key, {}), **dict(result.get("metadata") or {})}
    for result in results:
        result["spectrum_key"] = spectrum_identifier(
            result.get("frequency", []), result.get("z_real", []), result.get("z_imag", []),
            int(result["cycle"]), str(result.get("control") or "working"),
        )
    ml_payload = {
        "format": ML_RESULTS_FORMAT,
        "version": ML_RESULTS_VERSION,
        "source_project": str(output / "178.eisfit.json"),
        "schema_version": "stage5a-1.0",
        "source_file": str(raw_source_path), "inference_sample": INFERENCE_SAMPLE,
        "training_samples": list(TRAINING_SAMPLES), "initial_guess_only": True, "automatic_eec_fit": False,
        "frequency_model": str(FREQUENCY_MODEL), "frequency_preprocessing_input": str(input_path),
        "deterministic_preprocessing": input_root.get("stage2_configuration"), "topology_model": str(TOPOLOGY_MODEL),
        "topology_preprocessor": str(TOPOLOGY_PREPROCESSOR), "parameter_models": {p: {"path": str(STAGE4B_MODEL_DIR / f"{p}_{m.lower()}.joblib"), "model": m, "feature_configuration": c} for p, (m, c) in SELECTED_MODELS.items()},
        "drt_ridge": {"backend": "bayes_drt2.Inverter.ridge_fit", "used_only_for": ["R0", "L0"], "outlier_selection": False},
        "l0_decision": {"method": "deterministic_high_frequency_signature", "calibration": l0_calibration["report"], "validation_directory": str(output / "l0_decision_validation")},
        "parameter_limit_strategy": limit_strategy,
        "spectra": results,
    }
    output_path = output / "178.eisfit.json"
    results_path = output / "178_ml_results.json"
    output_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
    for result in ml_payload["spectra"]:
        for key in ("frequency", "z_real", "z_imag"):
            result.pop(key, None)
    results_path.write_text(json.dumps(ml_payload, indent=2), encoding="utf-8")
    rows = []
    for result in results:
        by_name = {spec["parameter_name"]: spec for spec in result["parameter_predictions"]}
        metadata = result.get("metadata") or {}
        row = {"source_name": result.get("source_name"), "cycle": result.get("cycle"), "spectrum_id": result.get("spectrum_id"), "voltage": result.get("voltage"), "current": result.get("current"), "Time": metadata.get("Time", result.get("time")), "Cycle mod 15": metadata.get("Cycle mod 15"), "predicted_f_min": result.get("predicted_f_min"), "predicted_f_max": result.get("predicted_f_max"), "topology": result["predicted_topology"], "L0_status": result["L0_status"], "l0_reason": result.get("l0_reason"), "hf_inductive_strength": result.get("l0_diagnostics", {}).get("high_frequency_inductive_strength"), "hf_negative_fraction": result.get("l0_diagnostics", {}).get("negative_imaginary_fraction"), "hf_negative_consecutive": result.get("l0_diagnostics", {}).get("negative_imaginary_consecutive_points")}
        for parameter in ("R0", "L0", "R1", "Q1", "alpha1", "R2", "Q2", "alpha2"):
            row[f"{parameter}_initial"] = by_name.get(parameter, {}).get("initial_value")
            row[f"{parameter}_lower_limit"] = by_name.get(parameter, {}).get("lower_limit")
            row[f"{parameter}_upper_limit"] = by_name.get(parameter, {}).get("upper_limit")
            row[f"{parameter}_source"] = by_name.get(parameter, {}).get("source")
            row[f"{parameter}_reliability"] = by_name.get(parameter, {}).get("reliability")
        rows.append(row)
    predictions_path = output / "predictions.csv"; pd.DataFrame(rows).to_csv(predictions_path, index=False)
    counts = pd.Series([r["predicted_topology"] for r in results]).value_counts().to_dict()
    l0_counts = pd.Series([r["L0_status"] for r in results]).value_counts().to_dict()
    report = {
        "number_of_spectra": len(results), "one_process": int(counts.get("ONE_PROCESS", 0)), "two_process": int(counts.get("TWO_PROCESS", 0)),
        "l0_required": int(l0_counts.get("required", 0)), "l0_not_required": int(l0_counts.get("not_required", 0)), "l0_unavailable": int(l0_counts.get("unavailable", 0)),
        "parameter_prediction_availability": {p: int(sum(p in {x["parameter_name"] for x in r["parameter_predictions"]} for r in results)) for p in ("R0", "L0", *PARAMETERS)},
        "input_preprocessed_file": str(input_path), "output_file": str(output_path), "ml_results_file": str(results_path), "predictions_csv": str(predictions_path),
        "frequency_model": str(FREQUENCY_MODEL), "topology_model": str(TOPOLOGY_MODEL), "parameter_models": materialized["manifest"],
        "training_samples": list(TRAINING_SAMPLES), "sample_178_used_for_training": False, "automatic_eec_fit": False,
        "bayes_drt2_used_only_for_r0_l0": True, "bayes_drt2_used_for_outliers": False, "raw_source_preserved": True,
        "l0_decision_validation": l0_calibration["report"], "parameter_limit_strategy": limit_strategy,
        "warnings": ["All stored values are initial guesses only.", "No conventional EEC fitting was started.", "L0 is omitted when the deterministic high-frequency rule does not require it."] ,
        "runtime_s": time.perf_counter() - started,
    }
    report_path = output / "report.json"; report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["deserialized_spectra"] = _validate_output(output_path, results_path, raw_by_id); report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_projects", nargs=6, type=Path)
    parser.add_argument("--input", type=Path, default=PREPROCESSED_INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(list(args.training_projects), args.input, args.output)


if __name__ == "__main__":
    main()
