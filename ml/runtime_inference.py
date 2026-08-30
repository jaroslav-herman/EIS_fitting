"""Runtime ML inference for spectra already open in the EIS application.

The application uses the non-selected, labelled spectra as a small local
training set.  Selected spectra are never included in fitting the runtime
models.  This is deliberately conservative: unavailable labels are reported
instead of being replaced with fabricated predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from circuit_structure import map_parameter_name, parameter_name_mapping
from eis_services import circuit_parameters

from .dataset import SpectrumRecord, canonical_electrochemical_topology
from .l0_decision import decide_l0, high_frequency_inductive_diagnostic
from .point_validity import detect_valid_points
from .preprocessing import SpectrumPreprocessor
from .frequency_range import _features, _fit_predict, _models as range_models, _targets
from .results_schema import spectrum_identifier, write_ml_results
from .topology_classifier import _models as topology_models
from .automatic_preprocessing import conservative_mask
from .evaluate_frequency_limit_ml import _clean_record
from .frequency_limit_ml import SpectrumFeatureExtractor
from .run_stage4b_parameters import calculate_absolute_impedance_features
from .parameter_prediction import inverse_target


@dataclass(frozen=True)
class RuntimeSpectrum:
    key: str
    record: SpectrumRecord
    circuit: str
    fitted_values: dict[str, float]


@dataclass(frozen=True)
class PretrainedArtifacts:
    frequency: Path
    topology: Path
    topology_preprocessor: Path
    parameter_models: dict[str, Path]
    fallback_parameters: dict[str, Path]
    topology_model_name: str


def discover_pretrained_artifacts(root: Path | None = None) -> tuple[PretrainedArtifacts | None, list[str]]:
    """Find the copied six-sample bundle without requiring user configuration."""
    root = Path(root or Path(__file__).resolve().parent / "analysis")
    frequency = root / "unseen_178_new_frequency" / "final_frequency_model.joblib"
    topology_dir = root / "unseen_178_new_topology"
    topology_hgb = topology_dir / "final_topology_hgb.joblib"
    topology_rf = topology_dir / "final_topology_rf.joblib"
    topology = None
    topology_name = ""
    for candidate, name in ((topology_hgb, "hist_gradient_boosting"), (topology_rf, "random_forest")):
        if not candidate.exists():
            continue
        try:
            _load_artifact(candidate)
        except Exception:
            continue
        topology, topology_name = candidate, name
    preprocessor = topology_dir / "topology_preprocessor.joblib"
    stage4b = root / "stage4b_parameters" / "models"
    stage4a = root / "stage4a_parameters" / "models"
    selected = {"R1": "R1_ridge.joblib", "Q1": "Q1_ridge.joblib", "R2": "R2_hgb.joblib", "Q2": "Q2_ridge.joblib", "alpha1": "alpha1_ridge.joblib", "alpha2": "alpha2_hgb.joblib"}
    parameters = {}
    for name, filename in selected.items():
        candidate = stage4b / filename
        if not candidate.exists():
            continue
        try:
            _load_artifact(candidate)
        except Exception:
            continue
        parameters[name] = candidate
    fallback = {name: stage4a / f"{name}_ridge.joblib" for name in selected if (stage4a / f"{name}_ridge.joblib").exists()}
    missing = []
    for label, path in (("frequency model", frequency), ("topology model", topology), ("topology preprocessor", preprocessor)):
        if path is None or not path.exists():
            missing.append(str(path))
    for name in selected:
        if name not in parameters and name not in fallback:
            missing.append(f"parameter model {name}")
    if missing:
        return None, missing
    return PretrainedArtifacts(frequency, topology, preprocessor, parameters, fallback, topology_name), []


def _load_artifact(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def _safe_voltage(value) -> float:
    value = _finite(value)
    return value if value is not None else 0.0


def infer_pretrained(artifacts: PretrainedArtifacts, targets: list[RuntimeSpectrum], *, operations: set[str]) -> list[dict]:
    """Run inference using the serialized six-sample artifacts only."""
    frequency_bundle = _load_artifact(artifacts.frequency)
    frequency_model = frequency_bundle["model"] if isinstance(frequency_bundle, dict) else frequency_bundle
    frequency_config = frequency_bundle.get("config", {}) if isinstance(frequency_bundle, dict) else {}
    extractor = SpectrumFeatureExtractor(int(frequency_config.get("grid_size", 16)))
    extractor.grid_ = np.asarray(frequency_bundle.get("feature_grid"), dtype=float) if isinstance(frequency_bundle, dict) and frequency_bundle.get("feature_grid") is not None else np.linspace(-0.5, 5.0, extractor.grid_size)
    extractor.fill_ = np.zeros(extractor.grid_size * 12, dtype=float)
    topology_model = _load_artifact(artifacts.topology)
    topology_preprocessor = _load_artifact(artifacts.topology_preprocessor)
    parameter_bundles = {name: _load_artifact(path) for name, path in artifacts.parameter_models.items()}
    parameter_fallbacks = {name: _load_artifact(path) for name, path in artifacts.fallback_parameters.items() if name not in parameter_bundles}
    results = []
    for target in targets:
        record = target.record
        stage1 = conservative_mask(record.frequency, record.impedance, threshold=10.0)
        cleaned = _clean_record(record, stage1.mask)
        base = extractor._one(cleaned.frequency, cleaned.impedance).reshape(1, -1)
        voltage = _safe_voltage(record.voltage)
        frequency_range = None
        if "frequency" in operations:
            prediction = np.asarray(frequency_model.predict(np.hstack([base, [[voltage]]]))[0], dtype=float)
            low, high = 10.0 ** prediction[0], 10.0 ** prediction[1]
            measured = (float(np.min(record.frequency)), float(np.max(record.frequency)))
            low, high = max(measured[0], low), min(measured[1], high)
            frequency_range = (low, high) if high > low else measured
        inside = np.ones(record.frequency.size, dtype=bool) if frequency_range is None else (record.frequency >= frequency_range[0]) & (record.frequency <= frequency_range[1])
        final_mask = stage1.mask & inside
        outlier_mask = np.zeros(record.frequency.size, dtype=bool)
        if "active_points" in operations:
            _detected, _score, diagnostics = detect_valid_points(record.frequency, record.impedance, frequency_range=frequency_range, threshold=4.0)
            outlier_mask = inside & (np.asarray(diagnostics["rejection_reason"], dtype=object) == "local_anomaly")
            final_mask &= ~outlier_mask
            if int(final_mask.sum()) < 3:
                final_mask = stage1.mask & inside
                outlier_mask = ~final_mask
        active_record = _clean_record(record, final_mask)
        topology = None
        confidence = None
        if "model" in operations or "initial_parameters" in operations:
            x_topology = topology_preprocessor.transform([active_record])
            topology = str(topology_model.predict(x_topology)[0])
            if hasattr(topology_model, "predict_proba"):
                confidence = float(np.max(topology_model.predict_proba(x_topology)[0]))
        if topology is None:
            topology = "TWO_PROCESS" if "p(R2,CPE2)" in target.circuit else "ONE_PROCESS"
        process_count = 2 if topology == "TWO_PROCESS" else 1
        l0_required = bool(decide_l0(high_frequency_inductive_diagnostic(record.frequency, record.impedance), strength_threshold=0.01)["l0_required"])
        circuit = ("R0-L0-" if l0_required else "R0-") + "p(R1,CPE1)" + ("-p(R2,CPE2)" if process_count == 2 else "")
        parameters = {}
        parameter_sources = {}
        if "initial_parameters" in operations:
            extra = calculate_absolute_impedance_features(active_record)
            current = _safe_voltage(record.current)
            topology_features = topology_preprocessor.transform([active_record])
            feature_sets = {
                # Stage 4B configurations A/B contain metadata only; C
                # contains the 192 spectral features plus scale features.
                "A": np.asarray([[voltage]], dtype=float),
                "B": np.asarray([[voltage, current]], dtype=float),
                "C": np.hstack([topology_features, [[voltage, current, *[extra[name] for name in ("log10_median_abs_Z", "log10_mean_abs_Z", "log10_max_abs_Z", "log10_min_abs_Z", "Re_Z_high", "Im_Z_high", "Re_Z_low", "Im_Z_low", "log10_abs_Z_high", "log10_abs_Z_low")]]]]),
                "stage4a": np.hstack([topology_features, [[voltage]]]),
            }
            canonical_names = ("R0", "R1", "Q1", "alpha1") + (("R2", "Q2", "alpha2") if process_count == 2 else ())
            for canonical in canonical_names:
                bundle = parameter_bundles.get(canonical) or parameter_fallbacks.get(canonical)
                if bundle is None:
                    continue
                configuration = bundle.get("feature_configuration", "stage4a")
                if canonical in parameter_fallbacks and canonical not in parameter_bundles:
                    x = feature_sets["stage4a"]
                else:
                    x = feature_sets.get(configuration, feature_sets["A"])
                raw = float(bundle["model"].predict(x)[0])
                transformation = bundle.get("target_transformation", "log10")
                if transformation in {"logit_alpha", "direct_alpha_clipped"}:
                    value = float(inverse_target([raw], canonical.lower() if canonical.startswith("alpha") else canonical)[0]) if transformation == "logit_alpha" else float(np.clip(raw, 1e-6, 1.0 - 1e-6))
                else:
                    value = float(10.0 ** raw)
                actual = {"Q1": "CPE1_0", "alpha1": "CPE1_1", "Q2": "CPE2_0", "alpha2": "CPE2_1"}.get(canonical, canonical)
                parameters[actual] = value
                parameter_sources[actual] = str(bundle.get("parameter", canonical))
        control = target.key.split("::")[1] if "::" in target.key else "working"
        result = {"spectrum_id": target.key, "spectrum_key": spectrum_identifier(record.frequency, record.z_real, record.z_imag, record.cycle, control), "source_name": target.key.split("::", 1)[0], "cycle": record.cycle, "control": control, "predicted_f_min": frequency_range[0] if frequency_range else None, "predicted_f_max": frequency_range[1] if frequency_range else None, "predicted_topology": topology, "predicted_eec_model": circuit, "suggested_EEC": circuit, "predicted_process_count": process_count, "predicted_l0_required": l0_required, "confidence": confidence, "ml_eec_parameters": parameters, "initial_sources": parameter_sources, "final_ml_active_mask": final_mask.tolist() if "active_points" in operations else None, "deterministic_outlier_mask": outlier_mask.tolist() if "active_points" in operations else None, "pretrained_artifacts": {"frequency": str(artifacts.frequency), "topology": str(artifacts.topology), "topology_model": artifacts.topology_model_name, "parameter_fallbacks": sorted(set(artifacts.fallback_parameters) - set(artifacts.parameter_models))}}
        results.append({key: value for key, value in result.items() if value is not None})
    return results


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def make_runtime_spectrum(key: str, cycle, circuit: str) -> RuntimeSpectrum:
    frequency = np.asarray(cycle.frequency_hz, dtype=float).reshape(-1)
    impedance = np.asarray(cycle.impedance, dtype=complex).reshape(-1)
    valid = np.isfinite(frequency) & (frequency > 0) & np.isfinite(impedance.real) & np.isfinite(impedance.imag)
    if int(valid.sum()) < 3:
        raise ValueError("spectrum has fewer than three valid points")
    window = cycle.frequency_window
    f_min, f_max = (None, None)
    if window is not None:
        f_min, f_max = sorted((float(window[0]), float(window[1])))
    topology = canonical_electrochemical_topology(circuit)
    if topology is None:
        raise ValueError(f"unsupported EEC topology: {circuit}")
    control = str(key).split("::")[1] if len(str(key).split("::")) > 1 else "working"
    fitted_values = {}
    if cycle.fit_parameters is not None:
        fitted_values = {
            parameter.name: float(value)
            for parameter, value in zip(cycle.parameters, np.asarray(cycle.fit_parameters).reshape(-1))
            if _finite(value) is not None
        }
    record = SpectrumRecord(
        spectrum_id=key,
        source_project=str(key.split("::", 1)[0]),
        sample_id=str(key.split("::", 1)[0]),
        cycle=int(cycle.cycle),
        voltage=_finite(cycle.potential_v),
        current=_finite(cycle.current_ma),
        time=_finite(cycle.time_s),
        frequency=frequency[valid],
        z_real=impedance.real[valid],
        z_imag=impedance.imag[valid],
        topology_label=topology,
        original_eec_topology=circuit,
        electrochemical_topology=topology,
        l0_required_in_manual_fit="-L0-" in circuit,
        device_setup=str(key.split("::")[0]),
        manual_f_min=f_min,
        manual_f_max=f_max,
        control=control,
    )
    return RuntimeSpectrum(key, record, circuit, fitted_values)


def _predict_range(training: list[RuntimeSpectrum], target: RuntimeSpectrum) -> tuple[float, float] | None:
    labelled = [item.record for item in training if item.record.manual_f_min and item.record.manual_f_max]
    if len(labelled) < 2:
        return None
    _x_train, _x_target = _features(labelled, [target.record], "spectrum_only", 64)
    model = range_models(42)["random_forest"]
    lower, upper = _fit_predict(model, _x_train, _targets(labelled), _x_target)
    minimum, maximum = sorted((10.0 ** float(lower[0]), 10.0 ** float(upper[0])))
    measured = target.record.frequency
    minimum = max(minimum, float(np.min(measured)))
    maximum = min(maximum, float(np.max(measured)))
    return (minimum, maximum) if maximum > minimum else None


def _predict_topology(training: list[RuntimeSpectrum], target: RuntimeSpectrum) -> tuple[str, float] | None:
    if len(training) < 2 or len({item.record.topology_label for item in training}) < 2:
        return None
    processor = SpectrumPreprocessor(grid_size=64, spectrum_mode="raw")
    x_train = processor.fit_transform([item.record for item in training])
    x_target = processor.transform([target.record])
    model = topology_models(42)["hist_gradient_boosting"]
    model.fit(x_train, [item.record.topology_label for item in training])
    prediction = str(model.predict(x_target)[0])
    probability = float(np.max(model.predict_proba(x_target)[0]))
    return prediction, probability


def _predict_parameters(training: list[RuntimeSpectrum], target: RuntimeSpectrum, circuit: str) -> dict[str, float]:
    processor = SpectrumPreprocessor(grid_size=64, spectrum_mode="raw")
    usable = [item for item in training if item.fitted_values]
    if len(usable) < 2:
        return {}
    x_train = processor.fit_transform([item.record for item in usable])
    x_target = processor.transform([target.record])
    target_parameters = circuit_parameters(circuit)
    values: dict[str, float] = {}
    for parameter in target_parameters:
        labels = []
        rows = []
        for index, item in enumerate(usable):
            mapping = parameter_name_mapping(item.circuit, circuit)
            source_name = next((name for name in item.fitted_values if map_parameter_name(name, mapping or {}) == parameter.name), None)
            if source_name is None:
                continue
            value = _finite(item.fitted_values[source_name])
            if value is None:
                continue
            rows.append(index)
            labels.append(value)
        if len(labels) < 2:
            continue
        model = RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=1)
        model.fit(x_train[rows], np.log10(np.maximum(labels, np.finfo(float).tiny)))
        prediction = float(10.0 ** model.predict(x_target)[0])
        values[parameter.name] = float(np.clip(prediction, parameter.lower, parameter.upper))
    return values


def infer_runtime(
    training: list[RuntimeSpectrum],
    targets: list[RuntimeSpectrum],
    *,
    operations: set[str],
) -> list[dict]:
    """Predict selected spectra and return GUI-facing sidecar records."""
    if not training:
        raise ValueError("at least one non-selected labelled spectrum is required")
    output = []
    for target in targets:
        record = target.record
        frequency_range = _predict_range(training, target) if "frequency" in operations else None
        topology_result = _predict_topology(training, target) if "model" in operations or "initial_parameters" in operations else None
        topology = topology_result[0] if topology_result else canonical_electrochemical_topology(target.circuit)
        confidence = topology_result[1] if topology_result else None
        diagnostic = high_frequency_inductive_diagnostic(record.frequency, record.impedance)
        l0_required = bool(decide_l0(diagnostic, strength_threshold=0.01)["l0_required"])
        if topology is not None:
            process_count = 2 if "-p(R2,CPE2)" in topology else 1
            circuit = ("R0-L0-" if l0_required else "R0-") + "p(R1,CPE1)"
            if process_count == 2:
                circuit += "-p(R2,CPE2)"
        else:
            circuit = None
        values = _predict_parameters(training, target, circuit) if circuit and "initial_parameters" in operations else {}
        active_mask = None
        outlier_mask = None
        if "active_points" in operations:
            active_mask, _scores, diagnostics = detect_valid_points(
                record.frequency,
                record.impedance,
                frequency_range=frequency_range,
                threshold=4.0,
            )
            outlier_mask = np.asarray(diagnostics["rejection_reason"], dtype=object) == "local_anomaly"
            if int(np.count_nonzero(active_mask)) < 3:
                minimum, maximum = frequency_range or (float(np.min(record.frequency)), float(np.max(record.frequency)))
                active_mask = (
                    np.isfinite(record.frequency)
                    & (record.frequency >= minimum)
                    & (record.frequency <= maximum)
                    & np.isfinite(record.z_real)
                    & np.isfinite(record.z_imag)
                )
                outlier_mask = ~active_mask
        control = target.key.split("::")[1] if "::" in target.key else "working"
        spectrum_key = spectrum_identifier(record.frequency, record.z_real, record.z_imag, record.cycle, control)
        item = {
            "spectrum_id": target.key,
            "spectrum_key": spectrum_key,
            "source_name": target.key.split("::", 1)[0],
            "cycle": record.cycle,
            "control": control,
            "predicted_f_min": frequency_range[0] if frequency_range else None,
            "predicted_f_max": frequency_range[1] if frequency_range else None,
            "predicted_topology": topology,
            "predicted_eec_model": circuit,
            "suggested_EEC": circuit,
            "predicted_process_count": (2 if circuit and "R2" in circuit else 1) if circuit else None,
            "predicted_l0_required": l0_required if circuit else None,
            "confidence": confidence,
            "ml_eec_parameters": values,
            "final_ml_active_mask": active_mask.tolist() if active_mask is not None else None,
            "deterministic_outlier_mask": outlier_mask.tolist() if outlier_mask is not None else None,
            "runtime_inference": True,
        }
        output.append({key: value for key, value in item.items() if value is not None})
    return output


def save_runtime_results(path: Path, spectra: list[dict], *, training_count: int, operations: set[str]) -> None:
    write_ml_results(
        Path(path), spectra,
        pipeline={"runtime_inference": True, "operations": sorted(operations), "training_spectra": int(training_count)},
    )
