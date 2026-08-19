"""Finalize the six-sample topology models and apply them to sample 178.

This is deliberately a separate deployment script.  It does not alter the
existing fitting, DRT, project, or GUI code, and it never estimates circuit
parameters.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from .automatic_preprocessing import conservative_mask
from .dataset import SpectrumRecord, canonical_electrochemical_topology, load_eisfit_projects
from .evaluate_frequency_limit_ml import _clean_record
from .evaluate_frequency_selection_voltage import _feature_matrix
from .evaluate_topology_automatic_preprocessing import class_label, topology_models
from .frequency_limit_ml import SpectrumFeatureExtractor
from .point_validity import detect_valid_points
from .preprocessing import SpectrumPreprocessor


TRAINING_SAMPLES = ("129", "140", "150", "157", "159", "181")
INFERENCE_SAMPLE = "178"
CLASSES = ("ONE_PROCESS", "TWO_PROCESS")
RECOMMENDED_MODEL = "hist_gradient_boosting"
SEED = 42
STAGE1_THRESHOLD = 10.0
STAGE2_THRESHOLD = 4.0
FREQUENCY_GRID_SIZE = 16
TOPOLOGY_GRID_SIZE = 64
FREQUENCY_MODEL_PATH = Path("ml/analysis/unseen_178_new_frequency/final_frequency_model.joblib")
PREPROCESSED_INPUT = Path("ml/analysis/unseen_178_preprocessed/178_ML_preprocessed_results.eisfit.json")
OUTPUT_DIR = Path("ml/analysis/unseen_178_new_topology")


def _clip_window(prediction: np.ndarray, record: SpectrumRecord) -> tuple[float, float, bool]:
    measured = (float(np.min(record.frequency)), float(np.max(record.frequency)))
    low, high = 10.0 ** float(prediction[0]), 10.0 ** float(prediction[1])
    clipped = low < measured[0] or high > measured[1] or high <= low
    low, high = max(measured[0], low), min(measured[1], high)
    if high <= low:
        low, high = measured
    return float(low), float(high), bool(clipped)


def _frequency_features(records: list[SpectrumRecord]):
    stage1 = {r.spectrum_id: conservative_mask(r.frequency, r.impedance, threshold=STAGE1_THRESHOLD) for r in records}
    cleaned = {r.spectrum_id: _clean_record(r, stage1[r.spectrum_id].mask) for r in records}
    extractor = SpectrumFeatureExtractor(FREQUENCY_GRID_SIZE)
    extractor.grid_ = np.linspace(-0.5, 5.0, FREQUENCY_GRID_SIZE)
    extractor.fill_ = np.zeros(FREQUENCY_GRID_SIZE * 12)
    base = {sid: extractor._one(r.frequency, r.impedance) for sid, r in cleaned.items()}
    return stage1, cleaned, base


def _automatic_record(record: SpectrumRecord, frequency_model, base, records_by_id, train_ids):
    prediction = frequency_model.predict(
        _feature_matrix([record.spectrum_id], base, records_by_id, "spectrum_plus_voltage", train_ids)
    )[0]
    low, high, clipped = _clip_window(prediction, record)
    inside = (record.frequency >= low) & (record.frequency <= high)
    detected, score, diagnostics = detect_valid_points(
        record.frequency,
        record.impedance,
        threshold=STAGE2_THRESHOLD,
        neighborhood=3,
        min_points=4,
        frequency_range=(low, high),
        max_iterations=2,
        return_diagnostics=True,
    )
    outlier = inside & (np.asarray(diagnostics["rejection_reason"], dtype=object) == "local_anomaly")
    stage1_result = conservative_mask(record.frequency, record.impedance, threshold=STAGE1_THRESHOLD)
    final = inside & stage1_result.mask & ~outlier
    return {
        "record": _clean_record(record, final),
        "prediction": prediction,
        "low": low,
        "high": high,
        "clipped": clipped,
        "stage1": stage1_result,
        "inside": inside,
        "stage2": detected,
        "score": score,
        "outlier": outlier,
        "final": final,
    }


def _input_record(item: dict, source_file: str) -> SpectrumRecord:
    frequency = np.asarray(item["frequency"], dtype=float)
    real = np.asarray(item["z_real"], dtype=float)
    imag = np.asarray(item["z_imag"], dtype=float)
    topology = canonical_electrochemical_topology(str(item.get("existing_eec_topology") or "")) or "R0-p(R1,CPE1)"
    return SpectrumRecord(
        spectrum_id=str(item["spectrum_id"]), source_project=source_file,
        sample_id=INFERENCE_SAMPLE, cycle=int(item["cycle"]),
        voltage=item.get("voltage"), current=item.get("current"), time=item.get("time"),
        frequency=frequency, z_real=real, z_imag=imag, topology_label=topology,
        original_eec_topology=item.get("existing_eec_topology"),
        electrochemical_topology=topology, source_name=str(item.get("source_name") or ""),
        metadata=dict(item.get("metadata") or {}),
    )


def _probabilities(model, x):
    values = model.predict_proba(x)[0]
    classes = list(model.classes_)
    probabilities = {label: float(values[classes.index(label)]) if label in classes else 0.0 for label in CLASSES}
    prediction = str(model.predict(x)[0])
    return prediction, probabilities, max(probabilities.values())


def _topology_row(model_name, model, x):
    prediction, probabilities, confidence = _probabilities(model, x)
    count = 2 if prediction == "TWO_PROCESS" else 1
    suggested = "R0-p(R1,CPE1)-p(R2,CPE2)" if count == 2 else "R0-p(R1,CPE1)"
    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "confidence": confidence,
        "process_count": count,
        "suggested_eec": suggested,
        "model": model_name,
    }


def _train_models(training_projects: list[Path], frequency_model, output: Path):
    mapping = {str(path): path.name.split(".")[0] for path in training_projects}
    report = load_eisfit_projects(training_projects, mapping, require_fit=True)
    records = report.records
    samples = sorted({r.sample_id for r in records})
    if tuple(samples) != tuple(sorted(TRAINING_SAMPLES)):
        raise RuntimeError(f"training samples are {samples}, expected {sorted(TRAINING_SAMPLES)}")
    if INFERENCE_SAMPLE in samples:
        raise RuntimeError("sample 178 entered the topology training set")
    stage1, cleaned, base = _frequency_features(records)
    records_by_id = {r.spectrum_id: r for r in records}
    train_ids = [r.spectrum_id for r in records]
    auto = {}
    for record in records:
        auto[record.spectrum_id] = _automatic_record(record, frequency_model, base, records_by_id, train_ids)
    topology_records = [auto[sid]["record"] for sid in train_ids]
    if any(len(r.frequency) < 3 for r in topology_records):
        raise RuntimeError("automatic preprocessing left fewer than three points in a training spectrum")
    preprocessor = SpectrumPreprocessor(grid_size=TOPOLOGY_GRID_SIZE, use_metadata=False, spectrum_mode="raw")
    x_train = preprocessor.fit_transform(topology_records)
    if x_train.shape[1] != 192:
        raise RuntimeError(f"unexpected topology feature dimension: {x_train.shape[1]}")
    labels = [class_label(r) for r in topology_records]
    models = topology_models(SEED)
    for model in models.values():
        model.fit(x_train, labels)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["random_forest"], output / "final_topology_rf.joblib")
    joblib.dump(models["hist_gradient_boosting"], output / "final_topology_hgb.joblib")
    joblib.dump(preprocessor, output / "topology_preprocessor.joblib")
    class_counts = {label: int(sum(value == label for value in labels)) for label in CLASSES}
    training_report = {
        "training_samples": list(TRAINING_SAMPLES), "excluded_samples": [INFERENCE_SAMPLE],
        "training_spectra": len(records), "class_counts": class_counts,
        "class_percentages": {key: 100.0 * value / len(labels) for key, value in class_counts.items()},
        "feature_dimension": int(x_train.shape[1]),
        "feature_representation": "64 common log-frequency grid + 64 median-|Z|-normalized Re(Z) + 64 median-|Z|-normalized Im(Z)",
        "normalization": "per-spectrum median absolute impedance",
        "metadata_used": False, "frequency_model": str(FREQUENCY_MODEL_PATH),
        "preprocessing": {"stage1_threshold": STAGE1_THRESHOLD, "stage2_threshold": STAGE2_THRESHOLD, "detector": "ml.point_validity.detect_valid_points", "max_iterations": 2},
        "models": {"random_forest": "300 trees, class_weight=balanced, random_state=42", "hist_gradient_boosting": "max_iter=150, random_state=42"},
        "sample_178_in_training": False, "topology_labels": list(CLASSES), "l0_classifier_trained": False,
    }
    (output / "training_report.json").write_text(json.dumps(training_report, indent=2), encoding="utf-8")
    config = {**training_report, "recommended_model": RECOMMENDED_MODEL, "seed": SEED, "topology_grid_size": TOPOLOGY_GRID_SIZE, "frequency_grid_size": FREQUENCY_GRID_SIZE}
    (output / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return records, auto, preprocessor, models, training_report


def _apply(input_path: Path, output: Path, preprocessor, models):
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    root = payload.get("ml_results", payload)
    items = root.get("spectra", [])
    records = [_input_record(item, str(root.get("source_file") or input_path)) for item in items]
    output_items, rows = [], []
    for item, record in zip(items, records):
        final = np.asarray(item.get("final_ml_active_mask", []), dtype=bool)
        if final.size != record.frequency.size:
            raise RuntimeError(f"final mask length mismatch for {record.spectrum_id}")
        if final.sum() < 3:
            raise RuntimeError(f"preprocessing genuinely failed for {record.spectrum_id}: {int(final.sum())} active points")
        active_record = replace(record, frequency=record.frequency[final], z_real=record.z_real[final], z_imag=record.z_imag[final])
        x = preprocessor.transform([active_record])
        rf = _topology_row("random_forest", models["random_forest"], x)
        hgb = _topology_row("hist_gradient_boosting", models["hist_gradient_boosting"], x)
        chosen = hgb
        result = dict(item)
        result.update({
            "predicted_process_count": chosen["process_count"], "predicted_topology": chosen["prediction"],
            "topology_model": RECOMMENDED_MODEL, "topology_probability": chosen["probabilities"],
            "prediction_probability": chosen["confidence"], "process_prediction_confidence": chosen["confidence"],
            "predicted_L0_required": None, "L0_prediction_confidence": None,
            "L0_prediction_status": "unavailable_no_serialized_model", "suggested_EEC": chosen["suggested_eec"],
            "predicted_eec_model": chosen["suggested_eec"], "topology_prediction_status": "serialized_model_prediction",
            "rf_topology": rf["prediction"], "rf_probabilities": rf["probabilities"], "rf_confidence": rf["confidence"],
            "hgb_topology": hgb["prediction"], "hgb_probabilities": hgb["probabilities"], "hgb_confidence": hgb["confidence"],
            "recommended_topology": chosen["prediction"], "final_active_point_count": int(final.sum()),
        })
        output_items.append(result)
        metadata = dict(item.get("metadata") or {})
        rows.append({"spectrum_id": item.get("spectrum_id"), "source_name": item.get("source_name"), "cycle": item.get("cycle"), "voltage": item.get("voltage"), "current": item.get("current"), "Time": metadata.get("Time", item.get("time")), "Cycle mod 15": metadata.get("Cycle mod 15"), "predicted_process_count": chosen["process_count"], "predicted_topology": chosen["prediction"], "prediction_probability": chosen["confidence"], "topology_model": RECOMMENDED_MODEL, "suggested_EEC": chosen["suggested_eec"], "rf_topology": rf["prediction"], "rf_confidence": rf["confidence"], "hgb_topology": hgb["prediction"], "hgb_confidence": hgb["confidence"], "P_ONE_PROCESS": chosen["probabilities"]["ONE_PROCESS"], "P_TWO_PROCESS": chosen["probabilities"]["TWO_PROCESS"], "predicted_L0_required": None})
    new_root = dict(root)
    new_root.update({"schema_version": "5.0", "topology_prediction": {"recommended_model": RECOMMENDED_MODEL, "classes": list(CLASSES), "feature_dimension": 192}, "topology_prediction_status": "serialized_model_prediction", "L0_prediction_status": "unavailable_no_serialized_model", "spectra": output_items})
    result_path = output / "178_ML_topology_results.eisfit.json"
    result_path.write_text(json.dumps({"ml_results": new_root}, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows)
    csv_path = output / "178_ML_topology_predictions.csv"
    frame.to_csv(csv_path, index=False)
    if not frame.empty:
        frame.assign(voltage_bin=frame["voltage"].round(2)).groupby(["voltage_bin", "predicted_topology"], dropna=False).size().reset_index(name="spectra").to_csv(output / "178_ML_topology_by_voltage.csv", index=False)
    return result_path, csv_path, frame


def run(training_projects: list[Path], input_path: Path, output: Path):
    started = time.perf_counter()
    bundle = joblib.load(FREQUENCY_MODEL_PATH)
    frequency_model = bundle["model"] if isinstance(bundle, dict) else bundle
    records, auto, preprocessor, models, training_report = _train_models(training_projects, frequency_model, output)
    result_path, csv_path, frame = _apply(input_path, output, preprocessor, models)
    # Re-load artifacts and verify deterministic inference against the saved output.
    saved_pre = joblib.load(output / "topology_preprocessor.joblib")
    saved_models = {"random_forest": joblib.load(output / "final_topology_rf.joblib"), "hist_gradient_boosting": joblib.load(output / "final_topology_hgb.joblib")}
    check_payload = json.loads(result_path.read_text(encoding="utf-8"))["ml_results"]
    check_records = [_input_record(item, str(check_payload.get("source_file") or input_path)) for item in check_payload["spectra"]]
    for item, record in zip(check_payload["spectra"], check_records):
        mask = np.asarray(item["final_ml_active_mask"], dtype=bool)
        x = saved_pre.transform([replace(record, frequency=record.frequency[mask], z_real=record.z_real[mask], z_imag=record.z_imag[mask])])
        prediction = str(saved_models[RECOMMENDED_MODEL].predict(x)[0])
        if prediction != item["predicted_topology"]:
            raise RuntimeError(f"saved-artifact reproduction failed for {item['spectrum_id']}")
    counts = frame["predicted_topology"].value_counts().to_dict()
    confidences = frame["prediction_probability"].to_numpy(float)
    report = {"training": training_report, "inference_sample": INFERENCE_SAMPLE, "classified_spectra": int(len(frame)), "one_process": int(counts.get("ONE_PROCESS", 0)), "two_process": int(counts.get("TWO_PROCESS", 0)), "confidence": {"min": float(confidences.min()), "mean": float(confidences.mean()), "median": float(np.median(confidences)), "max": float(confidences.max())}, "output_json": str(result_path), "output_csv": str(csv_path), "models": {"random_forest": str(output / "final_topology_rf.joblib"), "hist_gradient_boosting": str(output / "final_topology_hgb.joblib"), "preprocessor": str(output / "topology_preprocessor.joblib")}, "model_config": str(output / "model_config.json"), "no_l0_guess": True, "no_eec_parameter_fitting": True, "sample_178_used_for_training": False, "artifact_reproduction_check": True, "runtime_s": time.perf_counter() - started}
    (output / "inference_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_projects", nargs="+", type=Path)
    parser.add_argument("--input", type=Path, default=PREPROCESSED_INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if len(args.training_projects) != 6:
        raise SystemExit("exactly six training project paths are required")
    run(args.training_projects, args.input, args.output)


if __name__ == "__main__":
    main()
