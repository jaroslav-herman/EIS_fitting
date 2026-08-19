"""Blind inference of the automatic preprocessing/topology pipeline on sample 178."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .automatic_preprocessing import conservative_mask, sensitive_mask
from .dataset import load_eisfit_projects
from .evaluate_frequency_limit_ml import _clean_record, _manual_masks
from .evaluate_topology_automatic_preprocessing import _targets_for, class_label, topology_models
from .frequency_limit_ml import SpectrumFeatureExtractor, models as frequency_models, target_values
from .preprocessing import SpectrumPreprocessor


TRAINING_SAMPLES = ("129", "140", "150", "157", "159", "181")
INFERENCE_SAMPLE = "178"
OUTPUT_DIR = Path("ml/analysis/unseen_178")
CONFIG = {"stage1_threshold": 10.0, "stage2_threshold": 4.0, "frequency_model": "ridge", "frequency_grid_size": 16, "topology_grid_size": 64, "frequency_target": "robust_three_active_points", "topology_recommended_model": "hist_gradient_boosting", "seed": 42}


def _fit_frequency(records, manual_masks):
    stage1 = {r.spectrum_id: conservative_mask(r.frequency, r.impedance, threshold=CONFIG["stage1_threshold"]) for r in records}
    cleaned = {r.spectrum_id: _clean_record(r, stage1[r.spectrum_id].mask) for r in records}
    extractor = SpectrumFeatureExtractor(CONFIG["frequency_grid_size"]); extractor.grid_ = np.linspace(-0.5, 5.0, CONFIG["frequency_grid_size"]); extractor.fill_ = np.zeros(CONFIG["frequency_grid_size"] * 12)
    x = {sid: extractor._one(r.frequency, r.impedance) for sid, r in cleaned.items()}
    targets = {sid: _targets_for([r], [manual_masks[sid]])[0] for sid, r in ((r.spectrum_id, r) for r in records)}
    model = frequency_models(42)[CONFIG["frequency_model"]]; ids = [r.spectrum_id for r in records]; model.fit(np.vstack([x[sid] for sid in ids]), target_values([targets[sid] for sid in ids]))
    return model, extractor, stage1, cleaned, x


def _clip_window(predicted, measured):
    lo, hi = 10**float(predicted[0]), 10**float(predicted[1]); minimum, maximum = measured
    clipped = lo < minimum or hi > maximum or hi <= lo
    lo, hi = max(minimum, lo), min(maximum, hi)
    if hi <= lo: lo, hi = minimum, maximum
    return float(lo), float(hi), bool(clipped)


def _plot(record, result, output, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, z = record.frequency, record.impedance; raw = np.ones(f.size, dtype=bool); final = np.asarray(result["final_ml_active_mask"], dtype=bool); rejected = ~final
    fig, axes = plt.subplots(1, 3, figsize=(15, 4)); x = np.log10(f)
    axes[0].scatter(x[raw], z.real[raw], s=9, color="0.7", label="raw"); axes[0].scatter(x[final], z.real[final], s=14, color="tab:green", label="ML active"); axes[0].scatter(x[rejected], z.real[rejected], marker="x", color="tab:red", label="ML rejected"); axes[0].set_title("Re(Z)")
    axes[1].scatter(x, np.angle(z), s=9, color="0.7"); axes[1].scatter(x[final], np.angle(z)[final], s=14, color="tab:green"); axes[1].set_title("Bode phase")
    axes[2].scatter(z.real, -z.imag, s=9, color="0.7"); axes[2].scatter(z.real[final], -z.imag[final], s=14, color="tab:green"); axes[2].set_title("Nyquist")
    for ax in axes[:2]:
        ax.axvline(np.log10(result["predicted_f_min"]), color="tab:red", linestyle="--"); ax.axvline(np.log10(result["predicted_f_max"]), color="tab:purple", linestyle="--"); ax.set_xlabel("log10(f)")
    axes[0].legend(fontsize=8); fig.suptitle(f"{label}: RF={result['rf_topology']}, HGB={result['hgb_topology']}"); fig.tight_layout(); fig.savefig(output / f"{label}.png", dpi=130); plt.close(fig)


def _validate_ml_results(records, results):
    original = {(r.source_name, r.cycle): r for r in records}
    result_keys = [(item.get("source_name"), item.get("cycle")) for item in results]
    counts = pd.Series(result_keys).value_counts() if result_keys else pd.Series(dtype=int)
    duplicate_keys = [str(key) for key, count in counts.items() if count > 1]
    result_by_key = {key: item for key, item in zip(result_keys, results)}
    missing_results = [str(key) for key in original if key not in result_by_key]
    missing_metadata = {}
    metadata_columns = sorted({name for record in records for name in record.metadata})
    for key, record in original.items():
        missing = [name for name in metadata_columns if name not in record.metadata]
        if missing:
            missing_metadata[str(key)] = missing
        item = result_by_key.get(key)
        if item is None:
            continue
        lengths = [
            len(item.get("frequency", [])),
            len(item.get("z_real", [])),
            len(item.get("z_imag", [])),
            len(item.get("stage1_active_mask", [])),
            len(item.get("stage2_active_mask", [])),
            len(item.get("final_ml_active_mask", [])),
        ]
        if len(set(lengths)) != 1 or lengths[0] != len(record.frequency):
            raise ValueError(f"ML arrays do not match spectrum {key!r}")
        if not np.array_equal(np.asarray(item["frequency"]), record.frequency):
            raise ValueError(f"Frequency array changed for spectrum {key!r}")
        if not np.array_equal(np.asarray(item["z_real"]), record.z_real) or not np.array_equal(np.asarray(item["z_imag"]), record.z_imag):
            raise ValueError(f"Impedance array changed for spectrum {key!r}")
        if item.get("metadata", {}) != record.metadata:
            raise ValueError(f"Metadata changed for spectrum {key!r}")
    if duplicate_keys or missing_results:
        raise ValueError(f"Invalid ML result identity: duplicates={duplicate_keys}, missing={missing_results}")
    return {
        "source_names": len({key[0] for key in original}),
        "cycles": len({key[1] for key in original}),
        "unique_spectra": len(original),
        "ml_results": len(results),
        "metadata_columns": len(metadata_columns),
        "preserved_metadata_columns": metadata_columns,
        "missing_metadata": missing_metadata,
        "duplicate_spectrum_keys": duplicate_keys,
        "merged_spectra_detected": False,
    }


def infer(training_projects: list[Path], inference_project: Path, output: Path):
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=True); (output / "plots").mkdir(exist_ok=True)
    train_mapping = {str(p): p.name.split(".")[0] for p in training_projects}; train_report = load_eisfit_projects(training_projects, train_mapping, require_fit=True); train_records = train_report.records
    train_manual = _manual_masks(training_projects, {r.spectrum_id: r for r in train_records}); train_records = [r for r in train_records if r.spectrum_id in train_manual]
    infer_report = load_eisfit_projects([inference_project], {str(inference_project): INFERENCE_SAMPLE}, require_fit=False); inference_records = infer_report.records
    frequency_model, frequency_extractor, stage1_train, cleaned_train, _ = _fit_frequency(train_records, train_manual)
    train_auto = []
    for r in train_records:
        target = _targets_for([r], [train_manual[r.spectrum_id]])[0]; prediction = frequency_model.predict(frequency_extractor._one(cleaned_train[r.spectrum_id].frequency, cleaned_train[r.spectrum_id].impedance)[None, :])[0]; window, _, _ = _clip_window(prediction, (target["measured_f_min"], target["measured_f_max"]))
        hi = _clip_window(prediction, (target["measured_f_min"], target["measured_f_max"]))[1]; s2 = sensitive_mask(r.frequency, r.impedance, (window, hi), threshold=CONFIG["stage2_threshold"]); final = stage1_train[r.spectrum_id].mask & (r.frequency >= window) & (r.frequency <= hi) & s2.mask; train_auto.append(_clean_record(r, final))
    topology_pre = SpectrumPreprocessor(grid_size=CONFIG["topology_grid_size"], spectrum_mode="raw"); x_train = topology_pre.fit_transform(train_auto); y_train = [class_label(r) for r in train_auto]
    topology = topology_models(CONFIG["seed"]); [model.fit(x_train, y_train) for model in topology.values()]
    frequency_outputs = []; topology_outputs = []; summary_rows = []; plot_candidates = []
    for index, r in enumerate(inference_records):
        s1 = conservative_mask(r.frequency, r.impedance, threshold=CONFIG["stage1_threshold"]); cleaned = _clean_record(r, s1.mask); features = frequency_extractor._one(cleaned.frequency, cleaned.impedance)[None, :]; predicted_log = frequency_model.predict(features)[0]; measured = (float(np.min(r.frequency)), float(np.max(r.frequency))); lo, hi, clipped = _clip_window(predicted_log, measured); envelope = (r.frequency >= lo) & (r.frequency <= hi); s2 = sensitive_mask(r.frequency, r.impedance, (lo, hi), threshold=CONFIG["stage2_threshold"]); final = s1.mask & envelope & s2.mask; topo_record = _clean_record(r, final); x_test = topology_pre.transform([topo_record])
        predictions = {}
        for name, model in topology.items():
            probs = model.predict_proba(x_test)[0]; classes = list(model.classes_); probabilities = {c: float(probs[classes.index(c)]) if c in classes else 0.0 for c in ("ONE_PROCESS", "TWO_PROCESS")}; predictions[name] = {"topology": str(model.predict(x_test)[0]), "probabilities": probabilities, "confidence": max(probabilities.values())}
        existing = r.original_eec_topology
        result = {"spectrum_id": r.spectrum_id, "source_name": r.source_name, "cycle": r.cycle, "metadata": dict(r.metadata), "voltage": r.voltage, "current": r.current, "time": r.time, "frequency": r.frequency.tolist(), "z_real": r.z_real.tolist(), "z_imag": r.z_imag.tolist(), "existing_eec_topology": existing, "stage1_active_mask": s1.mask.tolist(), "stage1_rejection_score": s1.score.tolist(), "stage2_active_mask": s2.mask.tolist(), "stage2_rejection_score": s2.score.tolist(), "ml_envelope_mask": envelope.tolist(), "final_ml_active_mask": final.tolist(), "predicted_log_f_min": float(predicted_log[0]), "predicted_log_f_max": float(predicted_log[1]), "predicted_f_min": lo, "predicted_f_max": hi, "frequency_boundary_clipped": clipped, "measured_f_min": measured[0], "measured_f_max": measured[1], "first_active_frequency": float(r.frequency[envelope][0]) if envelope.any() else None, "last_active_frequency": float(r.frequency[envelope][-1]) if envelope.any() else None, "ml_envelope_points": int(envelope.sum()), "stage1_rejected_points": int((~s1.mask).sum()), "stage2_rejected_points_inside_envelope": int(np.sum(envelope & ~s2.mask)), "final_active_point_count": int(final.sum()), "rf_topology": predictions["random_forest"]["topology"], "rf_probabilities": predictions["random_forest"]["probabilities"], "hgb_topology": predictions["hist_gradient_boosting"]["topology"], "hgb_probabilities": predictions["hist_gradient_boosting"]["probabilities"], "recommended_topology": predictions[CONFIG["topology_recommended_model"]]["topology"]}
        topology_outputs.append(result); summary_rows.append({"spectrum_id": r.spectrum_id, "voltage": r.voltage, "time": r.time, "predicted_f_min": lo, "predicted_f_max": hi, "final_active_points": int(final.sum()), "rf_topology": result["rf_topology"], "rf_confidence": max(result["rf_probabilities"].values()), "hgb_topology": result["hgb_topology"], "hgb_confidence": max(result["hgb_probabilities"].values()), "recommended_topology": result["recommended_topology"], "existing_human_topology": existing})
        if index < 8: _plot(r, result, output / "plots", f"spectrum_{index+1:03d}")
    validation = _validate_ml_results(inference_records, topology_outputs)
    output_json = output / "178_ML.eisfit.json"; payload = {"ml_results": {"schema_version": "2.0", "source_file": str(inference_project), "training_samples": list(TRAINING_SAMPLES), "inference_sample": INFERENCE_SAMPLE, "model_finalization": "models fit once on all six training samples; sample 178 excluded", "preprocessing": CONFIG, "topology_prediction": {"recommended_model": CONFIG["topology_recommended_model"], "classes": ["ONE_PROCESS", "TWO_PROCESS"]}, "validation": validation, "spectra": topology_outputs}}
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8"); pd.DataFrame(summary_rows).to_csv(output / "178_summary.csv", index=False); (output / "178_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    config = {"input_file": str(inference_project), "training_samples": list(TRAINING_SAMPLES), "inference_sample": INFERENCE_SAMPLE, "frequency_model": CONFIG["frequency_model"], "topology_models": ["random_forest", "hist_gradient_boosting"], "no_bayes_drt2": True, "no_eec_fitting": True, "no_parameter_prediction": True, "runtime_s": time.perf_counter()-started, "training_spectra": len(train_records), "inference_spectra": len(topology_outputs), "excluded_inference": infer_report.exclusion_counts, "validation": validation}
    (output / "inference_report.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2)); return config


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--input", type=Path, default=Path(r"C:\Users\Herman\Desktop\Ti overlayer backup\178.eisfit.json")); parser.add_argument("--output", type=Path, default=OUTPUT_DIR); parser.add_argument("training_projects", nargs="+"); args=parser.parse_args(); infer([Path(p) for p in args.training_projects], args.input, args.output)


if __name__ == "__main__": main()
