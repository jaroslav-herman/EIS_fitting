"""Evaluate the new active-mask frequency-limit preprocessing experiment."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pandas as pd

from eis_project import dataframe_from_payload
from eis_services import load_cycle

from .automatic_preprocessing import active_boundary_targets, binary_metrics, conservative_mask, sensitive_mask
from .dataset import _payload_projects, load_eisfit_projects
from .frequency_limit_ml import MODEL_NAMES, SpectrumFeatureExtractor, models, regression_metrics, target_values


PROJECT_SAMPLES = ("129", "140", "150", "157", "159", "181")
TARGETS = ("literal", "robust", "ignore_isolated")
CONFIG = {"grid_size": 16, "seed": 42, "target_persistence": 3, "stage1_threshold": 10.0, "stage2_thresholds": (3.0, 4.0, 5.0), "primary_target": "robust", "final_stage2_threshold": 4.0, "trained_target_definitions": ("robust",), "benchmark_budget": "16-point grid, 12 RF trees, 12 HGB iterations; target-definition audit retains all candidates"}


def _manual_masks(projects, records):
    result = {}
    for path in projects:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for entry_index, (state, entry) in enumerate(_payload_projects(payload)):
            if not entry.get("dataframe"): continue
            dataframe = dataframe_from_payload(entry["dataframe"])
            control = str(state.get("control", payload.get("control", "cell")))
            dataset_key = str(entry.get("dataset_id") or f"dataset_{entry_index}")
            for cycle_text, saved in (state.get("cycles") or {}).items():
                sid = f"{Path(path).resolve()}::{dataset_key}::{control}::{int(cycle_text)}"
                if sid not in records or saved.get("manually_included") is None: continue
                cycle = load_cycle(dataframe, int(cycle_text), control)
                f = np.asarray(cycle.frequency_hz, dtype=float); z = np.asarray(cycle.impedance, dtype=complex)
                valid = np.isfinite(f) & (f > 0) & np.isfinite(z.real) & np.isfinite(z.imag)
                mask = np.asarray(saved["manually_included"], dtype=bool)
                if mask.size == f.size:
                    result[sid] = mask[valid]
    return result


def _clean_record(record, mask):
    return replace(record, frequency=record.frequency[mask], z_real=record.z_real[mask], z_imag=record.z_imag[mask])


def _prediction_row(record, target_name, model_name, held_out, target, pred):
    true_min, true_max = np.log10(target["f_min"]), np.log10(target["f_max"])
    row = {"spectrum_id": record.spectrum_id, "sample_id": record.sample_id, "voltage": record.voltage, "time": record.time, "topology": record.electrochemical_topology, "l0_required": record.l0_required_in_manual_fit,
           "target_definition": target_name, "model": model_name, "held_out_sample": held_out, "manual_f_min": target["f_min"], "manual_f_max": target["f_max"],
           "predicted_log_f_min": float(pred[0]), "predicted_log_f_max": float(pred[1]), "predicted_f_min": float(10**pred[0]), "predicted_f_max": float(10**pred[1]),
           "error_f_min_decades": float(pred[0] - true_min), "error_f_max_decades": float(pred[1] - true_max), "measured_f_min": target["measured_f_min"], "measured_f_max": target["measured_f_max"]}
    row["frequency_range_iou"] = _iou(true_min, true_max, pred[0], pred[1])
    return row


def _iou(a, b, c, d):
    intersection = max(0.0, min(b, d) - max(a, c)); union = max(b, d) - min(a, c)
    return intersection / union if union else 1.0


def _plot(record, manual, stage1, prediction, stage2, label, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, z = record.frequency, record.impedance; x = np.log10(f)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(x, z.real, "-", color="0.75"); axes[0].scatter(x[~manual], z.real[~manual], marker="x", color="tab:red", label="manual inactive"); axes[0].scatter(x[manual], z.real[manual], s=10, color="tab:green", label="manual active"); axes[0].set_title("Re(Z)")
    axes[1].plot(x, z.imag, "-", color="0.75"); axes[1].scatter(x[stage1], z.imag[stage1], s=10, color="tab:blue", label="stage 1 kept"); axes[1].scatter(x[stage2], z.imag[stage2], s=12, facecolors="none", edgecolors="tab:orange", label="final kept"); axes[1].set_title("Im(Z)")
    axes[2].plot(z.real, -z.imag, ".-"); axes[2].set_title("Nyquist")
    for ax in axes:
        ax.axvline(np.log10(prediction[0]), color="tab:red", linestyle="--", label="ML f_min")
        ax.axvline(np.log10(prediction[1]), color="tab:purple", linestyle="--", label="ML f_max")
        ax.set_xlabel("log10(f)" if ax is not axes[2] else "Re(Z)")
    axes[0].legend(fontsize=7); fig.suptitle(label); fig.tight_layout(); fig.savefig(output / f"{label}.png", dpi=130); plt.close(fig)


def evaluate(projects: list[Path], output: Path) -> dict:
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=True); (output / "plots").mkdir(exist_ok=True)
    mapping = {str(p): p.name.split(".")[0] for p in projects}; extraction = load_eisfit_projects(projects, mapping, require_fit=True)
    records = {r.spectrum_id: r for r in extraction.records}; manual = _manual_masks(projects, records)
    records = {sid: r for sid, r in records.items() if sid in manual}
    targets = {}; stage1 = {}; cleaned = {}; target_rows = []
    for sid, record in records.items():
        target_info = active_boundary_targets(record.frequency, manual[sid], persistence=CONFIG["target_persistence"])
        targets[sid] = {name: {"f_min": target_info[f"{name}_f_min"], "f_max": target_info[f"{name}_f_max"], "measured_f_min": target_info["measured_f_min"], "measured_f_max": target_info["measured_f_max"]} for name in TARGETS}
        target_rows.append({"spectrum_id": sid, "sample_id": record.sample_id, **{k: v for k, v in target_info.items() if not isinstance(v, list)}, "active_run_lengths": json.dumps(target_info["active_run_lengths"]), "inactive_run_lengths": json.dumps(target_info["inactive_run_lengths"])})
        result = conservative_mask(record.frequency, record.impedance, threshold=CONFIG["stage1_threshold"]); stage1[sid] = result
        keep = result.mask; cleaned[sid] = _clean_record(record, keep)
    target_frame = pd.DataFrame(target_rows); target_frame.to_csv(output / "target_definition_summary.csv", index=False)
    pattern_summary = {"literal_vs_robust_min_different_percent": float(100*np.mean(target_frame.literal_f_min != target_frame.robust_f_min)), "robust_vs_ignore_isolated_min_different_percent": float(100*np.mean(target_frame.robust_f_min != target_frame.ignore_isolated_f_min)), "lowest_active_run_1_percent": float(100*np.mean(target_frame.lowest_active_run_length == 1)), "lowest_active_run_2_percent": float(100*np.mean(target_frame.lowest_active_run_length == 2)), "lowest_active_run_3_percent": float(100*np.mean(target_frame.lowest_active_run_length == 3)), "lowest_active_run_ge4_percent": float(100*np.mean(target_frame.lowest_active_run_length >= 4)), "highest_active_run_1_percent": float(100*np.mean(target_frame.highest_active_run_length == 1)), "highest_active_run_ge4_percent": float(100*np.mean(target_frame.highest_active_run_length >= 4))}
    (output / "target_definition_report.json").write_text(json.dumps({"persistence": CONFIG["target_persistence"], "candidate_definitions": {"literal": "lowest active point", "robust": "first active run of at least 3 points", "ignore_isolated": "first active run of at least 4 points"}, "patterns": pattern_summary}, indent=2), encoding="utf-8")
    prediction_rows = []; runtimes = {}
    # The instrument frequency span is fixed by the measured dataset.  Using
    # this declared grid avoids rebuilding identical per-spectrum features in
    # every LOSO fold; no labels or sample metadata enter the representation.
    extractor = SpectrumFeatureExtractor(CONFIG["grid_size"])
    extractor.grid_ = np.linspace(-0.5, 5.0, CONFIG["grid_size"])
    extractor.fill_ = np.zeros(CONFIG["grid_size"] * 12, dtype=float)
    feature_by_id = {sid: extractor._one(r.frequency, r.impedance) for sid, r in cleaned.items()}
    for held_out in sorted({r.sample_id for r in records.values()}):
        train_ids = [sid for sid, r in records.items() if r.sample_id != held_out]; test_ids = [sid for sid, r in records.items() if r.sample_id == held_out]
        train = [cleaned[sid] for sid in train_ids]; test = [cleaned[sid] for sid in test_ids]
        x_train = np.vstack([feature_by_id[sid] for sid in train_ids]); x_test = np.vstack([feature_by_id[sid] for sid in test_ids])
        for target_name in CONFIG["trained_target_definitions"]:
            y = target_values([targets[sid][target_name] for sid in train_ids])
            for model_name in MODEL_NAMES:
                t0 = time.perf_counter(); model = models(CONFIG["seed"])[model_name]; model.fit(x_train, y); prediction = model.predict(x_test); runtimes[f"{target_name}:{model_name}"] = runtimes.get(f"{target_name}:{model_name}", 0.0) + time.perf_counter() - t0
                for record, sid, pred in zip(test, test_ids, prediction): prediction_rows.append(_prediction_row(records[sid], target_name, model_name, held_out, targets[sid][target_name], pred))
    predictions = pd.DataFrame(prediction_rows); predictions.to_csv(output / "per_spectrum_predictions.csv", index=False)
    metric_rows = []
    for keys, part in predictions.groupby(["target_definition", "model"]): metric_rows.append({"target_definition": keys[0], "model": keys[1], "spectra": len(part), **regression_metrics(part)})
    overall = pd.DataFrame(metric_rows); overall.to_csv(output / "overall_metrics.csv", index=False)
    per_sample = []
    for keys, part in predictions.groupby(["target_definition", "model", "sample_id"]): per_sample.append({"target_definition": keys[0], "model": keys[1], "sample_id": keys[2], "spectra": len(part), **regression_metrics(part)})
    pd.DataFrame(per_sample).to_csv(output / "per_sample_metrics.csv", index=False)
    def grouped_metrics(columns, filename):
        rows = []
        for keys, part in predictions.groupby(["target_definition", "model", *columns], dropna=False):
            if not isinstance(keys, tuple): keys = (keys,)
            row = dict(zip(["target_definition", "model", *columns], keys)); row.update({"spectra": len(part), **regression_metrics(part)}); rows.append(row)
        pd.DataFrame(rows).to_csv(output / filename, index=False)
    grouped_metrics(["voltage"], "per_voltage_metrics.csv"); grouped_metrics(["topology"], "per_topology_metrics.csv"); grouped_metrics(["l0_required"], "per_l0_metrics.csv")
    best = overall[overall.target_definition == CONFIG["primary_target"]].sort_values("f_min_MAE_decades").iloc[0]; best_model = str(best.model)
    best_predictions = predictions[(predictions.target_definition == CONFIG["primary_target"]) & (predictions.model == best_model)].set_index("spectrum_id")
    mask_rows = []; point_rows = []; stage2_stats = []
    for sid, record in records.items():
        manual_mask = manual[sid]; stage1_mask = stage1[sid].mask; pred = best_predictions.loc[sid]; envelope = (record.frequency >= pred.predicted_f_min) & (record.frequency <= pred.predicted_f_max)
        for threshold in CONFIG["stage2_thresholds"]:
            s2 = sensitive_mask(record.frequency, record.impedance, (pred.predicted_f_min, pred.predicted_f_max), threshold=threshold); final = stage1_mask & envelope & s2.mask
            metrics = binary_metrics(final, manual_mask); stage2_stats.append({"spectrum_id": sid, "sample_id": record.sample_id, "threshold": threshold, "stage1_rejected": int((~stage1_mask).sum()), "stage2_rejected_inside_envelope": int(np.sum(envelope & ~s2.mask)), "final_active_points": int(final.sum()), **metrics})
            if threshold == CONFIG["final_stage2_threshold"]:
                mask_rows.append({"spectrum_id": sid, "sample_id": record.sample_id, "model": best_model, "stage2_threshold": threshold, "manual_active_points": int(manual_mask.sum()), "stage1_rejected_points": int((~stage1_mask).sum()), "ml_envelope_points": int(envelope.sum()), "automatic_active_points": int(final.sum()), "manual_f_min": pred.manual_f_min, "ml_f_min": pred.predicted_f_min, "manual_f_max": pred.manual_f_max, "ml_f_max": pred.predicted_f_max, **metrics})
                for i, f in enumerate(record.frequency): point_rows.append({"spectrum_id": sid, "point_index": i, "frequency": f, "manual_active": bool(manual_mask[i]), "stage1_kept": bool(stage1_mask[i]), "ml_envelope": bool(envelope[i]), "automatic_active": bool(final[i])})
        if len(mask_rows) <= 9:
            _plot(record, manual_mask, stage1_mask, (pred.predicted_f_min, pred.predicted_f_max), final, f"sample_{record.sample_id}_{len(mask_rows)}", output / "plots")
    mask_frame = pd.DataFrame(mask_rows); mask_frame.to_csv(output / "mask_comparison.csv", index=False); pd.DataFrame(point_rows).to_csv(output / "automatic_active_masks.csv", index=False); pd.DataFrame(stage2_stats).to_csv(output / "stage2_sensitivity.csv", index=False)
    (output / "configuration.json").write_text(json.dumps({**CONFIG, "models": MODEL_NAMES, "primary_model_selected_by": "lowest LOSO robust f_min MAE", "primary_model": best_model, "features": "64-point fold-local log-frequency grid; normalized Re/Im/log-magnitude/phase and local slopes, curvature, residual, step"}, indent=2, default=list), encoding="utf-8")
    runtime = time.perf_counter() - started; report = {"records": len(records), "exclusions": extraction.exclusion_counts, "best_model": best_model, "best_target": CONFIG["primary_target"], "runtime_total_s": runtime, "model_runtime_s": runtimes, "stage1_rejected_points": int(sum((~v.mask).sum() for v in stage1.values())), "stage1_rejection_percent": float(100*sum((~v.mask).sum() for v in stage1.values())/sum(len(r.frequency) for r in records.values())), "stage2": "see stage2_sensitivity.csv", "topology_training_started": False}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md = ["# Active-mask frequency-limit ML preprocessing", "", f"Evaluated **{len(records)} spectra** with strict six-sample LOSO validation.", "", f"Primary target: **{CONFIG['primary_target']}**. Best model: **{best_model}**.", f"Best robust f_min MAE: **{best['f_min_MAE_decades']:.4f} decades**; within 0.10: **{best['f_min_within_0.10_percent']:.2f}%**; within 0.20: **{best['f_min_within_0.20_percent']:.2f}%**.", "", f"Stage 1 rejected **{report['stage1_rejected_points']} points ({report['stage1_rejection_percent']:.4f}%)** using threshold 10.0. Stage 2 results for thresholds 3, 4, and 5 are in `stage2_sensitivity.csv`.", "", "The target-definition report quantifies isolated active points at both boundaries. The saved automatic masks contain raw point identity, manual reference, Stage-1 status, ML envelope status, and final automatic status. No topology training was run.", "", f"Total runtime: **{runtime:.2f} s**."]
    (output / "report.md").write_text("\n".join(report_md), encoding="utf-8")
    print(overall.to_string(index=False)); print(json.dumps(report, indent=2)); return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("projects", nargs="+", type=Path); parser.add_argument("--output", type=Path, default=Path("ml/analysis/frequency_limit_ml")); args = parser.parse_args(); evaluate(args.projects, args.output)


if __name__ == "__main__": main()
