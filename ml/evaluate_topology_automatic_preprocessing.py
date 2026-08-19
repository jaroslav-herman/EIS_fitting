"""Strict nested LOSO topology evaluation using automatic preprocessing."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .automatic_preprocessing import conservative_mask, sensitive_mask
from .dataset import load_eisfit_projects
from .evaluate_frequency_limit_ml import PROJECT_SAMPLES, _clean_record, _manual_masks
from .frequency_limit_ml import SpectrumFeatureExtractor, models as frequency_models, target_values
from .preprocessing import SpectrumPreprocessor


MODELS = ("random_forest", "hist_gradient_boosting")
CONDITIONS = ("RAW", "MANUAL", "AUTOMATIC")
CLASSES = ("ONE_PROCESS", "TWO_PROCESS")
CONFIG = {"seed": 42, "frequency_model": "ridge", "frequency_grid_size": 16, "topology_grid_size": 64, "stage1_threshold": 10.0, "stage2_threshold": 4.0, "frequency_target": "robust", "validation": "strict nested six-fold LOSO", "topology_features": "64 log-frequency + 64 normalized Re(Z) + 64 normalized Im(Z)"}


def class_label(record):
    return "TWO_PROCESS" if "p(R2,CPE2)" in record.electrochemical_topology else "ONE_PROCESS"


def topology_models(seed):
    return {"random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1), "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, random_state=seed)}


def _targets_for(records, masks):
    rows = []
    for r, mask in zip(records, masks):
        f = np.sort(r.frequency[np.asarray(mask, dtype=bool)])
        active = np.asarray(mask, dtype=bool)[np.argsort(r.frequency, kind="mergesort")]
        if f.size < 3: f = np.sort(r.frequency)
        # Robust active-run target, matching the prior frequency-limit study.
        ordered = np.argsort(r.frequency, kind="mergesort"); a = np.asarray(mask, dtype=bool)[ordered]; fs = r.frequency[ordered]
        runs = []; starts = np.flatnonzero(a & ~np.r_[False, a[:-1]]); ends = np.flatnonzero(a & ~np.r_[a[1:], False])
        for s, e in zip(starts, ends):
            if e-s+1 >= 3: runs.append((s, e))
        low = fs[runs[0][0]] if runs else fs[np.flatnonzero(a)[0]]; high = fs[runs[-1][1]] if runs else fs[np.flatnonzero(a)[-1]]
        rows.append({"f_min": float(low), "f_max": float(high), "measured_f_min": float(fs[0]), "measured_f_max": float(fs[-1])})
    return rows


def _frequency_features(records, masks):
    cleaned = [_clean_record(r, m) for r, m in zip(records, masks)]
    extractor = SpectrumFeatureExtractor(CONFIG["frequency_grid_size"]); extractor.grid_ = np.linspace(-0.5, 5.0, CONFIG["frequency_grid_size"]); extractor.fill_ = np.zeros(CONFIG["frequency_grid_size"] * 12)
    return cleaned, np.vstack([extractor._one(r.frequency, r.impedance) for r in cleaned])


def _strict_windows(train_ids, test_ids, records, stage1, feature_by_id, target_by_id, model_name):
    def fit_predict(fit_ids, predict_ids):
        model = frequency_models(CONFIG["seed"])[model_name]; x = np.vstack([feature_by_id[s] for s in fit_ids]); y = target_values([target_by_id[s] for s in fit_ids]); model.fit(x, y); return model.predict(np.vstack([feature_by_id[s] for s in predict_ids]))
    test_prediction = fit_predict(train_ids, test_ids)
    train_prediction = {}
    for inner_sample in sorted({records[s].sample_id for s in train_ids}):
        inner_fit = [s for s in train_ids if records[s].sample_id != inner_sample]; inner_test = [s for s in train_ids if records[s].sample_id == inner_sample]
        for sid, pred in zip(inner_test, fit_predict(inner_fit, inner_test)): train_prediction[sid] = pred
    return {sid: (10**float(pred[0]), 10**float(pred[1])) for sid, pred in zip(test_ids, test_prediction)} | {sid: (10**float(pred[0]), 10**float(pred[1])) for sid, pred in train_prediction.items()}


def _metrics(frame):
    y = frame.true_class; p = frame.predicted_class; precision, recall, f1, support = precision_recall_fscore_support(y, p, labels=CLASSES, zero_division=0)
    probabilities = np.column_stack([frame.get(f"probability_{c}", pd.Series(0.0, index=frame.index)) for c in CLASSES]); truth = np.column_stack([(y == c).astype(float) for c in CLASSES])
    return {"spectra": len(frame), "accuracy": accuracy_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, p), "macro_f1": f1_score(y, p, labels=CLASSES, average="macro", zero_division=0), "brier": float(np.mean((probabilities-truth)**2)), "one_precision": precision[0], "one_recall": recall[0], "one_f1": f1[0], "two_precision": precision[1], "two_recall": recall[1], "two_f1": f1[1], "one_support": support[0], "two_support": support[1]}


def _plot(row, records, manual_masks, auto_masks, output, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = records[row.spectrum_id]; f = r.frequency; z = r.impedance; manual = manual_masks[row.spectrum_id]; auto = auto_masks[row.spectrum_id]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4)); axes[0].scatter(np.log10(f[manual]), np.abs(z[manual]), s=10, label="manual active"); axes[0].scatter(np.log10(f[~manual]), np.abs(z[~manual]), marker="x", color="tab:red", label="manual inactive"); axes[0].scatter(np.log10(f[auto]), np.abs(z[auto]), facecolors="none", edgecolors="tab:green", label="automatic active"); axes[0].set_title("Bode magnitude")
    axes[1].plot(np.log10(f), np.unwrap(np.angle(z)), ".-"); axes[1].set_title("Bode phase")
    axes[2].plot(z.real, -z.imag, ".-"); axes[2].set_title("Nyquist")
    true_class = row.get("true_class", row.get("true_class_manual", "unknown")); predicted_class = row.get("predicted_class", row.get("predicted_class_manual", "unknown"))
    fig.suptitle(f"{label}: true={true_class}, predicted={predicted_class}"); axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(output / f"{label}.png", dpi=130); plt.close(fig)


def evaluate(projects, output: Path):
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=True); (output / "plots").mkdir(exist_ok=True)
    mapping = {str(p): p.name.split(".")[0] for p in projects}; extraction = load_eisfit_projects(projects, mapping, require_fit=True); records = {r.spectrum_id: r for r in extraction.records}; manual_masks = _manual_masks(projects, records); records = {s:r for s,r in records.items() if s in manual_masks}
    stage1 = {}; stage1_masks = {}; auto_masks = {}; target_by_id = {}; feature_by_id = {}; cleaned_by_id = {}
    for sid, r in records.items():
        s1 = conservative_mask(r.frequency, r.impedance, threshold=CONFIG["stage1_threshold"]); stage1[sid] = s1; stage1_masks[sid] = s1.mask; cleaned_by_id[sid] = _clean_record(r, s1.mask); target_by_id[sid] = _targets_for([r], [manual_masks[sid]])[0]
    frequency_input = SpectrumFeatureExtractor(CONFIG["frequency_grid_size"]); frequency_input.grid_ = np.linspace(-0.5, 5.0, CONFIG["frequency_grid_size"]); frequency_input.fill_ = np.zeros(CONFIG["frequency_grid_size"]*12); feature_by_id = {s: frequency_input._one(r.frequency, r.impedance) for s,r in cleaned_by_id.items()}
    prediction_rows = []; mask_stats = []
    samples = sorted({r.sample_id for r in records.values()})
    for held_out in samples:
        train_ids = [s for s,r in records.items() if r.sample_id != held_out]; test_ids = [s for s,r in records.items() if r.sample_id == held_out]
        windows = _strict_windows(train_ids, test_ids, records, stage1, feature_by_id, target_by_id, CONFIG["frequency_model"])
        for sid, window in windows.items():
            r = records[sid]; s2 = sensitive_mask(r.frequency, r.impedance, window, threshold=CONFIG["stage2_threshold"]); envelope = (r.frequency >= window[0]) & (r.frequency <= window[1]); final = stage1_masks[sid] & envelope & s2.mask; auto_masks[sid] = final
            mask_stats.append({"spectrum_id": sid, "sample_id": r.sample_id, "stage1_rejected": int((~stage1_masks[sid]).sum()), "stage2_rejected_inside_envelope": int(np.sum(envelope & ~s2.mask)), "automatic_active_points": int(final.sum()), "manual_active_points": int(manual_masks[sid].sum()), "ml_f_min": window[0], "ml_f_max": window[1]})
        condition_records = {"RAW": {s: records[s] for s in train_ids+test_ids}, "MANUAL": {s: _clean_record(records[s], manual_masks[s]) for s in train_ids+test_ids}, "AUTOMATIC": {s: _clean_record(records[s], auto_masks[s]) for s in train_ids+test_ids}}
        for condition in CONDITIONS:
            train = [condition_records[condition][s] for s in train_ids]; test = [condition_records[condition][s] for s in test_ids]
            pre = SpectrumPreprocessor(grid_size=CONFIG["topology_grid_size"], spectrum_mode="raw"); x_train = pre.fit_transform(train); x_test = pre.transform(test); labels = [class_label(r) for r in train]
            for model_name, model in topology_models(CONFIG["seed"]).items():
                model.fit(x_train, labels); probs = model.predict_proba(x_test); model_classes = list(model.classes_); preds = model.predict(x_test)
                for r, pred, prob in zip(test, preds, probs):
                    row = {"spectrum_id": r.spectrum_id, "sample_id": r.sample_id, "voltage": r.voltage, "time": r.time, "current": r.current, "original_eec_topology": r.original_eec_topology, "canonical_topology": r.electrochemical_topology, "true_class": class_label(r), "predicted_class": str(pred), "preprocessing": condition, "model": model_name, "held_out_sample": held_out, "correct": bool(str(pred)==class_label(r))}
                    for c in CLASSES: row[f"probability_{c}"] = float(prob[model_classes.index(c)]) if c in model_classes else 0.0
                    prediction_rows.append(row)
    predictions = pd.DataFrame(prediction_rows); predictions.to_csv(output / "per_prediction.csv", index=False); pd.DataFrame(mask_stats).to_csv(output / "preprocessing_statistics.csv", index=False)
    overall_rows = []; sample_rows = []; class_rows = []
    for keys, part in predictions.groupby(["preprocessing", "model"]): overall_rows.append({"preprocessing": keys[0], "model": keys[1], **_metrics(part)})
    for keys, part in predictions.groupby(["preprocessing", "model", "held_out_sample"]): sample_rows.append({"preprocessing": keys[0], "model": keys[1], "held_out_sample": keys[2], **_metrics(part)})
    for keys, part in predictions.groupby(["preprocessing", "model", "true_class"]): class_rows.append({"preprocessing": keys[0], "model": keys[1], "true_class": keys[2], **_metrics(part)})
    overall = pd.DataFrame(overall_rows); overall.to_csv(output / "overall_metrics.csv", index=False); pd.DataFrame(sample_rows).to_csv(output / "per_sample_metrics.csv", index=False); pd.DataFrame(class_rows).to_csv(output / "per_class_metrics.csv", index=False)
    matrix_dir = output / "confusion_matrices"; matrix_dir.mkdir(exist_ok=True)
    for keys, part in predictions.groupby(["preprocessing", "model"]):
        target = matrix_dir / f"{keys[0]}_{keys[1]}"; target.mkdir(exist_ok=True); pd.DataFrame(confusion_matrix(part.true_class, part.predicted_class, labels=CLASSES), index=CLASSES, columns=CLASSES).to_csv(target / "aggregated.csv")
        for fold, fold_frame in part.groupby("held_out_sample"): pd.DataFrame(confusion_matrix(fold_frame.true_class, fold_frame.predicted_class, labels=CLASSES), index=CLASSES, columns=CLASSES).to_csv(target / f"held_out_{fold}.csv")
    manual = predictions[(predictions.preprocessing=="MANUAL")]; auto = predictions[(predictions.preprocessing=="AUTOMATIC")]; joined = manual.merge(auto, on=["spectrum_id", "model"], suffixes=("_manual", "_automatic")); errors = joined[(joined.correct_manual != joined.correct_automatic)].copy(); errors.to_csv(output / "error_analysis.csv", index=False)
    for label, row in [("automatic_correct", auto[auto.correct].iloc[0] if auto.correct.any() else None), ("automatic_incorrect", auto[~auto.correct].iloc[0] if (~auto.correct).any() else None), ("manual_correct_automatic_incorrect", errors[(errors.correct_manual)&(~errors.correct_automatic)].iloc[0] if not errors[(errors.correct_manual)&(~errors.correct_automatic)].empty else None), ("manual_incorrect_automatic_correct", errors[(~errors.correct_manual)&(errors.correct_automatic)].iloc[0] if not errors[(~errors.correct_manual)&(errors.correct_automatic)].empty else None)]:
        if row is not None:
            try: _plot(row, records, manual_masks, auto_masks, output / "plots", label)
            except Exception as error: (output / "plots" / "plot_failures.log").open("a", encoding="utf-8").write(f"{label}: {type(error).__name__}: {error}\n")
    config = {**CONFIG, "models": {"random_forest": "300 trees, balanced, seed 42", "hist_gradient_boosting": "150 iterations, seed 42"}, "dataset_samples": PROJECT_SAMPLES, "records": len(records), "preprocessing_source": "new automatic preprocessing primitives; fold-local robust frequency targets; nested inner LOSO for topology training spectra"}; (output / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    runtime = time.perf_counter()-started; report = {"records": len(records), "runtime_s": runtime, "strict_nested": True, "topology_training_started": True, "parameter_prediction_started": False}; (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(overall.to_string(index=False)); print(json.dumps(report, indent=2)); return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("projects", nargs="+", type=Path); parser.add_argument("--output", type=Path, default=Path("ml/analysis/topology_automatic_preprocessing")); args=parser.parse_args(); evaluate(args.projects, args.output)


if __name__ == "__main__": main()
