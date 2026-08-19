"""Strict six-sample LOSO benchmark for ML EEC parameter initialization."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eis_project import dataframe_from_payload
from eis_services import circuit_parameters, load_cycle

from .dataset import _payload_projects, load_eisfit_projects
from .evaluate_frequency_limit_ml import _manual_masks, _clean_record
from .parameter_prediction import (
    ALPHA_PARAMETERS, PARAMETERS, TOPOLOGY_PARAMETERS, FoldFeatureBuilder,
    bounds_from_residuals, bound_metrics, inverse_target, model_factories, residual_quantiles,
    parameter_mapping, residual_values, topology_for_circuit, transform_target,
)


TRAINING_SAMPLES = ("129", "140", "150", "157", "159", "181")
REPRESENTATIONS = ("MANUAL", "AUTOMATIC")
FEATURE_SETS = ("SPECTRUM_ONLY", "SPECTRUM_VOLTAGE", "SPECTRUM_VOLTAGE_CURRENT", "SPECTRUM_VOLTAGE_CURRENT_TIME", "VOLTAGE_ONLY")
MODELS = ("ridge", "random_forest", "hist_gradient_boosting")
AUTO_MASK_PATH = Path("ml/analysis/frequency_limit_ml/automatic_active_masks.csv")
TOPOLOGY_PREDICTIONS_PATH = Path("ml/analysis/topology_automatic_preprocessing/per_prediction.csv")


def extract_parameter_targets(projects, records):
    """Extract fit values by actual circuit parameter names; never infer L0."""
    record_ids = {r.spectrum_id for r in records}
    rows = {}
    for path in projects:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for index, (state, entry) in enumerate(_payload_projects(payload)):
            if not entry.get("dataframe"):
                continue
            dataset_key = str(entry.get("dataset_id") or f"dataset_{index}")
            control = str(state.get("control", payload.get("control", "cell")))
            for cycle_text, saved in (state.get("cycles") or {}).items():
                sid = f"{Path(path).resolve()}::{dataset_key}::{control}::{int(cycle_text)}"
                fit = saved.get("fit_parameters")
                circuit = str(saved.get("circuit") or state.get("circuit") or "")
                if sid not in record_ids or fit is None or not circuit:
                    continue
                names = [p.name for p in circuit_parameters(circuit)]
                values = np.asarray(fit, dtype=float)
                if values.size != len(names) or not np.isfinite(values).all():
                    continue
                targets = parameter_mapping(circuit, values, names)
                if any((not np.isfinite(value) or value <= 0) for name, value in targets.items() if name not in ALPHA_PARAMETERS):
                    continue
                if any((not np.isfinite(value) or not 0 < value < 1) for name, value in targets.items() if name in ALPHA_PARAMETERS):
                    continue
                rows[sid] = {"spectrum_id": sid, "sample": sid.split("::", 1)[0].split("\\")[-1].split(".")[0], "topology": topology_for_circuit(circuit), **targets}
    missing = record_ids - set(rows)
    if missing:
        raise RuntimeError(f"missing fitted parameter targets for {len(missing)} loaded spectra")
    return pd.DataFrame([rows[r.spectrum_id] for r in records])


def load_automatic_masks(records, path=AUTO_MASK_PATH):
    frame = pd.read_csv(path)
    required = {"spectrum_id", "point_index", "automatic_active"}
    if not required.issubset(frame.columns):
        raise ValueError(f"automatic mask artifact lacks {required - set(frame.columns)}")
    grouped = {sid: group for sid, group in frame.groupby("spectrum_id", sort=False)}
    result = {}
    for record in records:
        group = grouped.get(record.spectrum_id)
        if group is None or len(group) != len(record.frequency):
            raise RuntimeError(f"automatic mask missing or mis-sized for {record.spectrum_id}")
        group = group.sort_values("point_index")
        indices = group.point_index.to_numpy(int)
        if not np.array_equal(indices, np.arange(len(record.frequency))):
            raise RuntimeError(f"automatic mask point indices are not contiguous for {record.spectrum_id}")
        result[record.spectrum_id] = group.automatic_active.to_numpy(bool)
    return result


def _record_views(records, projects):
    manual = _manual_masks(projects, {r.spectrum_id: r for r in records})
    views = {"MANUAL": {}, "AUTOMATIC": {}}
    automatic = load_automatic_masks(records)
    for record in records:
        if record.spectrum_id not in manual:
            raise RuntimeError(f"manual mask missing for {record.spectrum_id}")
        mask = manual[record.spectrum_id] & (record.frequency >= record.manual_f_min) & (record.frequency <= record.manual_f_max)
        views["MANUAL"][record.spectrum_id] = _clean_record(record, mask)
        views["AUTOMATIC"][record.spectrum_id] = _clean_record(record, automatic[record.spectrum_id])
        if len(views["MANUAL"][record.spectrum_id].frequency) < 3 or len(views["AUTOMATIC"][record.spectrum_id].frequency) < 3:
            raise RuntimeError(f"too few active points for {record.spectrum_id}")
    return views, manual, automatic


def _model_metrics(frame):
    if frame.empty:
        return {"spectra": 0}
    parameter = frame.parameter.iloc[0]
    true = frame.true_value.to_numpy(float); pred = frame.predicted_value.to_numpy(float)
    if parameter in ALPHA_PARAMETERS:
        error = pred - true; absolute = np.abs(error); return {"spectra": len(frame), "mae": float(np.mean(absolute)), "rmse": float(np.sqrt(np.mean(error**2))), "median_absolute_error": float(np.median(absolute)), "within_0.02": float(np.mean(absolute <= .02)), "within_0.05": float(np.mean(absolute <= .05)), "within_0.10": float(np.mean(absolute <= .10)), "r2": float(1 - np.sum(error**2) / max(np.sum((true - true.mean())**2), np.finfo(float).eps))}
    log_true = np.log10(true); log_pred = np.log10(pred); error = log_pred - log_true; absolute = np.abs(error); return {"spectra": len(frame), "mae_log10": float(np.mean(absolute)), "rmse_log10": float(np.sqrt(np.mean(error**2))), "median_absolute_log10": float(np.median(absolute)), "within_x1.25": float(np.mean(absolute <= np.log10(1.25))), "within_x2": float(np.mean(absolute <= np.log10(2))), "within_x5": float(np.mean(absolute <= np.log10(5))), "within_x10": float(np.mean(absolute <= 1.0)), "r2": float(1 - np.sum(error**2) / max(np.sum((log_true - log_true.mean())**2), np.finfo(float).eps))}


def _fit_one(train_rows, test_rows, train_records, test_records, parameter, feature_set, model_name):
    builder = FoldFeatureBuilder(feature_set).fit(train_records)
    x_train, x_test = builder.transform(train_records), builder.transform(test_records)
    true_train = np.asarray([r[parameter] for r in train_rows], dtype=float)
    true_test = np.asarray([r[parameter] for r in test_rows], dtype=float)
    if model_name == "global_median":
        transformed_pred_train = np.full(true_train.size, np.median(transform_target(true_train, parameter)))
        transformed_pred_test = np.full(true_test.size, np.median(transform_target(true_train, parameter)))
    else:
        model = model_factories()[model_name](); model.fit(x_train, transform_target(true_train, parameter))
        transformed_pred_train = model.predict(x_train); transformed_pred_test = model.predict(x_test)
    pred_train, pred_test = inverse_target(transformed_pred_train, parameter), inverse_target(transformed_pred_test, parameter)
    residuals = residual_values(true_train, pred_train, parameter)
    bound_train = bounds_from_residuals(pred_test, residuals, parameter)
    return pred_test, bound_train, residuals, builder, (model if model_name != "global_median" else None)


def _fit_from_features(train_rows, test_rows, x_train, x_test, parameter, model_name):
    true_train = np.asarray([r[parameter] for r in train_rows], dtype=float)
    true_test = np.asarray([r[parameter] for r in test_rows], dtype=float)
    if model_name == "global_median":
        transformed_pred_train = np.full(true_train.size, np.median(transform_target(true_train, parameter)))
        transformed_pred_test = np.full(true_test.size, np.median(transform_target(true_train, parameter)))
    else:
        model = model_factories()[model_name](); model.fit(x_train, transform_target(true_train, parameter))
        transformed_pred_train = model.predict(x_train); transformed_pred_test = model.predict(x_test)
    pred_train, pred_test = inverse_target(transformed_pred_train, parameter), inverse_target(transformed_pred_test, parameter)
    residuals = residual_values(true_train, pred_train, parameter)
    return pred_test, bounds_from_residuals(pred_test, residuals, parameter), residuals


def _plot_best(frame, output, parameter):
    if frame.empty:
        return
    output.mkdir(parents=True, exist_ok=True)
    true, pred = frame.true_value.to_numpy(float), frame.predicted_value.to_numpy(float)
    fig, ax = plt.subplots(figsize=(5, 4)); ax.scatter(true, pred, s=5, alpha=.45); low, high = np.nanmin([true.min(), pred.min()]), np.nanmax([true.max(), pred.max()]); ax.plot([low, high], [low, high], "k--"); ax.set_xlabel("manually fitted"); ax.set_ylabel("predicted"); ax.set_title(parameter); fig.tight_layout(); fig.savefig(output / "predicted_vs_true" / f"{parameter}.png", dpi=130); plt.close(fig)
    for column, folder, xlabel in (("voltage", "residual_vs_voltage", "voltage"), ("time", "residual_vs_time", "time"), ("true_value", "residual_vs_parameter", "manually fitted")):
        fig, ax = plt.subplots(figsize=(5, 4)); ax.scatter(frame[column], frame.prediction_error, s=5, alpha=.45); ax.axhline(0, color="k", linestyle="--"); ax.set_xlabel(xlabel); ax.set_ylabel("prediction error"); fig.tight_layout(); fig.savefig(output / folder / f"{parameter}.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); ax.hist(frame.prediction_error, bins=40); ax.set_xlabel("prediction error"); fig.tight_layout(); fig.savefig(output / "residual_histogram" / f"{parameter}.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); ax.scatter(frame.predicted_value, frame.upper_95 - frame.lower_95, s=5, alpha=.45); ax.set_xlabel("predicted"); ax.set_ylabel("95% interval width"); fig.tight_layout(); fig.savefig(output / "bounds" / f"{parameter}.png", dpi=130); plt.close(fig)


def _metadata_feature_arrays(feature_set, train_rows, test_rows, train_spectrum, test_spectrum):
    if feature_set == "SPECTRUM_ONLY":
        return train_spectrum, test_spectrum
    names = {"SPECTRUM_VOLTAGE": ("voltage",), "SPECTRUM_VOLTAGE_CURRENT": ("voltage", "current"), "SPECTRUM_VOLTAGE_CURRENT_TIME": ("voltage", "current", "time"), "VOLTAGE_ONLY": ("voltage",)}[feature_set]
    train_meta = np.asarray([[row[name] for name in names] for row in train_rows], dtype=float)
    test_meta = np.asarray([[row[name] for name in names] for row in test_rows], dtype=float)
    fill = np.nanmedian(train_meta, axis=0); fill[~np.isfinite(fill)] = 0.0
    train_meta = np.where(np.isfinite(train_meta), train_meta, fill); test_meta = np.where(np.isfinite(test_meta), test_meta, fill)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train_meta); train_meta, test_meta = scaler.transform(train_meta), scaler.transform(test_meta)
    if feature_set == "VOLTAGE_ONLY":
        return train_meta, test_meta
    return np.hstack([train_spectrum, train_meta]), np.hstack([test_spectrum, test_meta])


def evaluate(projects, output: Path):
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=True)
    mapping = {str(p): p.name.split(".")[0] for p in projects}
    extraction = load_eisfit_projects(projects, mapping, require_fit=True)
    records = extraction.records
    if set(r.sample_id for r in records) != set(TRAINING_SAMPLES):
        raise RuntimeError("loaded dataset does not contain exactly the six training samples")
    targets = extract_parameter_targets(projects, records)
    record_by_id = {r.spectrum_id: r for r in records}
    views, manual_masks, auto_masks = _record_views(records, projects)
    for record in records:
        for representation in REPRESENTATIONS:
            record_by_id[record.spectrum_id].__dict__ if False else None
    observations = []
    for row in targets.to_dict("records"):
        row.update({"voltage": record_by_id[row["spectrum_id"]].voltage, "current": record_by_id[row["spectrum_id"]].current, "time": record_by_id[row["spectrum_id"]].time, "sample": record_by_id[row["spectrum_id"]].sample_id})
        observations.append(row)
    topology_predictions = None
    if TOPOLOGY_PREDICTIONS_PATH.exists():
        topology_predictions = pd.read_csv(TOPOLOGY_PREDICTIONS_PATH)
        topology_predictions = topology_predictions[(topology_predictions.preprocessing == "AUTOMATIC") & (topology_predictions.model == "hist_gradient_boosting")].set_index("spectrum_id")
    prediction_rows, bound_rows = [], []
    prediction_work = output / "_predictions_work.csv"; bound_work = output / "_bounds_work.csv"
    for work_path in (prediction_work, bound_work):
        if work_path.exists(): work_path.unlink()
    samples = sorted(set(TRAINING_SAMPLES))
    for representation in REPRESENTATIONS:
        fold_cache = {}
        # The spectral grid is independent of metadata feature set and of
        # parameter within each availability group; fit it once per fold.
        for held_out in samples:
            train_obs = [row for row in observations if row["sample"] != held_out]; test_obs = [row for row in observations if row["sample"] == held_out]
            all_train_records = [views[representation][row["spectrum_id"]] for row in train_obs]
            all_test_records = [views[representation][row["spectrum_id"]] for row in test_obs]
            spectrum_builder = FoldFeatureBuilder("SPECTRUM_ONLY").fit(all_train_records)
            all_spectrum_train = spectrum_builder.transform(all_train_records); all_spectrum_test = spectrum_builder.transform(all_test_records)
            group_cache = {}
            for group in ("ALL", "TWO_PROCESS"):
                group_train = [row for row in train_obs if group == "ALL" or row["topology"] == group]; group_test = [row for row in test_obs if group == "ALL" or row["topology"] == group]
                train_indices = [train_obs.index(row) for row in group_train]; test_indices = [test_obs.index(row) for row in group_test]
                group_cache[group] = (group_train, group_test, all_spectrum_train[train_indices], all_spectrum_test[test_indices])
            fold_cache[held_out] = group_cache
        for feature_set in FEATURE_SETS:
            for held_out in samples:
                for parameter in PARAMETERS:
                    group = "TWO_PROCESS" if parameter in {"R2", "Q2", "alpha2"} else "ALL"
                    group_train, group_test, spectrum_train, spectrum_test = fold_cache[held_out][group]
                    train_obs_p = [row for row in group_train if parameter in row and np.isfinite(row[parameter])]; test_obs_p = [row for row in group_test if parameter in row and np.isfinite(row[parameter])]
                    if not train_obs_p or not test_obs_p: continue
                    train_indices = [group_train.index(row) for row in train_obs_p]; test_indices = [group_test.index(row) for row in test_obs_p]
                    base_train, base_test = spectrum_train[train_indices], spectrum_test[test_indices]
                    x_train, x_test = _metadata_feature_arrays(feature_set, train_obs_p, test_obs_p, base_train, base_test)
                    model_names = ("global_median", "ridge", "random_forest", "hist_gradient_boosting")
                    for model_name in model_names:
                        if model_name == "global_median" and feature_set != "SPECTRUM_ONLY": continue
                        actual_feature_set = "SPECTRUM_ONLY" if model_name == "global_median" else feature_set
                        if model_name == "global_median":
                            pred, bounds, residuals = _fit_from_features(train_obs_p, test_obs_p, np.empty((len(train_obs_p), 0)), np.empty((len(test_obs_p), 0)), parameter, model_name)
                        else:
                            pred, bounds, residuals = _fit_from_features(train_obs_p, test_obs_p, x_train, x_test, parameter, model_name)
                        for row_index, (row, predicted) in enumerate(zip(test_obs_p, pred)):
                            base = {"sample": held_out, "spectrum_id": row["spectrum_id"], "topology": row["topology"], "parameter": parameter, "true_value": row[parameter], "predicted_value": float(predicted), "prediction_error": float(predicted - row[parameter]), "prediction_error_log": float(transform_target([predicted], parameter)[0] - transform_target([row[parameter]], parameter)[0]), "representation": representation, "feature_set": actual_feature_set, "model": model_name, "voltage": row["voltage"], "current": row["current"], "time": row["time"], "held_out_sample": held_out}
                            for level, (lower, upper, _clipped) in bounds.items(): base[f"lower_{level}"] = float(lower[row_index]); base[f"upper_{level}"] = float(upper[row_index])
                            prediction_rows.append(base)
                            for level, (lower, upper, clipped) in bounds.items():
                                metrics = bound_metrics(np.asarray([row[parameter]]), np.asarray([lower[row_index]]), np.asarray([upper[row_index]]))
                                bound_rows.append({"representation": representation, "feature_set": actual_feature_set, "model": model_name, "parameter": parameter, "sample": held_out, "interval": level, "true_value": row[parameter], "lower": lower[row_index], "upper": upper[row_index], "clipped": clipped, **metrics})
            if prediction_rows:
                pd.DataFrame(prediction_rows).to_csv(prediction_work, mode="a", header=not prediction_work.exists(), index=False)
                pd.DataFrame(bound_rows).to_csv(bound_work, mode="a", header=not bound_work.exists(), index=False)
                prediction_rows.clear(); bound_rows.clear()
    predictions = pd.read_csv(prediction_work); bounds = pd.read_csv(bound_work)
    predictions.to_csv(output / "predictions.csv", index=False)
    bounds.to_csv(output / "bound_metrics.csv", index=False)
    metric_rows = []
    for keys, frame in predictions.groupby(["representation", "feature_set", "model", "parameter"]): metric_rows.append({"representation": keys[0], "feature_set": keys[1], "model": keys[2], "parameter": keys[3], **_model_metrics(frame)})
    metrics = pd.DataFrame(metric_rows); metrics.to_csv(output / "per_parameter_metrics.csv", index=False)
    per_sample = []
    for keys, frame in predictions.groupby(["representation", "feature_set", "model", "parameter", "sample"]): per_sample.append({"representation": keys[0], "feature_set": keys[1], "model": keys[2], "parameter": keys[3], "sample": keys[4], **_model_metrics(frame)})
    pd.DataFrame(per_sample).to_csv(output / "per_sample_metrics.csv", index=False)
    overall = []
    for keys, frame in predictions.groupby(["representation", "feature_set", "model"]):
        values = {"representation": keys[0], "feature_set": keys[1], "model": keys[2], "parameters": int(frame.parameter.nunique()), "rows": len(frame)}
        values["mean_absolute_error"] = float(np.mean(np.abs(frame.prediction_error)))
        overall.append(values)
    pd.DataFrame(overall).to_csv(output / "overall_metrics.csv", index=False)
    predictions.groupby(["representation", "feature_set", "model"], as_index=False).prediction_error.agg(["mean", "std"]).to_csv(output / "model_comparison.csv", index=False)
    predictions.groupby(["representation", "feature_set", "parameter"], as_index=False).mean(numeric_only=True).to_csv(output / "feature_set_comparison.csv", index=False)
    # Select deployment candidates using automatic LOSO transformed-space MAE.
    auto_metrics = metrics[metrics.representation == "AUTOMATIC"].copy(); best_rows = []
    for parameter in PARAMETERS:
        part = auto_metrics[auto_metrics.parameter == parameter]
        if part.empty: continue
        score_col = "mae" if parameter in ALPHA_PARAMETERS else "mae_log10"
        best = part.sort_values(score_col).iloc[0]; best_rows.append(best.to_dict())
    best_frame = pd.DataFrame(best_rows); best_frame.to_csv(output / "best_models.csv", index=False)
    # Fit and save final artifacts only after the LOSO benchmark is complete.
    model_root = output / "parameter_models"
    final_model_rows, bound_config = [], {}
    for representation in REPRESENTATIONS:
        rep_dir = model_root / representation.lower(); rep_dir.mkdir(parents=True, exist_ok=True)
        rep_best = metrics[metrics.representation == representation]
        for parameter in PARAMETERS:
            choices = rep_best[rep_best.parameter == parameter]
            if choices.empty:
                continue
            score_col = "mae" if parameter in ALPHA_PARAMETERS else "mae_log10"
            selected = choices.sort_values(score_col).iloc[0]
            rows_for_parameter = [row for row in observations if parameter in row and np.isfinite(row[parameter])]
            final_records = [views[representation][row["spectrum_id"]] for row in rows_for_parameter]
            builder = FoldFeatureBuilder(str(selected.feature_set)).fit(final_records)
            x_final = builder.transform(final_records); y_final = transform_target([row[parameter] for row in rows_for_parameter], parameter)
            selected_model = str(selected.model)
            if selected_model == "global_median":
                final_model = {"kind": "global_median", "value_transformed": float(np.median(y_final))}; train_prediction_transformed = np.full(len(y_final), final_model["value_transformed"])
            else:
                final_model = model_factories()[selected_model](); final_model.fit(x_final, y_final); train_prediction_transformed = final_model.predict(x_final)
            train_prediction = inverse_target(train_prediction_transformed, parameter)
            residuals = residual_values([row[parameter] for row in rows_for_parameter], train_prediction, parameter)
            key = f"{representation}:{parameter}"
            bound_config[key] = {str(level): list(residual_quantiles(residuals, level)) for level in (0.90, 0.95, 0.99)}
            artifact = {"model": final_model, "feature_builder": builder, "parameter": parameter, "representation": representation, "feature_set": selected.feature_set, "model_name": selected_model, "target_transformation": "log10" if parameter not in ALPHA_PARAMETERS else "logit_alpha", "training_samples": list(TRAINING_SAMPLES), "training_spectra": len(rows_for_parameter), "residual_quantiles": bound_config[key]}
            path = rep_dir / f"parameter_{parameter}_{selected_model}_{str(selected.feature_set).lower()}.joblib"; joblib.dump(artifact, path)
            final_model_rows.append({"representation": representation, "parameter": parameter, "feature_set": selected.feature_set, "model": selected_model, "path": str(path), "training_spectra": len(rows_for_parameter)})
    (output / "parameter_model_config.json").write_text(json.dumps({"training_samples": list(TRAINING_SAMPLES), "models": final_model_rows, "feature_dimension": {"SPECTRUM_ONLY": 192, "SPECTRUM_VOLTAGE": 193, "SPECTRUM_VOLTAGE_CURRENT": 194, "SPECTRUM_VOLTAGE_CURRENT_TIME": 195, "VOLTAGE_ONLY": 1}, "sample_178_used": False}, indent=2), encoding="utf-8")
    (output / "parameter_bound_config.json").write_text(json.dumps({"method": "training residual quantiles", "levels": [0.90, 0.95, 0.99], "space": "log10 for positive parameters; raw alpha residuals", "bounds": bound_config}, indent=2), encoding="utf-8")
    pd.DataFrame(final_model_rows).to_csv(output / "final_model_manifest.csv", index=False)
    for parameter in PARAMETERS:
        part = predictions[(predictions.representation == "AUTOMATIC") & (predictions.parameter == parameter)]
        if part.empty: continue
        best = best_frame[best_frame.parameter == parameter].iloc[0]
        selected = part[(part.feature_set == best.feature_set) & (part.model == best.model)]
        _plot_best(selected, output / "plots", parameter)
    # Bounds summarized by interval and parameter for the deployment selection.
    selected_bound = bounds.merge(best_frame[["parameter", "feature_set", "model"]], on=["parameter", "feature_set", "model"], how="inner")
    selected_bound.groupby(["representation", "parameter", "interval"], as_index=False).agg(coverage=("coverage", "mean"), median_interval_width=("upper", lambda x: float(np.median(x))), mean_interval_width=("upper", "mean")).to_csv(output / "selected_bound_summary.csv", index=False)
    # Error-correlation diagnostic for parameters present in the same topology.
    corr_rows = []
    selected_predictions = predictions[(predictions.representation == "AUTOMATIC") & (predictions.model.isin(best_frame.model))]
    for keys, frame in selected_predictions.groupby(["representation", "feature_set", "model", "sample"]):
        wide = frame.pivot_table(index="spectrum_id", columns="parameter", values="prediction_error_log")
        for left, right in (("R1", "Q1"), ("R2", "Q2"), ("R1", "R2")):
            if left in wide and right in wide and wide[[left, right]].dropna().shape[0] >= 3: corr_rows.append({"representation": keys[0], "feature_set": keys[1], "model": keys[2], "sample": keys[3], "parameter_left": left, "parameter_right": right, "correlation": float(wide[[left, right]].corr().iloc[0, 1])})
    pd.DataFrame(corr_rows).to_csv(output / "parameter_error_correlations.csv", index=False)
    if topology_predictions is not None:
        diagnostic_rows = []
        for row in predictions.to_dict("records"):
            predicted_row = topology_predictions.loc[row["spectrum_id"]] if row["spectrum_id"] in topology_predictions.index else None
            if predicted_row is None:
                continue
            predicted_topology = "TWO_PROCESS" if str(predicted_row.predicted_class) == "TWO_PROCESS" else "ONE_PROCESS"
            if row["parameter"] not in TOPOLOGY_PARAMETERS[predicted_topology]:
                continue
            diagnostic_rows.append({**row, "topology_mode": "PREDICTED_TOPOLOGY", "topology_used": predicted_topology})
        diagnostic = pd.DataFrame(diagnostic_rows); diagnostic.to_csv(output / "predicted_topology_predictions.csv", index=False)
        diagnostic_metrics = []
        for keys, frame in diagnostic.groupby(["representation", "feature_set", "model", "parameter"]): diagnostic_metrics.append({"representation": keys[0], "feature_set": keys[1], "model": keys[2], "parameter": keys[3], "topology_mode": "PREDICTED_TOPOLOGY", "rows": len(frame), **_model_metrics(frame)})
        pd.DataFrame(diagnostic_metrics).to_csv(output / "predicted_topology_metrics.csv", index=False)
    config = {"training_samples": list(TRAINING_SAMPLES), "training_spectra": len(records), "representations": list(REPRESENTATIONS), "feature_sets": list(FEATURE_SETS), "models": list(MODELS) + ["global_median"], "model_budget": {"random_forest": "12 trees, min_samples_leaf=3, random_state=42", "hist_gradient_boosting": "12 iterations, learning_rate=0.06, l2_regularization=1.0, random_state=42", "ridge": "StandardScaler + Ridge(alpha=10)"}, "feature_dimension": {"SPECTRUM_ONLY": 192, "SPECTRUM_VOLTAGE": 193, "SPECTRUM_VOLTAGE_CURRENT": 194, "SPECTRUM_VOLTAGE_CURRENT_TIME": 195, "VOLTAGE_ONLY": 1}, "parameter_mapping": "CPE1_0->Q1, CPE1_1->alpha1, CPE2_0->Q2, CPE2_1->alpha2; L0 excluded", "target_transformations": {p: ("log10" if p not in ALPHA_PARAMETERS else "logit(alpha), inverse sigmoid") for p in PARAMETERS}, "bound_method": "training-fold residual quantiles at 90/95/99 percent; positive parameters in log10 space, alpha in raw space; physical clipping", "automatic_mask_artifact": str(AUTO_MASK_PATH), "topology_diagnostic_artifact": str(TOPOLOGY_PREDICTIONS_PATH), "sample_178_used": False, "conventional_eec_fitting": False, "runtime_s": time.perf_counter() - started}
    config["final_model_config"] = str(output / "parameter_model_config.json")
    config["final_bound_config"] = str(output / "parameter_bound_config.json")
    config["predicted_topology_diagnostic"] = bool(topology_predictions is not None)
    (output / "stage4_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps({**config, "extraction_exclusions": extraction.exclusion_counts, "target_parameters": sorted(set(targets.columns) - {"spectrum_id", "sample", "topology"}), "best_models": best_rows}, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "predictions": len(predictions), "runtime_s": config["runtime_s"], "best_models": best_rows}, indent=2))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projects", nargs=6, type=Path)
    parser.add_argument("--output", type=Path, default=Path("ml/analysis/stage4_parameters"))
    args = parser.parse_args()
    evaluate(list(args.projects), args.output)


if __name__ == "__main__":
    main()
