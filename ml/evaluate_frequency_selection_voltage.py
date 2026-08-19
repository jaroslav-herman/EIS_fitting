"""LOSO comparison of active-mask frequency targets and voltage-aware models."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .automatic_preprocessing import active_boundary_targets, binary_metrics, conservative_mask, sensitive_mask
from .dataset import load_eisfit_projects
from .evaluate_frequency_limit_ml import _clean_record, _manual_masks
from .frequency_limit_ml import SpectrumFeatureExtractor, models as frequency_models, target_values


FEATURE_SETS = ("spectrum_only", "spectrum_plus_voltage", "spectrum_plus_voltage_window", "voltage_only")
TARGET_SETS = ("old_frequency_window", "new_active_window")
MODELS = ("ridge", "random_forest", "hist_gradient_boosting")
CONFIG = {"seed": 42, "grid_size": 16, "stage1_threshold": 10.0, "stage2_threshold": 4.0, "active_persistence": 3, "features": FEATURE_SETS, "targets": TARGET_SETS}


def _targets(record, manual):
    info = active_boundary_targets(record.frequency, manual, persistence=CONFIG["active_persistence"]); stored_min, stored_max = float(record.manual_f_min), float(record.manual_f_max)
    robust_min = max(stored_min, float(info["robust_f_min"])); robust_max = min(stored_max, float(info["robust_f_max"]))
    if robust_max <= robust_min: robust_min, robust_max = stored_min, stored_max
    return {"old_frequency_window": {"f_min": stored_min, "f_max": stored_max, "measured_f_min": info["measured_f_min"], "measured_f_max": info["measured_f_max"]}, "new_active_window": {"f_min": robust_min, "f_max": robust_max, "measured_f_min": info["measured_f_min"], "measured_f_max": info["measured_f_max"]}, "literal_active": {"f_min": float(info["literal_f_min"]), "f_max": float(info["literal_f_max"])}}


def _feature_matrix(ids, base, records, feature_set, train_ids=None):
    if feature_set == "spectrum_only": return np.vstack([base[s] for s in ids])
    if feature_set == "voltage_only": raw = np.asarray([[records[s].voltage] for s in ids], dtype=float)
    else:
        raw = np.vstack([base[s] for s in ids]); extra = []
        voltage = np.asarray([[records[s].voltage] for s in ids], dtype=float); extra.append(voltage)
        if feature_set == "spectrum_plus_voltage_window": extra.append(np.asarray([[np.log10(records[s].manual_f_min), np.log10(records[s].manual_f_max)] for s in ids], dtype=float))
        raw = np.hstack([raw, *extra])
    fit_ids = train_ids or ids; scaler = StandardScaler(); fit_values = np.asarray([[records[s].voltage] if feature_set == "voltage_only" else [records[s].voltage, *([np.log10(records[s].manual_f_min), np.log10(records[s].manual_f_max)] if feature_set == "spectrum_plus_voltage_window" else [])] for s in fit_ids], dtype=float); scaler.fit(fit_values)
    if feature_set == "voltage_only": return scaler.transform(raw)
    base_values = np.asarray([[records[s].voltage, *([np.log10(records[s].manual_f_min), np.log10(records[s].manual_f_max)] if feature_set == "spectrum_plus_voltage_window" else [])] for s in ids], dtype=float)
    return np.hstack([np.vstack([base[s] for s in ids]), scaler.transform(base_values)])


def _metrics(frame, name):
    error = frame[f"error_{name}_decades"].to_numpy(float); absolute = np.abs(error)
    return {"MAE_decades": float(absolute.mean()), "median_abs_decades": float(np.median(absolute)), "RMSE_decades": float(np.sqrt(np.mean(error**2))), "mean_signed_error_decades": float(error.mean()), "median_signed_error_decades": float(np.median(error)), "within_0.05_percent": float(100*np.mean(absolute<=.05)), "within_0.10_percent": float(100*np.mean(absolute<=.10)), "within_0.20_percent": float(100*np.mean(absolute<=.20)), "within_0.50_percent": float(100*np.mean(absolute<=.50))}


def _plot_diagnostics(frame, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in ("f_min", "f_max"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4)); target = frame[f"target_{name}"].to_numpy(); pred = frame[f"predicted_{name}"].to_numpy(); error = frame[f"error_{name}_decades"].to_numpy()
        axes[0].scatter(frame.voltage, target, s=5); axes[0].set_title(f"Voltage vs manual {name}"); axes[0].set_xlabel("Voltage")
        axes[1].scatter(frame.voltage, pred, s=5); axes[1].set_title(f"Voltage vs predicted {name}"); axes[1].set_xlabel("Voltage"); fig.tight_layout(); fig.savefig(output / f"voltage_vs_{name}.png", dpi=130); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4)); axes[0].scatter(target, pred, s=5); axes[0].plot([target.min(), target.max()], [target.min(), target.max()], "k--"); axes[0].set_title(f"Predicted vs target {name}"); axes[1].hist(error, bins=40); axes[1].set_title(f"{name} signed error"); fig.tight_layout(); fig.savefig(output / f"{name}_prediction_diagnostics.png", dpi=130); plt.close(fig)


def evaluate(projects, output: Path):
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=True); plots = output / "plots"; plots.mkdir(exist_ok=True)
    mapping = {str(p): p.name.split(".")[0] for p in projects}; extraction = load_eisfit_projects(projects, mapping, require_fit=True); manual = _manual_masks(projects, {r.spectrum_id:r for r in extraction.records}); records = {r.spectrum_id:r for r in extraction.records if r.spectrum_id in manual}; stage1 = {s: conservative_mask(r.frequency, r.impedance, threshold=CONFIG["stage1_threshold"]) for s,r in records.items()}; cleaned = {s: _clean_record(r, stage1[s].mask) for s,r in records.items()}
    extractor = SpectrumFeatureExtractor(CONFIG["grid_size"]); extractor.grid_ = np.linspace(-0.5, 5.0, CONFIG["grid_size"]); extractor.fill_ = np.zeros(CONFIG["grid_size"]*12); base = {s: extractor._one(cleaned[s].frequency, cleaned[s].impedance) for s in records}; targets = {s: _targets(records[s], manual[s]) for s in records}; prediction_rows=[]; samples=sorted({r.sample_id for r in records.values()})
    for held_out in samples:
        train_ids=[s for s,r in records.items() if r.sample_id != held_out]; test_ids=[s for s,r in records.items() if r.sample_id == held_out]
        for target_name in TARGET_SETS:
            for feature_set in FEATURE_SETS:
                x_train=_feature_matrix(train_ids,base,records,feature_set,train_ids); x_test=_feature_matrix(test_ids,base,records,feature_set,train_ids)
                y=target_values([targets[s][target_name] for s in train_ids])
                for model_name in MODELS:
                    model=frequency_models(CONFIG["seed"])[model_name]; model.fit(x_train,y); pred=model.predict(x_test)
                    for sid,p in zip(test_ids,pred):
                        target=targets[sid][target_name]; measured=(target["measured_f_min"],target["measured_f_max"]); lo=max(measured[0],10**float(p[0])); hi=min(measured[1],10**float(p[1]));
                        if hi<=lo: lo,hi=measured
                        prediction_rows.append({"spectrum_id":sid,"sample_id":records[sid].sample_id,"voltage":records[sid].voltage,"time":records[sid].time,"topology":records[sid].electrochemical_topology,"held_out_sample":held_out,"target_set":target_name,"feature_set":feature_set,"model":model_name,"target_f_min":target["f_min"],"target_f_max":target["f_max"],"predicted_f_min":lo,"predicted_f_max":hi,"error_f_min_decades":np.log10(lo)-np.log10(target["f_min"]),"error_f_max_decades":np.log10(hi)-np.log10(target["f_max"]),"measured_f_min":measured[0],"measured_f_max":measured[1]})
    predictions=pd.DataFrame(prediction_rows); predictions.to_csv(output/"per_spectrum_predictions.csv",index=False); overall=[]
    for keys,part in predictions.groupby(["target_set","feature_set","model"]): overall.append({"target_set":keys[0],"feature_set":keys[1],"model":keys[2],"spectra":len(part),**{f"f_min_{k}":v for k,v in _metrics(part,"f_min").items()},**{f"f_max_{k}":v for k,v in _metrics(part,"f_max").items()}})
    overall_frame=pd.DataFrame(overall); overall_frame.to_csv(output/"overall_metrics.csv",index=False); overall_frame.to_csv(output/"model_comparison.csv",index=False)
    sample_rows=[]
    for keys,part in predictions.groupby(["target_set","feature_set","model","sample_id"]): sample_rows.append({"target_set":keys[0],"feature_set":keys[1],"model":keys[2],"sample_id":keys[3],"spectra":len(part),**{f"f_min_{k}":v for k,v in _metrics(part,"f_min").items()},**{f"f_max_{k}":v for k,v in _metrics(part,"f_max").items()}})
    pd.DataFrame(sample_rows).to_csv(output/"per_sample_metrics.csv",index=False)
    voltage_rows=[]
    for keys,part in predictions.dropna(subset=["voltage"]).groupby(["target_set","feature_set","model","sample_id"]):
        part=part.copy(); part["voltage_bin"]=part.voltage.round(2)
        for voltage,sub in part.groupby("voltage_bin"): voltage_rows.append({"target_set":keys[0],"feature_set":keys[1],"model":keys[2],"sample_id":keys[3],"voltage":voltage,"spectra":len(sub),**{f"f_min_{k}":v for k,v in _metrics(sub,"f_min").items()},**{f"f_max_{k}":v for k,v in _metrics(sub,"f_max").items()}})
    pd.DataFrame(voltage_rows).to_csv(output/"per_voltage_metrics.csv",index=False)
    primary=predictions[(predictions.target_set=="new_active_window")&(predictions.feature_set=="spectrum_plus_voltage")&(predictions.model=="ridge")].copy(); mask_rows=[]
    for sid,part in primary.groupby("spectrum_id"):
        r=records[sid]; row=part.iloc[0]; envelope=(r.frequency>=row.predicted_f_min)&(r.frequency<=row.predicted_f_max); s2=sensitive_mask(r.frequency,r.impedance,(row.predicted_f_min,row.predicted_f_max),threshold=CONFIG["stage2_threshold"]); final=stage1[sid].mask&envelope&s2.mask; metrics=binary_metrics(envelope,manual[sid]); mask_rows.append({"spectrum_id":sid,"sample_id":r.sample_id,"manual_active_fraction":float(np.mean(manual[sid])),"ml_envelope_fraction":float(np.mean(envelope)),"final_active_fraction":float(np.mean(final)),"active_fraction_difference":float(np.mean(final)-np.mean(manual[sid])),"stage1_rejected":int((~stage1[sid].mask).sum()),"stage2_rejected_inside_envelope":int(np.sum(envelope&~s2.mask)),"boundary_f_min_error_decades":row.error_f_min_decades,"boundary_f_max_error_decades":row.error_f_max_decades,**metrics})
    mask_frame=pd.DataFrame(mask_rows); mask_frame.to_csv(output/"boundary_error_summary.csv",index=False); _plot_diagnostics(primary,plots)
    config={**CONFIG,"dataset_samples":samples,"records":len(records),"primary_candidate":"new_active_window + spectrum_plus_voltage + ridge","manual_window_as_feature":"exploratory leakage ablation only","strict_validation":"six-fold sample LOSO"}; (output/"configuration.json").write_text(json.dumps(config,indent=2,default=list),encoding="utf-8")
    report={"records":len(records),"runtime_s":time.perf_counter()-started,"exclusions":extraction.exclusion_counts,"bayes_drt2":False,"topology_classification_run":False}; (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(overall_frame.sort_values(["target_set","f_min_MAE_decades"]).to_string(index=False)); print(json.dumps(report,indent=2)); return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("projects",nargs="+",type=Path); parser.add_argument("--output",type=Path,default=Path("ml/analysis/frequency_selection_voltage")); args=parser.parse_args(); evaluate(args.projects,args.output)


if __name__ == "__main__": main()
