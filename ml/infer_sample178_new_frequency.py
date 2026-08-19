"""Apply the latest spectrum-plus-voltage frequency model to unseen sample 178."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from .automatic_preprocessing import conservative_mask
from .dataset import load_eisfit_projects
from .evaluate_frequency_limit_ml import _clean_record, _manual_masks
from .evaluate_frequency_selection_voltage import _feature_matrix, _targets
from .frequency_limit_ml import SpectrumFeatureExtractor, models as frequency_models, target_values


TRAINING_SAMPLES = ("129", "140", "150", "157", "159", "181")
CONFIG = {"feature_set": "spectrum_plus_voltage", "target_set": "new_active_window", "model": "ridge", "stage1_threshold": 10.0, "grid_size": 16, "seed": 42, "stage2_applied": False, "topology_applied": False}


def _metadata_value(record, names, default=None):
    for name in names:
        if name in record.metadata:
            return record.metadata[name]
    return default


def _clip(prediction, measured):
    lo, hi = 10**float(prediction[0]), 10**float(prediction[1]); minimum, maximum = measured; clipped = lo < minimum or hi > maximum or hi <= lo
    lo, hi = max(minimum, lo), min(maximum, hi)
    if hi <= lo: lo, hi = minimum, maximum
    return float(lo), float(hi), bool(clipped)


def _plot(record, result, output, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, z = record.frequency, record.impedance; raw = np.ones(f.size, dtype=bool); active = np.asarray(result["ml_frequency_active_mask"], dtype=bool); rejected = ~active
    fig, axes = plt.subplots(1, 3, figsize=(15, 4)); x = np.log10(f)
    axes[0].scatter(x[raw], z.real[raw], s=9, color="0.72", label="raw"); axes[0].scatter(x[active], z.real[active], s=14, color="tab:green", label="ML envelope"); axes[0].scatter(x[rejected], z.real[rejected], marker="x", color="tab:red", label="outside envelope"); axes[0].set_title("Re(Z)")
    axes[1].scatter(x, np.angle(z), s=9, color="0.72"); axes[1].scatter(x[active], np.angle(z)[active], s=14, color="tab:green"); axes[1].set_title("Bode phase")
    axes[2].scatter(z.real, -z.imag, s=9, color="0.72"); axes[2].scatter(z.real[active], -z.imag[active], s=14, color="tab:green"); axes[2].set_title("Nyquist")
    for ax in axes[:2]: ax.axvline(np.log10(result["predicted_f_min"]), color="tab:red", linestyle="--"); ax.axvline(np.log10(result["predicted_f_max"]), color="tab:purple", linestyle="--"); ax.set_xlabel("log10(f)")
    axes[0].legend(fontsize=8); fig.suptitle(label); fig.tight_layout(); fig.savefig(output / f"{label}.png", dpi=130); plt.close(fig)


def infer(training_projects, input_project: Path, output: Path):
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=True); (output / "plots").mkdir(exist_ok=True)
    train_map = {str(p): p.name.split(".")[0] for p in training_projects}; train_report = load_eisfit_projects(training_projects, train_map, require_fit=True); train_records = train_report.records; train_manual = _manual_masks(training_projects, {r.spectrum_id:r for r in train_records}); train_records = [r for r in train_records if r.spectrum_id in train_manual]
    input_report = load_eisfit_projects([input_project], {str(input_project): "178"}, require_fit=False); input_records = input_report.records
    stage1_train = {r.spectrum_id: conservative_mask(r.frequency, r.impedance, threshold=CONFIG["stage1_threshold"]) for r in train_records}; cleaned_train = {s: _clean_record(r, stage1_train[s].mask) for s,r in ((r.spectrum_id,r) for r in train_records)}
    all_records = {r.spectrum_id:r for r in train_records}; all_records.update({r.spectrum_id:r for r in input_records}); stage1_all = {r.spectrum_id: conservative_mask(r.frequency, r.impedance, threshold=CONFIG["stage1_threshold"]) for r in input_records}; cleaned_all = dict(cleaned_train); cleaned_all.update({s:_clean_record(r,stage1_all[s].mask) for s,r in ((r.spectrum_id,r) for r in input_records)})
    extractor = SpectrumFeatureExtractor(CONFIG["grid_size"]); extractor.grid_ = np.linspace(-0.5, 5.0, CONFIG["grid_size"]); extractor.fill_ = np.zeros(CONFIG["grid_size"]*12); base = {s: extractor._one(r.frequency, r.impedance) for s,r in cleaned_all.items()}
    train_ids = list(cleaned_train); targets = {s: _targets(all_records[s], train_manual[s])["new_active_window"] for s in train_ids}; model = frequency_models(CONFIG["seed"])[CONFIG["model"]]; x_train = _feature_matrix(train_ids, base, all_records, CONFIG["feature_set"], train_ids); model.fit(x_train, target_values([targets[s] for s in train_ids])); joblib.dump({"model": model, "config": CONFIG, "training_samples": list(TRAINING_SAMPLES), "feature_grid": extractor.grid_}, output / "final_frequency_model.joblib")
    results=[]; summary=[]
    for index,r in enumerate(input_records):
        sid=r.spectrum_id; x_test=_feature_matrix([sid],base,all_records,CONFIG["feature_set"],train_ids); prediction=model.predict(x_test)[0]; measured=(float(np.min(r.frequency)),float(np.max(r.frequency))); lo,hi,clipped=_clip(prediction,measured); envelope=(r.frequency>=lo)&(r.frequency<=hi); s1=stage1_all[sid]
        metadata=dict(r.metadata); result={"spectrum_id":sid,"source_name":r.source_name,"cycle":r.cycle,"metadata":metadata,"voltage":r.voltage,"current":r.current,"time":r.time,"frequency":r.frequency.tolist(),"z_real":r.z_real.tolist(),"z_imag":r.z_imag.tolist(),"existing_eec_topology":r.original_eec_topology,"stage1_active_mask":s1.mask.tolist(),"stage1_rejection_score":s1.score.tolist(),"ml_frequency_active_mask":envelope.tolist(),"predicted_log_f_min":float(prediction[0]),"predicted_log_f_max":float(prediction[1]),"predicted_f_min":lo,"predicted_f_max":hi,"frequency_boundary_clipped":clipped,"measured_f_min":measured[0],"measured_f_max":measured[1],"stage1_rejected_points":int((~s1.mask).sum()),"ml_envelope_points":int(envelope.sum()),"stage2_applied":False,"topology_applied":False}
        results.append(result); summary.append({"spectrum_id":sid,"source_name":r.source_name,"cycle":r.cycle,"voltage":r.voltage,"current":r.current,"Time":_metadata_value(r,["Time","time"],r.time),"Cycle mod 15":_metadata_value(r,["Cycle mod 15","cycle mod 15"]),"measured_f_min":measured[0],"measured_f_max":measured[1],"predicted_f_min":lo,"predicted_f_max":hi,"predicted_f_min_log10":float(prediction[0]),"predicted_f_max_log10":float(prediction[1])});
        if index<8: _plot(r,result,output/"plots",f"spectrum_{index+1:03d}")
    payload={"ml_results":{"schema_version":"3.0","source_file":str(input_project),"training_samples":list(TRAINING_SAMPLES),"inference_sample":"178","model_finalization":"finalized on all six training samples only; sample 178 excluded","preprocessing":CONFIG,"stage2_note":"not applied in this visual frequency-envelope result","topology_note":"not applied","spectra":results}}; json_path=output/"178_ML_frequency_results.eisfit.json"; json_path.write_text(json.dumps(payload,indent=2),encoding="utf-8"); pd.DataFrame(summary).to_csv(output/"178_ML_frequency_predictions.csv",index=False); (output/"178_ML_frequency_predictions.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    report={"input_file":str(input_project),"output_json":str(json_path),"summary_csv":str(output/"178_ML_frequency_predictions.csv"),"training_samples":list(TRAINING_SAMPLES),"spectra":len(results),"voltage_range":[float(np.nanmin([r.voltage for r in input_records])),float(np.nanmax([r.voltage for r in input_records]))],"measured_frequency_range":[float(min(np.min(r.frequency) for r in input_records)),float(max(np.max(r.frequency) for r in input_records))],"predicted_fmin_range":[float(min(x["predicted_f_min"] for x in summary)),float(max(x["predicted_f_min"] for x in summary))],"predicted_fmax_range":[float(min(x["predicted_f_max"] for x in summary)),float(max(x["predicted_f_max"] for x in summary))],"fmin_outside_measured":sum(x["predicted_f_min"]<x["measured_f_min"] for x in summary),"fmax_outside_measured":sum(x["predicted_f_max"]>x["measured_f_max"] for x in summary),"stage1_rejected_points":sum(x["stage1_rejected_points"] for x in results),"stage2_applied":False,"topology_applied":False,"runtime_s":time.perf_counter()-started}; (output/"inference_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2)); return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("training_projects",nargs="+",type=Path); parser.add_argument("--input",type=Path,default=Path(r"C:\Users\Herman\Desktop\Ti overlayer backup\178.eisfit.json")); parser.add_argument("--output",type=Path,default=Path("ml/analysis/unseen_178_new_frequency")); args=parser.parse_args(); infer(args.training_projects,args.input,args.output)


if __name__=="__main__": main()
