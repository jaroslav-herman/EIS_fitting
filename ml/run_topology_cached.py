"""Topology-only LOSO evaluation using existing Bayes-DRT2 cache entries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .dataset import load_eisfit_projects, SpectrumRecord
from .frequency_range import _features, _fit_predict, _models as range_models, _targets
from .metrics import multiclass_brier, prediction_metrics
from .outlier_cache import OutlierCache
from .preprocessing import SpectrumPreprocessor
from .topology_classifier import _models as topology_models


def _display_topology(value: str) -> str:
    return "ONE_PROCESS" if "p(R1,CPE1)-p(R2,CPE2)" not in value else "TWO_PROCESS"


def _masked(record: SpectrumRecord, mask: np.ndarray) -> SpectrumRecord:
    return SpectrumRecord(
        spectrum_id=record.spectrum_id, source_project=record.source_project, sample_id=record.sample_id,
        cycle=record.cycle, voltage=record.voltage, current=record.current, time=record.time,
        frequency=record.frequency[mask], z_real=record.z_real[mask], z_imag=record.z_imag[mask],
        topology_label=_display_topology(record.electrochemical_topology),
        original_eec_topology=record.original_eec_topology, electrochemical_topology=_display_topology(record.electrochemical_topology),
        l0_required_in_manual_fit=record.l0_required_in_manual_fit, device_setup=record.device_setup,
        manual_f_min=record.manual_f_min, manual_f_max=record.manual_f_max,
    )


def _cached_record(cache: OutlierCache, record: SpectrumRecord, window):
    key = cache._key(record, tuple(map(float, window)))
    json_path, npz_path = cache._paths(key)
    if not json_path.exists() or not npz_path.exists():
        return None, {"spectrum_id": record.spectrum_id, "sample_id": record.sample_id,
                      "frequency_min": float(window[0]), "frequency_max": float(window[1]), "key": key,
                      "reason": "missing_cache_entry"}
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "success":
            return None, {**metadata, "key": key}
        mask = np.asarray(np.load(npz_path)["active_mask"], dtype=bool)
        if mask.size != record.frequency.size:
            return None, {**metadata, "key": key, "reason": "mask_length_mismatch"}
        return _masked(record, mask), None
    except Exception as error:
        return None, {"spectrum_id": record.spectrum_id, "sample_id": record.sample_id, "key": key,
                      "reason": f"cache_read:{type(error).__name__}:{error}"}


def _range_predict(train, test, name, grid_size, seed):
    x_train, x_test = _features(train, test, "spectrum_only", grid_size)
    lo, hi = _fit_predict(range_models(seed)[name], x_train, _targets(train), x_test)
    return {r.spectrum_id: (10**float(a), 10**float(b)) for r, a, b in zip(test, lo, hi)}


def _nested_ranges(train, name, grid_size, seed):
    result = {}
    for inner in sorted({r.sample_id for r in train}):
        inner_train = [r for r in train if r.sample_id != inner]
        result.update(_range_predict(inner_train, [r for r in train if r.sample_id == inner], name, grid_size, seed))
    return result


def _iou(record, predicted):
    a, b = np.log10(record.manual_f_min), np.log10(record.manual_f_max)
    c, d = np.log10(predicted[0]), np.log10(predicted[1])
    intersection = max(0.0, min(b, d) - max(a, c)); union = max(b, d) - min(a, c)
    return intersection / union if union else 1.0, c - a, d - b


def _fit_pipeline(train, test, pipeline, fold, grid_size, seed):
    pre = SpectrumPreprocessor(grid_size=grid_size, spectrum_mode="raw")
    x_train, x_test = pre.fit_transform(train), pre.transform(test)
    classes = ["ONE_PROCESS", "TWO_PROCESS"]
    rows = []
    for name in ("random_forest", "hist_gradient_boosting"):
        model = topology_models(seed)[name]
        model.fit(x_train, [r.topology_label for r in train])
        predictions = model.predict(x_test); probabilities = model.predict_proba(x_test); model_classes = list(model.classes_)
        for record, prediction, probability in zip(test, predictions, probabilities):
            row = {"spectrum_id": record.spectrum_id, "sample_id": record.sample_id, "voltage": record.voltage,
                   "time": record.time, "l0_required": record.l0_required_in_manual_fit,
                   "original_eec_string": record.original_eec_topology, "canonical_topology": record.topology_label,
                   "pipeline": pipeline, "topology_model": name, "validation_fold": fold,
                   "topology_prediction": str(prediction), "topology_correct": bool(str(prediction) == record.topology_label)}
            for cls in classes:
                row[f"probability_{cls}"] = float(probability[model_classes.index(cls)]) if cls in model_classes else 0.0
            rows.append(row)
    return rows


def _metric_row(frame, classes):
    view = frame.rename(columns={"canonical_topology": "true_topology", "topology_prediction": "predicted_topology"})
    values = prediction_metrics(view, classes)
    result = {k: v for k, v in values.items() if isinstance(v, (int, float, np.floating))}
    result["brier"] = multiclass_brier(view, classes)
    result["count"] = len(frame)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate topology models from existing Bayes-DRT2 masks")
    parser.add_argument("projects", nargs="+", type=Path)
    parser.add_argument("--sample", action="append", required=True, metavar="PROJECT=SAMPLE")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml_topology_cached_results"))
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    mapping = {}
    for item in args.sample:
        project, sample = item.split("=", 1); mapping[Path(project).name] = sample
    mapping.update({str(p): mapping[p.name] for p in args.projects})
    records = load_eisfit_projects(args.projects, mapping).records
    records = [SpectrumRecord(**{**r.__dict__, "topology_label": _display_topology(r.electrochemical_topology),
                                 "electrochemical_topology": _display_topology(r.electrochemical_topology)}) for r in records]
    cache = OutlierCache(args.cache_dir, threshold=1.0, workers=1)
    rows, missing = [], []
    started = time.perf_counter()
    for fold in sorted({r.sample_id for r in records}):
        train = [r for r in records if r.sample_id != fold]; test = [r for r in records if r.sample_id == fold]
        rows.extend(_fit_pipeline(train, test, "A_raw", fold, args.grid_size, args.seed))
        manual = {r.spectrum_id: (r.manual_f_min, r.manual_f_max) for r in train + test}
        b_train, b_test = [], []
        for r in train + test:
            item, failure = _cached_record(cache, r, manual[r.spectrum_id])
            (b_train if r.sample_id != fold else b_test).append(item) if item is not None else missing.append(failure)
        if b_train and b_test: rows.extend(_fit_pipeline(b_train, b_test, "B_manual_range", fold, args.grid_size, args.seed))
        for range_model in ("random_forest", "hist_gradient_boosting"):
            windows = {**_nested_ranges(train, range_model, args.grid_size, args.seed), **_range_predict(train, test, range_model, args.grid_size, args.seed)}
            c_train, c_test = [], []
            quality = {}
            for r in train + test:
                item, failure = _cached_record(cache, r, windows[r.spectrum_id])
                if item is None: missing.append(failure); continue
                if r.sample_id != fold: c_train.append(item)
                else:
                    c_test.append(item); quality[r.spectrum_id] = _iou(r, windows[r.spectrum_id])
            pipeline = f"C_ml_range_{range_model}"
            if c_train and c_test:
                new = _fit_pipeline(c_train, c_test, pipeline, fold, args.grid_size, args.seed)
                for row in new:
                    iou, dmin, dmax = quality[row["spectrum_id"]]
                    row.update({"manual_f_min": next(r.manual_f_min for r in test if r.spectrum_id == row["spectrum_id"]),
                                "manual_f_max": next(r.manual_f_max for r in test if r.spectrum_id == row["spectrum_id"]),
                                "predicted_f_min": windows[row["spectrum_id"]][0], "predicted_f_max": windows[row["spectrum_id"]][1],
                                "frequency_range_IoU": iou, "delta_log_fmin": dmin, "delta_log_fmax": dmax})
                rows.extend(new)
    frame = pd.DataFrame(rows); args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "per_spectrum_results.csv", index=False)
    classes = ["ONE_PROCESS", "TWO_PROCESS"]
    overall=[]; per_sample=[]; l0=[]
    for (pipeline, model), sub in frame.groupby(["pipeline", "topology_model"]):
        overall.append({"pipeline":pipeline,"topology_model":model,**_metric_row(sub,classes)})
        for fold, part in sub.groupby("validation_fold"): per_sample.append({"pipeline":pipeline,"topology_model":model,"held_out_sample":fold,**_metric_row(part,classes)})
        for l0_value, part in sub.groupby("l0_required", dropna=False): l0.append({"pipeline":pipeline,"topology_model":model,"l0_required":l0_value,**_metric_row(part,classes)})
        for sample in ("150", "157"):
            part=sub[sub.sample_id==sample]
            if not part.empty: l0.append({"pipeline":pipeline,"topology_model":model,"sample_id":sample,"l0_required":"all",**_metric_row(part,classes)})
    pd.DataFrame(overall).to_csv(args.output/"overall_metrics.csv",index=False); pd.DataFrame(per_sample).to_csv(args.output/"per_sample_metrics.csv",index=False); pd.DataFrame(l0).to_csv(args.output/"l0_metrics.csv",index=False)
    c=frame[frame.pipeline.str.startswith("C_")].copy(); groups=[]
    if not c.empty:
        c["iou_group"]=pd.cut(c.frequency_range_IoU,[-np.inf,.5,.75,.9,np.inf],labels=["IoU <= 0.50","0.50 < IoU <= 0.75","0.75 < IoU <= 0.90","IoU > 0.90"])
        for keys, part in c.groupby(["pipeline","topology_model","iou_group"],observed=False): groups.append({"pipeline":keys[0],"topology_model":keys[1],"iou_group":str(keys[2]),**_metric_row(part,classes)})
    pd.DataFrame(groups).to_csv(args.output/"iou_metrics.csv",index=False)
    matrix_dir=args.output/"confusion_matrices"
    for (pipeline,model),sub in frame.groupby(["pipeline","topology_model"]):
        dest=matrix_dir/pipeline/model;dest.mkdir(parents=True,exist_ok=True)
        for label,part in [("aggregated",sub),*[(str(f),g) for f,g in sub.groupby("validation_fold")]]:
            pd.DataFrame(confusion_matrix(part.canonical_topology,part.topology_prediction,labels=classes),index=classes,columns=classes).to_csv(dest/f"{label}.csv")
    pd.DataFrame(missing).to_csv(args.output/"missing_cache_entries.csv",index=False)
    (args.output/"runtime.json").write_text(json.dumps({"seconds":time.perf_counter()-started,"spectra":len(records),"cache_entries":len(list((args.cache_dir/"entries").glob("*.json"))),"missing_requests":len(missing),"bayes_drt2_calls":0},indent=2),encoding="utf-8")
    print(pd.DataFrame(overall).to_string(index=False)); print(f"Output: {args.output}; missing cache requests: {len(missing)}")


if __name__ == "__main__": raise SystemExit(main())
