"""Independently resumable stages for the cached topology experiment."""
from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from .dataset import SpectrumRecord
from .outlier_cache import OutlierCache
from .preprocessing import SpectrumPreprocessor

SAMPLES = ("181", "159", "140", "129", "150", "157")
FREQUENCY_MODELS = ("random_forest", "hist_gradient_boosting")


def stage1_frequency_predictions(source: Path, output: Path) -> dict:
    """Persist previously generated LOSO spectrum-only RF/HGB predictions."""
    started = time.perf_counter()
    source = Path(source); output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_file = source / "spectrum_only" / "predictions.csv"
    if not source_file.exists():
        raise FileNotFoundError(source_file)
    frame = pd.read_csv(source_file)
    frame = frame[frame["model_name"].isin(FREQUENCY_MODELS)].copy()
    frame = frame.rename(columns={"model_name": "frequency_model", "predicted_log_f_min": "predicted_log_fmin",
                                  "predicted_log_f_max": "predicted_log_fmax", "predicted_f_min": "predicted_fmin",
                                  "predicted_f_max": "predicted_fmax", "manual_f_min": "manual_fmin", "manual_f_max": "manual_fmax"})
    required = ["spectrum_id", "sample_id", "validation_fold", "frequency_model", "predicted_log_fmin", "predicted_log_fmax", "predicted_fmin", "predicted_fmax", "manual_fmin", "manual_fmax"]
    missing = [c for c in required if c not in frame.columns]
    if missing: raise ValueError(f"Stage 1 missing columns: {missing}")
    frame["sample_id"] = frame["sample_id"].astype(str); frame["validation_fold"] = frame["validation_fold"].astype(str)
    frame = frame[required + [c for c in ("range_iou", "error_log_f_min", "error_log_f_max") if c in frame.columns]]
    duplicates = frame.duplicated(["spectrum_id", "frequency_model"])
    expected = set(SAMPLES)
    report = {
        "source": str(source_file), "models": sorted(frame.frequency_model.unique()), "samples": sorted(frame.sample_id.unique()),
        "prediction_rows": int(len(frame)), "unique_spectra": int(frame.spectrum_id.nunique()),
        "duplicate_prediction_rows": int(duplicates.sum()), "fold_mismatch_rows": int((frame.sample_id != frame.validation_fold).sum()),
        "missing_models_or_folds": [], "runtime_s": time.perf_counter() - started,
    }
    for model in FREQUENCY_MODELS:
        for sample in SAMPLES:
            count = int(((frame.frequency_model == model) & (frame.validation_fold == sample)).sum())
            if count == 0: report["missing_models_or_folds"].append({"model": model, "fold": sample})
    if report["duplicate_prediction_rows"] or report["fold_mismatch_rows"] or report["missing_models_or_folds"]:
        raise ValueError(f"Stage 1 validation failed: {report}")
    frame.to_csv(output / "frequency_predictions.csv", index=False)
    (output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _canonical(record: SpectrumRecord) -> str:
    value = str(record.electrochemical_topology or record.topology_label)
    if value in {"ONE_PROCESS", "TWO_PROCESS"}:
        return value
    return "ONE_PROCESS" if "p(R1,CPE1)-p(R2,CPE2)" not in value else "TWO_PROCESS"


def _mapped(record: SpectrumRecord) -> SpectrumRecord:
    label = _canonical(record)
    return SpectrumRecord(
        spectrum_id=record.spectrum_id, source_project=record.source_project, sample_id=record.sample_id,
        cycle=record.cycle, voltage=record.voltage, current=record.current, time=record.time,
        frequency=record.frequency, z_real=record.z_real, z_imag=record.z_imag, topology_label=label,
        original_eec_topology=record.original_eec_topology, electrochemical_topology=label,
        l0_required_in_manual_fit=record.l0_required_in_manual_fit, device_setup=record.device_setup,
        manual_f_min=record.manual_f_min, manual_f_max=record.manual_f_max,
    )


def _cached_processed(cache: OutlierCache, record: SpectrumRecord, window):
    key = cache._key(record, tuple(map(float, window)))
    json_path, npz_path = cache._paths(key)
    if not json_path.exists() or not npz_path.exists():
        return None, {"spectrum_id": record.spectrum_id, "sample_id": record.sample_id, "key": key, "reason": "missing_cache_entry"}
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "success": return None, {**metadata, "key": key}
        mask = np.asarray(np.load(npz_path)["active_mask"], dtype=bool)
        if mask.size != record.frequency.size: return None, {**metadata, "key": key, "reason": "mask_length_mismatch"}
        return _mapped(SpectrumRecord(
            spectrum_id=record.spectrum_id, source_project=record.source_project, sample_id=record.sample_id,
            cycle=record.cycle, voltage=record.voltage, current=record.current, time=record.time,
            frequency=record.frequency[mask], z_real=record.z_real[mask], z_imag=record.z_imag[mask],
            topology_label=_canonical(record), original_eec_topology=record.original_eec_topology,
            electrochemical_topology=record.electrochemical_topology, l0_required_in_manual_fit=record.l0_required_in_manual_fit,
            device_setup=record.device_setup, manual_f_min=record.manual_f_min, manual_f_max=record.manual_f_max,
        )), None
    except Exception as error:
        return None, {"spectrum_id": record.spectrum_id, "sample_id": record.sample_id, "key": key, "reason": f"cache_read:{type(error).__name__}:{error}"}


def stage2_topology_datasets(
    records: list[SpectrumRecord],
    frequency_cache: Path,
    outlier_cache: Path,
    output: Path,
    grid_size: int = 64,
    samples: tuple[str, ...] | None = None,
) -> dict:
    started = time.perf_counter(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records = [_mapped(r) for r in records]
    samples = tuple(samples or sorted({r.sample_id for r in records}))
    predictions = pd.read_csv(Path(frequency_cache) / "frequency_predictions.csv")
    prediction_map = {(str(row.spectrum_id), str(row.frequency_model)): row for row in predictions.itertuples(index=False)}
    cache = OutlierCache(outlier_cache, threshold=1.0, workers=1)
    pipelines = {"topology_raw": None, "topology_manual": "manual", "topology_ml_rf": "random_forest", "topology_ml_hgb": "hist_gradient_boosting"}
    summary = {"datasets": {}, "failures": [], "runtime_s": None}
    for dataset_name, frequency_model in pipelines.items():
        chunks = []
        for fold in samples:
            train = [r for r in records if r.sample_id != fold]; test = [r for r in records if r.sample_id == fold]
            selected = train + test; processed = []; failed = []
            for record in selected:
                if frequency_model is None:
                    item = record; error = None
                else:
                    if frequency_model == "manual": window = (record.manual_f_min, record.manual_f_max)
                    else:
                        row = prediction_map.get((record.spectrum_id, frequency_model))
                        if row is None: item, error = None, {"spectrum_id": record.spectrum_id, "reason": "missing_frequency_prediction"}; processed.append(item); failed.append(error); continue
                        window = (row.predicted_fmin, row.predicted_fmax)
                    item, error = _cached_processed(cache, record, window)
                if item is None: failed.append(error)
                else: processed.append(item)
            train_ids = {r.spectrum_id for r in train}; train_processed = [r for r in processed if r.spectrum_id in train_ids]; test_processed = [r for r in processed if r.spectrum_id not in train_ids]
            if not train_processed or not test_processed:
                summary["failures"].extend([{**(failure or {}), "pipeline": dataset_name, "validation_fold": fold} for failure in failed]); continue
            pre = SpectrumPreprocessor(grid_size=grid_size, spectrum_mode="raw")
            x_train = pre.fit_transform(train_processed); x_test = pre.transform(test_processed)
            x = np.vstack([x_train, x_test]); ordered = train_processed + test_processed
            metadata_rows = []
            for record in ordered:
                row = prediction_map.get((record.spectrum_id, frequency_model)) if frequency_model in FREQUENCY_MODELS else None
                metadata_rows.append({"spectrum_id": record.spectrum_id, "sample_id": record.sample_id, "validation_fold": fold,
                                      "l0_required": record.l0_required_in_manual_fit, "canonical_topology": record.topology_label,
                                      "voltage": record.voltage, "time": record.time,
                                      "manual_fmin": record.manual_f_min, "manual_fmax": record.manual_f_max,
                                      "predicted_fmin": getattr(row, "predicted_fmin", np.nan) if row else np.nan,
                                      "predicted_fmax": getattr(row, "predicted_fmax", np.nan) if row else np.nan})
            chunks.append((x, metadata_rows))
        if chunks:
            matrix = np.vstack([chunk[0] for chunk in chunks]); metadata = pd.DataFrame([row for chunk in chunks for row in chunk[1]])
            np.savez_compressed(output / f"{dataset_name}.npz", X=matrix, y=metadata.canonical_topology.to_numpy(), spectrum_id=metadata.spectrum_id.to_numpy(), sample_id=metadata.sample_id.to_numpy(), validation_fold=metadata.validation_fold.to_numpy(), l0_required=metadata.l0_required.to_numpy(dtype=bool), voltage=metadata.voltage.to_numpy(dtype=float), time=metadata.time.to_numpy(dtype=float), manual_fmin=metadata.manual_fmin.to_numpy(dtype=float), manual_fmax=metadata.manual_fmax.to_numpy(dtype=float), predicted_fmin=metadata.predicted_fmin.to_numpy(dtype=float), predicted_fmax=metadata.predicted_fmax.to_numpy(dtype=float))
            metadata.to_csv(output / f"{dataset_name}_metadata.csv", index=False)
            summary["datasets"][dataset_name] = {"rows": int(len(metadata)), "features": int(matrix.shape[1]), "folds": sorted(metadata.validation_fold.unique())}
    summary["runtime_s"] = time.perf_counter() - started; summary["failure_count"] = len(summary["failures"])
    (output / "stage2_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["failures"]).to_csv(output / "stage2_failures.csv", index=False)
    return summary
