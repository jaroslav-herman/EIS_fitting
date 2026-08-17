"""Complete persistent Bayes-DRT2 entries for persisted Stage 1 ranges.

This is deliberately separate from the staged topology builder.  It only
materializes missing exact spectrum/range cache keys and never reruns Stage 1
or creates topology datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .dataset import load_eisfit_projects
from .outlier_cache import OutlierCache


SAMPLES = ("181", "159", "140", "129", "150", "157")
MODELS = ("random_forest", "hist_gradient_boosting")


def _cache_available(cache: OutlierCache, record, window: tuple[float, float]) -> bool:
    key = cache._key(record, window)
    json_path, npz_path = cache._paths(key)
    if not json_path.exists():
        return False
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "success" or not npz_path.exists():
            return False
        return np.asarray(np.load(npz_path)["active_mask"], dtype=bool).size == record.frequency.size
    except Exception:
        return False


def _records(projects: list[Path]):
    sample_ids = {str(path): path.stem for path in projects}
    report = load_eisfit_projects(projects, sample_ids)
    return {record.spectrum_id: record for record in report.records}


def complete_cache(
    projects: list[Path],
    frequency_cache: Path,
    outlier_cache: Path,
    *,
    progress_every: int = 25,
) -> dict:
    started = time.perf_counter()
    records = _records(projects)
    prediction_path = Path(frequency_cache) / "frequency_predictions.csv"
    frame = pd.read_csv(prediction_path)
    frame = frame[frame["frequency_model"].isin(MODELS)].copy()
    frame["spectrum_id"] = frame["spectrum_id"].astype(str)
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame.duplicated(["spectrum_id", "frequency_model"]).any():
        raise ValueError("Stage 1 contains duplicate spectrum/model predictions")
    missing_columns = {"spectrum_id", "sample_id", "validation_fold", "frequency_model", "predicted_fmin", "predicted_fmax"} - set(frame)
    if missing_columns:
        raise ValueError(f"Stage 1 is missing columns: {sorted(missing_columns)}")

    cache = OutlierCache(Path(outlier_cache), threshold=1.0, workers=1)
    unique = []
    seen = set()
    for row in frame.itertuples(index=False):
        record = records.get(str(row.spectrum_id))
        if record is None:
            raise ValueError(f"Stage 1 spectrum is absent from projects: {row.spectrum_id}")
        window = (float(row.predicted_fmin), float(row.predicted_fmax))
        key = cache._key(record, window)
        if key not in seen:
            seen.add(key)
            unique.append((record, window, row))

    # Stage 2 requests each persisted spectrum/model six times (once per
    # topology LOSO fold), while one physical cache entry is sufficient.
    logical_total = len(frame) * len(SAMPLES)
    available_before = sum(_cache_available(cache, record, window) for record, window, _ in unique)
    logical_available_before = available_before * len(SAMPLES)
    print(json.dumps({
        "required_ml_range_requests": logical_total,
        "unique_cache_identities": len(unique),
        "already_available_logical": logical_available_before,
        "missing_logical": logical_total - logical_available_before,
        "cache_hit_rate_before": logical_available_before / logical_total if logical_total else 1.0,
    }, indent=2), flush=True)

    # Validate ten missing entries before the full run.  The second lookup
    # proves that the persisted mask and key are readable and stable.
    verification = []
    for record, window, row in [item for item in unique if not _cache_available(cache, item[0], item[1])][:10]:
        expected_key = cache._key(record, window)
        result = cache.get_or_compute(record, window)
        json_path, npz_path = cache._paths(expected_key)
        reloaded = cache.get_or_compute(record, window)
        identical = result.mask is not None and reloaded.mask is not None and np.array_equal(result.mask, reloaded.mask)
        verification.append({"spectrum_id": record.spectrum_id, "frequency_model": row.frequency_model,
                             "validation_fold": str(row.validation_fold), "key_matches": result.metadata.get("key") == expected_key,
                             "reload_identical": bool(identical), "cache_files_exist": json_path.exists() and npz_path.exists()})
        if not verification[-1]["key_matches"] or not verification[-1]["reload_identical"]:
            raise RuntimeError(f"Cache verification failed: {verification[-1]}")
    print(json.dumps({"verification_count": len(verification), "verification": verification}, indent=2), flush=True)

    hits_before = cache.hits
    calls_before = cache.calls
    failures_before = cache.failures
    completed = 0
    last_report = time.perf_counter()
    for record, window, row in unique:
        result = cache.get_or_compute(record, window)
        completed += 1
        if result.mask is None:
            print(json.dumps({"status": "failure", "spectrum_id": record.spectrum_id,
                              "frequency_model": row.frequency_model, "error": result.metadata}), flush=True)
        if completed % progress_every == 0 or completed == len(unique):
            elapsed = time.perf_counter() - started
            rate = (completed / elapsed) if elapsed else 0.0
            remaining = (len(unique) - completed) / rate if rate else None
            print(json.dumps({"completed_unique": completed, "total_unique": len(unique),
                              "percentage": 100.0 * completed / len(unique) if unique else 100.0,
                              "cache_hits": cache.hits, "new_bayes_drt2_calls": cache.calls,
                              "elapsed_s": elapsed, "estimated_remaining_s": remaining,
                              "last_report_s": time.perf_counter() - last_report}), flush=True)
            last_report = time.perf_counter()

    available_after = sum(_cache_available(cache, record, window) for record, window, _ in unique)
    failed = len(unique) - available_after
    report = {
        "required_requests": logical_total,
        "unique_cache_identities": len(unique),
        "already_cached_logical": logical_available_before,
        "new_bayes_drt2_calls": cache.calls - calls_before,
        "cache_hits_this_run": cache.hits - hits_before,
        "cache_misses_this_run": cache.misses,
        "final_cache_size": len(list(cache.entries.glob("*.json"))),
        "remaining_missing_requests_logical": failed * len(SAMPLES),
        "failed_unique_requests": failed,
        "bayes_drt2_version": cache.bayes_drt2_version,
        "verification": verification,
        "runtime_s": time.perf_counter() - started,
    }
    Path(outlier_cache).mkdir(parents=True, exist_ok=True)
    (Path(outlier_cache) / "ml_range_completion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projects", nargs="+", type=Path)
    parser.add_argument("--frequency-cache", type=Path, required=True)
    parser.add_argument("--outlier-cache", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    complete_cache(args.projects, args.frequency_cache, args.outlier_cache, progress_every=max(1, args.progress_every))


if __name__ == "__main__":
    main()
