"""Persistent cache for unchanged EIS outlier preprocessing."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from eis_model import CycleState
from eis_services import analyze_outliers, circuit_parameters

from .dataset import SpectrumRecord


def _compute_outlier_job(args):
    record, window, threshold, version = args
    frequency = np.asarray(record.frequency, dtype=float)
    impedance = record.impedance
    minimum, maximum = sorted(map(float, window))
    minimum = max(minimum, float(np.min(frequency)))
    maximum = min(maximum, float(np.max(frequency)))
    if not maximum > minimum:
        raise ValueError("frequency window does not overlap measured data")
    state = CycleState(cycle=record.cycle, frequency_hz=frequency.copy(), impedance=impedance.copy(),
                       potential_v=float(record.voltage or 0.0), current_ma=float(record.current or 0.0),
                       time_s=record.time, frequency_window=(minimum, maximum), circuit=record.original_eec_topology)
    analysis = analyze_outliers(state, threshold, circuit_parameters(record.original_eec_topology))
    state.apply_outliers(analysis.outlier_indices)
    mask = state.included.astype(bool)
    if int(mask.sum()) < 3:
        raise ValueError("fewer than three active points after preprocessing")
    return mask, {"spectrum_id": record.spectrum_id, "sample_id": record.sample_id,
                  "frequency_min": float(window[0]), "frequency_max": float(window[1]),
                  "threshold": threshold, "bayes_drt2_version": version,
                  "raw_point_count": int(len(frequency)), "active_point_count": int(mask.sum()), "status": "success"}


def _compute_outlier_job_safe(args):
    started = time.perf_counter()
    try:
        mask, metadata = _compute_outlier_job(args)
        metadata["runtime_s"] = time.perf_counter() - started
        return mask, metadata
    except Exception as error:
        record, window, threshold, version = args
        return None, {"spectrum_id": record.spectrum_id, "sample_id": record.sample_id,
                      "frequency_min": float(window[0]), "frequency_max": float(window[1]),
                      "threshold": threshold, "bayes_drt2_version": version, "status": "failure",
                      "error_type": type(error).__name__, "error_message": str(error),
                      "runtime_s": time.perf_counter() - started}


def _version() -> str:
    try:
        return importlib.metadata.version("bayes-drt2")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@dataclass
class CacheResult:
    mask: np.ndarray | None
    metadata: dict
    hit: bool


class OutlierCache:
    """File-backed cache keyed by exact spectrum and exact frequency window."""

    def __init__(self, directory: Path, *, threshold: float = 1.0, workers: int = 1):
        self.directory = Path(directory)
        self.entries = self.directory / "entries"
        self.entries.mkdir(parents=True, exist_ok=True)
        self.threshold = float(threshold)
        self.workers = max(1, int(workers))
        self.bayes_drt2_version = _version()
        self.calls = 0
        self.hits = 0
        self.misses = 0
        self.failures = 0
        self.timings: list[float] = []
        self.requests: list[dict] = []

    def _key(self, record: SpectrumRecord, window: tuple[float, float]) -> str:
        payload = {
            "spectrum_id": record.spectrum_id,
            "frequency_min": float(window[0]),
            "frequency_max": float(window[1]),
            "threshold": self.threshold,
            "bayes_drt2_version": self.bayes_drt2_version,
            "active_rule": "CycleState.included then analyze_outliers(use_existing_fit=False)",
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _paths(self, key: str):
        return self.entries / f"{key}.json", self.entries / f"{key}.npz"

    def _compute(self, record: SpectrumRecord, window: tuple[float, float]):
        return _compute_outlier_job((record, window, self.threshold, self.bayes_drt2_version))

    def get_or_compute(self, record: SpectrumRecord, window: tuple[float, float]) -> CacheResult:
        key = self._key(record, window)
        json_path, npz_path = self._paths(key)
        if json_path.exists() and not npz_path.exists():
            try:
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
                if metadata.get("status") == "failure":
                    self.hits += 1
                    return CacheResult(None, metadata, True)
            except Exception:
                pass
        if json_path.exists() and npz_path.exists():
            try:
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
                mask = np.asarray(np.load(npz_path)["active_mask"], dtype=bool)
                if mask.size == record.frequency.size:
                    self.hits += 1
                    return CacheResult(mask, metadata, True)
            except Exception:
                pass
        self.misses += 1
        started = time.perf_counter()
        try:
            mask, metadata = self._compute(record, window)
            metadata.update({"key": key, "runtime_s": time.perf_counter() - started})
            np.savez_compressed(npz_path, active_mask=mask)
            json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self.timings.append(float(metadata["runtime_s"]))
            self.calls += 1
            return CacheResult(mask, metadata, False)
        except Exception as error:
            metadata = {"key": key, "spectrum_id": record.spectrum_id, "sample_id": record.sample_id,
                        "frequency_min": float(window[0]), "frequency_max": float(window[1]),
                        "threshold": self.threshold, "bayes_drt2_version": self.bayes_drt2_version,
                        "status": "failure", "error_type": type(error).__name__, "error_message": str(error),
                        "runtime_s": time.perf_counter() - started}
            json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self.timings.append(float(metadata["runtime_s"]))
            self.calls += 1
            self.failures += 1
            return CacheResult(None, metadata, False)

    def process(self, records: list[SpectrumRecord], windows: dict[str, tuple[float, float]], *, context: dict | None = None):
        jobs = [(record, tuple(map(float, windows[record.spectrum_id]))) for record in records]
        if self.workers == 1:
            results = [self.get_or_compute(record, window) for record, window in jobs]
        else:
            pending = []
            result_map = {}
            for record, window in jobs:
                key = self._key(record, window)
                json_path, npz_path = self._paths(key)
                if json_path.exists() and npz_path.exists():
                    result_map[key] = self.get_or_compute(record, window)
                else:
                    pending.append((record, window))
            # Consume the iterator incrementally so every completed result is
            # persisted immediately, while workers remain alive for the batch.
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                jobs_for_workers = [(record, window, self.threshold, self.bayes_drt2_version) for record, window in pending]
                for (record, window), (mask, metadata) in zip(pending, executor.map(_compute_outlier_job_safe, jobs_for_workers)):
                    key = self._key(record, window)
                    json_path, npz_path = self._paths(key)
                    metadata.update({"key": key})
                    if mask is not None:
                        np.savez_compressed(npz_path, active_mask=mask)
                    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    self.misses += 1
                    self.calls += 1
                    if metadata.get("runtime_s") is not None:
                        self.timings.append(float(metadata["runtime_s"]))
                    if metadata.get("status") == "failure":
                        self.failures += 1
                    result_map[key] = CacheResult(mask, metadata, False)
            results = [result_map[self._key(record, window)] for record, window in jobs]
        processed, failures = [], []
        for (record, window), result in zip(jobs, results):
            self.requests.append({"spectrum_id": record.spectrum_id, "sample_id": record.sample_id,
                                  "frequency_min": window[0], "frequency_max": window[1],
                                  "cache_hit": result.hit, "status": result.metadata.get("status"), **(context or {})})
            if result.mask is None:
                failures.append(result.metadata)
                continue
            mask = result.mask
            processed.append(record.__class__(
                spectrum_id=record.spectrum_id, source_project=record.source_project, sample_id=record.sample_id,
                cycle=record.cycle, voltage=record.voltage, current=record.current, time=record.time,
                frequency=record.frequency[mask], z_real=record.z_real[mask], z_imag=record.z_imag[mask],
                topology_label=record.topology_label, original_eec_topology=record.original_eec_topology,
                electrochemical_topology=record.electrochemical_topology, l0_required_in_manual_fit=record.l0_required_in_manual_fit,
                device_setup=record.device_setup, manual_f_min=record.manual_f_min, manual_f_max=record.manual_f_max,
            ))
        return processed, failures

    def write_report(self, path: Path):
        total = self.calls + self.hits
        report = {
            "bayes_drt2_version": self.bayes_drt2_version, "threshold": self.threshold,
            "workers": self.workers, "unique_calls_this_run": self.calls, "cache_hits_this_run": self.hits,
            "cache_misses_this_run": self.misses, "failures_this_run": self.failures,
            "requests_this_run": total, "mean_runtime_s": float(np.mean(self.timings)) if self.timings else 0.0,
            "median_runtime_s": float(np.median(self.timings)) if self.timings else 0.0,
            "total_preprocessing_runtime_s": float(np.sum(self.timings)),
        }
        if self.requests:
            import csv
            request_path = self.directory / "request_manifest.csv"
            fields = sorted({key for row in self.requests for key in row})
            with request_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if request_path.stat().st_size == 0:
                    writer.writeheader()
                writer.writerows(self.requests)
        Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
