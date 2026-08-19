"""Versioned storage helpers for ML predictions associated with EIS spectra."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ML_RESULTS_FORMAT = "eis-fitting-ml-results"
ML_RESULTS_VERSION = 1


def spectrum_identifier(
    frequency: object,
    z_real: object,
    z_imag: object,
    cycle: int,
    control: str,
) -> str:
    """Return a path-independent identifier for one measured spectrum."""
    digest = hashlib.sha256()
    for values in (frequency, z_real, z_imag):
        array = np.asarray(values, dtype="<f8").reshape(-1)
        digest.update(np.asarray([array.size], dtype="<u8").tobytes())
        digest.update(array.tobytes())
    digest.update(str(int(cycle)).encode("utf-8"))
    digest.update(str(control).casefold().encode("utf-8"))
    return f"eisfit-spectrum-v1:{digest.hexdigest()}"


def write_ml_results(
    path: Path,
    spectra: list[dict],
    *,
    source_project: str | None = None,
    pipeline: dict | None = None,
) -> None:
    payload = {
        "format": ML_RESULTS_FORMAT,
        "version": ML_RESULTS_VERSION,
        "source_project": source_project,
        "pipeline": pipeline or {},
        "spectra": spectra,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)

