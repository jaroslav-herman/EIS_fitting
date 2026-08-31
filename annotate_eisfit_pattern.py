"""Annotate repeated EIS voltage-loop metadata in an .eisfit JSON project.

The script preserves measured arrays and cycle order.  It adds two dataframe
columns to every dataset:

* ``Time``: one-based repeated-loop number;
* ``Cycle mod``: one-based cycle number within the detected loop.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from eis_project import dataframe_from_payload, load_json_payload


def _cycle_voltage_sequence(frame: pd.DataFrame) -> tuple[list[int], np.ndarray]:
    if "cycle_number" not in frame.columns or "ewe_ece_v" not in frame.columns:
        raise ValueError("dataset needs cycle_number and ewe_ece_v columns")
    grouped = frame.groupby("cycle_number", sort=True)["ewe_ece_v"].mean()
    cycles = [int(value) for value in grouped.index]
    voltages = pd.to_numeric(grouped, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(voltages).all():
        raise ValueError("cycle voltage metadata contains non-finite values")
    return cycles, voltages


def find_pattern_length(voltages: np.ndarray, tolerance: float = 0.01) -> int:
    """Return the shortest repeating voltage pattern length."""
    count = int(voltages.size)
    if count < 2:
        return count
    for length in range(1, count + 1):
        if count % length:
            continue
        blocks = voltages.reshape(count // length, length)
        if np.max(np.ptp(blocks, axis=0)) <= tolerance:
            return length
    return count


def annotate_payload(payload: dict[str, object], tolerance: float = 0.01) -> dict[str, object]:
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("project contains no datasets")

    ordered = sorted(
        datasets,
        key=lambda item: Path(str(item["state"]["source_path"])).name.casefold(),
    )
    for dataset in ordered:
        frame = dataframe_from_payload(str(dataset["dataframe"]))
        cycles, voltages = _cycle_voltage_sequence(frame)
        pattern_length = find_pattern_length(voltages, tolerance)
        if len(cycles) % pattern_length:
            raise ValueError("cycle count is not divisible by the detected pattern")
        cycle_to_position = {
            cycle: index % pattern_length + 1
            for index, cycle in enumerate(cycles)
        }
        cycle_to_loop = {
            cycle: index // pattern_length + 1
            for index, cycle in enumerate(cycles)
        }
        frame["Time"] = frame["cycle_number"].map(cycle_to_loop).astype("Int64")
        frame["Cycle mod"] = frame["cycle_number"].map(cycle_to_position).astype("Int64")
        dataset["dataframe"] = frame.to_json(orient="split", date_format="iso")
        dataset["pattern_length"] = pattern_length
        dataset["pattern_loops"] = len(cycles) // pattern_length
    payload["datasets"] = ordered
    return payload


def _write_payload(payload: dict[str, object], path: Path) -> None:
    raw = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    if path.suffix.casefold() == ".gz":
        raw = gzip.compress(raw, compresslevel=9, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input .eisfit JSON or .json.gz project")
    parser.add_argument("output", type=Path, help="annotated output project")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="maximum voltage difference between repeated cycles in volts",
    )
    args = parser.parse_args()
    payload = annotate_payload(load_json_payload(args.input), args.tolerance)
    _write_payload(payload, args.output)
    print(f"wrote {args.output}")
    for dataset in payload["datasets"]:
        source = Path(str(dataset["state"]["source_path"])).name
        print(f"{source}: {dataset['pattern_loops']} loops x {dataset['pattern_length']} cycles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
