"""Print a compact validation table for spectra stored in an .eisfit file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eis_project import dataframe_from_payload
from eis_services import load_cycle


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def inspect_eisfit(path: Path, sample: str = "Ti", limit: int = 30) -> pd.DataFrame:
    """Inspect up to ``limit`` saved cycles, including invalid rows."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("datasets") or [{"state": payload, "dataframe": payload.get("dataframe")}]
    rows: list[dict[str, object]] = []

    for entry in entries:
        state = entry["state"]
        dataframe_payload = entry.get("dataframe")
        control = str(state.get("control", payload.get("control", "cell")))
        default_topology = str(state.get("circuit", payload.get("circuit", "")) or "")
        try:
            dataframe = dataframe_from_payload(dataframe_payload)
        except Exception as error:
            rows.append({
                "spectrum_id": f"{path.name}::{control}",
                "sample": sample,
                "voltage": None,
                "topology": default_topology,
                "frequency_points": 0,
                "frequency_range": "",
                "valid": False,
                "validation_reason": f"invalid dataframe: {error}",
            })
            continue

        for cycle_text, saved in (state.get("cycles") or {}).items():
            if len(rows) >= limit:
                break
            cycle_number = int(cycle_text)
            spectrum_id = f"{path.name}::{control}::{cycle_number}"
            topology = str(saved.get("circuit") or default_topology).strip()
            row = {
                "spectrum_id": spectrum_id,
                "sample": sample,
                "voltage": None,
                "topology": topology,
                "frequency_points": 0,
                "frequency_range": "",
                "valid": False,
                "validation_reason": "",
            }
            try:
                cycle = load_cycle(dataframe, cycle_number, control)
                frequency = np.asarray(cycle.frequency_hz, dtype=float)
                impedance = np.asarray(cycle.impedance, dtype=complex)
                valid = (
                    np.isfinite(frequency)
                    & (frequency > 0)
                    & np.isfinite(impedance.real)
                    & np.isfinite(impedance.imag)
                )
                row["voltage"] = _number(cycle.potential_v)
                row["frequency_points"] = int(valid.sum())
                if valid.any():
                    row["frequency_range"] = f"{np.min(frequency[valid]):.6g}–{np.max(frequency[valid]):.6g} Hz"
                if not topology:
                    row["validation_reason"] = "missing topology"
                elif valid.sum() < 3:
                    row["validation_reason"] = "fewer than 3 valid frequency points"
                elif not np.all(valid):
                    row["validation_reason"] = f"{int((~valid).sum())} invalid point(s)"
                else:
                    row["valid"] = True
                    row["validation_reason"] = "ok"
            except Exception as error:
                row["validation_reason"] = f"cannot load spectrum: {error}"
            rows.append(row)

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect spectra extracted from an .eisfit project")
    parser.add_argument("project", type=Path)
    parser.add_argument("--sample", default="Ti")
    parser.add_argument("--limit", type=int, default=30, help="Number of spectra to inspect (recommended: 20–50)")
    parser.add_argument("--csv", type=Path, help="Optional output CSV path")
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 50:
        parser.error("--limit must be between 1 and 50")
    frame = inspect_eisfit(args.project, args.sample, args.limit)
    if args.csv:
        frame.to_csv(args.csv, index=False)
    print(frame.to_string(index=False))
    print(f"\nValid: {int(frame['valid'].sum())}/{len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
