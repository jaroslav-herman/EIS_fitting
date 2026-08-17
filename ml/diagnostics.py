from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dataset import SpectrumRecord


def records_frame(records: list[SpectrumRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "spectrum_id": record.spectrum_id,
            "source_project": record.source_project,
            "sample_id": record.sample_id,
            "cycle": record.cycle,
            "voltage": record.voltage,
            "current": record.current,
            "time": record.time,
            "device_setup": record.device_setup,
            "original_eec_topology": record.original_eec_topology,
            "electrochemical_topology": record.electrochemical_topology,
            "l0_required_in_manual_fit": record.l0_required_in_manual_fit,
            "manual_f_min": record.manual_f_min,
            "manual_f_max": record.manual_f_max,
            "manual_log_f_min": __import__("numpy").log10(record.manual_f_min) if record.manual_f_min else None,
            "manual_log_f_max": __import__("numpy").log10(record.manual_f_max) if record.manual_f_max else None,
            "raw_frequency_points": len(record.frequency),
            "cleaned_frequency_points": len(record.cleaned_frequency) if record.cleaned_frequency is not None else 0,
        }
        for record in records
    )


def export_diagnostics(records: list[SpectrumRecord], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame = records_frame(records)
    frame.to_csv(directory / "spectrum_diagnostics.csv", index=False)
    for name, columns in {
        "overall_topology_distribution.csv": ["electrochemical_topology"],
        "original_eec_distribution.csv": ["original_eec_topology"],
        "sample_topology_distribution.csv": ["sample_id", "electrochemical_topology"],
        "sample_original_eec_distribution.csv": ["sample_id", "original_eec_topology"],
        "sample_l0_distribution.csv": ["sample_id", "l0_required_in_manual_fit"],
        "setup_l0_distribution.csv": ["device_setup", "l0_required_in_manual_fit"],
        "voltage_l0_distribution.csv": ["voltage", "l0_required_in_manual_fit"],
        "voltage_topology_distribution.csv": ["voltage", "electrochemical_topology"],
        "voltage_frequency_range_distribution.csv": ["voltage", "manual_f_min", "manual_f_max"],
        "sample_frequency_range_distribution.csv": ["sample_id", "manual_f_min", "manual_f_max"],
    }.items():
        grouped = frame.groupby(columns, dropna=False).size().reset_index(name="count")
        if "electrochemical_topology" in columns or "original_eec_topology" in columns:
            denominators = grouped.groupby(columns[:-1], dropna=False)["count"].transform("sum") if len(columns) > 1 else grouped["count"].sum()
            grouped["percentage"] = grouped["count"] / denominators * 100.0
        grouped.to_csv(directory / name, index=False)
