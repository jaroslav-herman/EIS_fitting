from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from eis_model import CycleState
from eis_project import dataframe_from_payload
from eis_services import load_cycle


KNOWN_EEC_TOPOLOGIES = {
    "R0-L0-p(R1,CPE1)",
    "R0-L0-p(R1,CPE1)-p(R2,CPE2)",
    "R0-L0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)",
    "R0-L0-p(R1,CPE1)-p(R3,CPE3)",
    "R0-p(R1,CPE1)",
    "R0-p(R1,CPE1)-p(R2,CPE2)",
    "R0-p(R1,CPE1)-p(R3,CPE3)",
}


def canonical_electrochemical_topology(original: str) -> str | None:
    """Remove only the setup-parasitic L0 from a known manual EEC label."""
    mapping = {
        "R0-L0-p(R1,CPE1)": "R0-p(R1,CPE1)",
        "R0-p(R1,CPE1)": "R0-p(R1,CPE1)",
        "R0-L0-p(R1,CPE1)-p(R2,CPE2)": "R0-p(R1,CPE1)-p(R2,CPE2)",
        "R0-p(R1,CPE1)-p(R2,CPE2)": "R0-p(R1,CPE1)-p(R2,CPE2)",
        "R0-L0-p(R1,CPE1)-p(R3,CPE3)": "R0-p(R1,CPE1)-p(R3,CPE3)",
        "R0-p(R1,CPE1)-p(R3,CPE3)": "R0-p(R1,CPE1)-p(R3,CPE3)",
        "R0-L0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)": "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)",
        "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)": "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)",
    }
    return mapping.get(original.strip())


@dataclass(frozen=True)
class SpectrumRecord:
    spectrum_id: str
    source_project: str
    sample_id: str
    cycle: int
    voltage: float | None
    current: float | None
    time: float | None
    frequency: np.ndarray
    z_real: np.ndarray
    z_imag: np.ndarray
    topology_label: str
    original_eec_topology: str | None = None
    electrochemical_topology: str | None = None
    l0_required_in_manual_fit: bool | None = None
    cleaned_frequency: np.ndarray | None = None
    cleaned_z_real: np.ndarray | None = None
    cleaned_z_imag: np.ndarray | None = None
    device_setup: str | None = None
    manual_f_min: float | None = None
    manual_f_max: float | None = None
    control: str = "cell"

    def __post_init__(self) -> None:
        original = self.original_eec_topology or self.topology_label
        canonical = self.electrochemical_topology or canonical_electrochemical_topology(original) or self.topology_label
        object.__setattr__(self, "original_eec_topology", original)
        object.__setattr__(self, "electrochemical_topology", canonical)
        object.__setattr__(self, "l0_required_in_manual_fit", bool(self.l0_required_in_manual_fit) if self.l0_required_in_manual_fit is not None else "-L0-" in original)
        if self.manual_f_min is not None and self.manual_f_max is not None:
            minimum, maximum = sorted((float(self.manual_f_min), float(self.manual_f_max)))
            object.__setattr__(self, "manual_f_min", minimum)
            object.__setattr__(self, "manual_f_max", maximum)

    @property
    def impedance(self) -> np.ndarray:
        return self.z_real + 1j * self.z_imag

    def arrays(self, mode: str = "raw") -> tuple[np.ndarray, np.ndarray]:
        if mode == "cleaned" and self.cleaned_frequency is not None and self.cleaned_z_real is not None and self.cleaned_z_imag is not None:
            return self.cleaned_frequency, self.cleaned_z_real + 1j * self.cleaned_z_imag
        return self.frequency, self.impedance


def _setup_value(metadata: dict[str, object], fallback: str) -> str:
    for key, value in metadata.items():
        normalized = str(key).lower()
        match = re.search(r"\b(?:RDC|BDC)\d+\b", str(value), flags=re.IGNORECASE)
        if match:
            return match.group(0).upper()
        if any(token in normalized for token in ("device", "instrument", "setup", "technique")) and value not in (None, ""):
            return str(value)
    match = re.search(r"\b(?:RDC|BDC)\d+\b", str(fallback), flags=re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return fallback


@dataclass
class ExtractionReport:
    records: list[SpectrumRecord] = field(default_factory=list)
    exclusions: list[dict[str, str]] = field(default_factory=list)

    @property
    def exclusion_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.exclusions:
            reason = item["reason"]
            counts[reason] = counts.get(reason, 0) + 1
        return counts


def _number(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _read_project_payload(path: Path) -> dict:
    """Read a project payload from plain or gzip-compressed JSON."""
    if path.suffix.lower() == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle:
        return json.load(handle)


def _payload_projects(payload: dict) -> list[tuple[dict, dict]]:
    datasets = payload.get("datasets")
    if datasets:
        return [(entry["state"], entry) for entry in datasets]
    return [(payload, payload)]


def load_eisfit_projects(
    paths: list[Path],
    sample_ids: Mapping[str | Path, str],
    *,
    require_fit: bool = True,
    require_frequency_window: bool = True,
) -> ExtractionReport:
    """Extract labelled spectra from saved projects.

    ``sample_ids`` is intentionally explicit: sample identity is an important
    grouping variable and must not be guessed from an arbitrary filename.
    Saved project payloads are read directly, while spectrum construction uses
    the application's existing dataframe and ``load_cycle`` logic.
    """
    report = ExtractionReport()
    for path in paths:
        path = Path(path)
        sample = sample_ids.get(str(path), sample_ids.get(path, None))
        if sample is None:
            report.exclusions.append({"spectrum_id": str(path), "reason": "missing_sample_id"})
            continue
        try:
            payload = _read_project_payload(path)
        except Exception as error:
            report.exclusions.append({"spectrum_id": str(path), "reason": f"invalid_project:{type(error).__name__}"})
            continue
        for entry_index, (state_payload, entry) in enumerate(_payload_projects(payload)):
            dataframe_payload = entry.get("dataframe")
            if not dataframe_payload:
                report.exclusions.append({"spectrum_id": str(path), "reason": "missing_dataframe"})
                continue
            control = str(state_payload.get("control", payload.get("control", "cell")))
            circuit_default = state_payload.get("circuit") or payload.get("circuit")
            dataset_key = str(entry.get("dataset_id") or f"dataset_{entry_index}")
            try:
                dataframe = dataframe_from_payload(dataframe_payload)
            except Exception as error:
                report.exclusions.append({"spectrum_id": str(path), "reason": f"invalid_dataframe:{type(error).__name__}"})
                continue
            for cycle_text, saved in (state_payload.get("cycles") or {}).items():
                cycle_number = int(cycle_text)
                spectrum_id = f"{path.resolve()}::{dataset_key}::{control}::{cycle_number}"
                original_topology = str(saved.get("circuit") or circuit_default or "").strip()
                topology = canonical_electrochemical_topology(original_topology)
                if not original_topology:
                    report.exclusions.append({"spectrum_id": spectrum_id, "reason": "missing_topology"})
                    continue
                if original_topology not in KNOWN_EEC_TOPOLOGIES or topology is None:
                    report.exclusions.append({"spectrum_id": spectrum_id, "reason": "invalid_topology"})
                    continue
                if require_fit and saved.get("fit_parameters") is None:
                    report.exclusions.append({"spectrum_id": spectrum_id, "reason": "missing_fit"})
                    continue
                window = saved.get("frequency_window")
                if window is None and require_frequency_window:
                    report.exclusions.append({"spectrum_id": spectrum_id, "reason": "missing_frequency_window"})
                    continue
                if window is None:
                    manual_f_min = manual_f_max = None
                else:
                    try:
                        manual_f_min, manual_f_max = sorted((float(window[0]), float(window[1])))
                    except (TypeError, ValueError, IndexError):
                        report.exclusions.append({"spectrum_id": spectrum_id, "reason": "invalid_frequency_window"})
                        continue
                    if not np.isfinite(manual_f_min) or not np.isfinite(manual_f_max) or manual_f_min <= 0 or manual_f_max <= manual_f_min:
                        report.exclusions.append({"spectrum_id": spectrum_id, "reason": "invalid_frequency_window"})
                        continue
                try:
                    cycle: CycleState = load_cycle(dataframe, cycle_number, control)
                    frequency = np.asarray(cycle.frequency_hz, dtype=float)
                    impedance = np.asarray(cycle.impedance, dtype=complex)
                    valid = np.isfinite(frequency) & (frequency > 0) & np.isfinite(impedance.real) & np.isfinite(impedance.imag)
                    if valid.sum() < 3:
                        raise ValueError("too_few_valid_points")
                    metadata = cycle.custom_metadata
                    manual_mask = np.asarray(saved.get("manually_included", np.ones(frequency.size)), dtype=bool)
                    if manual_mask.size != frequency.size:
                        raise ValueError("manual_mask_length_mismatch")
                    window = saved.get("frequency_window")
                    if window is not None:
                        minimum, maximum = sorted((float(window[0]), float(window[1])))
                        manual_mask &= (frequency >= minimum) & (frequency <= maximum)
                    cleaned = valid & manual_mask
                    if cleaned.sum() < 3:
                        cleaned_frequency = cleaned_real = cleaned_imag = None
                    else:
                        cleaned_frequency = frequency[cleaned]
                        cleaned_real = impedance.real[cleaned]
                        cleaned_imag = impedance.imag[cleaned]
                    record = SpectrumRecord(
                        spectrum_id=spectrum_id,
                        source_project=str(path),
                        sample_id=str(sample),
                        cycle=cycle_number,
                        voltage=_number(cycle.potential_v),
                        current=_number(cycle.current_ma),
                        time=_number(cycle.time_s),
                        frequency=frequency[valid],
                        z_real=impedance.real[valid],
                        z_imag=impedance.imag[valid],
                        topology_label=topology,
                        original_eec_topology=original_topology,
                        electrochemical_topology=topology,
                        l0_required_in_manual_fit="-L0-" in original_topology,
                        cleaned_frequency=cleaned_frequency,
                        cleaned_z_real=cleaned_real,
                        cleaned_z_imag=cleaned_imag,
                        device_setup=_setup_value(metadata, f"{dataset_key}:{control}"),
                        manual_f_min=manual_f_min,
                        manual_f_max=manual_f_max,
                        control=control,
                    )
                    report.records.append(record)
                except Exception as error:
                    report.exclusions.append({"spectrum_id": spectrum_id, "reason": f"invalid_spectrum:{error}"})
    return report
