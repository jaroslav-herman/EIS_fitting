"""Import selected EIS files into an .eisfit project and label voltage loops.

The labels are written both as dataframe columns and as per-cycle custom
metadata so they survive project reload and are visible to the explorer.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import re

import numpy as np

from eis_project import save_project_file
from eis_services import load_cycle, load_projects


DEFAULT_PATTERN = r"^VIII_Day\d+_Procedure\d+_05_PEIS_C01\.mpr$"
DEFAULT_CIRCUIT = "R0-L0-p(R1,CPE1)"


def source_path_candidates(source: Path) -> list[Path]:
    """Return the exact Windows path plus a compatibility-normalized UNC path.

    Some paths are copied from Markdown or notes with escaped underscores,
    e.g. ``PEM-WE\\_measurements\\2026\\467\\_III``.  Windows Explorer displays
    the corresponding directory names as ``PEM-WE_measurements`` and
    ``467_III``.  Try the literal path first, then merge underscore-prefixed
    components without changing ordinary paths.
    """
    text = str(source)
    candidates = [Path(text)]
    if text.startswith("\\\\"):
        parts = text.split("\\")
        merged = []
        for part in parts:
            if part.startswith("_") and merged:
                merged[-1] += part
            else:
                merged.append(part)
        normalized = "\\".join(merged)
        if normalized != text:
            candidates.append(Path(normalized))
    return candidates


def resolve_source_path(source: Path) -> Path:
    """Resolve an existing source directory/file, including escaped UNC paths."""
    candidates = source_path_candidates(source)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    formatted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"source path is unavailable; tried: {formatted}")


def find_pattern_length(voltages: np.ndarray, tolerance: float = 0.01) -> int:
    """Return the shortest whole-number repeating pattern within tolerance."""
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


def _cycle_voltage_sequence(frame):
    if "cycle_number" not in frame or "ewe_ece_v" not in frame:
        raise ValueError("input needs cycle_number and ewe_ece_v columns")
    grouped = frame.groupby("cycle_number", sort=True)["ewe_ece_v"].mean()
    cycles = [int(value) for value in grouped.index]
    voltages = grouped.to_numpy(dtype=float)
    if not np.isfinite(voltages).all():
        raise ValueError("cycle voltage metadata contains non-finite values")
    return cycles, voltages


def _label_project(project, time_offset: int, tolerance: float):
    frame = project.dataframe.copy()
    cycles, voltages = _cycle_voltage_sequence(frame)
    pattern_length = find_pattern_length(voltages, tolerance)
    if pattern_length < 1 or len(cycles) % pattern_length:
        raise ValueError("cycle count is not divisible by the detected pattern")

    cycle_to_time = {
        cycle: time_offset + index // pattern_length + 1
        for index, cycle in enumerate(cycles)
    }
    cycle_to_mod = {
        cycle: index % pattern_length + 1 for index, cycle in enumerate(cycles)
    }
    frame["Time"] = frame["cycle_number"].map(cycle_to_time).astype("Int64")
    frame["Cycle mod"] = frame["cycle_number"].map(cycle_to_mod).astype("Int64")
    project.dataframe = frame

    # Rebuild every cycle from the imported dataframe, then add labels to the
    # metadata persisted by the project schema. Keep one-based cycle numbers.
    project.state.cycles = {}
    for cycle_number in cycles:
        cycle = load_cycle(frame, cycle_number, project.state.control)
        cycle.circuit = project.state.circuit
        cycle.parameters = deepcopy(project.state.default_parameters)
        cycle.custom_metadata.update(
            {"Time": cycle_to_time[cycle_number], "Cycle mod": cycle_to_mod[cycle_number]}
        )
        project.state.cycles[cycle_number] = cycle
    project.state.available_cycles = cycles
    project.state.active_cycle = cycles[0]
    return len(cycles) // pattern_length, pattern_length


def import_and_label(
    source: Path,
    output: Path,
    *,
    filename_pattern: str = DEFAULT_PATTERN,
    tolerance: float = 0.01,
    circuit: str = DEFAULT_CIRCUIT,
) -> dict[str, int]:
    """Import matching files from *source* and atomically save labeled Cell data."""
    source = resolve_source_path(Path(source))
    matcher = re.compile(filename_pattern, re.IGNORECASE)
    candidates = [source] if source.is_file() else sorted(
        (
            path
            for path in source.rglob("*")
            if path.is_file()
            and path.suffix.casefold() == ".mpr"
            and matcher.fullmatch(path.name)
        ),
        key=lambda path: path.name.casefold(),
    )
    if source.is_file() and not matcher.fullmatch(source.name):
        raise ValueError(f"file does not match --pattern: {source.name}")
    if not candidates:
        raise FileNotFoundError(f"no .mpr files matched in {source}")

    selection = {path.resolve(): ["cell"] for path in candidates}
    report = load_projects(
        candidates,
        control="cell",
        circuit=circuit,
        cycle=1,
        spectrum_kinds_by_path=selection,
    )
    if report.errors:
        details = "; ".join(f"{path.name}: {error}" for path, error in report.errors)
        raise RuntimeError(f"import failed: {details}")
    projects = [project for _dataset_id, project in report.loaded]
    if len(projects) != len(candidates):
        raise RuntimeError(
            f"expected one Cell dataset per file ({len(candidates)}), got {len(projects)}"
        )

    offset = 0
    loops = 0
    spectra = 0
    datasets = []
    for project in projects:
        pattern_loops, pattern_length = _label_project(project, offset, tolerance)
        offset += pattern_loops
        loops += pattern_loops
        spectra += len(project.state.available_cycles)
        datasets.append((project.dataset_id, project.state, project.dataframe))
        print(
            f"{project.state.source_path.name}: {pattern_loops} loops x "
            f"{pattern_length} cycles"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    save_project_file(projects[0].state, output, datasets=datasets)
    print(f"wrote {output} ({spectra} spectra, {loops} loops)")
    return {"files": len(projects), "spectra": spectra, "loops": loops}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help=".mpr file or directory to scan")
    parser.add_argument("output", type=Path, help="output .eisfit.json or .json.gz")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="regular expression for filenames")
    parser.add_argument("--tolerance", type=float, default=0.01, help="voltage tolerance in volts")
    parser.add_argument("--circuit", default=DEFAULT_CIRCUIT, help="initial EEC circuit")
    args = parser.parse_args()
    import_and_label(
        args.source,
        args.output,
        filename_pattern=args.pattern,
        tolerance=args.tolerance,
        circuit=args.circuit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
