"""Export EEC fit results and spectrum metadata from .eisfit projects."""

from __future__ import annotations

import argparse
from pathlib import Path

from eis_gui import EISApplication
from eis_project import export_fit_parameters_for_states


def default_output_path(project: Path) -> Path:
    """Return the app-style CSV name next to a project file."""
    name = project.name
    for suffix in (".eisfit.json.gz", ".eisfit.json", ".json.gz", ".json"):
        if name.casefold().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return project.with_name(f"{name}_fit_parameters.csv")


def export_project(project: Path, output: Path | None = None) -> int:
    """Load *project* and write the same fit-parameter CSV as the GUI."""
    project = Path(project)
    restored = EISApplication._load_saved_project(project)
    states = [state for _dataset_id, _loaded, state in restored]
    destination = Path(output) if output is not None else default_output_path(project)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = export_fit_parameters_for_states(states, destination)
    print(f"wrote {destination} ({count} fitted spectra)")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projects", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for CSVs; defaults to each project's directory",
    )
    args = parser.parse_args()
    total = 0
    for project in args.projects:
        output = (
            args.output_dir / default_output_path(project).name
            if args.output_dir is not None
            else None
        )
        total += export_project(project, output)
    print(f"exported {total} fitted spectra from {len(args.projects)} project(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
