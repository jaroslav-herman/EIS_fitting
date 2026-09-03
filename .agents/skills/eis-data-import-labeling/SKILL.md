---
name: eis-data-import-labeling
description: Import BioLogic EIS files into an .eisfit project and persist repeat-loop labels such as Time and Cycle mod; use for sample-specific file selection, Cell spectrum loading, voltage-pattern annotation, and project creation.
---

# EIS data import and labeling

Use the repository helper [load_and_label_eis.py](../../../load_and_label_eis.py) when a user asks to create an EIS project from raw files and label repeated measurement loops. It uses `wepy.load_folders` and `wepy.load_files` for Windows and network-folder inspection.

Always use this helper to import raw data and create a new `.eisfit` project. Do not replace it with ad-hoc file enumeration or direct project construction.

## Run

From the repository root, use the project interpreter:

```powershell
.\.venv\Scripts\python.exe load_and_label_eis.py `
  "\\server\share\sample-folder" `
  ".\sample_Cell.eisfit.json.gz"
```

The source may be one `.mpr` file or a directory. Directory contents are enumerated only through `wepy.load_files` after the sample folder is identified with `wepy.load_folders`. The import list is selected solely by `DEFAULT_FILE_CONTAINS = ("ay", "rocedure", "PEIS.mpr")` together with the `.mpr` extension. Empty files reported as having no cycles are skipped with a warning; other import errors stop the import. Use `--tolerance` to adjust the voltage similarity threshold used to detect the shortest repeated cycle pattern.

## Behavior to preserve

- Import only the `cell` spectrum (`spectrum_kinds_by_path={resolved_path: ["cell"]}`); do not silently import Working or Counter spectra.
- Skip files containing empty cycles with a warning; continue with the remaining files. Propagate other import errors.
- Sort files by source filename before assigning labels.
- Detect the shortest whole-number repeating pattern from the mean `ewe_ece_v` per cycle.
- Set `Time` to the one-based loop number continuing across files: `1, 2, 3, ...`.
- Set `Cycle mod` to the one-based position inside the detected pattern: `1, 2, ..., N, 1, 2, ...`.
- Store labels in both dataframe columns and every cycle's `custom_metadata`. The latter is required for labels to survive reload and appear in the project explorer.
- Rebuild cycle states through `load_cycle` so masks and raw frequency/impedance arrays retain repository conventions, then save with `save_project_file`, which performs atomic replacement.

After creation, reload the output and verify dataset count, cycle count, label columns, and label metadata. Do not overwrite an existing project unless the user explicitly requests replacement.
