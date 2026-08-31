---
name: eis-data-import-labeling
description: Import BioLogic EIS files into an .eisfit project and persist repeat-loop labels such as Time and Cycle mod; use for sample-specific file selection, Cell spectrum loading, voltage-pattern annotation, and project creation.
---

# EIS data import and labeling

Use the repository helper [load_and_label_eis.py](../../../load_and_label_eis.py) when a user asks to create an EIS project from raw files and label repeated measurement loops. It uses `wepy.load_folders` and `wepy.load_files` for Windows and network-folder inspection.

## Run

From the repository root, use the project interpreter:

```powershell
.\.venv\Scripts\python.exe load_and_label_eis.py `
  "\\server\share\sample-folder" `
  ".\sample_Cell.eisfit.json.gz"
```

The source may be one matching `.mpr` file or a directory. Directory contents are enumerated only through `wepy.load_files` after the sample folder is identified with `wepy.load_folders`. The default filename expression is case-insensitive and leaves the station prefix unrestricted; only `Day` and `Procedure` carry the numeric structure:

```text
^(?:.*_)?Day\d+_Procedure\d+_05_PEIS_C01\.mpr$
```

Override it with `--pattern` only when the requested sample uses a different naming convention. Use `--tolerance` to adjust the voltage similarity threshold used to detect the shortest repeated cycle pattern.

## Behavior to preserve

- Import only the `cell` spectrum (`spectrum_kinds_by_path={resolved_path: ["cell"]}`); do not silently import Working or Counter spectra.
- Sort files by source filename before assigning labels.
- Detect the shortest whole-number repeating pattern from the mean `ewe_ece_v` per cycle.
- Set `Time` to the one-based loop number continuing across files: `1, 2, 3, ...`.
- Set `Cycle mod` to the one-based position inside the detected pattern: `1, 2, ..., N, 1, 2, ...`.
- Store labels in both dataframe columns and every cycle's `custom_metadata`. The latter is required for labels to survive reload and appear in the project explorer.
- Rebuild cycle states through `load_cycle` so masks and raw frequency/impedance arrays retain repository conventions, then save with `save_project_file`, which performs atomic replacement.

After creation, reload the output and verify dataset count, cycle count, label columns, and label metadata. Do not overwrite an existing project unless the user explicitly requests replacement.
