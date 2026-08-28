---
name: eis-gui-development
description: Extend or repair the EIS Fitting Tkinter application, including plots, explorer actions, fitting/DRT workflows, persistence-visible state, and responsive background operations. Use for app-building work that touches eis_gui.py or connects UI behavior to EIS services; do not use for standalone ML training experiments.
---

# EIS GUI development

Build the feature at the narrowest owning layer while preserving responsive Tk behavior and per-spectrum state.

## Locate before reading

Do not load all of `eis_gui.py`. Use its method names as the index:

```powershell
rg -n "^(    )?def .*<term>|<visible label>|<state field>" eis_gui.py eis_model.py eis_services.py eis_project.py tests
```

Read the matching method, its direct callers, and the state/service types it uses.

- Read [references/app-map.md](references/app-map.md) only when a change crosses layers or adds a new workflow.
- Read [references/batch-and-cancellation.md](references/batch-and-cancellation.md) for batch, stop, progress, Up/Down, or chained background work.
- Read [references/testing-debugging.md](references/testing-debugging.md) when reproducing a GUI defect or choosing verification for a UI change.

## Place the change

- Persistent per-spectrum data or cache validity: `CycleState` in `eis_model.py`, then serialization in `eis_project.py`.
- Project-wide data: `ProjectState`.
- Reusable numerical, import, DRT, outlier, or fit logic: `eis_services.py` with GUI-free inputs/results.
- Circuit equivalence or parameter renumbering: `circuit_structure.py`.
- Widgets, dialogs, plotting artists, event binding, selection coordination, and status text: `eis_gui.py`.
- General `.mpt` parsing shared beyond this app: sibling `../wepy`.

Avoid adding more scientific computation to `EISApplication`. Prefer a small typed service result and a thin `_finish_*` UI method.

## Preserve application invariants

- Before switching a dataset/cycle or launching work based on entries, capture current controls with the established capture path.
- Run expensive work with `_submit(work, success, error_title, ...)`. Worker code must not touch Tk objects. Apply results and show dialogs only on the Tk thread.
- Respect `busy`, control disabling, the stop event, and partial-result semantics for batch operations.
- Keep `frequency_hz`, impedance, manual mask, outlier mask, residuals, and ML masks aligned by original point index.
- Use `cycle.included` for computation unless a workflow explicitly needs raw/manual masks.
- After changes to points, frequency range, model, or parameters, clear stale fits and invalidate DRT/KK caches as appropriate before refreshing.
- Preserve explorer multi-selection when focus changes unless the action explicitly replaces selection.
- Use structural circuit comparison/mapping for circuits that may have reordered branches or different element numbers.
- Do not silently apply automatic or ML suggestions. Preview or label them, require an explicit action, and retain a restoration path for destructive selection changes.

## Finish the vertical slice

Update every affected surface: state, service, persistence/migration, UI control/menu/shortcut, refresh/status behavior, and focused tests. Do not bump the project version for a field the existing tolerant loader can safely default; do bump it and retain old-version loading when the representation changes incompatibly.

Run the narrow test first, then:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

For visible or event-driven changes, manually exercise the feature with the included small data after automated tests.
