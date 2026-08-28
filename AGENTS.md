# EIS Fitting repository guidance

## Start here

- Read only the files that own the requested behavior; `eis_gui.py` is about 14k lines and should not be loaded wholesale.
- For Tk/UI work, use `.agents/skills/eis-gui-development/SKILL.md`.
- For ML inference, result reuse, or ML-to-GUI work, use `.agents/skills/eis-ml-integration/SKILL.md`.
- For circuit comparison, parameter transfer, or topology changes, use `.agents/skills/eec-circuit-reasoning/SKILL.md`.
- For state mutation, cache invalidation, project schema, or import/export compatibility, use `.agents/skills/eis-state-preservation/SKILL.md`.
- For impedance signs, units, frequency/tau handling, CPEs, Nyquist/Bode, or DRT semantics, use `.agents/skills/eis-drt-conventions/SKILL.md`.
- Preserve unrelated user changes. This repository may contain experimental data and generated ML artifacts that are intentionally untracked or large.

List the local skills without loading their bodies:

```powershell
rg -n "^name:|^description:" .agents\skills -g SKILL.md
```

Load only the matching `SKILL.md`; follow its reference links only when the routed condition applies. If a useful repository workflow recurs and no skill covers it, invoke `$skill-creator` to update the nearest skill or create one narrow skill. Do not create a skill from a one-off bug or preference.

## Architecture

- `eis_model.py`: GUI-independent mutable state (`CycleState`, `ProjectState`, `ParameterValue`).
- `eis_services.py`: loading, DRT/outlier analysis, model selection, and fitting. Put reusable computation here, not in Tk callbacks.
- `eis_project.py`: versioned `.eisfit.json` persistence and exports. Current project format/version: `eis-fitting-project` / `4`.
- `eis_gui.py`: Tk widgets, event coordination, plotting, and application of service results.
- `circuit_structure.py`: structural circuit equivalence and parameter-name mapping; do not compare equivalent circuits as raw strings.
- `ml/dataset.py`: canonical extraction of ML records from `.eisfit.json`.
- `ml/gui_results.py` and `ml/results_schema.py`: compatibility boundary between ML artifacts and the GUI.
- `wepy` is a sibling library at `../wepy`; fix general BioLogic parsing or reusable electrochemistry utilities there when that is the true owner.

## Non-negotiable behavior

- Keep Tk calls on the main thread. Long loading, fitting, DRT, and ML work goes through `EISApplication._submit()` or an equivalent worker/main-thread handoff.
- A point is active through `CycleState.included`, which intersects `manually_included` with the frequency window. `outliers` records why points were rejected; outlier actions also clear `manually_included`. Preserve original point order, mask length, and this synchronization.
- Any selection/model/parameter mutation that invalidates a fit or analysis must clear the affected fit and invalidate the appropriate caches.
- Keep training/evaluation out of the GUI. The GUI consumes serialized, validated predictions and always requires an explicit user action before predictions alter spectra.
- Match spectra with the path-independent `spectrum_identifier()` first. Path/name matching exists only for legacy artifacts.
- Keep raw measured arrays in the project, not duplicated in ML sidecars. Write JSON atomically through a temporary file and replacement.
- ML validation splits by physical sample. Fit preprocessing, imputation, scaling, feature selection, and models on training folds only; never leak held-out spectra or the unseen inference sample.

## Verification

Use the repository interpreter:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Find a symbol before opening a large file and run a focused test verbosely:

```powershell
rg -n "^(class|def) |^    def |<visible UI text>|<field_name>" eis_gui.py eis_model.py eis_services.py eis_project.py tests
.\.venv\Scripts\python.exe -m unittest tests.test_batch_stop -v
```

During iteration, run the smallest relevant test module first, then the full suite for changes to persistence, shared state, fitting, or ML contracts. For GUI changes, also launch the included small spectrum and exercise the changed interaction:

```powershell
.\.venv\Scripts\python.exe main.py .\PEIS_at_N2_flow_80_sccm_automated_01_PEIS.mpt --cycle 1
```

Unit tests do not verify Tk layout, focus, selection retention, or event ordering.
