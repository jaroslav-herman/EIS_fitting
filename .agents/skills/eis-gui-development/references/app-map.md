# App workflow map

Read only the section relevant to the requested feature.

## Loading and navigation

`main.py` parses startup options and calls `launch_nyquist_editor()`. Import uses `inspect_eis_file_spectrum_kinds()` followed by `load_projects()` in a worker. `_register_dataset()`, `_switch_dataset()`, `_capture_controls()`, and `_restore_controls()` coordinate the active dataset and cycle. The explorer maps rows to `(dataset_id, LoadedProject, SpectrumMetadata)`.

## State and active points

`CycleState` owns measured arrays, `manually_included`, `outliers`, `frequency_window`, fit arrays, DRT caches, KK caches, parameters, circuit, and custom metadata. `included` is the effective mask. `ProjectState` owns available/loaded cycles and defaults; cycles are loaded lazily.

## Async work

`EISApplication._submit()` sets busy state, disables controls, submits one callable to its executor, and polls the future with `root.after()`. `_poll_future()` handles errors on the main thread and invokes a `_finish_*` callback. Batch services accept a stop event and return completed plus skipped work; keep completed results when stopped.

## Plot and explorer refresh

Plot configuration and artist updates live around `_build_plot()`, `_configure_*_plot()`, and `_refresh_plot()`. Explorer schema, sorting, selection, and row values live between `_build_explorer()` and `_activate_explorer_item()`. Mutating actions normally finish with control restoration, plot refresh, explorer value refresh when displayed metadata changed, and a concise status update.

## Persistence

`eis_project.py` converts state to a versioned JSON payload, embeds dataframes with pandas `orient="split"`, and writes atomically. Loading accepts project versions 1–4 and supplies compatibility defaults. Tests that construct older payloads are valuable migration checks.

## Existing high-value test anchors

- `tests/test_batch_stop.py`: cancellation and partial batch results.
- `tests/test_circuit_structure.py`: circuit equivalence and mapping.
- `tests/test_relaxis_project_load.py` and `test_relaxis_mask_limits_models.py`: imported state and mask alignment.
- `tests/test_plot_export.py`: displayed-series export.
- `tests/test_spectrum_simulator.py`: circuit simulation behavior.
