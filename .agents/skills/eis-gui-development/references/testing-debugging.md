# GUI testing and debugging

## Cheap investigation

Search by visible label, callback, state field, and refresh method before reading code:

```powershell
rg -n "<button text>|<callback>|<state field>|_refresh_|_restore_" eis_gui.py eis_model.py eis_services.py tests
```

Trace one vertical slice: event binding or menu entry → callback → capture/validation → service or state mutation → `_finish_*` → restore/refresh/status. Most GUI regressions are a missing edge in that slice, stale state, or work performed on the wrong thread.

## Test selection

- Circuit/model transfer: `tests.test_circuit_structure`, `tests.test_batch_stop`.
- Project/load/masks: `tests.test_relaxis_project_load`, `tests.test_relaxis_mask_limits_models`.
- ML consumer contract: `tests.test_ml`, `tests.test_ml_results_architecture`.
- Plot data/export: `tests.test_plot_export`.
- Simulator: `tests.test_spectrum_simulator`.
- Point detection/frequency selection: `tests.test_point_validity`, `tests.test_low_frequency_selector`.

Run the narrow module with `-v`, then the suite when shared state or persistence changed:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_circuit_structure -v
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Manual GUI check

```powershell
.\.venv\Scripts\python.exe main.py .\PEIS_at_N2_flow_80_sccm_automated_01_PEIS.mpt --cycle 1
```

Exercise the changed action, cancel/close path, cycle switch, explorer multi-selection, save/reload when state persists, and a second invocation. Check status text and terminal tracebacks. Do not treat a passing unit suite as evidence that layout, focus, event order, or background-thread behavior is correct.
