---
name: eis-deterministic-ml-fit
description: Run deterministic EIS outlier selection followed by a selected ML model, frequency/model/parameter application, conventional EEC fitting, and robust refinement while preserving project state and provenance.
---

# Deterministic outliers + ML/EEC fitting

Use [run_deterministic_ml_pipeline.py](../../../run_deterministic_ml_pipeline.py) for the complete ordered workflow on every spectrum in an `.eisfit.json` or `.eisfit.json.gz` project.

## Standard command

```powershell
.\.venv\Scripts\python.exe run_deterministic_ml_pipeline.py `
  ".\467_III_cathode_etching_series_20min_Cell.eisfit.json.gz" `
  --model ".\ml\analysis\number_aware_pipeline_455\pipeline.joblib" `
  --results ".\467_III_cathode_etching_series_20min_Cell_ml_results.json" `
  --report ".\467_III_cathode_etching_series_20min_Cell_ml_fit_report.json"
```

Defaults reproduce the workflow used in this chat: deterministic outlier threshold `3`, Sputtered cathode bundle, refine z-threshold `3.5`, and at most `5` refinement iterations.

## Required order and state rules

1. Load the saved project through the application loader.
2. For every spectrum, run `detect_outliers_in_active_points` on the current included points with threshold `3`, then apply those indices with `CycleState.apply_outliers`.
3. Build runtime records from the unchanged raw arrays and run the selected serialized ML bundle. The model predicts frequency limits, deterministic ML active masks, EEC topology, and initial parameters.
4. Apply predictions, fit conventionally with local least squares, then run robust Refine fit.
5. If fitting fails, retry with fitted parameters copied from the nearest successful spectrum by voltage and time, using structural EEC parameter mapping.
6. Save atomically and retain a report of fits, refinements, retries, and failures.

The pre-ML deterministic exclusions must not be discarded when the ML active mask is applied: combine masks by intersection and preserve existing outlier provenance. Do not alter raw frequency or impedance arrays, truncate masks, or run a second standalone deterministic pass after refinement. Frequency-window exclusions are not automatically labeled as deterministic outliers.

Validate result count and path-independent spectrum identifiers before applying predictions. After saving, reload the compressed project and verify raw arrays, full-length masks, fit curves, fitted parameters, and refinement provenance. Do not silently hide nonconverged or failed spectra.
