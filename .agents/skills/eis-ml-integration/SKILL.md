---
name: eis-ml-integration
description: Reuse, evaluate, serialize, or integrate EIS machine-learning outputs with the EIS Fitting app. Use for .eisfit extraction, leakage-safe validation, inference artifacts, spectrum identity, ML sidecars, GUI previews, and explicit application of predictions; do not use for ordinary manual fitting or plot-only UI work.
---

# EIS ML integration

Treat ML as a producer of portable suggestions and the GUI as a conservative consumer. Preserve reproducibility, sample-level validation, spectrum identity, and user control.

## Choose the path

- Dataset extraction or labels: start at `ml/dataset.py` and `SpectrumRecord`.
- Preprocessing/features: reuse `ml/preprocessing.py`, `ml/automatic_preprocessing.py`, or the stage-specific reusable module; avoid copying logic from `run_*`/`evaluate_*` scripts.
- GUI consumption: start at `ml/gui_results.py`, `ml/results_schema.py`, and the `eis_gui.py` methods containing `_ml_`.
- Artifact migration: use `ml/migrate_embedded_results.py` and retain compatibility tests.
- A new training/evaluation stage: read [references/ml-contracts.md](references/ml-contracts.md) before implementation.

Inspect only the selected modules and their tests. The `ml/` directory contains historical experiment runners and generated-stage code; do not infer the current contract from a single runner.

## Reuse boundaries

- Extract spectra through `load_eisfit_projects()` rather than parsing project JSON independently.
- Put stable transformations and inference helpers in reusable modules. Keep CLI runners thin and guarded by `if __name__ == "__main__"`.
- Do not import experiment runners into `eis_gui.py`.
- The GUI loads a sidecar into `MLResult`, previews overlays/metadata, and applies frequency masks, active points, topology, or initial parameters only through separate explicit actions.
- Validate suggested circuits through `suggested_eec()` and map equivalent circuit parameter names structurally.

## Identity and artifact contract

- Use `spectrum_identifier(frequency, z_real, z_imag, cycle, control)` as the primary, path-independent key.
- Preserve `spectrum_id`, `spectrum_key`, cycle, control, source metadata, pipeline/model versions, and training sample provenance.
- New GUI-facing output uses `format: eis-fitting-ml-results`, version `1`, normally named `*_ml_results.json`.
- Do not duplicate raw frequency/impedance arrays in the sidecar merely to identify a spectrum. Legacy readers may accept them, but new writers should emit the hash key.
- Write through `write_ml_results()` or the same temporary-file replacement pattern.
- Reject or mark unavailable masks whose length differs from the measured spectrum; never truncate or reorder to force a match.

## Scientific validity

- The independent unit is the physical sample, not a spectrum. Use leave-one-sample-out validation for model choice and reported performance.
- Fit every learned preprocessing step on the training fold only, including grids, fills, scaling, feature selection, thresholds learned from data, and calibration.
- Keep any declared unseen inference sample out of all training, tuning, and selection.
- Preserve logarithmic frequency treatment and avoid extrapolating outside a spectrum's measured range.
- Record exclusions with reasons instead of silently dropping spectra.
- Separate conventional EEC fitting results from ML predictions in names and reports.
- Use fixed seeds where supported and save enough configuration and provenance to reproduce inference.

## Verification

Add tests at both sides of a changed contract: writer/schema behavior and `load_ml_results()`/GUI matching behavior. Prefer synthetic arrays for unit tests and a small `.eisfit.json` fixture for integration.

Run focused tests such as:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_ml tests.test_ml_results_architecture
```

Then run the full suite when changing shared extraction, identifiers, project persistence, masks, or GUI application behavior.
