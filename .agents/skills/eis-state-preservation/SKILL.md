---
name: eis-state-preservation
description: Change EIS Fitting state, masks, fits, caches, project persistence, imports, or exports without losing or misaligning scientific results. Use for CycleState/ProjectState mutations, .eisfit schema evolution, save/load compatibility, unsaved-change behavior, and result invalidation; do not use for display-only UI edits.
---

# EIS state preservation

Preserve raw measurements and valid derived work; invalidate only results whose inputs changed. Read [references/state-contract.md](references/state-contract.md) before changing fields, serialization, masks, or cache rules.

## Mutation rules

- Raw `frequency_hz` and `impedance` are measured identity. Never edit or reorder them in place to implement display order, filtering, or fitting.
- All point-indexed arrays must retain the raw point count and ordering. Reject incompatible masks/results rather than truncating, padding, or guessing.
- `included` is `manually_included` intersected with the inclusive frequency window. `outliers` is provenance; outlier operations set both `outliers[index] = True` and `manually_included[index] = False`.
- Use copying helpers for parameter lists and arrays when transferring between cycles. Avoid shared mutable state.
- If measured inputs, included points, circuit, or fit inputs change, clear stale fit fields and the dependent caches indicated by the state contract.
- Cached DRT/KK results are reusable only when their stored included mask and other cache keys match current inputs.
- Background workers should return results; apply mutations on the Tk thread after confirming the target cycle/dataset.

## Persistence rules

- Update serialization and loading together. Supply defaults for older supported versions and validate lengths before assigning arrays.
- Bump `PROJECT_VERSION` only for an incompatible representation or required migration; keep all previously supported versions readable when practical.
- Preserve complex arrays as paired real/imaginary arrays and reconstruct them without display-sign conversion.
- Keep project writes atomic through a sibling temporary file and `replace()`.
- Keep ML artifacts in their sidecar contract; do not embed or duplicate raw measurements in new sidecars.
- Add a round-trip test plus an older-payload compatibility test for meaningful schema changes.

Run persistence/mask tests, then the full suite:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_relaxis_project_load tests.test_relaxis_mask_limits_models tests.test_ml_results_architecture -v
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
