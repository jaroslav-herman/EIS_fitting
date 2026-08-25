# ML contracts and stage checklist

## Canonical records

`ml.dataset.SpectrumRecord` carries spectrum/source/sample identity, cycle, metadata, measured arrays, manual topology, and optional manual frequency targets. Its `arrays()` modes keep raw and cleaned representations explicit. `ExtractionReport` retains exclusions and their reasons.

Before creating a new parser, check whether the needed value is already available through `eis_project.dataframe_from_payload()`, `eis_services.load_cycle()`, or `SpectrumRecord`.

## Sidecar fields

The stable outer contract is owned by `ml/results_schema.py`. GUI compatibility is intentionally broader in `ml/gui_results.py`; it reads current sidecars, legacy embedded `ml_results`, tabular outputs, and ML-initialization projects. This tolerance is a reader concern, not permission for new writers to invent another shape.

A new sidecar spectrum should include the canonical hash key and the smallest relevant prediction fields. Examples include frequency bounds, active/outlier masks, process count, L0 requirement, suggested EEC, model parameters, parameter limits/reliability, confidence, and metadata. Omit unavailable predictions or state why they are unavailable; do not fabricate defaults.

## Training/evaluation checklist

1. Declare physical sample IDs and the held-out/unseen policy.
2. Define exclusions and target derivation before model comparison.
3. Build preprocessing inside each training fold.
4. Compare baselines and candidates with the same folds and metrics.
5. Select without consulting the unseen inference sample.
6. Refit the chosen pipeline on permitted training samples only.
7. Serialize the model/preprocessor or emit predictions with model and pipeline provenance.
8. Verify round-trip loading and spectrum matching in the GUI consumer.

## GUI application semantics

Frequency ranges and masks change which measured points are active, so preserve the previous manual selection before applying them and offer restoration. Applying a model or initial parameters should clear stale fits. Applying point masks should also invalidate DRT/KK caches that depend on active points. A prediction overlay may be displayed without mutating scientific state.
