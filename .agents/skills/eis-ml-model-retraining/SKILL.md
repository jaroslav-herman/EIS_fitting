---
name: eis-ml-model-retraining
description: Retrain the named GUI ML models (Sputtered cathode, AEM-WE, Flex) in EIS Fitting on manually fitted .eisfit.json training projects without changing model structure. Use for adding training samples, regenerating pipeline.joblib artifacts, and reproducing report.json provenance; do not use for new experiment runners or GUI inference work.
---

# EIS ML model retraining

Retrain a deployment bundle in place, keeping the pipeline structure and the GUI
consumption path untouched. The GUI never trains: it loads
`ml/analysis/<artifact>/pipeline.joblib` through `ML_TRAINED_MODELS` in
`eis_gui.py`, then `load_pipeline_bundle()` and `infer_bundle_records()` from
`ml/number_aware_pipeline.py`. Retraining only replaces the fitted weights and
the `report.json` provenance.

## Named models and their trainers

| GUI key | Artifact directory | Trainer |
| --- | --- | --- |
| Sputtered cathode | `ml/analysis/number_aware_pipeline_453_455_457_467_voltage` | historical Windows-path run; retrain with `ml/number_aware_pipeline.py` CLI |
| AEM-WE | `ml/analysis/number_aware_pipeline_aem_we` | `ml/train_aem_we_model.py` |
| Flex | `ml/analysis/number_aware_pipeline_flex_181` | `ml/train_flex_model.py` |

If a named model has no dedicated trainer, add one that mirrors
`ml/train_flex_model.py` / `ml/train_aem_we_model.py` (thin CLI, guarded by
`if __name__ == "__main__"`, defaults pointing at the committed training data)
instead of hand-running ad hoc code. Keep the existing `report.json` key
structure of that artifact so downstream readers do not break.

## Training data contract

- Training projects are manually fitted `.eisfit.json(.gz)` projects; for AEM-WE
  they live under `training_data/AEM/`. One project = one physical sample, named
  `<sample>.eisfit.json.gz`.
- The physical sample is the independent unit. Every project gets an explicit
  sample ID passed in the path→sample mapping; never let a spectrum's sample
  appear in both training and validation roles.
- Extraction goes through `load_eisfit_projects()`
  (`require_fit=True, require_frequency_window=False,
  allow_invalid_frequency_window=True`) and `train_bundle()`; never parse
  project JSON directly or copy preprocessing logic.
- Commit new training projects to the repo (the original artifacts reference
  Windows OneDrive paths that are unreachable to other machines).

## Retraining steps

1. Prepare the environment: `uv sync`, then use `.venv/bin/python`. The repo
   targets Python 3.14; system Python lacks the dependencies.
2. Run the model's trainer, e.g.
   `.venv/bin/python -m ml.train_aem_we_model` (optional
   `--source <project>` repeated per sample, `--output`, `--seed 42`).
   For AEM-WE the defaults already point at `training_data/AEM/*.eisfit.json.gz`
   and `ml/analysis/number_aware_pipeline_aem_we`.
3. The trainer writes `pipeline.joblib` and `report.json` atomically (tmp file +
   replace) into the artifact directory. Do not hand-edit these.
4. Verify before committing:
   - `load_pipeline_bundle(<artifact>/pipeline.joblib)` unpickles and
     `bundle.training_samples` lists all intended samples.
   - `infer_bundle_records()` on a few spectra from one training project
     returns sensible windows/topologies (smoke test of the GUI path).
   - `report.json` keeps the artifact's previous top-level keys and records the
     new samples/records; LOSO becomes available once ≥2 samples are present.
   - `.venv/bin/python -m unittest tests.test_number_aware_pipeline -v`,
     then the full suite for shared-contract changes.

## Committing

- `ml/analysis/` is partially gitignored; the tracked artifact files must be
  added with `git add -f ml/analysis/<artifact>/pipeline.joblib
  ml/analysis/<artifact>/report.json`. Check `git ls-files` first to confirm
  which files are already tracked.
- Never stage `eisyfit.egg-info/` churn or unrelated untracked analysis output.
- Commit trainer code together with the regenerated artifacts in one focused
  commit describing the samples added (see prior AEM-WE retrain commits for
  style).

## Things that are not retraining

- Changing features, transforms, model families, or the `number_aware_staged_eis`
  version is a pipeline change, not a retrain; route it through
  `ml/number_aware_pipeline.py` plus tests first, and only then retrain.
- Parameter limits and reliability labels are derived from LOSO across
  samples; adding samples legitimately changes them from
  `low_single_sample` to LOSO intervals. Do not fake this by editing reports.
