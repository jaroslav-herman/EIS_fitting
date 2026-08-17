# ML topology classification

This package performs the first ML experiment only: predicting the manually
selected EEC circuit string from an impedance spectrum. It does not fit
circuits, predict parameters, select points, use DRT features, or change the
Tkinter application.

`load_eisfit_projects()` reads the existing `.eisfit` JSON payload, reuses the
application's dataframe deserializer and `load_cycle()`, and excludes invalid
or unlabeled spectra with a reason. Supply an explicit mapping from project
path to physical sample ID; sample identity must be trusted for leave-one-
sample-out validation.

`SpectrumPreprocessor` uses a fold-local logarithmic frequency grid. Impedance
is scaled per spectrum by its median magnitude. Points outside a spectrum's
range are not extrapolated; missing grid values are filled using statistics
learned from the training fold. Metadata is optional and separately
standardized from training-fold values.

The public API is intentionally small:

```python
from pathlib import Path
from ml import load_eisfit_projects, run_topology_experiment

paths = [Path("project_a.eisfit.json"), Path("project_b.eisfit.json")]
report = load_eisfit_projects(paths, {str(paths[0]): "sample_1", str(paths[1]): "sample_2"})
experiment = run_topology_experiment(report.records, use_metadata=False)
experiment.save(Path("ml_results/spectrum_only"))
```

The result contains per-spectrum probabilities, confidence, correctness, fold
identifiers, fold metrics, overall metrics, confidence-threshold summaries,
and Brier scores. The primary validation is leave-one-sample-out; random
splits are deliberately not part of this first implementation.
