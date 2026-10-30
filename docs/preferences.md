# Preferences

Open **Preferences** from the menu bar or with `Ctrl+P`. Preferences are stored per user, outside the project file.

## EEC models

- Choose which EEC circuits appear in the fitting-model preset lists.
- Add custom `impedance.py` circuit strings to the presets.

## EEC optimizer

These settings control single, selected, and batch EEC fitting:

- **Pipeline** — choose `local only`, `PSO → local`, `GA → local`, `PSO only`, or `GA only`.
- **Random seed** — blank means random.
- **Population / iterations** — global optimizer population size and iteration count.
- **Weight residuals by |Z|** — weight residuals by impedance modulus.
- **Local Jacobian** — numerical only, automatic, or analytical when supported.
- **Show last fit diagnostics** — displays objective, RMSE, convergence, elapsed time, and pipeline stages from the last fit.

## Explorers

- Choose default X and Y axes for the EEC fit-parameter and DRT parameter explorers.
- Reorder Spectra Explorer columns and hide columns you do not need.
- Choose which metadata is shown when hovering Fit/DRT explorer points.
- Choose whether new columns are inserted at the end or the beginning.

## Auto model selection

- **EEC fit time limit** — maximum seconds for an individual EEC fit during automatic model selection.
- Criterion, maximum number of peaks, peak prominence, and peak height settings guide Hybrid DRT-based automatic model selection.

## Thresholds & guidance

- Default outlier threshold.
- Default deterministic threshold.
- Robust-refinement z threshold and maximum iterations.
- Suggested defaults are shown next to advanced settings.

## ML pipelines

- Configure, per trained model (Sputtered cathode, AEM-WE, Flex), the ordered pipeline actions (frequency limits, outlier removal, model selection, parameter limits, initial parameters, fit, refine) and the outlier threshold, refine Z threshold, and refine iterations.
- **Reset to default pipeline** restores the built-in full pipeline for the selected model.
- These pipelines are applied in the **ML processing** dialog when **Apply saved pipeline for the selected model** is enabled there.

## EEC parameter limits

Open **Parameters limits** from the Circuit parameters group (`Alt+W`) to set global bounds for:

- resistors (`r`),
- inductors (`l`),
- CPE constants (`cpe_q`),
- CPE exponents (`cpe_alpha`).

## Saving

Press **Save** in the Preferences window to write the settings to the per-user preferences file. Settings are loaded the next time the application starts.
