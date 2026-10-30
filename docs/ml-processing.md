# ML processing

eisyFIT can use pretrained machine-learning pipelines to suggest EEC topology and initial parameters. ML assistance is always explicit: predictions are previewed or labelled and never silently replace your data.

## What ML can do

- Suggest an EEC model/topology for selected spectra.
- Predict initial EEC parameter values.
- Apply saved ML results to spectra.
- Fit spectra after ML suggestions are applied.

## Open the ML dialog

Use the **ML processing…** menu item in the menu bar.

The dialog lets you choose operations for the selected spectra. Typical choices are:

- infer topology/initial parameters with the available pretrained pipelines,
- apply previously saved ML results,
- fit affected spectra after applying suggestions.

## Saved pipelines per model

Each trained model (Sputtered cathode, AEM-WE, Flex) can have its own pipeline — the ordered actions plus outlier/refinement settings — saved in **Preferences → ML pipelines**.

In the ML processing dialog, enable **Apply saved pipeline for the selected model (from Preferences)**. The dialog then loads the pipeline saved for the selected model, disables the manual action list and settings, and **Run** executes exactly that saved pipeline. Switching the model while the checkbox is enabled reloads the newly selected model's pipeline.

When the checkbox is off, the dialog works manually as before: you build the action list and set the outlier/refinement values yourself.

## Load ML results

Use **File → Load ML results file…** to load a saved ML results file. The results become available to the ML workflow and can be matched to spectra by the path-independent spectrum identifier.

## Safety rules

- ML suggestions do not change spectra until you explicitly apply them.
- The GUI consumes serialized, validated predictions.
- Applying ML suggestions requires an explicit user action.
- A restoration path is retained where a suggestion replaces existing values.
- Raw measured arrays stay in the project; ML sidecars do not duplicate them.

## Practical workflow

1. Select one or more spectra in the Spectra Explorer.
2. Open **ML processing…**.
3. Choose the ML operations to run.
4. Review which spectra are available and which are unavailable.
5. Apply suggestions explicitly.
6. Fit the affected spectra.
7. Save the project.

## Related actions

- **Suggest EEC initials…** in the Fit menu or Actions group suggests initial values from compatible fitted spectra without using the ML pipelines.
- **Detect EEC parameter anomalies…** compares selected fitted parameter vectors against a calibration set.
- **Auto model: selected (Hybrid DRT)** suggests a model from Hybrid DRT analysis rather than ML.
