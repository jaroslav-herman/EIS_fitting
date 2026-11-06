# Button and menu reference

This page explains what the visible buttons and menu entries do. Menu labels ending with `…` open a dialog.

## Menu bar

### File

| Menu item | What it does |
| --- | --- |
| New project (`Ctrl+N`) | Clears the current project and starts an empty one. |
| Import data… (`Ctrl+I`) | Imports one or more BioLogic `.mpt` files, or a delimited text/CSV spectrum file, and keeps previously loaded spectra. |
| Load project… (`Ctrl+L`) | Loads a saved `.eisfit.json` project with spectra, models, parameters, fits, and DRT results. |
| Load RelaxIS 3 project | Imports a project exported by RelaxIS 3. |
| Save project… (`Ctrl+S`) | Saves the current project to a `.eisfit.json` file. |
| Load ML results file… | Loads saved machine-learning predictions and makes them available to the ML workflow. |
| Exit | Closes the application and asks whether to save unsaved changes. |

### Fit

| Menu item | What it does |
| --- | --- |
| Fit selected spectrum (`Alt+S`) | Fits the displayed spectrum with the current model and parameter-table values. |
| Suggest EEC initials… | Suggests initial EEC parameters from compatible fitted spectra and optionally fits immediately. |
| Detect EEC parameter anomalies… | Compares selected fitted spectra against a calibration set and highlights anomalous parameter vectors. |
| Batch down | Fits spectra below the current spectrum in explorer order, using each result to initialize the next. |
| Batch fit selected down | Batch-fits the selected spectra toward lower cycles. |
| Batch fit selected up | Batch-fits the selected spectra toward higher cycles. |
| Batch selected up and down | Batch-fits the selected spectra in both directions. |
| Batch up | Fits spectra above the current spectrum in explorer order, using each result to initialize the next. |
| Batch down to metadata value… | Batch-fits downward until a chosen numeric explorer metadata column reaches a target value. |
| Batch up to metadata value… | Batch-fits upward until a chosen numeric explorer metadata column reaches a target value. |

Batch operations run in the background and can be cancelled with **Stop**.

### Procedures

| Menu item | What it does |
| --- | --- |
| Procedure builder… | Opens the builder for saving reusable blocks and chaining actions such as frequency limits, fitting, and batch fitting. |

### ML processing…

Opens the machine-learning dialog. It can apply pretrained topology/parameter predictions to selected spectra, preview or apply ML results, and optionally fit the affected spectra. Read [ml-processing.md](ml-processing.md).

### Preferences (`Ctrl+P`)

Opens the settings dialog. Read [preferences.md](preferences.md).

### Export

| Menu item | What it does |
| --- | --- |
| Export fit parameters - all spectra… | Saves fitted EEC parameters and spectrum metadata for all spectra to CSV. |
| Export fit parameters - selected spectra… (`Ctrl+E`) | Saves fitted EEC parameters for the selected spectra to CSV. |
| Export to Python - all spectra… | Creates fit-parameter CSV files plus a ready-to-run pandas script, and opens it in an available editor. |
| Export to Python - selected spectra… (`Ctrl+Shift+E`) | Same as above for the selected spectra. |
| Export DRTs - all spectra… | Saves computed DRT points for all spectra. |
| Export DRTs - selected spectra… | Saves computed DRT points for the selected spectra. |

### Help

| Menu item | What it does |
| --- | --- |
| Getting started | Opens the in-app viewer on the quick-start guide. |
| eisyFIT overview | Opens the overview and concepts guide. |
| Button and menu reference | Opens this reference. |
| Procedures and batch fitting | Opens the automation guide. |
| DRT and outliers | Opens the DRT/outlier guide. |
| Explorers and export | Opens the explorer/export guide. |
| ML processing | Opens the ML guide. |
| Preferences | Opens the settings guide. |
| Open documentation file… (`F1`) | Opens any `.md` documentation file in the in-app viewer. |
| About eisyFIT | Shows the application name and version. |

## Plot controls

| Control | What it does |
| --- | --- |
| Edit points: Off/On (`Alt+E`) | Toggles point editing. In edit mode, click a point in the plot to include or exclude it. |
| Edit points and fit: Off/On (`Alt+Shift+E`) | Toggles point editing with an automatic fit after each point change. |
| Active points | Resets the plot view to the active points. |
| Show all points | Shows excluded points in addition to active points. |
| Hide legends (`Alt+H`) | Hides or shows plot legends. |
| Show Bode / Show Nyquist (`Alt+B`) | Switches the main plot between Bode and Nyquist representation. |
| Show spectrum | Shows or hides the measured spectrum. |
| Show DRT | Shows or hides the DRT panel. |
| Show KK residuals | Shows or hides Kramers–Kronig residuals. |
| Show EEC fit | Shows or hides the fitted equivalent-circuit curve. |
| Show DRT fit | Shows or hides the fitted DRT model curve. |
| Show DRT recovered | Shows or hides the spectrum recovered from the DRT model. |

Right-click a plot for **Save graph**, **Copy to Clipboard**, **Export data**, and **Reset view**.

## Spectra explorer

| Control | What it does |
| --- | --- |
| Explorer rows | One row per spectrum. Click to select; `Ctrl+A` selects all spectra. |
| Column headers | Sort rows by that column. Click again to reverse the sort. Drag headers to reorder columns. |
| Natural sort | Uses natural filename/number ordering instead of plain lexicographic ordering. |
| Restore | Restores the saved explorer column order. |
| Edit metadata… (`Alt+M`) | Creates or edits a metadata column for selected spectra, optionally deriving values from filenames. |
| Delete selected spectra (`Delete`) | Removes the selected spectra from the project. |
| Plot selected spectra | Plots the selected spectra in the main plot. |
| Plot selected DRTs | Plots DRTs for the selected spectra. |
| Plot cell/WE/CE | Plots cell, working-electrode, and counter-electrode spectra together when available. |
| Fit parameters explorer | Opens the EEC fit-parameter explorer across spectra. |
| DRT parameters explorer | Opens the DRT fit-parameter explorer across spectra. |

## Fitting model group

| Control | What it does |
| --- | --- |
| Model box | Select a preset EEC or type an `impedance.py` circuit string. |
| Set model | Applies the model box value to the displayed spectrum. |
| Set model for selected | Applies the model box value to all selected spectra. |
| Sort by tau | Orders selected circuit parameter blocks by their characteristic time constant. |
| Switch blocks | Exchanges the complete parameter groups (values, limits, errors, fixed flags, and fitted values) of two chosen R/CPE blocks, e.g. block 1 with block 3, without changing the circuit. |
| Open EEC analysis window | Opens a detached EEC model/parameter editor for the active spectrum. |
| Auto model: selected (Hybrid DRT) | Selects an EEC model automatically from a Hybrid DRT analysis of the selected spectra. |

## Circuit parameters group

| Control | What it does |
| --- | --- |
| Parameter table | Edit each parameter's initial value, lower bound, upper bound, and fixed state. |
| Apply all to selected | Copies the table's parameter values to all selected spectra. |
| Parameters limits (`Alt+W`) | Opens global EEC parameter bounds. |

## Selection group

| Control | What it does |
| --- | --- |
| Min. freq. / Max. freq. | Set the active frequency window in hertz. |
| Auto | Derives the maximum frequency automatically. |
| Apply lower to selected spectra | Copies the lower frequency limit to selected spectra. |
| Apply upper to selected spectra | Copies the upper frequency limit to selected spectra. |
| Outlier threshold | Threshold used by ridge-analysis outlier detection. |
| Outliers | Runs bayes-drt2 ridge analysis on active points, applies its outlier mask, and initializes EEC parameters. |
| Deterministic threshold | Threshold used by deterministic outlier removal. |
| Remove deterministic outliers | Removes deterministic outliers from active points. |
| Reset points | Includes all points and clears detected outliers. |

## Actions group

| Control | What it does |
| --- | --- |
| Fit spectrum (`Alt+S`) | Fits the displayed spectrum. |
| Fit selected | Fits every selected spectrum with its own model and initial parameters. |
| Initial values | Runs ridge analysis and initializes EEC parameters from its result. |
| Suggest EEC initials… | Suggests initial values from compatible fitted spectra. |
| Detect EEC parameter anomalies… | Detects anomalous fitted parameter vectors in selected spectra. |
| Robust z threshold / Maximum refine iterations | Settings for robust refitting. |
| Refine fit | Refits selected spectra while down-weighting large residuals. |
| Stop | Cancels the running fit, batch, DRT, or ML operation. |

## DRT analysis group

| Control | What it does |
| --- | --- |
| Method | Choose **Ridge DRT** or **Hybrid DRT**. `Alt+C` switches the method. |
| Calculate DRT / Fit selected | Calculates DRT for the selected spectra, or fits the DRT peak model when DRT peaks are shown. |
| Add Gaussian peak | Adds a Gaussian peak to the DRT model. |
| Add Lorentzian peak | Adds a Lorentzian peak to the DRT model. |
| Add Voigt peak | Adds a Voigt peak to the DRT model. |
| Add HN peak | Adds a Havriliak–Negami peak to the DRT model. |
| Fit peaks | Fits the DRT peak parameters to the measured spectrum. |
| Peak parameters table | Edit peak parameters and bounds. |
| Apply all to selected | Applies peak-table values to selected spectra. |
| Apply Fix / Initial / Lower / Upper | Applies only the fixed flag, initial value, lower bound, or upper bound to selected spectra. |
| Send initials | Sends fitted DRT peak information to the EEC circuit as initial values. |
| Open DRT analysis window | Opens a detached DRT peak editor. |
| Remove all peaks | Deletes all DRT peaks. |

## Spectra Simulator group

| Control | What it does |
| --- | --- |
| Circuit | EEC model used to simulate a spectrum. |
| Import Fit Parameters | Copies fitted parameters, model, and frequency limits from the active spectrum into the simulator. Enabled when a spectrum is loaded. |
| Min. freq. / Max. freq. | Simulated frequency range in hertz. |
| Points/decade | Frequency sampling density. |
| Add noise / Noise (%) | Adds reproducible random noise to the simulated spectrum. |
| Seed | Random seed; blank means random. |
| Calculate spectrum | Generates the simulated spectrum and displays it. |
| Add to Spectra Explorer | Adds the simulated spectrum to the project so it can be fitted and exported. |
| Calculate DRT | Calculates Ridge or Hybrid DRT for the simulated spectrum. |

## Parameter explorers

Both the **Fit parameters explorer** and **DRT parameters explorer** provide:

- X/Y axis selectors for any fitted parameter or metadata value,
- point selection in the scatter plot,
- an interactive curve-fit tool that fits a typed equation (for example `a*x + b`) through picked points,
- hover metadata for the plotted points.

## Keyboard shortcuts

| Shortcut | Action |
| --- | --- |
| `Up` / `Down` | Move to the next/previous cycle |
| `Shift+Up` / `Shift+Down` | Move cycle and keep explorer selection |
| `Ctrl+Up` / `Ctrl+Down` | Move cycle focus without changing explorer selection |
| `Alt+A` | Copy fitted values from the previous cycle into current initial values |
| `Alt+D` | Copy fitted values from the next cycle into current initial values |
| `Alt+S` | Fit the selected spectrum |
| `Alt+E` | Toggle point editing |
| `Alt+Shift+E` | Toggle point editing with automatic fitting |
| `Alt+H` | Toggle legends |
| `Alt+B` | Toggle Nyquist/Bode plot mode |
| `Alt+C` | Toggle Ridge/Hybrid DRT method |
| `Alt+Shift+C` | Open the circuit picker |
| `Alt+R` | Calculate DRT for the current spectrum |
| `Alt+W` | Open parameter limits |
| `Alt+M` | Edit a metadata column from the clipboard |
| `Alt+Q` | Activate zoom |
| `Alt+V` | Initialize values from ridge analysis |
| `Alt+F` | Open the batch-fit direction menu |
| `Alt+Y` | Toggle the analysis mode |
| `Ctrl+N` | New project |
| `Ctrl+I` | Import data |
| `Ctrl+L` | Load project |
| `Ctrl+S` | Save project |
| `Ctrl+E` | Export selected fit parameters |
| `Ctrl+Shift+E` | Export selected Python workspace |
| `Ctrl+P` | Open Preferences |
| `Ctrl+A` | Select all spectra |
| `Delete` | Delete selected spectra |
| `F1` | Open a documentation file |
