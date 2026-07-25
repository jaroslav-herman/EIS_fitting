## EIS_fitting

Responsive Nyquist editor for BioLogic `.mpt` (PEIS) files:

- Displays the spectrum beside an editable circuit-parameter table
- Displays read-only percentage fit errors with shaded quality cells instead of units
- Keeps point selections, parameter values, and fit curves for every visited cycle
- Runs fitting and manual outlier detection without blocking the GUI
- Lets you click points to include or exclude them
- Compares measured and fitted points at identical frequencies with residual connectors
- Provides a sortable spectra explorer for cycle and acquisition metadata

### Run

Double-click `run_gui.ps1`, or run this from `EIS_fitting/`:

`python main.py PEIS_at_N2_flow_80_sccm_automated_01_PEIS.mpt --cycle 1 --threshold 1.0 --circuit "R0-L0-p(R1,CPE1)"`

### GUI controls

- Click a point: toggle include/exclude
- Cycle arrows or selector: move between cycles
- Spectra explorer: select a spectrum or sort rows by source file, cycle, voltage, current, point count, or frequency limits
- Fitting model: select a preset or enter an `impedance.py` circuit string, then set it
- Apply to current cycle: use the frequency range only for the displayed spectrum
- Apply to all cycles: use the frequency range for every spectrum in the file
- Outliers: current: run the bayes-drt2 ridge analysis on active points only, apply its outlier mask without reactivating excluded points, and initialize the EEC fit from its ohmic resistance, inductance, and strongest DRT peaks
- Outliers: all cycles: perform the same active-point analysis and EEC initialization for every spectrum in the background
- Fit spectrum: fit included points with the parameter table values
- Batch fit from current: fit toward higher cycles, using each result to initialize the next
- Export to Python: save fitted parameters and spectrum metadata from all loaded files to CSV, create a ready-to-run pandas script, and open it in VS Code or another available editor
- Reset points: include all points and clear detected outliers

The **File** menu contains additive multi-file data import, project load/save, mask saving, fit-parameter and Python-workspace exports, and exit commands. Import accepts several `.mpt` files at once and keeps previously loaded spectra.

The **Fit** menu fits the selected spectrum or batch-fits upward/downward through the explorer's visible order. Metadata-limited batches use the last clicked numeric explorer column and stop at its nearest available target value.

Outlier detection never runs automatically. Cycle state is retained while the GUI is open.

Keyboard shortcuts:

- `Alt+A`: copy fitted values from the previous cycle into the current initial values
- `Alt+D`: copy fitted values from the next cycle into the current initial values
- `Alt+S`: fit the selected spectrum
- `Ctrl+O`: import another `.mpt` data file
- `Ctrl+Shift+O`: load an EIS fitting project
- `Ctrl+S`: save the EIS fitting project

### Code structure

- `eis_model.py`: GUI-independent project and per-cycle state
- `eis_services.py`: file loading, outlier detection, and fitting operations
- `eis_project.py`: versioned project persistence and fit CSV export
- `eis_gui.py`: Tk interface, plot rendering, and event coordination
- `main.py`: command-line entry point
