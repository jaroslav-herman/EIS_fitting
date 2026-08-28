## EIS Fitting

EIS Fitting is a desktop application for exploring, cleaning, and fitting
electrochemical impedance spectroscopy (EIS) data exported by BioLogic as
`.mpt` (PEIS) files. It combines an interactive Nyquist plot with circuit
parameter editing, cycle-aware analysis, DRT-assisted outlier detection, and
batch fitting, so spectra can be reviewed and fitted without leaving the GUI.

The application:

- Supports multiple files and multiple measurement cycles
- Displays spectra beside an editable equivalent-circuit parameter table
- Lets you include or exclude individual points while preserving cycle state
- Runs fitting and outlier detection in the background so the GUI stays responsive
- Uses `bayes-drt2` ridge analysis to identify outliers and initialize circuit fits
- Compares measured and fitted points at identical frequencies with residual connectors
- Provides a sortable spectra explorer for cycle and acquisition metadata
- Exports fitted parameters and spectrum metadata to CSV and a ready-to-run Python script

## Installation

### Requirements

- Windows, macOS, or Linux with a desktop environment
- Python 3.14 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/), the environment and dependency manager used by this project

### Install from PyPI

To install the released application into an isolated environment and launch it:

```text
uvx eisyfit
```

To add it to an existing uv project:

```text
uv add eisyfit
eisyfit
```

The PyPI release depends on public distributions of the project’s DRT
backends. Until the maintained forks of `bayes-drt2`, `hybrid-drt`, and the
EIS-specific `wepy` package are published under installable PyPI names, use
the GitHub installation below.

The application requires Python 3.14 or newer and a desktop environment with
Tk support. On some Linux distributions, Tk must be installed separately
(for example, the `python3-tk` system package).

### Run from GitHub

In PowerShell, Terminal, or a shell:

```text
git clone https://github.com/jaroslav-herman/EIS_fitting.git
cd EIS_fitting
uv sync
```

`uv sync` creates the project environment and installs the versions recorded
in `uv.lock`, including the required EIS and DRT libraries. Re-run it after
pulling changes to update the environment.

### Start the application

From the repository directory, run:

```text
uv run python main.py
```

To open a BioLogic file immediately and select a cycle:

```text
uv run python main.py PEIS_at_N2_flow_80_sccm_automated_01_PEIS.mpt --cycle 1
```

On Windows, `run_gui.cmd` can also be double-clicked. It checks that `uv` is
available and starts the application from the repository directory.

If `uv` is not available, install it using the official instructions linked
above, then open a new terminal and repeat `uv sync`.

## License

EIS Fitting is licensed under the GNU General Public License v3.0 or later.
Its dependencies remain under their respective licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the dependency license
summary.

## Run

The command-line entry point also accepts options for the initial control
channel, outlier threshold, and equivalent-circuit model:

```text
uv run python main.py PEIS_at_N2_flow_80_sccm_automated_01_PEIS.mpt --cycle 1 --threshold 1.0 --circuit "R0-L0-p(R1,CPE1)"
```

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
