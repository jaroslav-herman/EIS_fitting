## EIS_fitting

Minimal interactive Nyquist editor for BioLogic `.mpt` (PEIS) files:

- Automatically flags outliers via `wepy.eis.find_outliers` (excluded by default)
- Lets you click points to include/exclude them
- Fits an equivalent electrical circuit (EEC) using `wepy.eis.fit_spectrum`

### Run

From `EIS_fitting/`:

`python main.py PEIS_at_N2_flow_80_sccm_automated_01_PEIS.mpt --cycle 1 --threshold 1.0 --circuit "R0-L0-p(R1,CPE1)"`

### Controls

- Click a point: toggle include/exclude
- `r`: reset to auto-outlier selection
- `f`: fit the currently included points and overlay the model curve
- `s`: save the current include-mask as `*.npy` in the working directory
