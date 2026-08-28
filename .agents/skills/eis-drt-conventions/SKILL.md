---
name: eis-drt-conventions
description: Apply this repository's electrochemical impedance and DRT conventions when loading, computing, plotting, exporting, or reviewing EIS behavior. Use for impedance signs, Nyquist/Bode axes, frequency windows, tau/gamma, CPE parameters, DRT/KK inputs, and unit-sensitive calculations; do not use for generic Tk layout or ML validation design.
---

# EIS and DRT conventions

Use the internal scientific representation consistently; convert signs only at data-source or display boundaries.

## Impedance and plots

- Internal impedance is complex `Z = Z' + j Z''`. Capacitive spectra therefore commonly have negative `Z.imag`.
- BioLogic columns named `-Im(Z)` are converted on load with `real - 1j * minus_imaginary`. Do not retain `-Im` as the internal imaginary component.
- Nyquist displays `Z.real` versus `-Z.imag` in ohms.
- Bode displays `abs(Z)` and `-degrees(angle(Z))`; `_phase_degrees()` owns this display convention.
- Saved fit and ML complex values store mathematical real/imaginary components, not Nyquist display coordinates.

## Frequency and point identity

- Frequency is positive Hz. Numerical services sort working copies descending when required; raw `CycleState` point order remains the identity for masks, residuals, and sidecars.
- Frequency windows are inclusive and tolerate reversed endpoints by sorting `(min, max)`.
- Use logarithmic frequency treatment for interpolation, range errors, and ML features where established. Never take a logarithm of non-positive frequency.
- Fitting requires at least three included points. DRT and KK operate on the current effective included set, not silently on every measured point.

## DRT and circuit quantities

- Tau is seconds and frequency is Hz, related by `tau = 1 / (2πf)` for characteristic-frequency reasoning.
- DRT x-values are positive tau; peak centers are stored as `center_log10 = log10(tau)`. Gamma is in ohms.
- Keep Ridge and Hybrid DRT caches distinct and store the included mask used to calculate each result.
- `R*` parameters are ohms, `L*` are henries, and CPE compound parameters use `_0 = Q`, `_1 = alpha`. Alpha is dimensionless and normally bounded within `(0, 1]` according to the existing configured bounds.
- For an R‖CPE process, use the existing `wepy.eis.tau`/`cpe_tau` implementation rather than recreating the Q/alpha relationship.
- Treat fitted EEC curves, DRT-recovered impedance, and Lin-KK curves as different derived products; label and invalidate them separately.

## Review checklist

For sign/unit-sensitive changes, test a small synthetic capacitive spectrum and assert internal complex values separately from displayed/exported coordinates. Check ascending/descending input, mask alignment, positive-frequency filtering, tau ordering, and save/reload of complex arrays. Use the simulator and plot-export tests when relevant:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_spectrum_simulator tests.test_plot_export tests.test_mpr_import -v
```
