from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import TextBox
from impedance.models.circuits import CustomCircuit
from impedance.models.circuits.circuits import calculateCircuitLength

from wepy import read_mpt_dataframe
from wepy.eis import find_outliers
from wepy.eis import fit_spectrum as wepy_fit_spectrum
from wepy.eis import show_fit as wepy_show_fit


def _as_1d_array(x) -> np.ndarray:
    arr = np.asarray(x)
    return arr.reshape(-1)


def _sort_by_freq_desc(freq: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(freq)[::-1]
    return freq[idx], Z[idx]

def _load_cycle_spectrum(
    df,
    *,
    cycle: int,
    control: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if "freq_hz" not in df.columns:
        raise KeyError(
            "Missing 'freq_hz' column (is this an EC-Lab PEIS .mpt?). "
            f"Available columns include: {list(df.columns)[:20]}"
        )

    if "cycle_number" in df.columns:
        rows = (df["cycle_number"] == cycle) & (df["freq_hz"] != 0)
    else:
        rows = df["freq_hz"] != 0

    if control == "Ewe":
        re_col = "re_z_ohm"
        mi_col = "minus_im_z_ohm"
        e_col = "ewe_v"
    else:
        re_col = "re_zwe_ce_ohm"
        mi_col = "minus_im_zwe_ce_ohm"
        e_col = "ewe_ece_v"

    missing = [c for c in (re_col, mi_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing expected impedance columns {missing}. "
            f"Try switching --control, or verify the file contains EIS columns."
        )

    f = df.loc[rows, "freq_hz"].to_numpy()
    Z = df.loc[rows, re_col].to_numpy() - 1j * df.loc[rows, mi_col].to_numpy()
    E = float(np.nanmean(df.loc[rows, e_col].to_numpy())) if e_col in df.columns else 0.0
    I = float(np.nanmean(df.loc[rows, "i_ma"].to_numpy())) if "i_ma" in df.columns else 0.0
    return f, Z, E, I


@dataclass
class SpectrumSelection:
    freq_hz: np.ndarray
    Z: np.ndarray
    manual_included: np.ndarray  # bool mask, same length
    auto_outliers: np.ndarray  # bool mask, same length
    freq_window: tuple[float, float] | None = None  # (f_min, f_max) in Hz
    included: np.ndarray | None = None  # derived mask
    E_V: float = 0.0
    I_mA: float = 0.0
    source_path: Path | None = None
    cycle: int | None = None

    def update_included(self) -> None:
        freq = self.freq_hz
        base = self.manual_included
        if self.freq_window is None:
            self.included = base.copy()
            return
        f_min, f_max = self.freq_window
        lo = min(float(f_min), float(f_max))
        hi = max(float(f_min), float(f_max))
        freq_mask = (freq >= lo) & (freq <= hi)
        self.included = base & freq_mask

    def included_freq(self) -> np.ndarray:
        self.update_included()
        assert self.included is not None
        return self.freq_hz[self.included]

    def included_Z(self) -> np.ndarray:
        self.update_included()
        assert self.included is not None
        return self.Z[self.included]


class NyquistEditor:
    def __init__(
        self,
        selection: SpectrumSelection,
        *,
        circuit: str,
        title: str,
        df=None,
        cycles: list[int] | None = None,
        control: str = "Ewe",
        outlier_threshold: float = 1.0,
    ) -> None:
        # Matplotlib default keymap binds 'f' to fullscreen in some backends.
        # Clear it so our 'f' hotkey always triggers fitting.
        mpl.rcParams["keymap.fullscreen"] = []

        self.selection = selection
        self.circuit = circuit
        self.df = df
        self.cycles = cycles or ([] if df is not None else [])
        self.control = control
        self.outlier_threshold = float(outlier_threshold)
        self._base_title = title

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title("EIS Nyquist Editor")
        self.ax.set_title(title)
        self.ax.set_xlabel("Re(Z) / Ω")
        self.ax.set_ylabel("-Im(Z) / Ω")
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True, alpha=0.25)

        self.fig.subplots_adjust(right=0.72)

        (self.scatter_included,) = self.ax.plot(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=5,
            color="#1f77b4",
            label="Included",
            picker=True,
            pickradius=6,
        )
        (self.scatter_excluded,) = self.ax.plot(
            [],
            [],
            linestyle="",
            marker="x",
            markersize=6,
            color="#d62728",
            label="Excluded",
            picker=True,
            pickradius=6,
        )
        (self.fit_line,) = self.ax.plot(
            [],
            [],
            linestyle="-",
            linewidth=2,
            color="black",
            alpha=0.65,
            label="Fit",
        )

        self.status = self.fig.text(0.01, 0.01, "", ha="left", va="bottom")
        self.ax.legend(loc="best")

        self._pressed_keys: set[str] = set()
        self._last_fit_params: np.ndarray | None = None
        self._param_names: list[str] = []
        self._param_init_boxes: dict[str, TextBox] = {}
        self._param_lo_boxes: dict[str, TextBox] = {}
        self._param_hi_boxes: dict[str, TextBox] = {}
        self._fmin_box: TextBox | None = None
        self._fmax_box: TextBox | None = None

        self._init_param_panel()

        self.selection.update_included()
        self._redraw_points()
        self._update_status()

        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)

    def _infer_init_value(self, name: str) -> float:
        n = (name or "").lower()
        if n.startswith("r"):
            return 0.1
        if n.startswith("l"):
            return 1e-8
        if n.startswith("cpe") and n.endswith("_0"):
            return 1e-6
        if n.startswith("cpe") and n.endswith("_1"):
            return 0.9
        if n.startswith("c"):
            return 1e-6
        return 0.1

    def _infer_bounds(self, name: str) -> tuple[float, float]:
        n = (name or "").lower()
        if n.startswith("cpe") and n.endswith("_1"):
            return 0.0, 1.0
        if n.startswith("r"):
            return 0.0, 1e6
        if n.startswith("l"):
            return 0.0, 1.0
        if n.startswith("cpe") and n.endswith("_0"):
            return 1e-12, 1e3
        if n.startswith("c"):
            return 1e-12, 1e3
        return 0.0, 1e6

    def _init_param_panel(self) -> None:
        circuit_len = int(calculateCircuitLength(self.circuit))
        circuit = CustomCircuit(self.circuit, initial_guess=[1.0] * circuit_len)
        names, _units = circuit.get_param_names()
        self._param_names = list(names)

        x0 = 0.74
        w = 0.24
        y = 0.92
        h = 0.035
        dy = 0.05

        self.fig.text(x0, y + 0.04, "Params / bounds", ha="left", va="bottom")
        self.fig.text(x0 + 0.10, y + 0.04, "init", ha="left", va="bottom", fontsize=9)
        self.fig.text(x0 + 0.165, y + 0.04, "lo", ha="left", va="bottom", fontsize=9)
        self.fig.text(x0 + 0.215, y + 0.04, "hi", ha="left", va="bottom", fontsize=9)

        for name in self._param_names:
            if y - h < 0.06:
                break
            lo, hi = self._infer_bounds(name)
            self.fig.text(x0, y + 0.005, name, ha="left", va="bottom", fontsize=9)
            ax_init = self.fig.add_axes([x0 + 0.10, y, 0.055, h])
            ax_lo = self.fig.add_axes([x0 + 0.160, y, 0.045, h])
            ax_hi = self.fig.add_axes([x0 + 0.210, y, 0.045, h])
            init_box = TextBox(ax_init, "", initial=f"{self._infer_init_value(name):g}")
            lo_box = TextBox(ax_lo, "", initial=f"{lo:g}")
            hi_box = TextBox(ax_hi, "", initial=f"{hi:g}")
            self._param_init_boxes[name] = init_box
            self._param_lo_boxes[name] = lo_box
            self._param_hi_boxes[name] = hi_box
            y -= dy

        # Frequency window controls (points outside are always excluded).
        y -= 0.01
        self.fig.text(x0, y + 0.03, "Freq window (Hz)", ha="left", va="bottom")
        self.fig.text(x0, y + 0.005, "min", ha="left", va="bottom", fontsize=9)
        self.fig.text(x0 + 0.12, y + 0.005, "max", ha="left", va="bottom", fontsize=9)
        ax_fmin = self.fig.add_axes([x0 + 0.03, y, 0.09, h])
        ax_fmax = self.fig.add_axes([x0 + 0.15, y, 0.09, h])
        fmin0 = float(np.nanmin(self.selection.freq_hz)) if self.selection.freq_hz.size else 0.0
        fmax0 = float(np.nanmax(self.selection.freq_hz)) if self.selection.freq_hz.size else 0.0
        self._fmin_box = TextBox(ax_fmin, "", initial=f"{fmin0:g}")
        self._fmax_box = TextBox(ax_fmax, "", initial=f"{fmax0:g}")

        def _apply_freq_window(_txt: str) -> None:
            if self._fmin_box is None or self._fmax_box is None:
                return
            try:
                fmin = float((self._fmin_box.text or "").strip())
                fmax = float((self._fmax_box.text or "").strip())
            except Exception:
                return
            self.selection.freq_window = (fmin, fmax)
            self.selection.update_included()
            self.fit_line.set_data([], [])
            self._last_fit_params = None
            self._redraw_points()
            self._update_status()

        self._fmin_box.on_submit(_apply_freq_window)
        self._fmax_box.on_submit(_apply_freq_window)

        self.fig.text(
            x0,
            0.06,
            "Keys: f=fit (uses init+lo+hi); enter=apply freq",
            ha="left",
            va="bottom",
            fontsize=8,
            alpha=0.8,
        )

    def _get_init_params_from_ui(self) -> list[float]:
        init: list[float] = []
        for name in self._param_names:
            box = self._param_init_boxes.get(name)
            if box is None:
                continue
            txt = (box.text or "").strip()
            init.append(float(txt))
        return init

    def _get_bounds_from_ui(self) -> tuple[list[float], list[float]]:
        lo_vals: list[float] = []
        hi_vals: list[float] = []
        for name in self._param_names:
            lo_box = self._param_lo_boxes.get(name)
            hi_box = self._param_hi_boxes.get(name)
            if lo_box is None or hi_box is None:
                continue
            lo_txt = (lo_box.text or "").strip()
            hi_txt = (hi_box.text or "").strip()
            lo_vals.append(float(lo_txt))
            hi_vals.append(float(hi_txt))
        return lo_vals, hi_vals

    def _set_init_params_in_ui(self, params: np.ndarray) -> None:
        vals = _as_1d_array(params).tolist()
        if len(vals) != len(self._param_names):
            return
        for name, val in zip(self._param_names, vals):
            box = self._param_init_boxes.get(name)
            if box is None:
                continue
            box.set_val(f"{float(val):g}")

    def _update_status(self) -> None:
        self.selection.update_included()
        assert self.selection.included is not None
        total = int(self.selection.freq_hz.size)
        included = int(np.count_nonzero(self.selection.included))
        excluded = total - included
        auto = int(np.count_nonzero(self.selection.auto_outliers))
        cycle_txt = (
            f" | cycle: {self.selection.cycle}"
            if self.selection.cycle is not None
            else ""
        )
        freq_txt = ""
        if self.selection.freq_window is not None:
            fmin, fmax = self.selection.freq_window
            freq_txt = f" | f=[{min(fmin,fmax):g},{max(fmin,fmax):g}] Hz"
        self.status.set_text(
            f"points: {total} | included: {included} | excluded: {excluded} "
            f"(auto outliers: {auto}){cycle_txt}{freq_txt} | keys: "
            f"[click=toggle, \u2190/\u2192=cycle, r=reset, f=fit, s=save mask]"
        )

    def _redraw_points(self) -> None:
        self.selection.update_included()
        assert self.selection.included is not None
        Z = self.selection.Z
        x = Z.real
        y = -Z.imag

        inc = self.selection.included
        exc = ~inc

        self.scatter_included.set_data(x[inc], y[inc])
        self.scatter_excluded.set_data(x[exc], y[exc])

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()

    def _nearest_point_index(self, xdata: float, ydata: float) -> int | None:
        if not np.isfinite(xdata) or not np.isfinite(ydata):
            return None
        Z = self.selection.Z
        x = Z.real
        y = -Z.imag
        dx = x - xdata
        dy = y - ydata
        dist2 = dx * dx + dy * dy
        if dist2.size == 0:
            return None
        i = int(np.argmin(dist2))
        return i

    def _toggle_index(self, i: int) -> None:
        if i < 0 or i >= self.selection.manual_included.size:
            return
        # Points outside the freq window are always excluded (not toggleable in).
        if self.selection.freq_window is not None:
            f_min, f_max = self.selection.freq_window
            lo = min(float(f_min), float(f_max))
            hi = max(float(f_min), float(f_max))
            fi = float(self.selection.freq_hz[i])
            if not (lo <= fi <= hi):
                return
        self.selection.manual_included[i] = ~self.selection.manual_included[i]

    def _reset_to_auto_outliers(self) -> None:
        self.selection.manual_included = ~self.selection.auto_outliers.copy()
        self.selection.update_included()
        self.fit_line.set_data([], [])
        self._last_fit_params = None
        self._set_init_params_in_ui(np.array([self._infer_init_value(n) for n in self._param_names]))
        self._refresh_title()

    def _refresh_title(self) -> None:
        cycle = self.selection.cycle
        if cycle is None:
            self.ax.set_title(self._base_title)
            return
        self.ax.set_title(f"{self._base_title}\ncycle={cycle}")

    def _cycle_step(self, direction: int) -> None:
        if not self.cycles or self.df is None:
            return
        if self.selection.cycle is None:
            current = self.cycles[0]
        else:
            current = int(self.selection.cycle)

        try:
            idx = self.cycles.index(current)
        except ValueError:
            idx = 0

        next_idx = idx + int(direction)
        if next_idx < 0:
            next_idx = 0
        if next_idx >= len(self.cycles):
            next_idx = len(self.cycles) - 1

        next_cycle = self.cycles[next_idx]
        if next_cycle == current:
            return

        f, Z, E, I = _load_cycle_spectrum(self.df, cycle=next_cycle, control=self.control)
        f = _as_1d_array(f)
        Z = _as_1d_array(Z)
        f, Z = _sort_by_freq_desc(f, Z)

        outlier_indices = find_outliers(f, Z, threshold=self.outlier_threshold)
        auto_outliers = np.zeros(f.size, dtype=bool)
        if outlier_indices is not None and len(outlier_indices) > 0:
            auto_outliers[np.asarray(outlier_indices, dtype=int)] = True

        self.selection.freq_hz = f
        self.selection.Z = Z
        self.selection.auto_outliers = auto_outliers
        self.selection.manual_included = ~auto_outliers.copy()
        if self._fmin_box is not None and self._fmax_box is not None:
            self._fmin_box.set_val(f"{float(np.nanmin(f)):g}")
            self._fmax_box.set_val(f"{float(np.nanmax(f)):g}")
            self.selection.freq_window = (float(np.nanmin(f)), float(np.nanmax(f)))
        self.selection.update_included()
        self.selection.E_V = float(E)
        self.selection.I_mA = float(I)
        self.selection.cycle = int(next_cycle)
        self.fit_line.set_data([], [])
        self._last_fit_params = None
        self._set_init_params_in_ui(np.array([self._infer_init_value(n) for n in self._param_names]))
        self._refresh_title()

    def _fit(self) -> None:
        f = self.selection.included_freq()
        Z = self.selection.included_Z()
        if f.size < 3:
            self.fit_line.set_data([], [])
            self._last_fit_params = None
            return

        f, Z = _sort_by_freq_desc(_as_1d_array(f), _as_1d_array(Z))

        try:
            init = self._get_init_params_from_ui()
            bounds = self._get_bounds_from_ui()
            params, _errors = wepy_fit_spectrum(
                f,
                Z,
                cir=self.circuit,
                init=init,
                bounds=bounds,
                outliers=False,
                E=float(self.selection.E_V),
                I=float(self.selection.I_mA),
            )
            self._last_fit_params = _as_1d_array(params)
        except Exception as e:
            self.fit_line.set_data([], [])
            self._last_fit_params = None
            self.status.set_text(self.status.get_text() + f" | fit error: {type(e).__name__}: {e}")
            self.fig.canvas.draw_idle()
            return

        # wepy.eis.fit_spectrum prepends [E, I] to the circuit parameters.
        circuit_params = self._last_fit_params[2:]
        f_fit, Z_fit = wepy_show_fit(f, self.circuit, circuit_params, points=200)
        self.fit_line.set_data(Z_fit.real, -Z_fit.imag)
        self._set_init_params_in_ui(circuit_params)

    def _save_mask(self, out_path: Path | None = None) -> Path:
        if out_path is None:
            stem = "mask_included"
            if self.selection.source_path is not None:
                stem = self.selection.source_path.stem
                if self.selection.cycle is not None:
                    stem = f"{stem}_cycle{self.selection.cycle}"
                stem = f"{stem}_mask_included"
            out_path = Path(f"{stem}.npy")
        self.selection.update_included()
        assert self.selection.included is not None
        np.save(out_path, self.selection.included.astype(bool))
        return out_path

    def _on_pick(self, event) -> None:
        mouse = getattr(event, "mouseevent", None)
        if mouse is None:
            return
        if mouse.xdata is None or mouse.ydata is None:
            return

        i = self._nearest_point_index(float(mouse.xdata), float(mouse.ydata))
        if i is None:
            return

        self._toggle_index(i)
        self._redraw_points()
        self._update_status()

    def _on_key_press(self, event) -> None:
        if not getattr(event, "key", None):
            return
        self._pressed_keys.add(event.key)
        key = event.key.lower()

        if key in {"left", "a"}:
            self._cycle_step(-1)
            self._redraw_points()
            self._update_status()
            return

        if key in {"right", "d"}:
            self._cycle_step(+1)
            self._redraw_points()
            self._update_status()
            return

        if key == "r":
            self._reset_to_auto_outliers()
            self._redraw_points()
            self._update_status()
            return

        if key == "f":
            self._fit()
            self._redraw_points()
            self._update_status()
            return

        if key == "s":
            out = self._save_mask()
            self.status.set_text(self.status.get_text() + f" | saved: {out}")
            self.fig.canvas.draw_idle()
            return

    def _on_key_release(self, event) -> None:
        if not getattr(event, "key", None):
            return
        self._pressed_keys.discard(event.key)

    def show(self) -> None:
        plt.show()


def _safe_unique_ints(values: Iterable[object]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for v in values:
        try:
            i = int(v)
        except Exception:
            continue
        if i in seen:
            continue
        out.append(i)
        seen.add(i)
    return sorted(out)


def launch_nyquist_editor(
    *,
    mpt_path: Path,
    cycle: int = 1,
    control: str = "Ewe",
    outlier_threshold: float = 1.0,
    circuit: str = "R0-L0-p(R1,CPE1)",
) -> None:
    df, meta, technique = read_mpt_dataframe(mpt_path)

    if "cycle_number" in df.columns:
        cycles = _safe_unique_ints(df["cycle_number"].values)
    else:
        cycles = []

    f, Z, E, I = _load_cycle_spectrum(df, cycle=cycle, control=control)

    f = _as_1d_array(f)
    Z = _as_1d_array(Z)
    f, Z = _sort_by_freq_desc(f, Z)

    outlier_indices = find_outliers(f, Z, threshold=outlier_threshold)
    auto_outliers = np.zeros(f.size, dtype=bool)
    if outlier_indices is not None and len(outlier_indices) > 0:
        auto_outliers[np.asarray(outlier_indices, dtype=int)] = True

    manual_included = ~auto_outliers.copy()
    freq_window = (float(np.nanmin(f)), float(np.nanmax(f))) if f.size else None
    selection = SpectrumSelection(
        freq_hz=f,
        Z=Z,
        manual_included=manual_included,
        auto_outliers=auto_outliers,
        freq_window=freq_window,
        E_V=E,
        I_mA=I,
        source_path=mpt_path,
        cycle=cycle if "cycle_number" in df.columns else None,
    )
    selection.update_included()

    tech = technique or "Unknown"
    base_title = (
        f"{mpt_path.name} | technique={tech}"
        + (f" | cycles={len(cycles)}" if cycles else "")
        + f"\nAuto-outlier threshold={outlier_threshold:g} | circuit={circuit}"
    )

    editor = NyquistEditor(
        selection,
        circuit=circuit,
        title=base_title,
        df=df if cycles else None,
        cycles=cycles,
        control=control,
        outlier_threshold=outlier_threshold,
    )
    editor.show()
