from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import copy
from io import BytesIO
import json
import joblib
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Callable

import matplotlib
import numpy as np
from natsort import natsort_keygen, ns
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from wepy.eis import tau as cpe_tau

from eis_model import CycleState, ParameterValue, ProjectState
from eis_project import (
    _dataframe_to_payload,
    _state_to_payload,
    _derived_block_values,
    _external_column_name,
    _external_parameter_name,
    _externalize_record,
    _dataframe_to_payload,
    _state_to_payload,
    dataframe_from_payload,
    export_drts_for_states,
    export_fit_parameters,
    export_fit_parameters_for_states,
    export_python_workspace as write_python_workspace,
    load_project_file,
    load_json_payload,
    save_project_file,
)
from eis_services import (
    AutomaticEECModel,
    BatchFitReport,
    DRTComputation,
    FitTimeoutError,
    FitOptions,
    KKResiduals,
    LoadedProject,
    ProjectImportReport,
    RidgeInitialization,
    SPECTRUM_METADATA_COLUMN,
    WORKING_POTENTIAL_COLUMN,
    COUNTER_POTENTIAL_COLUMN,
    SpectrumBatchReport,
    SpectrumFitTarget,
    SpectrumMetadata,
    batch_fit_from_cycle,
    batch_fit_spectra,
    calculate_hybrid_drt,
    calculate_lin_kk_residuals,
    catalog_spectra,
    circuit_parameters,
    analyze_outliers,
    find_outliers_for_all_cycles,
    fit_cycle,
    fit_cycle_with_timeout,
    inspect_eis_file_spectrum_kinds,
    refine_fit_cycle,
    load_cycle,
    load_project_from_dataframe,
    load_project,
    load_projects,
    select_eec_model_from_hybrid_drt,
)
from ml.gui_results import MLResult, load_ml_results, load_ml_results_payload, suggested_eec
from ml.results_schema import spectrum_identifier, write_ml_results
from ml.point_validity import detect_outliers_in_active_points
from ml.runtime_inference import (
    discover_pretrained_artifacts,
    infer_pretrained,
    make_runtime_spectrum,
    save_runtime_results,
)
from ml.number_aware_pipeline import infer_bundle_records, load_pipeline_bundle
from spectrum_simulator import logarithmic_frequencies, simulate_spectrum


def extract_metadata_value_from_filename(
    filename: str, expression: str, capture_group: str | None = None
) -> object:
    """Extract and sensibly type one metadata value from a source filename."""
    try:
        pattern = re.compile(expression)
    except re.error as error:
        raise ValueError(f"invalid regular expression: {error}") from error
    group_names = list(pattern.groupindex)
    if not group_names:
        raise ValueError("the regular expression must contain a named capture group")
    group = (capture_group or "").strip()
    if not group:
        if len(group_names) != 1:
            raise ValueError(
                "specify the capture group when the expression has multiple named groups"
            )
        group = group_names[0]
    if group not in pattern.groupindex:
        raise ValueError(f"named capture group '{group}' was not found")

    names = [Path(filename).stem, Path(filename).name]
    match = next(
        (match for candidate in names if candidate for match in [pattern.search(candidate)] if match),
        None,
    )
    if match is None:
        raise ValueError("filename did not match the expression")
    value = match.group(group).strip()
    if not value:
        raise ValueError("the named capture group produced an empty value")
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    try:
        numeric_value = float(value)
    except ValueError:
        return value
    if not np.isfinite(numeric_value):
        raise ValueError("the named capture group produced a non-finite number")
    return numeric_value


def suggest_metadata_filename_regex(filenames: list[str]) -> str | None:
    """Suggest a regex for the single variable portion shared by filenames."""
    stems = list(dict.fromkeys(Path(filename).stem for filename in filenames if filename))
    if len(stems) < 2:
        return None
    prefix = os.path.commonprefix(stems)
    suffix = os.path.commonprefix([stem[::-1] for stem in stems])[::-1]
    # Prefer complete filename tokens over a coincidental shared suffix such
    # as the ``2`` in ``N2``/``O2``.
    while suffix and suffix[0].isalnum():
        suffix = suffix[1:]
    if len(prefix) + len(suffix) >= min(len(stem) for stem in stems):
        return None
    values = [stem[len(prefix) : len(stem) - len(suffix) or None] for stem in stems]
    if len(set(values)) < 2 or any(not value for value in values):
        return None
    if all(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value) for value in values):
        value_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    else:
        value_pattern = r".+?"
    expression = f"^{re.escape(prefix)}(?P<value>{value_pattern}){re.escape(suffix)}$"
    try:
        if any(
            extract_metadata_value_from_filename(filename, expression, "value")
            is None
            for filename in stems
        ):
            return None
    except ValueError:
        return None
    return expression
from extract_relaxis import export_to_eisfit_json
from explorer_filter import (
    FilterCondition,
    FilterDefinition,
    apply_filters,
    field_is_numeric,
    field_operators,
)
from plot_export import extract_displayed_series, write_displayed_csv
from circuit_structure import (
    circuits_equivalent,
    map_parameter_name,
    parameter_name_mapping,
    parse_circuit,
)


# Configure Matplotlib before any application figure or text artist is created.
# ``wepy`` installs its own publication style as an import side effect.  The
# GUI should use Matplotlib's normal defaults instead of inheriting that style
# (or a user's external TeX configuration).  Keep Matplotlib's built-in
# math-text parser enabled because its default log tick formatter uses
# ``\\mathdefault{...}`` internally; disabling it displays that implementation
# detail literally on the graph.
matplotlib.rcdefaults()
matplotlib.rcParams["text.usetex"] = False

MODEL_PRESETS = (
    "R0-L0-p(R1,CPE1)",
    "R0-p(R1,CPE1)",
    "R0-L0-p(R1,CPE1)-p(R2,CPE2)",
    "R0-p(R1,C1)",
    "R0-p(R1,CPE1)-W1",
)

ML_TRAINED_MODELS = {
    "Sputtered cathode": Path(__file__).resolve().parent / "ml" / "analysis" / "number_aware_pipeline_455" / "pipeline.joblib",
}


def _configure_matplotlib_without_tex() -> None:
    """Restore Matplotlib defaults and use plain text for GUI plots."""
    # Reapply this at application construction time because importing another
    # plotting backend can change the process-wide Matplotlib rcParams.
    matplotlib.rcdefaults()
    matplotlib.rcParams["text.usetex"] = False


_configure_matplotlib_without_tex()


class _ExplorerLineTool:
    """Interactive reference line shared by the parameter explorers."""

    def __init__(self, axes, canvas, parent, refresh_callback) -> None:
        self.axes = axes
        self.canvas = canvas
        self.refresh_callback = refresh_callback
        self.visible = tk.BooleanVar(value=False)
        self.slope_var = tk.StringVar(value="")
        self.intersection_var = tk.StringVar(value="")
        self.slope_function_var = tk.StringVar(value="k")
        self.intersection_function_var = tk.StringVar(value="q")
        self._points: list[tuple[float, float]] | None = None
        self._data_available = False
        self._active_point: int | None = None
        self._slope: float | None = None
        self._intersection: float | None = None

        frame = ttk.LabelFrame(parent, text="Reference line y = kx + q", padding=4)
        frame.grid(row=5, column=0, columnspan=5, padx=3, pady=(6, 0), sticky="ew")
        for column in range(8):
            frame.columnconfigure(column, weight=1 if column in {1, 3, 5, 7} else 0)
        ttk.Checkbutton(
            frame,
            text="Show line",
            variable=self.visible,
            command=self._on_visibility_changed,
        ).grid(row=0, column=0, padx=(0, 8), sticky="w")
        ttk.Label(frame, text="Slope k").grid(row=0, column=1, sticky="e")
        ttk.Entry(
            frame, textvariable=self.slope_var, state="readonly", width=12
        ).grid(row=0, column=2, padx=(3, 8), sticky="ew")
        ttk.Label(frame, text="Intersection q").grid(row=0, column=3, sticky="e")
        ttk.Entry(
            frame, textvariable=self.intersection_var, state="readonly", width=12
        ).grid(row=0, column=4, padx=(3, 8), sticky="ew")
        ttk.Button(
            frame, text="Recalculate", command=self._recalculate
        ).grid(row=0, column=5, padx=(0, 8), sticky="w")
        ttk.Label(frame, text="k function").grid(row=1, column=0, sticky="e")
        ttk.Entry(
            frame, textvariable=self.slope_function_var, width=14
        ).grid(row=1, column=1, columnspan=2, padx=(3, 8), sticky="ew")
        ttk.Label(frame, text="q function").grid(row=1, column=3, sticky="e")
        ttk.Entry(
            frame, textvariable=self.intersection_function_var, width=14
        ).grid(row=1, column=4, columnspan=2, padx=(3, 8), sticky="ew")
        ttk.Label(
            frame, text="Drag the two markers in the graph; functions use k, q, and np"
        ).grid(row=1, column=6, columnspan=2, padx=(0, 3), sticky="w")

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    @staticmethod
    def _format_value(value: float | None) -> str:
        return "" if value is None or not np.isfinite(value) else f"{value:.8g}"

    def _update_line_parameters(self) -> None:
        if self._points is None:
            self._slope = None
            self._intersection = None
        else:
            (x1, y1), (x2, y2) = self._points
            if not np.isfinite([x1, y1, x2, y2]).all() or x1 == x2:
                self._slope = None
                self._intersection = None
            else:
                self._slope = (y2 - y1) / (x2 - x1)
                self._intersection = y1 - self._slope * x1
        self.slope_var.set(self._format_value(self._slope))
        self.intersection_var.set(self._format_value(self._intersection))

    def set_data(self, values: list[tuple[float, float]]) -> None:
        finite_values = [
            (float(x), float(y))
            for x, y in values
            if np.isfinite(x) and np.isfinite(y)
        ]
        self._data_available = len(finite_values) >= 2
        if self._points is None and self._data_available:
            ordered = sorted(finite_values)
            self._points = [ordered[0], ordered[-1]]
            self._update_line_parameters()

    def _on_visibility_changed(self) -> None:
        self.refresh_callback()

    def _recalculate(self) -> None:
        if self._points is None or self._slope is None or self._intersection is None:
            return
        try:
            scope = {
                "__builtins__": {},
                "np": np,
                "k": self._slope,
                "q": self._intersection,
            }
            slope = float(eval(self.slope_function_var.get().strip() or "k", scope))
            intersection = float(
                eval(self.intersection_function_var.get().strip() or "q", scope)
            )
        except (TypeError, ValueError, SyntaxError, NameError, ZeroDivisionError):
            return
        if not np.isfinite([slope, intersection]).all():
            return
        x1, _y1 = self._points[0]
        x2, _y2 = self._points[1]
        self._slope = slope
        self._intersection = intersection
        self._points = [
            (x1, slope * x1 + intersection),
            (x2, slope * x2 + intersection),
        ]
        self._update_line_parameters()
        self.redraw()
        self.canvas.draw_idle()

    def _nearest_point(self, event) -> int | None:
        if not self.visible.get() or self._points is None:
            return None
        if event.x is None or event.y is None:
            return None
        try:
            display_points = self.axes.transData.transform(self._points)
        except (ValueError, OverflowError):
            return None
        distances = [
            float(np.hypot(display_x - event.x, display_y - event.y))
            for display_x, display_y in display_points
        ]
        nearest = int(np.argmin(distances))
        return nearest if distances[nearest] <= 12.0 else None

    def _on_press(self, event) -> None:
        if event.inaxes is self.axes and event.button == 1:
            self._active_point = self._nearest_point(event)

    def _on_motion(self, event) -> None:
        if self._active_point is None or event.inaxes is not self.axes:
            return
        if event.xdata is None or event.ydata is None or self._points is None:
            return
        if (
            self.axes.get_xscale() == "log" and event.xdata <= 0
        ) or (self.axes.get_yscale() == "log" and event.ydata <= 0):
            return
        self._points[self._active_point] = (float(event.xdata), float(event.ydata))
        self._update_line_parameters()
        self.redraw()
        self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if event.button == 1:
            self._active_point = None

    def redraw(self) -> None:
        if not self.visible.get() or not self._data_available or self._points is None:
            return
        (x1, y1), (x2, y2) = self._points
        if self._slope is None or self._intersection is None:
            x_values = np.full(200, x1)
            y_values = np.linspace(y1, y2, 200)
        else:
            x_limits = self.axes.get_xlim()
            x_values = np.linspace(x_limits[0], x_limits[1], 200)
            y_values = self._slope * x_values + self._intersection
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        if self.axes.get_xscale() == "log":
            finite &= x_values > 0
        if self.axes.get_yscale() == "log":
            finite &= y_values > 0
        if np.count_nonzero(finite) >= 2:
            self.axes.plot(
                x_values[finite], y_values[finite], "--", color="black",
                linewidth=1.2, label="Reference line"
            )
        self.axes.plot(
            [x1, x2], [y1, y2], "o", color="black", markersize=7,
            markerfacecolor="white", markeredgewidth=1.4,
            label="Line control points",
        )


class ParameterTable(ttk.Frame):
    def __init__(self, parent: tk.Misc, name_double_click=None, display_name=None) -> None:
        super().__init__(parent)
        self._name_double_click = name_double_click
        self._display_name = display_name or _external_parameter_name
        self._name_labels = {}
        self._highlighted_names: set[str] = set()
        self._rows: list[
            tuple[
                str,
                str,
                float | None,
                tk.BooleanVar,
                tk.StringVar,
                tk.StringVar,
                tk.StringVar,
            ]
        ] = []
        headers = ("Parameter", "Fix", "Initial", "Error (%)", "Lower", "Upper")
        for column, text in enumerate(headers):
            ttk.Label(self, text=text, style="Heading.TLabel").grid(
                row=0, column=column, padx=3, pady=(0, 4), sticky="ew"
            )
        for column in range(6):
            self.columnconfigure(column, weight=1 if column >= 2 else 0)

    def set_parameters(self, parameters: list[ParameterValue]) -> None:
        for child in self.grid_slaves():
            if int(child.grid_info()["row"]) > 0:
                child.destroy()
        self._rows.clear()
        self._name_labels.clear()
        for row, parameter in enumerate(parameters, start=1):
            fixed = tk.BooleanVar(value=parameter.fixed)
            initial = tk.StringVar(value=f"{parameter.initial:g}")
            lower = tk.StringVar(value=f"{parameter.lower:g}")
            upper = tk.StringVar(value=f"{parameter.upper:g}")
            label = tk.Label(
                self,
                text=self._display_name(parameter.name),
                anchor="w",
                background="#fff2a8" if parameter.name in self._highlighted_names else "#f0f0f0",
            )
            label.grid(
                row=row, column=0, padx=3, pady=2
            )
            self._name_labels[parameter.name] = label
            if self._name_double_click is not None:
                label.bind(
                    "<Double-Button-1>",
                    lambda _event, name=parameter.name: self._name_double_click(name),
                )
            ttk.Checkbutton(self, variable=fixed).grid(
                row=row, column=1, padx=3, pady=2
            )

            ttk.Entry(self, textvariable=initial, width=10).grid(
                row=row, column=2, padx=3, pady=2, sticky="ew"
            )
            error_text, error_color = self._format_error(parameter.error_percent)
            tk.Label(
                self,
                text=error_text,
                background=error_color,
                relief=tk.SOLID,
                borderwidth=1,
                width=10,
            ).grid(row=row, column=3, padx=3, pady=2, sticky="ew")
            for column, variable in enumerate((lower, upper), start=4):
                ttk.Entry(self, textvariable=variable, width=10).grid(
                    row=row, column=column, padx=3, pady=2, sticky="ew"
                )
            self._rows.append(
                (
                    parameter.name,
                    parameter.unit,
                    parameter.error_percent,
                    fixed,
                    initial,
                    lower,
                    upper,
                )
            )

    def set_highlighted_names(self, names: set[str]) -> None:
        self._highlighted_names = set(names)
        for name, label in self._name_labels.items():
            label.configure(
                background="#fff2a8" if name in self._highlighted_names else "#f0f0f0"
            )

    @staticmethod
    def _format_error(error_percent: float | None) -> tuple[str, str]:
        if error_percent is None or np.isnan(error_percent):
            return "—", "#eeeeee"
        if np.isinf(error_percent):
            return "∞", "#f8d7da"
        if error_percent <= 5.0:
            color = "#d4edda"
        elif error_percent <= 20.0:
            color = "#fff3cd"
        else:
            color = "#f8d7da"
        return f"{error_percent:.3g}", color

    def values(self) -> list[ParameterValue]:
        parameters = []
        for name, unit, error_percent, fixed, initial, lower, upper in self._rows:
            parameters.append(
                ParameterValue(
                    name,
                    unit,
                    float(initial.get()),
                    float(lower.get()),
                    float(upper.get()),
                    error_percent,
                    bool(fixed.get()),
                )
            )
        return parameters


class MetadataColumnDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, spectrum_count: int) -> None:
        super().__init__(parent)
        self.result: tuple[str, list[str | None]] | None = None
        self.spectrum_count = spectrum_count
        self.title("Add metadata column")
        self.geometry("520x430")
        self.minsize(420, 320)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(3, weight=1)

        ttk.Label(body, text="Column name").grid(row=0, column=0, sticky="w")
        self.name_entry = ttk.Entry(body)
        self.name_entry.grid(row=1, column=0, sticky="ew", pady=(3, 10))
        ttk.Label(
            body,
            text=(
                f"Paste one value per line for the {spectrum_count} selected "
                "spectra in explorer order. Blank lines create empty cells."
            ),
            wraplength=470,
            justify=tk.LEFT,
        ).grid(row=2, column=0, sticky="w", pady=(0, 6))

        text_frame = ttk.Frame(body)
        text_frame.grid(row=3, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self.values_text = tk.Text(text_frame, wrap="none", undo=True)
        scrollbar = ttk.Scrollbar(
            text_frame, orient=tk.VERTICAL, command=self.values_text.yview
        )
        self.values_text.configure(yscrollcommand=scrollbar.set)
        self.values_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        buttons = ttk.Frame(body)
        buttons.grid(row=4, column=0, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(
            side=tk.RIGHT, padx=(6, 0)
        )
        ttk.Button(buttons, text="Add column", command=self._accept).pack(
            side=tk.RIGHT
        )
        ttk.Button(
            buttons,
            text="Repeat pattern",
            command=lambda: self._accept(repeat_pattern=True),
        ).pack(side=tk.RIGHT, padx=(0, 6))

        self.bind("<Control-Return>", lambda _event: self._accept())
        self.grab_set()
        self.name_entry.focus_set()

    def _accept(self, repeat_pattern: bool = False) -> None:
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror(
                "Missing column name", "Enter a name for the new column.", parent=self
            )
            return

        raw = self.values_text.get("1.0", "end-1c").replace("\r\n", "\n")
        lines = raw.split("\n")
        if repeat_pattern:
            while lines and lines[-1] == "":
                lines.pop()
        else:
            while len(lines) > self.spectrum_count and lines[-1] == "":
                lines.pop()
        if repeat_pattern:
            valid_count = 1 <= len(lines) <= self.spectrum_count
        else:
            valid_count = len(lines) == self.spectrum_count
        if not valid_count:
            messagebox.showerror(
                "Wrong number of values",
                (
                    f"Paste between 1 and {self.spectrum_count} values for a "
                    f"repeating pattern; {len(lines)} were found."
                    if repeat_pattern
                    else f"Paste exactly {self.spectrum_count} values; "
                    f"{len(lines)} were found."
                ),
                parent=self,
            )
            return

        values: list[str | None] = []
        for line_number, line in enumerate(lines, start=1):
            cells = line.split("\t")
            if len(cells) > 1 and any(cell.strip() for cell in cells[1:]):
                messagebox.showerror(
                    "Multiple columns pasted",
                    f"Line {line_number} contains more than one value. Paste one column only.",
                    parent=self,
                )
                return
            value = cells[0].strip()
            values.append(value if value else None)

        if repeat_pattern:
            values = [
                values[index % len(values)]
                for index in range(self.spectrum_count)
            ]

        self.result = (name, values)
        self.destroy()


class MetadataEditDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Tk,
        spectrum_count: int,
        columns: list[str],
        initial_column: str | None = None,
        source_filenames: list[str] | None = None,
        suggestion_filenames: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.result: tuple[str, list[object], bool] | None = None
        self.spectrum_count = spectrum_count
        self.source_filenames = source_filenames or []
        self.suggestion_filenames = suggestion_filenames or self.source_filenames
        self._filename_preview_key: tuple[str, str, str] | None = None
        self._filename_preview_values: list[object] | None = None
        self.title("Edit metadata column")
        self.geometry("700x620")
        self.minsize(560, 430)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(5, weight=1)
        ttk.Label(body, text="Metadata column").grid(row=0, column=0, sticky="w")
        default_column = initial_column if initial_column in columns else columns[0]
        self.column_var = tk.StringVar(value=default_column)
        ttk.Combobox(
            body,
            textvariable=self.column_var,
            values=columns,
            state="readonly",
        ).grid(row=1, column=0, sticky="ew", pady=(3, 10))
        new_column_frame = ttk.Frame(body)
        new_column_frame.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        new_column_frame.columnconfigure(2, weight=1)
        self.new_column_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            new_column_frame,
            text="Create new column",
            variable=self.new_column_var,
            command=self._toggle_new_column,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(new_column_frame, text="Name").grid(
            row=0, column=1, padx=(10, 4), sticky="e"
        )
        self.new_column_entry = ttk.Entry(new_column_frame, state="disabled")
        self.new_column_entry.grid(row=0, column=2, sticky="ew")
        self.filename_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            new_column_frame,
            text="Create column from source filename",
            variable=self.filename_mode_var,
            command=self._toggle_filename_mode,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        filename_frame = ttk.LabelFrame(body, text="Source filename rule", padding=8)
        filename_frame.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        filename_frame.columnconfigure(1, weight=1)
        ttk.Label(filename_frame, text="Regular expression").grid(
            row=0, column=0, padx=(0, 8), sticky="w"
        )
        self.filename_expression_entry = ttk.Entry(filename_frame, state="disabled")
        self.filename_expression_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(
            filename_frame,
            text="Suggest from loaded filenames",
            command=self._suggest_filename_expression,
        ).grid(row=0, column=2, padx=(8, 0), sticky="e")
        ttk.Label(filename_frame, text="Value capture group").grid(
            row=1, column=0, padx=(0, 8), pady=(6, 0), sticky="w"
        )
        self.filename_group_box = ttk.Combobox(filename_frame, state="disabled")
        self.filename_group_box.grid(row=1, column=1, pady=(6, 0), sticky="ew")
        ttk.Button(
            filename_frame,
            text="Preview filename values",
            command=self._preview_filename_values,
        ).grid(row=2, column=1, pady=(8, 0), sticky="e")
        self.filename_preview = tk.Text(
            filename_frame, height=7, state="disabled", wrap="none"
        )
        self.filename_preview.grid(row=3, column=0, columnspan=2, pady=(8, 0), sticky="ew")
        ttk.Label(
            body,
            text=(
                f"Paste one value per line for the {spectrum_count} selected "
                "spectra in explorer order."
            ),
            wraplength=470,
            justify=tk.LEFT,
        ).grid(row=4, column=0, sticky="w", pady=(0, 6))
        text_frame = ttk.Frame(body)
        text_frame.grid(row=5, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self.values_text = tk.Text(text_frame, wrap="none", undo=True)
        scrollbar = ttk.Scrollbar(
            text_frame, orient=tk.VERTICAL, command=self.values_text.yview
        )
        self.values_text.configure(yscrollcommand=scrollbar.set)
        self.values_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        buttons = ttk.Frame(body)
        buttons.grid(row=6, column=0, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(
            side=tk.RIGHT, padx=(6, 0)
        )
        ttk.Button(buttons, text="Apply changes", command=self._accept).pack(
            side=tk.RIGHT
        )
        ttk.Button(
            buttons,
            text="Repeat pattern",
            command=lambda: self._accept(repeat_pattern=True),
        ).pack(side=tk.RIGHT, padx=(0, 6))
        self.bind("<Control-Return>", lambda _event: self._accept())
        self.grab_set()

    def _toggle_new_column(self) -> None:
        enabled = self.new_column_var.get()
        self.new_column_entry.configure(state="normal" if enabled else "disabled")
        if enabled:
            self.new_column_entry.focus_set()

    def _toggle_filename_mode(self) -> None:
        enabled = self.filename_mode_var.get()
        self.new_column_var.set(enabled)
        self.new_column_entry.configure(state="normal" if enabled else "disabled")
        self.filename_expression_entry.configure(
            state="normal" if enabled else "disabled"
        )
        self.filename_group_box.configure(state="normal" if enabled else "disabled")
        if enabled:
            self.new_column_entry.focus_set()

    def _set_filename_preview(self, text: str) -> None:
        self.filename_preview.configure(state="normal")
        self.filename_preview.delete("1.0", tk.END)
        self.filename_preview.insert("1.0", text)
        self.filename_preview.configure(state="disabled")

    def _suggest_filename_expression(self) -> None:
        expression = suggest_metadata_filename_regex(self.suggestion_filenames)
        if expression is None:
            messagebox.showinfo(
                "No filename pattern found",
                "At least two loaded filenames with one shared variable portion are needed.",
                parent=self,
            )
            return
        self.filename_expression_entry.delete(0, tk.END)
        self.filename_expression_entry.insert(0, expression)
        self.filename_group_box.set("value")
        self.filename_group_box.configure(values=("value",))
        self._set_filename_preview(
            "Suggested from the loaded filenames. Review the expression, then preview values."
        )

    def _preview_filename_values(self) -> bool:
        column = self.new_column_entry.get().strip()
        expression = self.filename_expression_entry.get().strip()
        group = self.filename_group_box.get().strip()
        if not column or not expression:
            messagebox.showerror(
                "Incomplete filename rule",
                "Enter a new column name and regular expression.",
                parent=self,
            )
            return False
        try:
            compiled = re.compile(expression)
            self.filename_group_box.configure(values=list(compiled.groupindex))
        except re.error as error:
            self._filename_preview_key = None
            self._filename_preview_values = None
            self._set_filename_preview(f"ERROR: {error}")
            messagebox.showerror("Invalid filename rule", str(error), parent=self)
            return False
        grouped: dict[str, tuple[object | None, int, str | None]] = {}
        values: list[object] = []
        for filename in self.source_filenames:
            try:
                value = extract_metadata_value_from_filename(
                    filename, expression, group or None
                )
                values.append(value)
                error_text = None
            except ValueError as error:
                value = None
                error_text = str(error)
            previous = grouped.get(filename)
            grouped[filename] = (
                value if previous is None or previous[2] is not None else previous[0],
                previous[1] + 1 if previous else 1,
                error_text or (previous[2] if previous else None),
            )
        lines = ["Source file\tValue\tSelected spectra"]
        lines.extend(
            f"{filename}\t{('ERROR: ' + error) if error else value}\t{count}"
            for filename, (value, count, error) in grouped.items()
        )
        self._set_filename_preview("\n".join(lines))
        if any(error for _value, _count, error in grouped.values()):
            self._filename_preview_key = None
            self._filename_preview_values = None
            return False
        self._filename_preview_key = (column, expression, group)
        self._filename_preview_values = values
        return True

    def _selected_column_name(self) -> tuple[str, bool] | None:
        if not self.new_column_var.get():
            return self.column_var.get(), False
        name = self.new_column_entry.get().strip()
        if not name:
            messagebox.showerror(
                "Missing column name",
                "Enter a name for the new column.",
                parent=self,
            )
            return None
        return name, True

    def _accept(self, repeat_pattern: bool = False) -> None:
        if self.filename_mode_var.get():
            if not self._preview_filename_values():
                return
            if not messagebox.askyesno(
                "Apply filename metadata",
                "Apply the previewed values to the selected spectra?",
                parent=self,
            ):
                return
            self.result = (
                self.new_column_entry.get().strip(),
                list(self._filename_preview_values or []),
                True,
            )
            self.destroy()
            return
        selected_column = self._selected_column_name()
        if selected_column is None:
            return
        raw = self.values_text.get("1.0", "end-1c").replace("\r\n", "\n")
        lines = raw.split("\n")
        if repeat_pattern:
            while lines and lines[-1] == "":
                lines.pop()
        else:
            while len(lines) > self.spectrum_count and lines[-1] == "":
                lines.pop()
        if repeat_pattern:
            valid_count = 1 <= len(lines) <= self.spectrum_count
        else:
            valid_count = len(lines) == self.spectrum_count
        if not valid_count:
            messagebox.showerror(
                "Wrong number of values",
                (
                    f"Paste between 1 and {self.spectrum_count} values for a "
                    f"repeating pattern; {len(lines)} were found."
                    if repeat_pattern
                    else f"Paste exactly {self.spectrum_count} values; "
                    f"{len(lines)} were found."
                ),
                parent=self,
            )
            return
        values: list[str | None] = []
        for line_number, line in enumerate(lines, start=1):
            cells = line.split("\t")
            if len(cells) > 1 and any(cell.strip() for cell in cells[1:]):
                messagebox.showerror(
                    "Multiple columns pasted",
                    f"Line {line_number} contains more than one value.",
                    parent=self,
                )
                return
            value = cells[0].strip()
            values.append(value if value else None)
        if repeat_pattern:
            values = [
                values[index % len(values)]
                for index in range(self.spectrum_count)
            ]
        self.result = (selected_column[0], values, selected_column[1])
        self.destroy()


class ElectrodeSelectionDialog(tk.Toplevel):
    _labels = {
        "working": "WE–RE",
        "cell": "WE–CE",
        "counter": "CE–RE",
    }

    def __init__(self, parent: tk.Tk, path: Path, available: list[str]) -> None:
        super().__init__(parent)
        self.result: tuple[list[str], bool] | None = None
        self.title(f"Select spectra — {path.name}")
        self.transient(parent)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            body,
            text="This file contains multiple electrode-pair spectra.\nSelect the spectra to import:",
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 8))
        self._variables = {
            kind: tk.BooleanVar(value=True) for kind in available
        }
        for kind in ("working", "cell", "counter"):
            if kind in self._variables:
                ttk.Checkbutton(
                    body,
                    text=self._labels[kind],
                    variable=self._variables[kind],
                ).pack(anchor="w")
        self.apply_to_all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            body,
            text="Apply to all subsequent applicable files in this import",
            variable=self.apply_to_all_var,
        ).pack(anchor="w", pady=(8, 0))
        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Import selected", command=self._accept).pack(
            side=tk.RIGHT, padx=(0, 6)
        )
        self.bind("<Return>", lambda _event: self._accept())
        self.grab_set()

    def _accept(self) -> None:
        selected = [kind for kind, variable in self._variables.items() if variable.get()]
        if not selected:
            messagebox.showerror(
                "No spectra selected",
                "Select at least one electrode-pair spectrum.",
                parent=self,
            )
            return
        self.result = (selected, self.apply_to_all_var.get())
        self.destroy()


def _compatible_spectrum_selection(
    selection: list[str], available: list[str]
) -> list[str] | None:
    compatible = [kind for kind in selection if kind in available]
    return compatible or None


class EISApplication:
    def __init__(
        self,
        root: tk.Tk,
        path: Path | None,
        cycle: int,
        control: str,
        threshold: float,
        circuit: str,
    ) -> None:
        self.root = root
        self.path = path.resolve() if path is not None else None
        self.project_path: Path | None = None
        self._saved_project_signature: str | None = None
        self.current_dataset_id: str | None = None
        self.requested_cycle = cycle
        self.control = control
        self.circuit = circuit
        self._preferences_path = self._preferences_file_path()
        self._procedure_library_blocks: dict[str, list[dict[str, str]]] = {}
        self._procedure_library: dict[str, list[dict[str, object]]] = {}
        self._model_presets = self._load_preferences()
        self.analysis_mode_var = tk.StringVar(value="EEC")
        self.loaded: LoadedProject | None = None
        self.state: ProjectState | None = None
        self.loaded_projects: dict[str, LoadedProject] = {}
        self._dataset_order: list[str] = []
        self._custom_metadata_columns: list[str] = []
        self._last_metadata_edit_column: dict[str, str] = {}
        self._explorer_rows: dict[str, tuple[str, LoadedProject, SpectrumMetadata]] = (
            {}
        )
        self._explorer_current_column_order: list[str] | None = None
        self._fit_explorer_filter = FilterDefinition()
        self._drt_explorer_filter = FilterDefinition()
        self._explorer_lookup: dict[tuple[str, int], str] = {}
        self._explorer_anchor_item: str | None = None
        self._explorer_primary_item: str | None = None
        self._suspend_explorer_select = False
        self._explorer_shift_double_click = False
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="eis-worker"
        )
        self.busy = False
        self._fit_cancel_requested = False
        self._fit_parameter_snapshot = None
        self._stop_event = threading.Event()
        self._operation_labels: list[str] = []
        self._operation_name = "operation"
        self.drt_peak_parameters: list[dict[str, float]] = []
        self._drt_peak_cycle_key = None
        self._drt_peak_artists = []
        self._drt_peak_sum_artist = None
        self._drt_peak_drag = None
        self._drt_peak_drag_moved = False
        self._selected_drt_peak_index = None
        self._drt_aux_parameter_limits = {}
        self._plot_imports = None
        self.plot_mode = "nyquist"
        self.procedure_blocks: dict[str, list[dict[str, str]]] = {}
        self.procedures: dict[str, list[dict[str, object]]] = {}
        self.simulator_spectrum = None
        self.simulator_parameters: list[ParameterValue] = []
        self.simulator_drt_result = None
        self.simulator_drt_mode_var = tk.StringVar(value="Ridge DRT")

        initial_threshold = (
            self._bayes_drt2_threshold_preference
            if np.isclose(threshold, 1.0)
            else f"{threshold:g}"
        )
        self.threshold_var = tk.StringVar(value=initial_threshold)
        self.deterministic_threshold_var = tk.StringVar(
            value=self._deterministic_threshold_preference
        )
        self.refine_z_threshold_var = tk.StringVar(
            value=self._refine_z_threshold_preference
        )
        self.refine_max_iterations_var = tk.StringVar(
            value=self._refine_max_iterations_preference
        )
        self.fit_pipeline_var = tk.StringVar(value=self._fit_pipeline_preference)
        self.fit_seed_var = tk.StringVar(value=self._fit_seed_preference)
        self.fit_population_var = tk.StringVar(value=self._fit_population_preference)
        self.fit_iterations_var = tk.StringVar(value=self._fit_iterations_preference)
        self.fit_weight_modulus_var = tk.BooleanVar(value=self._fit_weight_modulus_preference)
        self.fit_jacobian_mode_var = tk.StringVar(value=self._fit_jacobian_mode_preference)
        self._last_fit_result = None
        self.model_var = tk.StringVar(value=circuit)
        self.show_drt_var = tk.BooleanVar(value=False)
        self.show_kk_var = tk.BooleanVar(value=False)
        self.show_spectrum_var = tk.BooleanVar(value=True)
        self.show_all_points_var = tk.BooleanVar(value=False)
        self.show_eec_fit_var = tk.BooleanVar(value=True)
        self.show_drt_fit_var = tk.BooleanVar(value=False)
        self.show_drt_recovered_var = tk.BooleanVar(value=False)
        self.hide_legends_var = tk.BooleanVar(value=False)
        self.show_ml_frequency_ranges_var = tk.BooleanVar(value=False)
        self.show_ml_active_points_var = tk.BooleanVar(value=False)
        self.show_ml_model_var = tk.BooleanVar(value=False)
        self.show_ml_residuals_var = tk.BooleanVar(value=False)
        self.ml_results: dict[str, MLResult] = {}
        self.ml_results_directory: Path | None = None
        self.ml_results_status_var = tk.StringVar(value="No ML results loaded")
        self.minimum_frequency_var = tk.StringVar()
        self.maximum_frequency_var = tk.StringVar()
        self.auto_max_frequency_var = tk.BooleanVar(value=False)
        self._frequency_control_guard = False
        self._frequency_apply_after_id = None
        self.cycle_var = tk.StringVar(value=str(cycle))
        self.status_var = tk.StringVar(value="Opening application…")

        self._configure_window()
        self._build_menu()
        _configure_matplotlib_without_tex()
        self._build_interface()
        self._base_refresh_plot = self._refresh_plot
        self._refresh_plot = self._refresh_plot_with_drt_recovery
        self._analysis_windows_cycle_key = None
        self._analysis_windows_last_busy = False
        self.root.after(300, self._poll_analysis_windows)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Up>", lambda _event: self.change_cycle(-1))
        self.root.bind("<Down>", lambda _event: self.change_cycle(1))
        self.root.bind("<Shift-Up>", lambda _event: self.change_cycle(-1, True))
        self.root.bind("<Shift-Down>", lambda _event: self.change_cycle(1, True))
        self.root.bind("<Control-Up>", lambda _event: self.change_cycle(-1, focus_only=True))
        self.root.bind("<Control-Down>", lambda _event: self.change_cycle(1, focus_only=True))
        self.root.bind("<Control-a>", self.select_all_spectra)
        self.root.bind("<Delete>", self._on_delete_key)
        self.root.bind("<Control-e>", lambda _event: self.export_selected_fits())
        self.root.bind("<Control-E>", lambda _event: self.export_selected_fits())
        self.root.bind(
            "<Control-Shift-e>",
            lambda _event: self.export_selected_python_workspace(),
        )
        self.root.bind(
            "<Control-Shift-E>",
            lambda _event: self.export_selected_python_workspace(),
        )
        self.root.bind("<Control-s>", self._on_control_s)
        self.root.bind("<Control-S>", self._on_control_s)
        self.root.bind("<Control-l>", lambda _event: self.load_project())
        self.root.bind("<Control-L>", lambda _event: self.load_project())
        self.root.bind("<Control-i>", lambda _event: self.import_data())
        self.root.bind("<Control-I>", lambda _event: self.import_data())
        self.root.bind("<Alt-a>", self._on_alt_a)
        self.root.bind("<Alt-A>", self._on_alt_a)
        self.root.bind("<Alt-d>", self._on_alt_d)
        self.root.bind("<Alt-D>", self._on_alt_d)
        self.root.bind("<Alt-e>", self.toggle_point_edit_mode)
        self.root.bind("<Alt-m>", self.edit_metadata_column_from_clipboard)
        self.root.bind("<Alt-M>", self.edit_metadata_column_from_clipboard)
        self.root.bind("<Alt-q>", self._active_zoom_key)
        self.root.bind("<Alt-Q>", self._active_zoom_key)
        self.root.bind("<Alt-v>", self._initial_values_key)
        self.root.bind("<Alt-V>", self._initial_values_key)
        self.root.bind("<Alt-Shift-e>", self._toggle_auto_fit_points_key)
        self.root.bind("<Alt-Shift-E>", self._toggle_auto_fit_points_key)
        self.root.bind("<Alt-h>", self._toggle_legends_key)
        self.root.bind("<Alt-H>", self._toggle_legends_key)
        self.root.bind("<Alt-f>", self._open_batch_fit_key_menu)
        self.root.bind("<Alt-F>", self._open_batch_fit_key_menu)
        self.root.bind("<Alt-b>", self._toggle_plot_mode_key)
        self.root.bind("<Alt-B>", self._toggle_plot_mode_key)
        self.root.bind_all("<Alt-KeyPress>", self._handle_alt_keypad)
        self.root.bind("<Alt-y>", self._toggle_analysis_mode_key)
        self.root.bind("<Alt-Y>", self._toggle_analysis_mode_key)
        self.root.bind("<Alt-c>", self._toggle_drt_method_key)
        self.root.bind("<Alt-C>", self._toggle_drt_method_key)
        self.root.bind("<Alt-Shift-c>", self._open_circuit_picker_key)
        self.root.bind("<Alt-Shift-C>", self._open_circuit_picker_key)
        self.root.bind("<Alt-r>", self._calculate_current_drt_key)
        self.root.bind("<Alt-R>", self._calculate_current_drt_key)
        self.root.bind("<Alt-s>", self._on_analysis_alt_s)
        self.root.bind("<Alt-S>", self._on_analysis_alt_s)
        if self.path is not None:
            self.root.after(30, self._begin_loading)
        else:
            self.status_var.set("Ready — import data or load a project.")

    def _current_directory(self) -> Path:
        if self.path is not None:
            return self.path.parent
        return Path.cwd()

    def _current_stem(self) -> str:
        if self.path is not None:
            return self.path.stem
        return "eis_project"

    def _current_name(self) -> str:
        if self.path is not None:
            return self.path.name
        return "No file loaded"

    def _configure_window(self) -> None:
        self._update_window_title()
        self.root.geometry("1220x760")
        self.root.minsize(940, 620)
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Heading.TLabel", font=("Segoe UI", 9, "bold"))

    def _build_menu(self) -> None:
        menu_bar = tk.Menu(self.root)
        self.file_menu = tk.Menu(menu_bar, tearoff=False)
        self.file_menu.add_command(
            label="Import data…",
            accelerator="Ctrl+I",
            command=self.import_data,
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(
            label="Load project…",
            accelerator="Ctrl+L",
            command=self.load_project,
        )
        self.file_menu.add_command(
            label="Load RelaxIS 3 project",
            command=self.load_relaxis_project,
        )
        self.file_menu.add_command(
            label="Save project…",
            accelerator="Ctrl+S",
            command=self.save_project,
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Save current mask…", command=self.save_mask)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.close)
        menu_bar.add_cascade(label="File", menu=self.file_menu)
        self._project_menu_actions = (
            "Load RelaxIS 3 project",
            "Load project…",
            "Save project…",
            "Save current mask…",
        )
        self.fit_menu = tk.Menu(menu_bar, tearoff=False)
        self.fit_menu.add_command(
            label="Fit selected spectrum",
            accelerator="Alt+S",
            command=self.fit,
        )
        self.fit_menu.add_separator()
        self.fit_menu.add_command(
            label="Batch down",
            command=lambda: self.batch_fit_explorer(1),
        )
        self.fit_menu.add_command(
            label="Batch fit selected down",
            command=self.batch_fit_selected_down,
        )
        self.fit_menu.add_command(
            label="Batch fit selected up",
            command=lambda: self.batch_fit_selected_down(-1),
        )
        self.fit_menu.add_command(
            label="Batch selected up and down",
            command=self.batch_fit_selected_up_down,
        )
        self.fit_menu.add_command(
            label="Batch up",
            command=lambda: self.batch_fit_explorer(-1),
        )
        self.fit_menu.add_separator()
        self.fit_menu.add_command(
            label="Batch down to metadata value…",
            command=lambda: self.batch_fit_explorer(1, to_metadata_value=True),
        )
        self.fit_menu.add_command(
            label="Batch up to metadata value…",
            command=lambda: self.batch_fit_explorer(-1, to_metadata_value=True),
        )
        menu_bar.add_cascade(label="Fit", menu=self.fit_menu)
        self.procedure_menu = tk.Menu(menu_bar, tearoff=False)
        self.procedure_menu.add_command(
            label="Procedure builder…",
            command=self.open_procedure_builder,
        )
        menu_bar.add_cascade(label="Procedures", menu=self.procedure_menu)
        self.preferences_menu = tk.Menu(menu_bar, tearoff=False)
        self.preferences_menu.add_command(
            label="Preferences…",
            command=self.open_preferences,
        )
        menu_bar.add_cascade(label="Preferences", menu=self.preferences_menu)
        self.export_menu = tk.Menu(menu_bar, tearoff=False)
        self.export_menu.add_command(
            label="Export fit parameters - all spectra…",
            command=self.export_fits,
        )
        self.export_menu.add_command(
            label="Export fit parameters - selected spectra…",
            accelerator="Ctrl+E",
            command=self.export_selected_fits,
        )
        self.export_menu.add_separator()
        self.export_menu.add_command(
            label="Export to Python - all spectra…",
            command=self.export_python_workspace,
        )
        self.export_menu.add_command(
            label="Export to Python - selected spectra…",
            accelerator="Ctrl+Shift+E",
            command=self.export_selected_python_workspace,
        )
        self.export_menu.add_separator()
        self.export_menu.add_command(
            label="Export DRTs - all spectra…",
            command=self.export_drts,
        )
        self.export_menu.add_command(
            label="Export DRTs - selected spectra…",
            command=self.export_selected_drts,
        )
        menu_bar.add_cascade(label="Export", menu=self.export_menu)
        self.root.configure(menu=menu_bar)
        self._fit_menu_actions = (
            "Fit selected spectrum",
            "Batch down",
            "Batch fit selected down",
            "Batch fit selected up",
            "Batch selected up and down",
            "Batch up",
            "Batch down to metadata value…",
            "Batch up to metadata value…",
        )
        self._export_menu_actions = (
            "Export fit parameters - all spectra…",
            "Export fit parameters - selected spectra…",
            "Export to Python - all spectra…",
            "Export to Python - selected spectra…",
            "Export DRTs - all spectra…",
            "Export DRTs - selected spectra…",
        )

    @staticmethod
    def _preferences_file_path() -> Path:
        app_data = os.environ.get("APPDATA")
        base = Path(app_data) if app_data else Path.home() / ".config"
        return base / "EIS_fitting" / "preferences.json"

    @staticmethod
    def _validate_procedure_data(
        blocks_payload, procedures_payload
    ) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, object]]]]:
        blocks: dict[str, list[dict[str, str]]] = {}
        if isinstance(blocks_payload, dict):
            for name, raw_steps in blocks_payload.items():
                if not isinstance(name, str) or not isinstance(raw_steps, list):
                    continue
                steps = [
                    {"action": str(step["action"]), "parameter": str(step.get("parameter", ""))}
                    for step in raw_steps
                    if isinstance(step, dict) and isinstance(step.get("action"), str)
                ]
                if len(steps) == len(raw_steps):
                    blocks[name] = steps
        procedures: dict[str, list[dict[str, object]]] = {}
        if isinstance(procedures_payload, dict):
            for name, raw_entries in procedures_payload.items():
                if not isinstance(name, str) or not isinstance(raw_entries, list):
                    continue
                entries = []
                valid = True
                for entry in raw_entries:
                    if not isinstance(entry, dict) or not isinstance(entry.get("block"), str):
                        valid = False
                        break
                    raw_steps = entry.get("steps", [])
                    if not isinstance(raw_steps, list):
                        valid = False
                        break
                    steps = [
                        {"action": str(step["action"]), "parameter": str(step.get("parameter", ""))}
                        for step in raw_steps
                        if isinstance(step, dict) and isinstance(step.get("action"), str)
                    ]
                    if len(steps) != len(raw_steps):
                        valid = False
                        break
                    entries.append({"block": entry["block"], "steps": steps})
                if valid:
                    procedures[name] = entries
        return blocks, procedures

    def _load_preferences(self) -> tuple[str, ...]:
        self._bayes_drt2_threshold_preference = "1.0"
        self._deterministic_threshold_preference = "4"
        self._refine_z_threshold_preference = "3.5"
        self._refine_max_iterations_preference = "5"
        self._fit_timeout_seconds = 10.0
        self._fit_pipeline_preference = "local only"
        self._fit_seed_preference = ""
        self._fit_population_preference = "30"
        self._fit_iterations_preference = "200"
        self._fit_weight_modulus_preference = False
        self._fit_jacobian_mode_preference = "Numerical only"
        self._last_import_directory = Path.cwd()
        self._last_project_directory = Path.cwd()
        self._fit_explorer_x_preference = "I_mA"
        self._drt_explorer_x_preference = "I_mA"
        self._fit_explorer_y_preference = "R0"
        self._drt_explorer_y_preference = "R0"
        self._explorer_column_order_preference: list[str] = []
        self._explorer_hidden_columns_preference: list[str] = []
        self._explorer_new_columns_position = "end"
        self._eec_parameter_bounds = {
            "r": (0.0, 1e6),
            "l": (0.0, 1.0),
            "cpe_q": (1e-6, 1e3),
            "cpe_alpha": (0.5, 1.0),
        }
        self._auto_model_settings = {
            "criterion": "lml-bic",
            "max_num_peaks": 10,
            "peak_prominence": None,
            "peak_height": None,
            "prior": True,
            "prior_strength": None,
            "min_r0": None,
            "min_l0": None,
        }
        try:
            payload = json.loads(self._preferences_path.read_text(encoding="utf-8"))
            saved_timeout = float(payload.get("fit_timeout_seconds", 10.0))
            if np.isfinite(saved_timeout) and saved_timeout > 0:
                self._fit_timeout_seconds = saved_timeout
            optimizer = payload.get("eec_optimizer", {})
            if isinstance(optimizer, dict):
                pipeline = str(optimizer.get("pipeline", "local only"))
                if pipeline in {"local only", "PSO → local", "GA → local", "PSO only", "GA only"}:
                    self._fit_pipeline_preference = pipeline
                self._fit_seed_preference = str(optimizer.get("seed", ""))
                self._fit_population_preference = str(optimizer.get("population", "30"))
                self._fit_iterations_preference = str(optimizer.get("iterations", "200"))
                self._fit_weight_modulus_preference = bool(optimizer.get("weight_by_modulus", False))
                jacobian_mode = str(optimizer.get("jacobian_mode", "Numerical only"))
                if jacobian_mode in {"Automatic", "Analytical when supported", "Numerical only"}:
                    self._fit_jacobian_mode_preference = jacobian_mode
            thresholds = payload.get("outlier_thresholds", {})
            if isinstance(thresholds, dict):
                for key, attribute_name, default in (
                    ("bayes_drt2", "_bayes_drt2_threshold_preference", "1.0"),
                    ("deterministic", "_deterministic_threshold_preference", "4"),
                    ("refine_z", "_refine_z_threshold_preference", "3.5"),
                ):
                    value = float(thresholds.get(key, default))
                    if np.isfinite(value) and value > 0:
                        setattr(self, attribute_name, f"{value:g}")
                value = int(thresholds.get("refine_max_iterations", 5))
                if value >= 1:
                    self._refine_max_iterations_preference = str(value)
            for preference_name, attribute_name in (
                ("last_import_directory", "_last_import_directory"),
                ("last_project_directory", "_last_project_directory"),
            ):
                saved_directory = payload.get(preference_name)
                if saved_directory:
                    candidate = Path(str(saved_directory)).expanduser()
                    if candidate.is_dir():
                        setattr(self, attribute_name, candidate.resolve())
            self._fit_explorer_x_preference = str(
                payload.get("fit_explorer_x", "I_mA")
            ).strip() or "I_mA"
            self._drt_explorer_x_preference = str(
                payload.get("drt_explorer_x", "I_mA")
            ).strip() or "I_mA"
            self._fit_explorer_y_preference = str(
                payload.get("fit_explorer_y", "R0")
            ).strip() or "R0"
            self._drt_explorer_y_preference = str(
                payload.get("drt_explorer_y", "R0")
            ).strip() or "R0"
            saved_column_order = payload.get("explorer_column_order", [])
            if isinstance(saved_column_order, list):
                self._explorer_column_order_preference = [
                    str(value).strip() for value in saved_column_order if str(value).strip()
                ]
            saved_hidden_columns = payload.get("explorer_hidden_columns", [])
            if isinstance(saved_hidden_columns, list):
                self._explorer_hidden_columns_preference = [
                    str(value).strip()
                    for value in saved_hidden_columns
                    if str(value).strip()
                ]
            position = str(payload.get("explorer_new_columns_position", "end")).casefold()
            if position in {"beginning", "end"}:
                self._explorer_new_columns_position = position
            saved_bounds = payload.get("eec_parameter_bounds", {})
            if isinstance(saved_bounds, dict):
                for category, default in self._eec_parameter_bounds.items():
                    value = saved_bounds.get(category)
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        self._eec_parameter_bounds[category] = (
                            float(value[0]),
                            float(value[1]),
                        )
            saved_auto = payload.get("auto_model", {})
            if isinstance(saved_auto, dict):
                for key in self._auto_model_settings:
                    if key in saved_auto:
                        self._auto_model_settings[key] = saved_auto[key]
                self._auto_model_settings["max_num_peaks"] = max(
                    int(self._auto_model_settings["max_num_peaks"]), 1
                )
            (
                self._procedure_library_blocks,
                self._procedure_library,
            ) = self._validate_procedure_data(
                payload.get("procedure_blocks"), payload.get("procedures")
            )
            circuits = payload.get("eec_circuits", [])
            if isinstance(circuits, list):
                values = tuple(
                    dict.fromkeys(
                        str(circuit).strip()
                        for circuit in circuits
                        if str(circuit).strip()
                    )
                )
                if values:
                    return values
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return MODEL_PRESETS

    def _remember_dialog_directory(self, preference_name: str, selected: str) -> None:
        directory = Path(selected).resolve().parent
        setattr(self, f"_{preference_name}", directory)
        try:
            payload = json.loads(self._preferences_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            payload = {}
        payload[preference_name] = str(directory)
        try:
            self._preferences_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._preferences_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(self._preferences_path)
        except OSError:
            # The current session still uses the selected folder even if the
            # optional cross-session preference cannot be written.
            pass

    def _dialog_directory(self, preference_name: str) -> Path:
        directory = getattr(self, f"_{preference_name}", None)
        if isinstance(directory, Path) and directory.is_dir():
            return directory
        return Path.cwd()

    def _save_preferences(
        self,
        circuits: tuple[str, ...],
        fit_explorer_x: str,
        drt_explorer_x: str,
        fit_explorer_y: str,
        drt_explorer_y: str,
        explorer_column_order: list[str],
        explorer_hidden_columns: list[str],
        explorer_new_columns_position: str,
        procedure_blocks: dict[str, list[dict[str, str]]] | None = None,
        procedures: dict[str, list[dict[str, object]]] | None = None,
    ) -> None:
        self._preferences_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._preferences_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "eec_circuits": list(circuits),
                    "fit_explorer_x": fit_explorer_x,
                    "drt_explorer_x": drt_explorer_x,
                    "fit_explorer_y": fit_explorer_y,
                    "drt_explorer_y": drt_explorer_y,
                    "explorer_column_order": explorer_column_order,
                    "explorer_hidden_columns": explorer_hidden_columns,
                    "explorer_new_columns_position": explorer_new_columns_position,
                    "eec_parameter_bounds": self._eec_parameter_bounds,
                    "outlier_thresholds": {
                        "bayes_drt2": self.threshold_var.get(),
                        "deterministic": self.deterministic_threshold_var.get(),
                        "refine_z": self.refine_z_threshold_var.get(),
                        "refine_max_iterations": self.refine_max_iterations_var.get(),
                    },
                    "auto_model": self._auto_model_settings,
                    "last_import_directory": str(self._last_import_directory),
                    "last_project_directory": str(self._last_project_directory),
                    "fit_timeout_seconds": self._fit_timeout_seconds,
                    "procedure_blocks": procedure_blocks if procedure_blocks is not None else self._procedure_library_blocks,
                    "procedures": procedures if procedures is not None else self._procedure_library,
                    "eec_optimizer": {
                        "pipeline": self.fit_pipeline_var.get(),
                        "seed": self.fit_seed_var.get(),
                        "population": self.fit_population_var.get(),
                        "iterations": self.fit_iterations_var.get(),
                        "weight_by_modulus": bool(self.fit_weight_modulus_var.get()),
                        "jacobian_mode": self.fit_jacobian_mode_var.get(),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(self._preferences_path)

    def save_procedures_to_preferences(self) -> None:
        """Store the current block/procedure library alongside other preferences."""
        try:
            self._save_preferences(
                self._model_presets,
                self._fit_explorer_x_preference,
                self._drt_explorer_x_preference,
                self._fit_explorer_y_preference,
                self._drt_explorer_y_preference,
                self._explorer_column_order_preference,
                self._explorer_hidden_columns_preference,
                self._explorer_new_columns_position,
                self.procedure_blocks,
                self.procedures,
            )
            self._procedure_library_blocks = copy.deepcopy(self.procedure_blocks)
            self._procedure_library = copy.deepcopy(self.procedures)
            self._update_status("blocks and procedures saved to Preferences")
        except (OSError, TypeError, ValueError) as error:
            messagebox.showerror("Preferences save failed", str(error), parent=self.root)

    def _autosave_procedures_to_project(self) -> None:
        if self.project_path is not None and self.state is not None and not self.busy:
            self.save_project(self.project_path)

    def open_preferences(self) -> None:
        existing = getattr(self, "preferences_popup", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return

        popup = tk.Toplevel(self.root)
        self.preferences_popup = popup
        popup.title("Preferences")
        popup.geometry("700x620")
        popup.minsize(560, 420)
        popup.transient(self.root)
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(popup)
        notebook.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        eec_tab = ttk.Frame(notebook, padding=10)
        optimizer_tab = ttk.Frame(notebook, padding=10)
        explorer_tab = ttk.Frame(notebook, padding=10)
        auto_tab = ttk.Frame(notebook, padding=10)
        thresholds_tab = ttk.Frame(notebook, padding=10)
        notebook.add(eec_tab, text="EEC models")
        notebook.add(optimizer_tab, text="EEC optimizer")
        notebook.add(explorer_tab, text="Explorers")
        notebook.add(auto_tab, text="Auto model selection")
        notebook.add(thresholds_tab, text="Thresholds & guidance")

        eec_tab.columnconfigure(0, weight=1)
        eec_tab.rowconfigure(1, weight=1)
        optimizer_tab.columnconfigure(1, weight=1)
        explorer_tab.columnconfigure(1, weight=1)
        explorer_tab.rowconfigure(5, weight=1)
        auto_tab.columnconfigure(1, weight=1)
        thresholds_tab.columnconfigure(0, weight=1)

        ttk.Label(
            optimizer_tab,
            text="These settings control single, selected, and batch EEC fitting.",
            wraplength=480,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        ttk.Label(optimizer_tab, text="Pipeline").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            optimizer_tab,
            textvariable=self.fit_pipeline_var,
            state="readonly",
            values=("local only", "PSO → local", "GA → local", "PSO only", "GA only"),
        ).grid(row=1, column=1, sticky="ew")
        ttk.Label(optimizer_tab, text="Random seed (blank = random)").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(optimizer_tab, textvariable=self.fit_seed_var).grid(
            row=2, column=1, sticky="ew", pady=(8, 0)
        )
        ttk.Label(optimizer_tab, text="Population / iterations").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        limits = ttk.Frame(optimizer_tab)
        limits.grid(row=3, column=1, sticky="ew", pady=(8, 0))
        limits.columnconfigure(0, weight=1)
        limits.columnconfigure(1, weight=1)
        ttk.Entry(limits, textvariable=self.fit_population_var).grid(row=0, column=0, sticky="ew")
        ttk.Entry(limits, textvariable=self.fit_iterations_var).grid(row=0, column=1, padx=(8, 0), sticky="ew")
        ttk.Checkbutton(
            optimizer_tab, text="Weight residuals by |Z|", variable=self.fit_weight_modulus_var
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(optimizer_tab, text="Local Jacobian").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            optimizer_tab,
            textvariable=self.fit_jacobian_mode_var,
            state="readonly",
            values=("Numerical only", "Automatic", "Analytical when supported"),
        ).grid(row=5, column=1, sticky="ew", pady=(8, 0))
        ttk.Button(
            optimizer_tab, text="Show last fit diagnostics", command=self._show_fit_diagnostics
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(14, 0))

        fit_x_values = {"I_mA", "Ecell_V", "time_s", "cycle"}
        drt_x_values = set(fit_x_values) | {"R0", "L0"}
        fit_y_values = {"R0"}
        drt_y_values = {"R0", "L0"}
        if self.state is not None:
            for record in self._collect_fit_parameter_records():
                fit_x_values.update(record)
                fit_y_values.update(record)
            for record in self._collect_drt_parameter_records(self._selected_drt_mode()):
                drt_x_values.update(record)
                drt_y_values.update(record)
        fit_x_var = tk.StringVar(value=self._fit_explorer_x_preference)
        drt_x_var = tk.StringVar(value=self._drt_explorer_x_preference)
        fit_y_var = tk.StringVar(value=self._fit_explorer_y_preference)
        drt_y_var = tk.StringVar(value=self._drt_explorer_y_preference)
        ttk.Label(explorer_tab, text="EEC default X axis").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Combobox(
            explorer_tab,
            textvariable=fit_x_var,
            values=sorted(fit_x_values),
            state="normal",
        ).grid(row=0, column=1, padx=(8, 0), sticky="ew")
        ttk.Label(explorer_tab, text="DRT default X axis").grid(
            row=1, column=0, pady=(6, 0), sticky="w"
        )
        ttk.Combobox(
            explorer_tab,
            textvariable=drt_x_var,
            values=sorted(drt_x_values),
            state="normal",
        ).grid(row=1, column=1, padx=(8, 0), pady=(6, 0), sticky="ew")
        ttk.Label(explorer_tab, text="EEC default Y axis").grid(
            row=2, column=0, pady=(6, 0), sticky="w"
        )
        ttk.Combobox(
            explorer_tab,
            textvariable=fit_y_var,
            values=sorted(fit_y_values),
            state="normal",
        ).grid(row=2, column=1, padx=(8, 0), pady=(6, 0), sticky="ew")
        ttk.Label(explorer_tab, text="DRT default Y axis").grid(
            row=3, column=0, pady=(6, 0), sticky="w"
        )
        ttk.Combobox(
            explorer_tab,
            textvariable=drt_y_var,
            values=sorted(drt_y_values),
            state="normal",
        ).grid(row=3, column=1, padx=(8, 0), pady=(6, 0), sticky="ew")
        self._sync_custom_metadata_columns()
        preference_columns = self._explorer_columns()
        saved_order = [
            column for column in self._explorer_column_order_preference
            if column in preference_columns
        ]
        saved_order.extend(column for column in preference_columns if column not in saved_order)
        hidden_columns = {
            column
            for column in self._explorer_hidden_columns_preference
            if column in preference_columns
        }
        visible_order = [column for column in saved_order if column not in hidden_columns]
        hidden_order = [
            column
            for column in self._explorer_hidden_columns_preference
            if column in preference_columns
        ]
        ttk.Label(explorer_tab, text="Spectra Explorer column order").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(12, 4)
        )
        order_frame = ttk.Frame(explorer_tab)
        order_frame.grid(row=5, column=0, columnspan=2, sticky="nsew")
        order_frame.columnconfigure(0, weight=1)
        order_frame.columnconfigure(1, weight=0)
        order_frame.columnconfigure(2, weight=1)
        order_frame.rowconfigure(1, weight=1)
        ttk.Label(order_frame, text="Visible columns").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        ttk.Label(order_frame, text="Hidden columns").grid(
            row=0, column=2, sticky="w", padx=(8, 0)
        )
        visible_list = tk.Listbox(order_frame, exportselection=False, height=6)
        visible_list.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        hidden_list = tk.Listbox(order_frame, exportselection=False, height=6)
        hidden_list.grid(row=1, column=2, sticky="nsew", padx=(8, 0))
        for column in visible_order:
            visible_list.insert(tk.END, f"{column} — {self._explorer_headings.get(column, column)}")
        for column in hidden_order:
            hidden_list.insert(tk.END, f"{column} — {self._explorer_headings.get(column, column)}")
        transfer_buttons = ttk.Frame(order_frame)
        transfer_buttons.grid(row=1, column=1, padx=2)

        def move_columns(source: tk.Listbox, target: tk.Listbox) -> None:
            selected = source.curselection()
            if not selected:
                return
            values = [source.get(index) for index in selected]
            for index in reversed(selected):
                source.delete(index)
            for value in values:
                target.insert(tk.END, value)
            target.selection_clear(0, tk.END)
            target.selection_set(target.size() - len(values), tk.END)
            target.see(target.size() - 1)

        ttk.Button(
            transfer_buttons,
            text="Hide →",
            command=lambda: move_columns(visible_list, hidden_list),
        ).pack(pady=2)
        ttk.Button(
            transfer_buttons,
            text="← Show",
            command=lambda: move_columns(hidden_list, visible_list),
        ).pack(pady=2)
        ttk.Label(explorer_tab, text="Unlisted columns:").grid(
            row=6, column=0, sticky="w", pady=(6, 0)
        )
        new_columns_position_var = tk.StringVar(
            value=self._explorer_new_columns_position.title()
        )
        ttk.Combobox(
            explorer_tab,
            textvariable=new_columns_position_var,
            values=("End", "Beginning"),
            state="readonly",
        ).grid(row=6, column=1, sticky="ew", pady=(6, 0))

        def move_order_item(direction: int) -> None:
            selection = visible_list.curselection()
            if not selection:
                return
            index = selection[0]
            target = index + direction
            if not 0 <= target < visible_list.size():
                return
            value = visible_list.get(index)
            visible_list.delete(index)
            visible_list.insert(target, value)
            visible_list.selection_set(target)
            visible_list.see(target)

        order_buttons = ttk.Frame(explorer_tab)
        order_buttons.grid(row=6, column=0, columnspan=2, sticky="e", pady=(4, 0))
        ttk.Button(order_buttons, text="Move up", command=lambda: move_order_item(-1)).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(order_buttons, text="Move down", command=lambda: move_order_item(1)).pack(
            side=tk.LEFT, padx=3
        )

        ttk.Label(
            eec_tab,
            text="EEC circuits shown in the fitting-model lists",
        ).grid(row=0, column=0, sticky="w")
        list_frame = ttk.Frame(eec_tab, padding=(0, 8, 0, 0))
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        circuit_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED, exportselection=False)
        circuit_list.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=circuit_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        circuit_list.configure(yscrollcommand=scrollbar.set)
        for circuit in self._model_presets:
            circuit_list.insert(tk.END, circuit)

        entry = ttk.Entry(eec_tab)
        entry.grid(row=2, column=0, pady=(8, 6), sticky="ew")

        bound_vars: dict[str, tuple[tk.StringVar, tk.StringVar]] = {}
        bounds_frame = ttk.LabelFrame(
            eec_tab, text="Default parameter limits", padding=8
        )
        bounds_frame.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        bounds_frame.columnconfigure(1, weight=1)
        bounds_frame.columnconfigure(3, weight=1)
        ttk.Label(bounds_frame, text="Parameter").grid(row=0, column=0, sticky="w")
        ttk.Label(bounds_frame, text="Lower").grid(row=0, column=1, padx=4, sticky="w")
        ttk.Label(bounds_frame, text="Upper").grid(row=0, column=3, padx=4, sticky="w")
        for row, (category, label) in enumerate(
            (("r", "R"), ("l", "L"), ("cpe_q", "CPE Q"), ("cpe_alpha", "CPE alpha")),
            start=1,
        ):
            lower, upper = self._eec_parameter_bounds[category]
            lower_var = tk.StringVar(value=f"{lower:g}")
            upper_var = tk.StringVar(value=f"{upper:g}")
            bound_vars[category] = (lower_var, upper_var)
            ttk.Label(bounds_frame, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(bounds_frame, textvariable=lower_var).grid(
                row=row, column=1, padx=4, pady=2, sticky="ew"
            )
            ttk.Label(bounds_frame, text="–").grid(row=row, column=2, sticky="w")
            ttk.Entry(bounds_frame, textvariable=upper_var).grid(
                row=row, column=3, padx=4, pady=2, sticky="ew"
            )

        criterion_var = tk.StringVar(value=str(self._auto_model_settings["criterion"]))
        max_peaks_var = tk.StringVar(value=str(self._auto_model_settings["max_num_peaks"]))
        prominence_var = tk.StringVar(
            value="" if self._auto_model_settings["peak_prominence"] is None
            else str(self._auto_model_settings["peak_prominence"])
        )
        height_var = tk.StringVar(
            value="" if self._auto_model_settings["peak_height"] is None
            else str(self._auto_model_settings["peak_height"])
        )
        prior_var = tk.BooleanVar(value=bool(self._auto_model_settings["prior"]))
        prior_strength_var = tk.StringVar(
            value="" if self._auto_model_settings["prior_strength"] is None
            else str(self._auto_model_settings["prior_strength"])
        )
        min_r0_var = tk.StringVar(
            value="" if self._auto_model_settings["min_r0"] is None
            else str(self._auto_model_settings["min_r0"])
        )
        min_l0_var = tk.StringVar(
            value="" if self._auto_model_settings["min_l0"] is None
            else str(self._auto_model_settings["min_l0"])
        )
        fit_timeout_var = tk.StringVar(value=f"{self._fit_timeout_seconds:g}")
        ttk.Label(
            auto_tab,
            text="These settings are used for every Auto model selection run.",
            wraplength=430,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        ttk.Label(auto_tab, text="Selection criterion").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            auto_tab,
            textvariable=criterion_var,
            values=("bic", "lml", "lml-bic"),
            state="readonly",
        ).grid(row=1, column=1, sticky="ew")
        ttk.Label(auto_tab, text="Maximum DRT peaks / EEC blocks").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Entry(auto_tab, textvariable=max_peaks_var).grid(
            row=2, column=1, sticky="ew", pady=(6, 0)
        )
        ttk.Label(auto_tab, text="Minimum peak prominence (blank = package default)").grid(
            row=3, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Entry(auto_tab, textvariable=prominence_var).grid(
            row=3, column=1, sticky="ew", pady=(6, 0)
        )
        ttk.Label(auto_tab, text="Minimum peak height (blank = package default)").grid(
            row=4, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Entry(auto_tab, textvariable=height_var).grid(
            row=4, column=1, sticky="ew", pady=(6, 0)
        )
        ttk.Checkbutton(auto_tab, text="Use DRT prior", variable=prior_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Label(auto_tab, text="Prior strength (blank = package default)").grid(
            row=6, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Entry(auto_tab, textvariable=prior_strength_var).grid(
            row=6, column=1, sticky="ew", pady=(6, 0)
        )
        ttk.Label(auto_tab, text="Minimum R0 threshold (Ω; blank = always include)").grid(
            row=7, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Entry(auto_tab, textvariable=min_r0_var).grid(
            row=7, column=1, sticky="ew", pady=(6, 0)
        )
        ttk.Label(auto_tab, text="Minimum L0 threshold (H; blank = always include)").grid(
            row=8, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Entry(auto_tab, textvariable=min_l0_var).grid(
            row=8, column=1, sticky="ew", pady=(6, 0)
        )
        ttk.Label(auto_tab, text="EEC fit time limit (seconds)").grid(
            row=9, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Entry(auto_tab, textvariable=fit_timeout_var).grid(
            row=9, column=1, sticky="ew", pady=(6, 0)
        )

        ttk.Label(
            thresholds_tab,
            text=(
                "These are the defaults used by the main analysis controls on a fresh "
                "application start. Adjust the corresponding fields on the main tab "
                "when a particular spectrum needs a different sensitivity."
            ),
            wraplength=600,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))
        default_thresholds = ttk.LabelFrame(
            thresholds_tab, text="Outlier detection and refinement defaults", padding=8
        )
        default_thresholds.grid(row=1, column=0, sticky="ew")
        default_thresholds.columnconfigure(1, weight=1)
        threshold_vars = {
            "bayes_drt2": tk.StringVar(value=self.threshold_var.get()),
            "deterministic": tk.StringVar(value=self.deterministic_threshold_var.get()),
            "refine_z": tk.StringVar(value=self.refine_z_threshold_var.get()),
            "refine_max_iterations": tk.StringVar(value=self.refine_max_iterations_var.get()),
        }
        threshold_rows = (
            (
                "bayes_drt2",
                "Bayes-DRT2 outlier threshold",
                "Lower values flag more points; increase to make detection more conservative.",
            ),
            (
                "deterministic",
                "Deterministic outlier threshold",
                "Robust-score cutoff; 3–4 is more sensitive, 5 is more conservative.",
            ),
            (
                "refine_z",
                "Robust z threshold (Refine fit)",
                "Residual cutoff used when iteratively deactivating bad points.",
            ),
            (
                "refine_max_iterations",
                "Maximum iterations (Refine fit)",
                "Maximum points-removal/refit passes; increase only for difficult spectra.",
            ),
        )
        for row, (key, label, help_text) in enumerate(threshold_rows):
            ttk.Label(default_thresholds, text=label).grid(
                row=row, column=0, sticky="w", pady=3
            )
            ttk.Entry(default_thresholds, textvariable=threshold_vars[key], width=12).grid(
                row=row, column=1, sticky="w", padx=(16, 8), pady=3
            )
            ttk.Label(
                default_thresholds, text=help_text, wraplength=390, justify=tk.LEFT
            ).grid(row=row, column=2, sticky="w", pady=3)

        other_thresholds = ttk.LabelFrame(
            thresholds_tab, text="Other useful starting values", padding=8
        )
        other_thresholds.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        other_thresholds.columnconfigure(1, weight=1)
        other_rows = (
            ("Maximum DRT peaks / EEC blocks", "10", "Auto model selection"),
            ("EEC fit time limit", "10 s", "Auto model selection"),
            ("Optimizer population", "30", "EEC optimizer"),
            ("Optimizer iterations", "200", "EEC optimizer"),
            ("CPE alpha bounds", "0.5–1.0", "EEC parameter limits"),
        )
        for row, (label, value, location) in enumerate(other_rows):
            ttk.Label(other_thresholds, text=label).grid(
                row=row, column=0, sticky="w", pady=3
            )
            ttk.Label(other_thresholds, text=value).grid(
                row=row, column=1, sticky="w", padx=(16, 8), pady=3
            )
            ttk.Label(other_thresholds, text=location).grid(
                row=row, column=2, sticky="w", pady=3
            )
        ttk.Label(
            thresholds_tab,
            text=(
                "Practical tuning: start with the defaults, inspect the excluded points, "
                "and change one threshold at a time. A lower cutoff increases sensitivity "
                "but can remove valid measurements; a higher cutoff may leave artifacts "
                "in the fit."
            ),
            wraplength=600,
            justify=tk.LEFT,
        ).grid(row=3, column=0, sticky="w", pady=(12, 0))

        def add_circuit(_event=None) -> None:
            value = entry.get().strip()
            if value and value not in circuit_list.get(0, tk.END):
                circuit_list.insert(tk.END, value)
                entry.delete(0, tk.END)
            entry.focus_set()

        def remove_circuit() -> None:
            for index in reversed(circuit_list.curselection()):
                circuit_list.delete(index)

        def restore_defaults() -> None:
            circuit_list.delete(0, tk.END)
            for circuit in MODEL_PRESETS:
                circuit_list.insert(tk.END, circuit)

        def save_and_close() -> None:
            circuits = tuple(
                circuit.strip()
                for circuit in circuit_list.get(0, tk.END)
                if circuit.strip()
            )
            if not circuits:
                messagebox.showerror(
                    "No circuits",
                    "Keep at least one EEC circuit in the list.",
                    parent=popup,
                )
                return
            try:
                max_num_peaks = int(max_peaks_var.get())
                if max_num_peaks < 1:
                    raise ValueError("maximum peaks must be at least 1")

                def optional_float(variable: tk.StringVar, label: str):
                    value = variable.get().strip()
                    if not value:
                        return None
                    number = float(value)
                    if not np.isfinite(number) or number < 0:
                        raise ValueError(f"{label} must be a non-negative number")
                    return number

                parameter_bounds = {}
                for category, (lower_var, upper_var) in bound_vars.items():
                    lower = float(lower_var.get())
                    upper = float(upper_var.get())
                    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                        raise ValueError(
                            f"{category} lower limit must be smaller than its upper limit"
                        )
                    parameter_bounds[category] = (lower, upper)

                auto_settings = {
                    "criterion": criterion_var.get(),
                    "max_num_peaks": max_num_peaks,
                    "peak_prominence": optional_float(prominence_var, "peak prominence"),
                    "peak_height": optional_float(height_var, "peak height"),
                    "prior": bool(prior_var.get()),
                    "prior_strength": optional_float(prior_strength_var, "prior strength"),
                    "min_r0": optional_float(min_r0_var, "R0 threshold"),
                    "min_l0": optional_float(min_l0_var, "L0 threshold"),
                }
                bayes_drt2_threshold = float(threshold_vars["bayes_drt2"].get())
                deterministic_threshold = float(threshold_vars["deterministic"].get())
                refine_z_threshold = float(threshold_vars["refine_z"].get())
                refine_max_iterations = int(
                    threshold_vars["refine_max_iterations"].get()
                )
                if not np.isfinite(bayes_drt2_threshold) or bayes_drt2_threshold <= 0:
                    raise ValueError("Bayes-DRT2 threshold must be positive")
                if not np.isfinite(deterministic_threshold) or deterministic_threshold <= 0:
                    raise ValueError("deterministic-outlier threshold must be positive")
                if not np.isfinite(refine_z_threshold) or refine_z_threshold <= 0:
                    raise ValueError("robust z threshold must be positive")
                if refine_max_iterations < 1:
                    raise ValueError("maximum refine iterations must be at least 1")
                fit_timeout_seconds = float(fit_timeout_var.get())
                if (
                    not np.isfinite(fit_timeout_seconds)
                    or fit_timeout_seconds <= 0
                ):
                    raise ValueError("EEC fit time limit must be positive")
                self._fit_options_from_controls()
                self._auto_model_settings = auto_settings
                self._eec_parameter_bounds = parameter_bounds
                self._fit_timeout_seconds = fit_timeout_seconds
                self.threshold_var.set(f"{bayes_drt2_threshold:g}")
                self.deterministic_threshold_var.set(f"{deterministic_threshold:g}")
                self.refine_z_threshold_var.set(f"{refine_z_threshold:g}")
                self.refine_max_iterations_var.set(str(refine_max_iterations))
                self._bayes_drt2_threshold_preference = self.threshold_var.get()
                self._deterministic_threshold_preference = self.deterministic_threshold_var.get()
                self._refine_z_threshold_preference = self.refine_z_threshold_var.get()
                self._refine_max_iterations_preference = self.refine_max_iterations_var.get()
                self._save_preferences(
                    circuits,
                    fit_x_var.get().strip() or "I_mA",
                    drt_x_var.get().strip() or "I_mA",
                    fit_y_var.get().strip() or "R0",
                    drt_y_var.get().strip() or "R0",
                    [
                        saved_order_item.split(" — ", 1)[0]
                        for saved_order_item in visible_list.get(0, tk.END)
                    ],
                    [
                        hidden_order_item.split(" — ", 1)[0]
                        for hidden_order_item in hidden_list.get(0, tk.END)
                    ],
                    new_columns_position_var.get().casefold(),
                    self.procedure_blocks,
                    self.procedures,
                )
            except (OSError, TypeError, ValueError) as error:
                messagebox.showerror("Preferences save failed", str(error), parent=popup)
                return
            self._model_presets = tuple(dict.fromkeys(circuits))
            self._fit_explorer_x_preference = fit_x_var.get().strip() or "I_mA"
            self._drt_explorer_x_preference = drt_x_var.get().strip() or "I_mA"
            self._fit_explorer_y_preference = fit_y_var.get().strip() or "R0"
            self._drt_explorer_y_preference = drt_y_var.get().strip() or "R0"
            self._explorer_column_order_preference = [
                saved_order_item.split(" — ", 1)[0]
                for saved_order_item in visible_list.get(0, tk.END)
            ]
            self._explorer_hidden_columns_preference = [
                hidden_order_item.split(" — ", 1)[0]
                for hidden_order_item in hidden_list.get(0, tk.END)
            ]
            self._explorer_new_columns_position = new_columns_position_var.get().casefold()
            self._explorer_current_column_order = None
            self._apply_explorer_column_order()
            self._procedure_library_blocks = copy.deepcopy(self.procedure_blocks)
            self._procedure_library = copy.deepcopy(self.procedures)
            if hasattr(self, "model_box"):
                self.model_box.configure(values=self._model_presets)
            self._update_status("preferences saved")
            close_popup()

        def close_popup() -> None:
            self.preferences_popup = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", close_popup)
        entry.bind("<Return>", add_circuit)
        buttons = ttk.Frame(popup, padding=(0, 0, 0, 0))
        buttons.grid(row=1, column=0, pady=(0, 10), sticky="e")
        ttk.Button(buttons, text="Add", command=add_circuit).pack(side=tk.LEFT, padx=3)
        ttk.Button(buttons, text="Remove selected", command=remove_circuit).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(buttons, text="Restore defaults", command=restore_defaults).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(buttons, text="Save", command=save_and_close).pack(side=tk.LEFT, padx=3)
        ttk.Button(buttons, text="Cancel", command=close_popup).pack(side=tk.LEFT, padx=3)
        popup.grab_set()
        popup.focus_force()

    def _build_interface(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.grid(row=0, column=0, sticky="nsew")

        left_panel = ttk.Panedwindow(body, orient=tk.VERTICAL)
        self.plot_frame = ttk.Frame(left_panel, padding=8)
        explorer_frame = ttk.Frame(left_panel, padding=(8, 0, 8, 8))
        controls = ttk.Frame(body, padding=(8, 10, 12, 8), width=390)
        left_panel.add(self.plot_frame, weight=4)
        left_panel.add(explorer_frame, weight=1)
        body.add(left_panel, weight=4)
        body.add(controls, weight=0)
        self._build_plot()
        self._build_explorer(explorer_frame)
        self._build_controls(controls)

        ttk.Separator(self.root).grid(row=1, column=0, sticky="ew")
        ttk.Label(self.root, textvariable=self.status_var, padding=(8, 5)).grid(
            row=2, column=0, sticky="ew"
        )

    def _build_plot(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.collections import LineCollection
        from matplotlib.figure import Figure
        from matplotlib.widgets import RectangleSelector

        self._line_collection_class = LineCollection
        self._rectangle_selector_class = RectangleSelector
        self.point_toggle_mode = False
        self.point_auto_fit = False
        self._pan_state = None
        self.plot_controls = ttk.Frame(self.plot_frame)
        self.plot_controls.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        self.toggle_points_button = ttk.Button(
            self.plot_controls,
            text="Edit points: Off",
            command=self.toggle_point_edit_mode,
        )
        self.toggle_points_button.pack(side=tk.LEFT)
        self.auto_fit_points_button = ttk.Button(
            self.plot_controls,
            text="Edit points and fit: Off",
            command=self.toggle_auto_fit_points,
        )
        self.auto_fit_points_button.pack(side=tk.LEFT, padx=(6, 0))
        self.reset_view_button = ttk.Button(
            self.plot_controls,
            text="Active zoom",
            command=self.reset_plot_view,
        )
        self.reset_view_button.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Checkbutton(
            self.plot_controls,
            text="Show all points",
            variable=self.show_all_points_var,
            command=self.toggle_show_all_points,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            self.plot_controls,
            text="Hide legends",
            variable=self.hide_legends_var,
            command=self._update_legend_visibility,
        ).pack(side=tk.LEFT, padx=(8, 0))
        self.toggle_plot_mode_button = ttk.Button(
            self.plot_controls,
            text="Show Bode",
            command=self.toggle_plot_mode,
        )
        self.toggle_plot_mode_button.pack(side=tk.LEFT, padx=(6, 0))
        self.drt_mode_var = tk.StringVar(value="Ridge DRT")
        ttk.Checkbutton(
            self.plot_controls,
            text="Show spectrum",
            variable=self.show_spectrum_var,
            command=self.toggle_spectrum_view,
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Checkbutton(
            self.plot_controls,
            text="Show DRT",
            variable=self.show_drt_var,
            command=self.toggle_drt_view,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            self.plot_controls,
            text="Show KK residuals",
            variable=self.show_kk_var,
            command=self.toggle_kk_view,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            self.plot_controls,
            text="Show EEC fit",
            variable=self.show_eec_fit_var,
            command=self.toggle_fit_visibility,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            self.plot_controls,
            text="Show DRT fit",
            variable=self.show_drt_fit_var,
            command=self.toggle_drt_fit_visibility,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            self.plot_controls,
            text="Show DRT recovered",
            variable=self.show_drt_recovered_var,
            command=self.toggle_drt_recovered_visibility,
        ).pack(side=tk.LEFT, padx=(8, 0))
        self.ml_controls = ttk.LabelFrame(
            self.plot_frame, text="ML results", padding=(6, 3)
        )
        self.ml_controls.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        ttk.Button(
            self.ml_controls,
            text="ML processing…",
            command=self.open_ml_processing,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            self.ml_controls,
            text="Load ML results file",
            command=self.load_ml_results_file,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            self.ml_controls,
            text="Apply ML EEC Model",
            command=self.apply_ml_eec_to_selected,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            self.ml_controls,
            text="Load ML Frequency Selection",
            command=self.apply_ml_frequency_to_selected,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            self.ml_controls,
            text="Apply ML Initial Parameters",
            command=self.load_and_apply_ml_initial_parameters,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            self.ml_controls,
            text="Restore Original Selection",
            command=self.restore_ml_original_selection,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(
            self.ml_controls, textvariable=self.ml_results_status_var
        ).pack(side=tk.LEFT, padx=(10, 0))
        self.figure = Figure(figsize=(7.5, 6.5), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.draw()
        self.toolbar = self._create_toolbar(
            self.canvas,
            self.plot_frame,
            self.reset_plot_view,
        )
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_plot_button_press)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.canvas.mpl_connect("button_release_event", self._on_plot_button_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.mpl_connect("scroll_event", self._on_plot_scroll)
        self._attach_plot_export_menu(self.canvas, self.plot_frame)
        self._configure_plot_layout()

    def _create_toolbar(
        self,
        canvas,
        master: tk.Misc,
        home_callback: Callable[[], None] | None = None,
    ):
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        class _ActiveZoomToolbar(NavigationToolbar2Tk):
            def __init__(self, *args, home_callback=None, **kwargs):
                self._home_callback = home_callback
                super().__init__(*args, **kwargs)

            def home(self, *args):
                if self._home_callback is not None:
                    self._home_callback()
                    return
                super().home(*args)

        return _ActiveZoomToolbar(
            canvas,
            master,
            pack_toolbar=False,
            home_callback=home_callback,
        )

    def _attach_plot_export_menu(self, canvas, owner: tk.Misc | None = None) -> None:
        """Add the common displayed-data export menu to a Matplotlib canvas."""
        if getattr(canvas, "_eis_plot_export_bound", False):
            return
        canvas._eis_plot_export_bound = True
        menu_owner = owner or self.root
        widget = canvas.get_tk_widget()

        def show_menu(event) -> str | None:
            if event.button != 3:
                return None
            if self.point_toggle_mode or self.point_auto_fit:
                # Let Matplotlib deliver the right-click event to the point
                # editor instead of opening the graph context menu.
                return None
            axes = event.inaxes
            if axes is None:
                return None
            gui_event = getattr(event, "guiEvent", None)
            x_root = getattr(gui_event, "x_root", None)
            y_root = getattr(gui_event, "y_root", None)
            if x_root is None or y_root is None:
                figure_width, figure_height = canvas.figure.canvas.get_width_height()
                x_root = widget.winfo_rootx() + event.x * widget.winfo_width() / max(figure_width, 1)
                y_root = widget.winfo_rooty() + widget.winfo_height() - event.y * widget.winfo_height() / max(figure_height, 1)
            menu = tk.Menu(menu_owner, tearoff=False)
            # Keep the Tcl/Tk menu alive after this callback returns.  A local
            # Menu can otherwise be garbage-collected while the popup is open,
            # especially after repeated right-clicks on recreated plot canvases.
            canvas._eis_plot_context_menu = menu
            menu.add_command(
                label="Save graph",
                command=lambda: self._save_plot_graph(
                    canvas.figure, axes, menu_owner
                ),
            )
            menu.add_command(
                label="Copy to Clipboard",
                command=lambda: self._copy_plot_to_clipboard(
                    canvas.figure, menu_owner
                ),
            )
            menu.add_separator()
            menu.add_command(
                label="Export data",
                command=lambda: self._export_displayed_plot_data(axes, menu_owner),
            )
            try:
                menu.tk_popup(int(x_root), int(y_root))
            finally:
                menu.grab_release()
            return None

        canvas.mpl_connect("button_press_event", show_menu)

    def _save_plot_graph(self, figure, axes, owner: tk.Misc | None = None) -> None:
        title = axes.get_title().strip() or "plot"
        title = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_") or "plot"
        path = filedialog.asksaveasfilename(
            parent=owner or self.root,
            title="Save graph",
            initialfile=f"{title}.png",
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("PDF document", "*.pdf"),
                ("SVG image", "*.svg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            figure.savefig(path, bbox_inches="tight")
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Save graph",
                f"Could not save the graph:\n{error}",
                parent=owner or self.root,
            )
            return
        self._update_status(f"saved graph to {Path(path).name}")

    def _copy_plot_to_clipboard(
        self, figure, owner: tk.Misc | None = None
    ) -> None:
        """Copy a rendered PNG to the native clipboard on Windows."""
        if os.name != "nt":
            messagebox.showinfo(
                "Copy to Clipboard",
                "Graph image clipboard copying is currently supported on Windows only.",
                parent=owner or self.root,
            )
            return
        try:
            from PIL import Image

            image_buffer = BytesIO()
            figure.savefig(image_buffer, format="png", dpi=figure.dpi)
            image_buffer.seek(0)
            image = Image.open(image_buffer).convert("RGBA")
            width, height = image.size
            pixels = image.tobytes("raw", "BGRA")
            dib = struct.pack(
                "<IiiHHIIiiII",
                40,
                width,
                -height,
                1,
                32,
                0,
                len(pixels),
                0,
                0,
                0,
                0,
            ) + pixels

            import ctypes

            kernel32 = ctypes.windll.kernel32
            user32 = ctypes.windll.user32
            handle_type = ctypes.c_void_p
            kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
            kernel32.GlobalAlloc.restype = handle_type
            kernel32.GlobalLock.argtypes = [handle_type]
            kernel32.GlobalLock.restype = handle_type
            kernel32.GlobalUnlock.argtypes = [handle_type]
            kernel32.GlobalUnlock.restype = ctypes.c_bool
            kernel32.GlobalFree.argtypes = [handle_type]
            kernel32.GlobalFree.restype = handle_type
            user32.OpenClipboard.argtypes = [handle_type]
            user32.OpenClipboard.restype = ctypes.c_bool
            user32.EmptyClipboard.argtypes = []
            user32.EmptyClipboard.restype = ctypes.c_bool
            user32.SetClipboardData.argtypes = [ctypes.c_uint, handle_type]
            user32.SetClipboardData.restype = handle_type
            user32.CloseClipboard.argtypes = []
            user32.CloseClipboard.restype = ctypes.c_bool
            GMEM_MOVEABLE = 0x0002
            GMEM_ZEROINIT = 0x0040
            CF_DIB = 8
            handle = kernel32.GlobalAlloc(GMEM_MOVEABLE | GMEM_ZEROINIT, len(dib))
            if not handle:
                raise OSError("Windows could not allocate clipboard memory")
            locked = kernel32.GlobalLock(handle)
            if not locked:
                kernel32.GlobalFree(handle)
                raise OSError("Windows could not lock clipboard memory")
            try:
                ctypes.memmove(locked, dib, len(dib))
            finally:
                kernel32.GlobalUnlock(handle)
            if not user32.OpenClipboard(self.root.winfo_id()):
                kernel32.GlobalFree(handle)
                raise OSError("Windows could not open the clipboard")
            try:
                user32.EmptyClipboard()
                if not user32.SetClipboardData(CF_DIB, handle):
                    kernel32.GlobalFree(handle)
                    raise OSError("Windows could not set the clipboard image")
                handle = None
            finally:
                user32.CloseClipboard()
            self._update_status("copied graph to clipboard")
        except (ImportError, OSError, ValueError) as error:
            messagebox.showerror(
                "Copy to Clipboard",
                f"Could not copy the graph image:\n{error}",
                parent=owner or self.root,
            )

    def _export_displayed_plot_data(self, axes, owner: tk.Misc | None = None) -> None:
        series = extract_displayed_series(axes)
        if not series:
            messagebox.showinfo(
                "Export data",
                "There are no displayed data series to export.",
                parent=owner or self.root,
            )
            return
        title = axes.get_title().strip() or "plot"
        title = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_") or "plot"
        path = filedialog.asksaveasfilename(
            parent=owner or self.root,
            title="Export displayed plot data",
            initialfile=f"{title}_data.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            rows = write_displayed_csv(axes, path)
        except OSError as error:
            messagebox.showerror(
                "Export data",
                f"Could not write the export file:\n{error}",
                parent=owner or self.root,
            )
            return
        self._update_status(f"exported {rows} displayed plot points to {Path(path).name}")

    @staticmethod
    def _phase_degrees(values: np.ndarray) -> np.ndarray:
        return -np.degrees(np.angle(values))

    def _update_point_hover(self, event) -> None:
        if (
            self.state is None
            or self.plot_mode != "nyquist"
            or self.point_toggle_mode
            or self.point_auto_fit
        ):
            return self._hide_point_hover()
        if event.inaxes is not self.axes or event.x is None or event.y is None:
            return self._hide_point_hover()
        cycle = self.state.active
        if cycle.impedance.size == 0:
            return self._hide_point_hover()
        display_points = self.axes.transData.transform(
            np.column_stack((cycle.impedance.real, -cycle.impedance.imag))
        )
        distances = np.hypot(
            display_points[:, 0] - event.x,
            display_points[:, 1] - event.y,
        )
        if distances.size == 0:
            return self._hide_point_hover()
        index = int(np.argmin(distances))
        if distances[index] > 9.0:
            return self._hide_point_hover()
        annotation = getattr(self, "point_hover_annotation", None)
        if annotation is None or annotation.axes is not self.axes:
            annotation = self.axes.annotate(
                "",
                xy=(0.0, 0.0),
                xytext=(10, 10),
                textcoords="offset points",
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "#777777",
                    "alpha": 0.9,
                },
                zorder=20,
            )
            self.point_hover_annotation = annotation
        frequency = float(cycle.frequency_hz[index])
        annotation.xy = (
            float(cycle.impedance.real[index]),
            float(-cycle.impedance.imag[index]),
        )
        annotation.set_text(f"f = {frequency:.6g} Hz")
        annotation.set_visible(True)
        self.canvas.draw_idle()

    def _hide_point_hover(self) -> None:
        annotation = getattr(self, "point_hover_annotation", None)
        if annotation is not None and annotation.get_visible():
            annotation.set_visible(False)
            if hasattr(self, "canvas"):
                self.canvas.draw_idle()

    def _refresh_plot_with_drt_recovery(self, *args, **kwargs) -> None:
        self._base_refresh_plot(*args, **kwargs)
        self._update_drt_recovered_plot()

    def _drt_recovered_impedance(
        self,
        cycle,
        frequencies: np.ndarray | None = None,
    ) -> np.ndarray | None:
        mode = self._selected_drt_mode()
        if mode == "hybrid":
            tau_values = cycle.saved_hybrid_tau_s
            gamma_values = cycle.saved_hybrid_gamma_ohm
            resistance = cycle.saved_hybrid_ohmic_resistance
        else:
            tau_values = cycle.saved_ridge_tau_s
            gamma_values = cycle.saved_ridge_gamma_ohm
            resistance = cycle.saved_ridge_ohmic_resistance
        if tau_values is None or gamma_values is None:
            return None
        tau_values = np.asarray(tau_values, dtype=float)
        gamma_values = np.asarray(gamma_values, dtype=float)
        valid = (
            np.isfinite(tau_values)
            & np.isfinite(gamma_values)
            & (tau_values > 0.0)
        )
        if np.count_nonzero(valid) < 2:
            return None
        tau_values = tau_values[valid]
        gamma_values = gamma_values[valid]
        order = np.argsort(tau_values)
        tau_values = tau_values[order]
        gamma_values = gamma_values[order]
        if frequencies is None:
            frequencies = cycle.frequency_hz
        frequencies = np.asarray(frequencies, dtype=float)
        angular_frequency = 2.0 * np.pi * frequencies
        integrand = gamma_values[None, :] / (
            1.0 + 1j * angular_frequency[:, None] * tau_values[None, :]
        )
        impedance = np.trapezoid(
            integrand,
            x=np.log(tau_values),
            axis=1,
        )
        if resistance is not None:
            impedance = impedance + float(resistance)
        inductance = (
            cycle.saved_hybrid_inductance
            if mode == "hybrid"
            else cycle.saved_ridge_inductance
        )
        if inductance is not None:
            impedance = impedance + 1j * angular_frequency * float(inductance)
        return impedance

    def _drt_peak_impedance(
        self,
        cycle,
        frequencies: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if not self.drt_peak_parameters:
            return None
        if frequencies is None:
            frequencies = cycle.frequency_hz
        frequencies = np.asarray(frequencies, dtype=float)
        angular_frequency = 2.0 * np.pi * frequencies
        peaks = []
        for peak in self.drt_peak_parameters:
            try:
                center = float(peak["center_log10"])
                sigma = max(float(peak["sigma_log10"]), 1e-8)
                height = max(float(peak["height"]), 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            peaks.append((center, sigma, height))
        if not peaks:
            return None
        lower = min(center - 6.0 * sigma for center, sigma, _height in peaks)
        upper = max(center + 6.0 * sigma for center, sigma, _height in peaks)
        tau_values = np.logspace(lower, upper, 1200)
        gamma_values = np.zeros_like(tau_values)
        log_tau = np.log10(tau_values)
        for center, sigma, height in peaks:
            gamma_values += height * np.exp(-0.5 * ((log_tau - center) / sigma) ** 2)
        impedance = np.trapezoid(
            gamma_values[None, :] / (
                1.0 + 1j * angular_frequency[:, None] * tau_values[None, :]
            ),
            x=np.log(tau_values),
            axis=1,
        )
        mode = self._selected_drt_mode()
        resistance = (
            cycle.saved_hybrid_ohmic_resistance
            if mode == "hybrid"
            else cycle.saved_ridge_ohmic_resistance
        )
        if resistance is not None:
            impedance = impedance + float(resistance)
        if cycle.saved_ridge_inductance is not None:
            impedance = impedance + 1j * angular_frequency * float(
                cycle.saved_ridge_inductance
            )
        return impedance

    def _update_drt_fit_curve(self, cycle, impedance: np.ndarray | None) -> None:
        if not hasattr(self, "drt_fit_artist"):
            return
        if not self.show_drt_fit_var.get() or impedance is None:
            self.drt_fit_artist.set_data([], [])
            if getattr(self, "drt_phase_fit_artist", None) is not None:
                self.drt_phase_fit_artist.set_data([], [])
            return
        curve_source_frequency = np.asarray(cycle.frequency_hz, dtype=float)
        curve_source_frequency = curve_source_frequency[
            np.isfinite(curve_source_frequency) & (curve_source_frequency > 0)
        ]
        if curve_source_frequency.size < 2:
            self.drt_fit_artist.set_data([], [])
            return
        curve_frequency = np.geomspace(
            float(np.min(curve_source_frequency)),
            float(np.max(curve_source_frequency)),
            600,
        )

        curve_impedance = self._drt_peak_impedance(cycle, curve_frequency)
        if curve_impedance is None:
            self.drt_fit_artist.set_data([], [])
            return
        self.drt_fit_artist.set_visible(True)
        self.drt_fit_artist.set_zorder(4)
        if self.plot_mode == "nyquist":
            self.drt_fit_artist.set_data(
                curve_impedance.real, -curve_impedance.imag
            )
        else:
            self.drt_fit_artist.set_data(
                curve_frequency, np.abs(curve_impedance)
            )
            if getattr(self, "drt_phase_fit_artist", None) is not None:
                self.drt_phase_fit_artist.set_data(
                    curve_frequency, self._phase_degrees(curve_impedance)
                )

    def _update_drt_curve_comparison(
        self,
        prefix: str,
        impedance: np.ndarray | None,
        color: str,
        label: str,
    ) -> None:
        cycle = self.state.active if self.state is not None else None
        if cycle is None or impedance is None:
            impedance = np.asarray([], dtype=complex)
        frequency = np.asarray(cycle.frequency_hz, dtype=float) if cycle is not None else np.asarray([])
        measured = np.asarray(cycle.impedance, dtype=complex) if cycle is not None else np.asarray([])
        included = np.asarray(cycle.included, dtype=bool) if cycle is not None else np.asarray([], dtype=bool)
        names = (
            "points_included",
            "points_excluded",
            "residual",
            "excluded_residual",
            "phase_points_included",
            "phase_points_excluded",
            "phase_residual",
            "phase_excluded_residual",
        )
        artists = {name: f"{prefix}_{name}" for name in names}
        if not hasattr(self, artists["points_included"]) or getattr(
            self, artists["points_included"]
        ).axes is not self.axes:
            setattr(self, artists["points_included"], self.axes.plot(
                [], [], "o", color=color, markersize=3, alpha=0.55,
                label=f"{label} at measured frequencies",
            )[0])
            setattr(self, artists["points_excluded"], self.axes.plot(
                [], [], "o", color=color, markersize=2, alpha=0.18,
                label="_nolegend_",
            )[0])
            setattr(self, artists["residual"], self._line_collection_class(
                [], colors="#777777", linewidths=0.8, linestyles="dashed",
                alpha=0.3, label=f"Measured-to-{label.lower()} difference",
            ))
            setattr(self, artists["excluded_residual"], self._line_collection_class(
                [], colors="#777777", linewidths=0.7, linestyles="dashed",
                alpha=0.1, label="_nolegend_",
            ))
            self.axes.add_collection(getattr(self, artists["residual"]))
            self.axes.add_collection(getattr(self, artists["excluded_residual"]))
        if self.plot_mode == "bode" and self.phase_axes is not None:
            phase_artist = getattr(self, artists["phase_points_included"], None)
            if phase_artist is None or phase_artist.axes is not self.phase_axes:
                setattr(self, artists["phase_points_included"], self.phase_axes.plot(
                    [], [], "o", color=color, markersize=2.5, alpha=0.5,
                    label=f"{label} phase at measured frequencies",
                )[0])
                setattr(self, artists["phase_points_excluded"], self.phase_axes.plot(
                    [], [], "o", color=color, markersize=2, alpha=0.16,
                    label="_nolegend_",
                )[0])
                setattr(self, artists["phase_residual"], self._line_collection_class(
                    [], colors="#777777", linewidths=0.7, linestyles="dashed",
                    alpha=0.3, label=f"Measured-to-{label.lower()} phase difference",
                ))
                setattr(self, artists["phase_excluded_residual"], self._line_collection_class(
                    [], colors="#777777", linewidths=0.6, linestyles="dashed",
                    alpha=0.1, label="_nolegend_",
                ))
                self.phase_axes.add_collection(getattr(self, artists["phase_residual"]))
                self.phase_axes.add_collection(getattr(self, artists["phase_excluded_residual"]))
        visible = (
            self.show_drt_recovered_var.get()
            if prefix == "drt_recovered"
            else self.show_drt_fit_var.get()
        )
        if not visible or impedance.size != frequency.size or measured.size != frequency.size:
            for name in (
                "points_included",
                "points_excluded",
                "phase_points_included",
                "phase_points_excluded",
            ):
                artist = getattr(self, artists[name], None)
                if artist is not None:
                    artist.set_data([], [])
            for name in (
                "residual",
                "excluded_residual",
                "phase_residual",
                "phase_excluded_residual",
            ):
                artist = getattr(self, artists[name], None)
                if artist is not None:
                    artist.set_segments([])
            return
        calculated = impedance
        if self.plot_mode == "nyquist":
            x_measured, y_measured = measured.real, -measured.imag
            x_calculated, y_calculated = calculated.real, -calculated.imag
            getattr(self, artists["points_included"]).set_data(x_calculated[included], y_calculated[included])
            getattr(self, artists["points_excluded"]).set_data(x_calculated[~included], y_calculated[~included])
            getattr(self, artists["residual"]).set_segments([
                [(x_measured[index], y_measured[index]), (x_calculated[index], y_calculated[index])]
                for index in np.flatnonzero(included)
            ])
            getattr(self, artists["excluded_residual"]).set_segments([
                [(x_measured[index], y_measured[index]), (x_calculated[index], y_calculated[index])]
                for index in np.flatnonzero(~included)
            ])
        elif self.phase_axes is not None:
            magnitude_measured = np.abs(measured)
            magnitude_calculated = np.abs(calculated)
            phase_measured = self._phase_degrees(measured)
            phase_calculated = self._phase_degrees(calculated)
            getattr(self, artists["points_included"]).set_data(frequency[included], magnitude_calculated[included])
            getattr(self, artists["points_excluded"]).set_data(frequency[~included], magnitude_calculated[~included])
            getattr(self, artists["residual"]).set_segments([
                [(frequency[index], magnitude_measured[index]), (frequency[index], magnitude_calculated[index])]
                for index in np.flatnonzero(included)
            ])
            getattr(self, artists["excluded_residual"]).set_segments([
                [(frequency[index], magnitude_measured[index]), (frequency[index], magnitude_calculated[index])]
                for index in np.flatnonzero(~included)
            ])
            getattr(self, artists["phase_points_included"]).set_data(frequency[included], phase_calculated[included])
            getattr(self, artists["phase_points_excluded"]).set_data(frequency[~included], phase_calculated[~included])
            getattr(self, artists["phase_residual"]).set_segments([
                [(frequency[index], phase_measured[index]), (frequency[index], phase_calculated[index])]
                for index in np.flatnonzero(included)
            ])
            getattr(self, artists["phase_excluded_residual"]).set_segments([
                [(frequency[index], phase_measured[index]), (frequency[index], phase_calculated[index])]
                for index in np.flatnonzero(~included)
            ])

    def _update_drt_recovered_plot(self) -> None:
        if not hasattr(self, "axes"):
            return
        recovered = None
        if self.show_drt_recovered_var.get() and self.state is not None:
            recovered = self._drt_recovered_impedance(self.state.active)
        if recovered is None:
            if hasattr(self, "drt_recovered_artist"):
                self.drt_recovered_artist.set_data([], [])
            if hasattr(self, "phase_drt_recovered_artist"):
                self.phase_drt_recovered_artist.set_data([], [])
            peak_impedance = (
                self._drt_peak_impedance(self.state.active)
                if self.state is not None and self.show_drt_fit_var.get()
                else None
            )
            self._update_drt_fit_curve(self.state.active, peak_impedance) if self.state is not None else None
            self._update_drt_curve_comparison(
                "drt_recovered", None, "#00838f", "DRT recovered"
            )
            self._update_drt_curve_comparison(
                "drt_fit",
                peak_impedance,
                "#6a1b9a",
                "DRT fit",
            )
            if hasattr(self, "canvas"):
                self.canvas.draw_idle()
            return
        frequency = np.asarray(self.state.active.frequency_hz, dtype=float)
        included = self.state.active.included
        if (
            not hasattr(self, "drt_recovered_artist")
            or self.drt_recovered_artist.axes is not self.axes
        ):
            (self.drt_recovered_artist,) = self.axes.plot(
                [], [], "-", color="#00838f", linewidth=1.8,
                alpha=0.85, label="DRT recovered",
            )
        self.drt_recovered_artist.set_visible(True)
        self.drt_recovered_artist.set_zorder(3)
        valid_frequency = np.isfinite(frequency) & (frequency > 0.0)
        smooth_frequency = None
        recovered_smooth = None
        if np.count_nonzero(valid_frequency) >= 2:
            smooth_frequency = np.geomspace(
                float(np.min(frequency[valid_frequency])),
                float(np.max(frequency[valid_frequency])),
                600,
            )
            recovered_smooth = self._drt_recovered_impedance(
                self.state.active, smooth_frequency
            )
        if hasattr(self, "phase_drt_recovered_artist"):
            self.phase_drt_recovered_artist.set_data([], [])
        if recovered_smooth is None:
            self.drt_recovered_artist.set_data([], [])
        elif self.plot_mode == "nyquist":
            self.drt_recovered_artist.set_data(
                recovered_smooth.real, -recovered_smooth.imag
            )
        else:
            self.drt_recovered_artist.set_data(
                smooth_frequency, np.abs(recovered_smooth)
            )
            if (
                not hasattr(self, "phase_drt_recovered_artist")
                or self.phase_drt_recovered_artist.axes is not self.phase_axes
            ):
                if self.phase_axes is None:
                    return
                (self.phase_drt_recovered_artist,) = self.phase_axes.plot(
                    [], [], "-", color="#00695c", linewidth=1.5,
                    alpha=0.85, label="DRT recovered phase",
                )
            if recovered_smooth is not None:
                self.phase_drt_recovered_artist.set_data(
                    smooth_frequency, self._phase_degrees(recovered_smooth)
                )
        peak_impedance = (
            self._drt_peak_impedance(self.state.active)
            if self.show_drt_fit_var.get() and self.state is not None
            else None
        )
        self._update_drt_fit_curve(self.state.active, peak_impedance) if self.state is not None else None
        self._update_drt_curve_comparison(
            "drt_recovered", recovered, "#00838f", "DRT recovered"
        )
        self._update_drt_curve_comparison(
            "drt_fit",
            peak_impedance,
            "#6a1b9a",
            "DRT fit",
        )
        self._update_legend_visibility()
        self.canvas.draw_idle()

    def toggle_drt_recovered_visibility(self) -> None:
        self._refresh_plot(rescale=False)

    def _update_plot_mode_button(self) -> None:
        if not hasattr(self, "toggle_plot_mode_button"):
            return
        self.toggle_plot_mode_button.configure(
            text="Show Bode" if self.plot_mode == "nyquist" else "Show Nyquist"
        )

    def _update_legend_visibility(self) -> None:
        for axis in self._active_plot_axes():
            legend = axis.get_legend()
            if legend is not None:
                legend.set_visible(not self.hide_legends_var.get())

    def _active_plot_axes(self) -> set:
        return {
            axis
            for axis in (
                self.axes,
                getattr(self, "phase_axes", None),
                getattr(self, "kk_axes", None),
                self.drt_axes,
            )
            if axis is not None
        }

    def _configure_nyquist_plot(self) -> None:
        self.phase_axes = None
        self.axes.set_xlabel("Re(Z) / Ohm")
        self.axes.set_ylabel("-Im(Z) / Ohm")
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.grid(True, alpha=0.25)
        self.axes.axhline(0.0, color="#444444", linewidth=1.2, alpha=0.85, zorder=0)
        self.axes.axvline(0.0, color="#444444", linewidth=1.2, alpha=0.85, zorder=0)
        (self.included_artist,) = self.axes.plot(
            [], [], "o", color="#1769aa", markersize=5, label="Included"
        )
        (self.excluded_artist,) = self.axes.plot(
            [], [], "x", color="#c62828", markersize=6, label="Excluded"
        )
        (self.fit_artist,) = self.axes.plot(
            [], [], "-", color="#202020", linewidth=2, alpha=0.8, label="Fit"
        )
        (self.fit_points_included_artist,) = self.axes.plot(
            [],
            [],
            "o",
            color="#f57c00",
            markersize=3,
            alpha=0.6,
            label="Fit at measured frequencies",
        )
        (self.fit_points_excluded_artist,) = self.axes.plot(
            [],
            [],
            "o",
            color="#f57c00",
            markersize=2.5,
            alpha=0.2,
            label="_nolegend_",
        )
        self.residual_artist = self._line_collection_class(
            [],
            colors="#777777",
            linewidths=0.9,
            linestyles="dashed",
            alpha=0.3,
            zorder=1,
            label="Measured-to-fit difference",
        )
        self.excluded_residual_artist = self._line_collection_class(
            [],
            colors="#777777",
            linewidths=0.8,
            linestyles="dashed",
            alpha=0.1,
            zorder=1,
            label="_nolegend_",
        )
        self.axes.add_collection(self.residual_artist)
        self.axes.add_collection(self.excluded_residual_artist)
        self.phase_included_artist = None
        self.phase_excluded_artist = None
        self.phase_fit_artist = None
        self.phase_fit_points_included_artist = None
        self.phase_fit_points_excluded_artist = None
        self.phase_residual_artist = None
        self.phase_excluded_residual_artist = None
        self.axes.legend(loc="best")

    def _configure_bode_plot(self) -> None:
        self.axes.set_xscale("log")
        self.axes.set_xlabel("Frequency / Hz")
        self.axes.set_ylabel("|Z| / Ohm")
        self.axes.grid(True, alpha=0.25)
        self.phase_axes = self.axes.twinx()
        self.phase_axes.set_ylabel("-Phase / deg")
        self.phase_axes.grid(False)
        (self.included_artist,) = self.axes.plot(
            [], [], "o", color="#1769aa", markersize=5, label="|Z| included"
        )
        (self.excluded_artist,) = self.axes.plot(
            [], [], "x", color="#c62828", markersize=6, label="|Z| excluded"
        )
        (self.fit_artist,) = self.axes.plot(
            [], [], "-", color="#202020", linewidth=2, alpha=0.8, label="|Z| fit"
        )
        (self.fit_points_included_artist,) = self.axes.plot(
            [],
            [],
            "o",
            color="#f57c00",
            markersize=3,
            alpha=0.6,
            label="|Z| fit at measured frequencies",
        )
        (self.fit_points_excluded_artist,) = self.axes.plot(
            [],
            [],
            "o",
            color="#f57c00",
            markersize=2.5,
            alpha=0.2,
            label="_nolegend_",
        )
        (self.phase_included_artist,) = self.phase_axes.plot(
            [], [], "s", color="#6a1b9a", markersize=4, alpha=0.85, label="-Phase included"
        )
        (self.phase_excluded_artist,) = self.phase_axes.plot(
            [], [], "x", color="#ab47bc", markersize=4.5, alpha=0.45, label="-Phase excluded"
        )
        (self.phase_fit_artist,) = self.phase_axes.plot(
            [], [], "-", color="#4a148c", linewidth=1.8, alpha=0.8, label="-Phase fit"
        )
        (self.phase_fit_points_included_artist,) = self.phase_axes.plot(
            [],
            [],
            "o",
            color="#8e24aa",
            markersize=2.5,
            alpha=0.55,
            label="-Phase fit at measured frequencies",
        )
        (self.phase_fit_points_excluded_artist,) = self.phase_axes.plot(
            [],
            [],
            "o",
            color="#8e24aa",
            markersize=2,
            alpha=0.18,
            label="_nolegend_",
        )
        self.residual_artist = self._line_collection_class(
            [],
            colors="#777777",
            linewidths=0.9,
            linestyles="dashed",
            alpha=0.3,
            zorder=1,
            label="|Z| measured-to-fit difference",
        )
        self.excluded_residual_artist = self._line_collection_class(
            [],
            colors="#777777",
            linewidths=0.8,
            linestyles="dashed",
            alpha=0.1,
            zorder=1,
            label="_nolegend_",
        )
        self.phase_residual_artist = self._line_collection_class(
            [],
            colors="#8d6e63",
            linewidths=0.8,
            linestyles="dashed",
            alpha=0.26,
            zorder=1,
            label="-Phase measured-to-fit difference",
        )
        self.phase_excluded_residual_artist = self._line_collection_class(
            [],
            colors="#8d6e63",
            linewidths=0.7,
            linestyles="dashed",
            alpha=0.1,
            zorder=1,
            label="_nolegend_",
        )
        self.axes.add_collection(self.residual_artist)
        self.axes.add_collection(self.excluded_residual_artist)
        self.phase_axes.add_collection(self.phase_residual_artist)
        self.phase_axes.add_collection(self.phase_excluded_residual_artist)
        magnitude_handles, magnitude_labels = self.axes.get_legend_handles_labels()
        phase_handles, phase_labels = self.phase_axes.get_legend_handles_labels()
        self.axes.legend(
            magnitude_handles + phase_handles,
            magnitude_labels + phase_labels,
            loc="best",
        )

    def _configure_plot_layout(self) -> None:
        self.figure.clear()
        self.phase_axes = None
        self.kk_axes = None
        self._drt_peak_artists = []
        self._drt_peak_sum_artist = None
        self._update_plot_mode_button()
        show_spectrum = self.show_spectrum_var.get()
        if not show_spectrum:
            self.axes = self.figure.add_axes([0.0, 0.0, 0.0, 0.0])
        if show_spectrum and self.show_drt_var.get() and self.show_kk_var.get():
            grid = self.figure.add_gridspec(
                2,
                2,
                width_ratios=[1.55, 1.0],
                height_ratios=[1.0, 0.42],
            )
            self.axes = self.figure.add_subplot(grid[0, 0])
            self.kk_axes = self.figure.add_subplot(grid[1, 0])
            self.drt_axes = self.figure.add_subplot(grid[:, 1])
        elif show_spectrum and self.show_drt_var.get():
            grid = self.figure.add_gridspec(1, 2, width_ratios=[1.55, 1.0])
            self.axes = self.figure.add_subplot(grid[0, 0])
            self.drt_axes = self.figure.add_subplot(grid[0, 1])
        elif show_spectrum and self.show_kk_var.get():
            grid = self.figure.add_gridspec(2, 1, height_ratios=[1.0, 0.42])
            self.axes = self.figure.add_subplot(grid[0, 0])
            self.kk_axes = self.figure.add_subplot(grid[1, 0])
            self.drt_axes = None
        elif show_spectrum:
            self.axes = self.figure.add_subplot(111)
            self.drt_axes = None
        elif self.show_drt_var.get() and self.show_kk_var.get():
            grid = self.figure.add_gridspec(2, 1, height_ratios=[0.42, 1.0])
            self.kk_axes = self.figure.add_subplot(grid[0, 0])
            self.drt_axes = self.figure.add_subplot(grid[1, 0])
        elif self.show_drt_var.get():
            self.drt_axes = self.figure.add_subplot(111)
        elif self.show_kk_var.get():
            self.kk_axes = self.figure.add_subplot(111)
        else:
            self.drt_axes = None
        if self.plot_mode == "bode":
            self._configure_bode_plot()
        else:
            self._configure_nyquist_plot()
        if not show_spectrum:
            self.axes.set_visible(False)
        (self.drt_fit_artist,) = self.axes.plot(
            [], [], "-", color="#00897b", linewidth=1.8, alpha=0.9, label="DRT fit"
        )
        self.drt_phase_fit_artist = None
        (self.ml_range_artist,) = self.axes.plot(
            [], [], "o", markerfacecolor="none", markeredgecolor="#00838f",
            markersize=10, markeredgewidth=1.2, alpha=0.35,
            label="_nolegend_",
        )
        (self.ml_active_artist,) = self.axes.plot(
            [], [], "D", color="#2e7d32", markersize=5, alpha=0.8,
            label="_nolegend_",
        )
        (self.ml_rejected_artist,) = self.axes.plot(
            [], [], "x", color="#ef6c00", markersize=6, alpha=0.75,
            label="_nolegend_",
        )
        (self.ml_model_artist,) = self.axes.plot(
            [], [], "--", color="#d81b60", linewidth=2.0, alpha=0.9,
            label="_nolegend_",
        )
        self.ml_residual_artist = self._line_collection_class(
            [], colors="#d81b60", linewidths=0.8, linestyles="dotted",
            alpha=0.45, zorder=1, label="_nolegend_",
        )
        self.axes.add_collection(self.ml_residual_artist)
        self.ml_phase_active_artist = None
        self.ml_phase_rejected_artist = None
        self.ml_phase_model_artist = None
        self.ml_phase_residual_artist = None
        if self.phase_axes is not None:
            (self.drt_phase_fit_artist,) = self.phase_axes.plot(
                [], [], "-", color="#00897b", linewidth=1.6, alpha=0.9, label="-Phase DRT fit"
            )
        if self.phase_axes is not None:
            (self.ml_phase_active_artist,) = self.phase_axes.plot(
                [], [], "D", color="#388e3c", markersize=4, alpha=0.75,
                label="_nolegend_",
            )
            (self.ml_phase_rejected_artist,) = self.phase_axes.plot(
                [], [], "x", color="#f57c00", markersize=5, alpha=0.7,
                label="_nolegend_",
            )
            (self.ml_phase_model_artist,) = self.phase_axes.plot(
                [], [], "--", color="#c2185b", linewidth=1.7, alpha=0.9,
                label="_nolegend_",
            )
            self.ml_phase_residual_artist = self._line_collection_class(
                [], colors="#c2185b", linewidths=0.7, linestyles="dotted",
                alpha=0.4, zorder=1, label="_nolegend_",
            )
            self.phase_axes.add_collection(self.ml_phase_residual_artist)
        self.axes.legend(loc="best")
        if self.phase_axes is not None:
            self.phase_axes.legend(loc="best")
        self.axes.axhline(
            0.0, color="#444444", linewidth=1.2, alpha=0.85, zorder=0
        )
        if self.phase_axes is not None:
            self.phase_axes.axhline(
                0.0, color="#444444", linewidth=1.2, alpha=0.85, zorder=0
            )
        if self.kk_axes is not None:
            self.kk_axes.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.5)
            self.kk_axes.axhline(
                0.0, color="#444444", linewidth=1.2, alpha=0.85, zorder=0
            )
            self.kk_axes.set_xscale("log")
            self.kk_axes.grid(True, alpha=0.2)
            self.kk_axes.set_xlabel("Frequency / Hz")
            self.kk_axes.set_ylabel("KK residual / %")
            (self.kk_real_artist,) = self.kk_axes.plot(
                [], [], "o-", color="#00897b", markersize=3, linewidth=1.0, label="Real"
            )
            (self.kk_imag_artist,) = self.kk_axes.plot(
                [], [], "s-", color="#8e24aa", markersize=2.8, linewidth=1.0, label="Imag"
            )
            self.kk_axes.legend(loc="best")
        else:
            self.kk_real_artist = None
            self.kk_imag_artist = None
        if self.drt_axes is not None:
            self.drt_axes.set_xscale("log")
            self.drt_axes.set_xlabel("Tau / s")
            self.drt_axes.set_ylabel("Gamma / Ohm")
            self.drt_axes.grid(True, alpha=0.25)
            self.drt_axes.axhline(
                0.0, color="#444444", linewidth=1.2, alpha=0.85, zorder=0
            )
            (self.drt_artist,) = self.drt_axes.plot(
                [], [], "-", color="#6a1b9a", linewidth=1.8, alpha=0.9, label="Ridge DRT"
            )
            self.drt_axes.legend(loc="best")
        else:
            self.drt_artist = None
        self._update_legend_visibility()
        self.zoom_selector = self._rectangle_selector_class(
            self.axes,
            self._on_zoom_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )
        self.zoom_selector.set_active(not self.point_toggle_mode)
        self.edit_selector = self._rectangle_selector_class(
            self.axes,
            self._on_edit_area_select,
            useblit=True,
            button=[3],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )
        self.edit_selector.set_active(self.point_toggle_mode)
        self.canvas.draw_idle()

    def _build_explorer(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, padding=6)
        explorer_header = ttk.Frame(group)
        ttk.Label(explorer_header, text="Spectra explorer").pack(side=tk.LEFT)
        self.natural_sort_var = tk.BooleanVar(value=False)
        self.edit_metadata_button = ttk.Button(
            explorer_header,
            text="Edit metadata…",
            command=self.edit_metadata_column_from_clipboard,
        )
        self.edit_metadata_button.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Checkbutton(
            explorer_header,
            text="Natural sort",
            variable=self.natural_sort_var,
            command=self._toggle_natural_sort,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            explorer_header,
            text="Restore",
            command=self.restore_explorer_column_order,
        ).pack(side=tk.LEFT, padx=(8, 0))
        group.configure(labelwidget=explorer_header)
        group.pack(fill=tk.BOTH, expand=True)
        group.columnconfigure(0, weight=1)
        group.rowconfigure(0, weight=1)
        group.rowconfigure(1, weight=0)
        columns = (
            "fitted",
            "drt",
            "model",
            "source",
            "cycle",
            "potential",
            "current",
            "time",
            "points",
            "f_min",
            "f_max",
        )
        self.explorer = ttk.Treeview(
            group,
            columns=columns,
            show="headings",
            selectmode="extended",
            height=7,
        )
        self._explorer_headings = {
            "fitted": "Fitted",
            "drt": "DRT",
            "model": "EEC model",
            "source": "Source file",
            "cycle": "Cycle",
            "potential": "Ecell_V",
            "current": "I_mA",
            "time": "time/s",
            "points": "Points",
            "f_min": "fmin_act_Hz",
            "f_max": "fmax_act_Hz",
        }
        self._explorer_attributes = {
            "fitted": None,
            "drt": None,
            "model": None,
            "source": None,
            "cycle": "cycle",
            "potential": "potential_v",
            "current": "current_ma",
            "time": "time_s",
            "points": "point_count",
            "f_min": "minimum_frequency_hz",
            "f_max": "maximum_frequency_hz",
        }
        self._explorer_sort_reverse: dict[str, bool] = {}
        self._explorer_sort_columns: list[tuple[str, bool]] = []
        self._explorer_selected_column = "cycle"
        self._explorer_header_drag: dict[str, object] | None = None
        widths = {
            "fitted": 62,
            "drt": 48,
            "model": 220,
            "source": 190,
            "cycle": 65,
            "potential": 105,
            "current": 110,
            "time": 95,
            "points": 65,
            "f_min": 125,
            "f_max": 125,
        }
        for column in columns:
            self.explorer.heading(
                column,
                text=self._explorer_headings[column],
                command=lambda selected=column: self._sort_explorer(selected),
            )
            anchor = tk.W if column == "source" else tk.E
            self.explorer.column(
                column, width=widths[column], minwidth=55, anchor=anchor, stretch=False
            )
        self.explorer.configure(displaycolumns=self._explorer_display_columns())
        scrollbar = ttk.Scrollbar(
            group, orient=tk.VERTICAL, command=self.explorer.yview
        )
        self.explorer_xscrollbar = ttk.Scrollbar(
            group, orient=tk.HORIZONTAL, command=self.explorer.xview
        )
        self.explorer.configure(
            yscrollcommand=scrollbar.set,
            xscrollcommand=self.explorer_xscrollbar.set,
        )
        self.explorer.grid(row=0, column=0, sticky="nsew")
        self.explorer.tag_configure(
            "current_row",
            background="#dff3df",
            font=("Segoe UI", 9, "bold"),
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.explorer_xscrollbar.grid(row=1, column=0, sticky="ew")
        self.explorer.bind(
            "<Configure>",
            lambda _event: self._schedule_explorer_horizontal_scrollbar_update(),
            add="+",
        )
        self.explorer.bind("<Button-1>", self._on_explorer_click, add="+")
        self.explorer.bind(
            "<Shift-Double-Button-1>", self._on_explorer_shift_double_click, add="+"
        )
        self.explorer.bind("<Double-Button-1>", self._on_explorer_double_click, add="+")
        self.explorer.bind("<Button-1>", self._on_explorer_heading_click, add="+")
        self.explorer.bind("<<TreeviewSelect>>", self._select_explorer_spectrum)
        self.explorer.bind(
            "<B1-Motion>",
            self._on_explorer_header_drag_motion,
            add="+",
        )
        self.explorer.bind(
            "<ButtonRelease-1>",
            self._on_explorer_header_release,
            add="+",
        )
        self.explorer.bind("<Up>", lambda event: self._on_explorer_arrow(event, -1))
        self.explorer.bind("<Down>", lambda event: self._on_explorer_arrow(event, 1))
        self.explorer.bind("<Control-a>", self.select_all_spectra)

        explorer_actions = ttk.Frame(group)
        explorer_actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        explorer_actions.columnconfigure(0, weight=1)
        explorer_actions.columnconfigure(1, weight=1)
        explorer_actions.columnconfigure(2, weight=1)
        explorer_actions.columnconfigure(3, weight=1)
        self.delete_spectrum_button = ttk.Button(
            explorer_actions,
            text="Delete selected spectra",
            command=self.delete_selected_spectrum,
        )
        self.delete_spectrum_button.grid(row=0, column=0, sticky="ew")
        self.plot_selected_button = ttk.Button(
            explorer_actions,
            text="Plot selected spectra",
            command=self.plot_selected_spectra,
        )
        self.plot_selected_button.grid(row=0, column=1, padx=(6, 0), sticky="ew")
        self.plot_selected_drts_button = ttk.Button(
            explorer_actions,
            text="Plot selected DRTs",
            command=self.plot_selected_drts,
        )
        self.plot_selected_drts_button.grid(row=0, column=2, padx=(6, 0), sticky="ew")
        self.plot_three_electrode_button = ttk.Button(
            explorer_actions,
            text="Plot cell/WE/CE",
            command=self.plot_three_electrode_spectra,
        )
        self.plot_three_electrode_button.grid(
            row=0, column=3, padx=(6, 0), sticky="ew"
        )
        self.plot_fit_parameters_button = ttk.Button(
            explorer_actions,
            text="Fit parameters explorer",
            command=self.open_fit_parameters_explorer,
        )
        self.plot_fit_parameters_button.grid(
            row=1, column=0, columnspan=2, padx=(0, 3), pady=(6, 0), sticky="ew"
        )
        self.plot_drt_parameters_button = ttk.Button(
            explorer_actions,
            text="DRT parameters explorer",
            command=self.open_drt_parameters_explorer,
        )
        self.plot_drt_parameters_button.grid(
            row=1, column=2, columnspan=2, padx=(3, 0), pady=(6, 0), sticky="ew"
        )
        self.explorer_selection_var = tk.StringVar(value="0 spectra selected")
        ttk.Label(
            group,
            textvariable=self.explorer_selection_var,
            anchor="w",
        ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        self._schedule_explorer_horizontal_scrollbar_update()

    def _schedule_explorer_horizontal_scrollbar_update(self) -> None:
        if not hasattr(self, "explorer_xscrollbar"):
            return
        self.root.after_idle(self._update_explorer_horizontal_scrollbar)

    def _update_explorer_horizontal_scrollbar(self) -> None:
        if not hasattr(self, "explorer_xscrollbar"):
            return
        first, last = self.explorer.xview()
        if first <= 0.0 and last >= 1.0:
            self.explorer_xscrollbar.grid_remove()
        else:
            self.explorer_xscrollbar.grid()

    def _explorer_base_columns(self) -> tuple[str, ...]:
        return (
            "fitted",
            "drt",
            "model",
            "source",
            "cycle",
            "potential",
            "current",
            "time",
            "points",
            "f_min",
            "f_max",
        )

    def _explorer_columns(self) -> list[str]:
        return [*self._explorer_base_columns(), *self._custom_metadata_columns]

    def _explorer_display_columns(self) -> list[str]:
        hidden = set(self._explorer_hidden_columns_preference)
        available = [
            column for column in self._explorer_columns() if column not in hidden
        ]
        preferred = self._explorer_current_column_order
        if preferred is None:
            preferred = self._explorer_column_order_preference
        ordered = [column for column in preferred if column in available]
        unlisted = [column for column in available if column not in ordered]
        if self._explorer_new_columns_position == "beginning":
            ordered = [*unlisted, *ordered]
        else:
            ordered.extend(unlisted)
        return ordered

    def _apply_explorer_column_order(self, order: list[str] | None = None) -> None:
        available = self._explorer_columns()
        if order is not None:
            self._explorer_current_column_order = [
                column for column in order if column in available
            ]
        display_columns = self._explorer_display_columns()
        self.explorer.configure(displaycolumns=display_columns)
        self._schedule_explorer_horizontal_scrollbar_update()

    def restore_explorer_column_order(self) -> None:
        self._explorer_current_column_order = None
        self._apply_explorer_column_order()

    def _sync_custom_metadata_columns(self) -> None:
        columns: list[str] = []
        for dataset_id in self._dataset_order:
            loaded = self.loaded_projects[dataset_id]
            is_ewe_data = (
                loaded.state.control == "working"
                and "ewe_ece_v" not in loaded.dataframe.columns
            )
            for spectrum in loaded.spectra:
                for name in spectrum.custom_metadata:
                    if str(name).casefold().startswith("_ml_"):
                        continue
                    if is_ewe_data and name in {
                        WORKING_POTENTIAL_COLUMN,
                        COUNTER_POTENTIAL_COLUMN,
                    }:
                        continue
                    if name in {
                        "Ecell_V",
                        "I_mA",
                        "time_s",
                        "time/s",
                        "fmin_Hz",
                        "fmax_Hz",
                        "fmin_act_Hz",
                        "fmax_act_Hz",
                    }:
                        continue
                    if name not in columns:
                        columns.append(name)
        self._custom_metadata_columns = columns

    def _refresh_explorer_schema(self) -> None:
        if not hasattr(self, "explorer"):
            return
        # Keep the Treeview's internal column order aligned with the order used
        # when row values are constructed in _populate_explorer().  The visual
        # order belongs in displaycolumns; using it for columns makes values
        # appear under the wrong headings as soon as a custom column is added
        # (especially when an explorer column-order preference is active).
        columns = self._explorer_columns()
        self.explorer.configure(
            columns=columns,
            displaycolumns=self._explorer_display_columns(),
        )
        headings = {
            "fitted": "Fitted",
            "drt": "DRT",
            "model": "EEC model",
            "source": "Source file",
            "cycle": "Cycle",
            "potential": "Voltage (V)",
            "current": "Current (mA)",
            "time": "time/s",
            "points": "Points",
            "f_min": "Min frequency (Hz)",
            "f_max": "Max frequency (Hz)",
        }
        widths = {
            "fitted": 62,
            "drt": 48,
            "model": 220,
            "source": 190,
            "cycle": 65,
            "potential": 105,
            "current": 110,
            "time": 95,
            "points": 65,
            "f_min": 125,
            "f_max": 125,
        }
        for column in self._custom_metadata_columns:
            headings[column] = column
            widths[column] = max(110, min(220, 12 * max(len(column), 6)))
        self._explorer_headings = headings
        for column in columns:
            label = headings[column]
            self.explorer.heading(
                column,
                text=label,
                command=lambda selected=column: self._sort_explorer(selected),
            )
            anchor = (
                tk.W
                if column == "source" or column in self._custom_metadata_columns
                else tk.E
            )
            if column in {"fitted", "drt"}:
                anchor = tk.CENTER
            self.explorer.column(
                column, width=widths[column], minwidth=55, anchor=anchor, stretch=False
            )
        self._schedule_explorer_horizontal_scrollbar_update()

    @staticmethod
    def _format_explorer_value(value, column: str | None = None) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return ""
        numeric_column = column in {"potential", "current", "time", "f_min", "f_max"}
        if column is not None:
            numeric_column |= any(
                keyword in column.casefold()
                for keyword in ("potential", "voltage", "frequency")
            )
        if numeric_column:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                pass
            else:
                if np.isfinite(numeric_value):
                    return f"{numeric_value:.5g}"
        return str(value)

    def _explorer_value(
        self,
        loaded: LoadedProject,
        spectrum: SpectrumMetadata,
        column: str,
    ):
        if column == "fitted":
            cycle = loaded.state.cycles.get(spectrum.cycle)
            return "Yes" if cycle is not None and cycle.fit_parameters is not None else "No"
        if column == "drt":
            cycle = loaded.state.cycles.get(spectrum.cycle)
            if cycle is None:
                return "N"
            if cycle.saved_hybrid_tau_s is not None and cycle.saved_hybrid_gamma_ohm is not None:
                return "H"
            if cycle.saved_ridge_tau_s is not None and cycle.saved_ridge_gamma_ohm is not None:
                return "R"
            return "N"
        if column == "model":
            cycle = loaded.state.cycles.get(spectrum.cycle)
            return cycle.model(loaded.state.circuit) if cycle is not None else loaded.state.circuit
        if column == "source":
            return loaded.state.source_path.name
        if column == "cycle":
            return spectrum.cycle
        if column == "potential":
            cycle = loaded.state.cycles.get(spectrum.cycle)
            return cycle.potential_v if cycle is not None else spectrum.potential_v
        if column == "current":
            cycle = loaded.state.cycles.get(spectrum.cycle)
            return cycle.current_ma if cycle is not None else spectrum.current_ma
        if column == "time":
            cycle = loaded.state.cycles.get(spectrum.cycle)
            return cycle.time_s if cycle is not None else spectrum.time_s
        if column == "points":
            return spectrum.point_count
        if column == "f_min":
            return self._explorer_frequency_range(loaded, spectrum)[0]
        if column == "f_max":
            return self._explorer_frequency_range(loaded, spectrum)[1]
        return spectrum.custom_metadata.get(column)

    @staticmethod
    def _explorer_frequency_range(
        loaded: LoadedProject,
        spectrum: SpectrumMetadata,
    ) -> tuple[float, float]:
        cycle = loaded.state.cycles.get(spectrum.cycle)
        frequency_window = (
            cycle.frequency_window
            if cycle is not None and cycle.frequency_window is not None
            else loaded.state.all_frequency_window
        )
        if frequency_window is not None:
            return float(frequency_window[0]), float(frequency_window[1])
        return spectrum.minimum_frequency_hz, spectrum.maximum_frequency_hz

    def _populate_explorer(self) -> None:
        self._sync_custom_metadata_columns()
        self._refresh_explorer_schema()
        self.explorer.delete(*self.explorer.get_children())
        self._explorer_rows.clear()
        self._explorer_lookup.clear()
        self._explorer_anchor_item = None
        self._explorer_primary_item = None
        self.drt_peak_parameters = []
        self._drt_peak_cycle_key = None
        self._drt_peak_drag = None
        self._drt_aux_parameter_limits = {}
        for dataset_index, dataset_id in enumerate(self._dataset_order):
            loaded = self.loaded_projects[dataset_id]
            for spectrum in loaded.spectra:
                item = f"dataset_{dataset_index}_cycle_{spectrum.cycle}"
                self._explorer_rows[item] = (dataset_id, loaded, spectrum)
                self._explorer_lookup[(dataset_id, spectrum.cycle)] = item
                values = [
                    self._format_explorer_value(
                        self._explorer_value(loaded, spectrum, column), column
                    )
                    for column in self._explorer_columns()
                ]
                self.explorer.insert(
                    "",
                    tk.END,
                    iid=item,
                    values=values,
                )
        self._refresh_explorer_focus_tag()
        self._update_explorer_selection_status()
        if self._explorer_sort_columns:
            self._apply_explorer_sort()
        self._schedule_explorer_horizontal_scrollbar_update()

    def _sort_explorer(self, column: str) -> None:
        if not self._explorer_rows:
            return
        reverse = self._explorer_sort_reverse.get(column, False)
        self._explorer_selected_column = column
        self._explorer_sort_columns = [(column, reverse)]
        self._apply_explorer_sort()
        self._explorer_sort_reverse[column] = not reverse

    def _on_explorer_heading_click(self, event):
        if self.explorer.identify_region(event.x, event.y) != "heading":
            return
        column_index = self.explorer.identify_column(event.x)
        columns = list(self._explorer_display_columns())
        if not column_index.startswith("#"):
            return "break"
        index = int(column_index[1:]) - 1
        if index < 0 or index >= len(columns):
            return "break"
        column = columns[index]
        self._explorer_header_drag = {
            "column": column,
            "start_x": event.x,
            "state": event.state,
            "moved": False,
        }
        return "break"

    def _on_explorer_header_drag_motion(self, event):
        drag = self._explorer_header_drag
        if drag is None:
            return
        if abs(event.x - int(drag["start_x"])) < 5:
            return
        columns = self._explorer_display_columns()
        target_index_text = self.explorer.identify_column(event.x)
        if not target_index_text.startswith("#"):
            return
        target_index = int(target_index_text[1:]) - 1
        if not 0 <= target_index < len(columns):
            return
        source_column = str(drag["column"])
        source_index = columns.index(source_column)
        if source_index == target_index:
            return
        columns.insert(target_index, columns.pop(source_index))
        self._explorer_current_column_order = columns
        drag["moved"] = True
        self._apply_explorer_column_order(columns)

    def _on_explorer_header_release(self, event):
        drag = self._explorer_header_drag
        self._explorer_header_drag = None
        if drag is None:
            self._schedule_explorer_horizontal_scrollbar_update()
            return
        if not bool(drag["moved"]):
            self._sort_explorer_heading(
                str(drag["column"]), int(drag["state"])
            )
        self._schedule_explorer_horizontal_scrollbar_update()

    def _sort_explorer_heading(self, column: str, state: int) -> None:
        if not self._explorer_rows:
            return
        shift_pressed = bool(state & 0x0001)
        if not shift_pressed:
            reverse = self._explorer_sort_reverse.get(column, False)
            self._explorer_sort_columns = [(column, reverse)]
            self._explorer_selected_column = column
            self._apply_explorer_sort()
            self._explorer_sort_reverse[column] = not reverse
            return
        existing = {name: reverse for name, reverse in self._explorer_sort_columns}
        if column in existing:
            reverse = not existing[column]
            self._explorer_sort_columns = [
                (name, reverse if name == column else current_reverse)
                for name, current_reverse in self._explorer_sort_columns
            ]
        else:
            self._explorer_sort_columns.append((column, False))
            reverse = False
        self._explorer_selected_column = column
        self._apply_explorer_sort()
        self._explorer_sort_reverse[column] = not reverse

    def _toggle_natural_sort(self) -> None:
        if not self._explorer_rows:
            return
        if not self._explorer_sort_columns:
            self._explorer_sort_columns = [
                (
                    self._explorer_selected_column,
                    self._explorer_sort_reverse.get(
                        self._explorer_selected_column, False
                    ),
                )
            ]
        self._apply_explorer_sort()

    def _apply_explorer_sort(self) -> None:
        ordered = list(self._explorer_rows.items())
        for column, reverse in reversed(self._explorer_sort_columns):
            ordered.sort(
                key=lambda item, selected_column=column: self._explorer_sort_key(
                    item[1][1], item[1][2], selected_column
                ),
                reverse=reverse,
            )
        for index, (item, _row) in enumerate(ordered):
            self.explorer.move(item, "", index)
        priorities = {
            column: (position, reverse)
            for position, (column, reverse) in enumerate(self._explorer_sort_columns, 1)
        }
        for name, label in self._explorer_headings.items():
            marker = ""
            if name in priorities:
                position, reverse = priorities[name]
                marker = f" {position}{' ▼' if reverse else ' ▲'}"
            self.explorer.heading(name, text=f"{label}{marker}")

    def _explorer_sort_key(
        self,
        loaded: LoadedProject,
        spectrum: SpectrumMetadata,
        column: str,
    ):
        if column == "fitted":
            cycle = loaded.state.cycles.get(spectrum.cycle)
            return (0, 1 if cycle is not None and cycle.fit_parameters is not None else 0)
        elif column == "drt":
            cycle = loaded.state.cycles.get(spectrum.cycle)
            if cycle is None:
                return (0, 0)
            if cycle.saved_hybrid_tau_s is not None and cycle.saved_hybrid_gamma_ohm is not None:
                return (0, 2)
            if cycle.saved_ridge_tau_s is not None and cycle.saved_ridge_gamma_ohm is not None:
                return (0, 1)
            return (0, 0)
        elif column == "model":
            cycle = loaded.state.cycles.get(spectrum.cycle)
            value = cycle.model(loaded.state.circuit) if cycle is not None else loaded.state.circuit
        elif column == "source":
            value = loaded.state.source_path.name
        elif column == "cycle":
            value = spectrum.cycle
        elif column == "potential":
            value = spectrum.potential_v
        elif column == "current":
            value = spectrum.current_ma
        elif column == "time":
            value = spectrum.time_s
        elif column == "points":
            value = spectrum.point_count
        elif column == "f_min":
            value = self._explorer_frequency_range(loaded, spectrum)[0]
        elif column == "f_max":
            value = self._explorer_frequency_range(loaded, spectrum)[1]
        else:
            value = spectrum.custom_metadata.get(column)
        if value is None:
            return (2, "")
        if isinstance(value, (int, float, np.integer, np.floating)):
            if isinstance(value, float) and np.isnan(value):
                return (2, "")
            return (0, float(value))
        text = str(value).strip()
        try:
            number = float(text)
        except ValueError:
            if self.natural_sort_var.get():
                return (1, natsort_keygen(alg=ns.IGNORECASE)(text))
            return (1, text.casefold())
        if np.isfinite(number):
            return (0, number)
        if self.natural_sort_var.get():
            return (1, natsort_keygen(alg=ns.IGNORECASE)(text))
        return (2, "")

    def _refresh_explorer_values(self) -> None:
        if not hasattr(self, "explorer"):
            return
        for item, (_dataset_id, loaded, spectrum) in self._explorer_rows.items():
            if not self.explorer.exists(item):
                continue
            values = [
                self._format_explorer_value(
                    self._explorer_value(loaded, spectrum, column), column
                )
                for column in self._explorer_columns()
            ]
            self.explorer.item(item, values=values)

    def _select_explorer_spectrum(self, _event=None) -> None:
        if self._suspend_explorer_select or self.busy or self.state is None:
            return
        selected = self.explorer.selection()
        if not selected:
            return
        primary = self.explorer.focus()
        if primary not in selected:
            primary = (
                self._explorer_primary_item
                if self._explorer_primary_item in selected
                else selected[-1]
            )
        self._explorer_primary_item = primary
        self._explorer_anchor_item = primary
        self._refresh_explorer_focus_tag()
        self._activate_explorer_item(primary)

    def _on_explorer_arrow(self, event, direction: int):
        shift_pressed = bool(event.state & 0x0001)
        control_pressed = bool(event.state & 0x0004)
        self.change_cycle(
            direction,
            preserve_selection=shift_pressed,
            focus_only=control_pressed and not shift_pressed,
        )
        return "break"

    def select_all_spectra(self, _event=None):
        if self.busy or self.state is None or not hasattr(self, "explorer"):
            return "break"
        items = list(self.explorer.get_children(""))
        if not items:
            return "break"
        primary = (
            self._explorer_primary_item
            if self._explorer_primary_item in items
            else items[0]
        )
        self._set_explorer_selection(items, primary=primary)
        return "break"

    def open_export_menu(self, _event=None):
        if not hasattr(self, "export_menu"):
            return "break"
        x_position = self.root.winfo_rootx() + max(self.root.winfo_width() // 2, 240)
        y_position = self.root.winfo_rooty() + 35
        try:
            self.export_menu.tk_popup(x_position, y_position)
        finally:
            self.export_menu.grab_release()
        return "break"

    def open_procedure_builder(self) -> None:
        existing = getattr(self, "procedure_builder_popup", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return
        popup = tk.Toplevel(self.root)
        self.procedure_builder_popup = popup
        popup.title("Procedure builder")
        popup.geometry("860x760")
        popup.minsize(700, 600)
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(5, weight=1)

        def close_popup() -> None:
            self.procedure_builder_popup = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", close_popup)
        steps: list[dict[str, str]] = []
        action_names = [
            "Select spectra by metadata",
            "Sort spectra by metadata",
            "Set frequency limits for selected",
            "Find outliers in selected",
            "Choose EEC model",
            "Go to spectrum…",
            "Initial values",
            "Fit current spectrum",
            "Fit selected",
            "Refine fits",
            "Find deterministic outliers",
            "Auto model (Hybrid DRT)",
            "Batch fit selected",
        ]
        action_help = {
            "Select spectra by metadata": "column=value, e.g. cycle=14",
            "Sort spectra by metadata": "column, e.g. time",
            "Set frequency limits for selected": "minimum,maximum",
            "Find outliers in selected": "threshold (blank uses current value)",
            "Choose EEC model": "circuit, e.g. R0-L0-p(R1,CPE1)",
            "Go to spectrum…": "position in selected spectra: first, middle, last, or a number",
            "Initial values": "no parameters",
            "Fit current spectrum": "no parameters",
            "Fit selected": "fits selected spectra using each spectrum's model and initial parameters (no parameters)",
            "Refine fits": "refines the existing fits for selected spectra using the current refinement settings",
            "Find deterministic outliers": "detects and deactivates deterministic outliers in selected spectra using the current threshold",
            "Auto model (Hybrid DRT)": "selects a Hybrid DRT discrete model for selected spectra (no parameters)",
            "Batch fit selected": "direction: up, down, or up and down",
        }
        top = ttk.Frame(popup, padding=10)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Block name").grid(row=0, column=0, sticky="w")
        name_var = tk.StringVar(value="Block 1")
        ttk.Entry(top, textvariable=name_var).grid(row=0, column=1, padx=8, sticky="ew")
        block_var = tk.StringVar()
        block_box = ttk.Combobox(
            top,
            textvariable=block_var,
            values=sorted(self.procedure_blocks),
            state="readonly",
            width=22,
        )
        block_box.grid(row=0, column=2, sticky="e")

        add_frame = ttk.LabelFrame(popup, text="Add action", padding=8)
        add_frame.grid(row=1, column=0, padx=10, sticky="ew")
        add_frame.columnconfigure(1, weight=1)
        action_var = tk.StringVar(value=action_names[0])
        action_box = ttk.Combobox(
            add_frame, textvariable=action_var, values=action_names, state="readonly"
        )
        action_box.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        parameter_var = tk.StringVar()
        parameter_controls = ttk.Frame(add_frame)
        parameter_controls.grid(row=0, column=1, sticky="ew")
        parameter_controls.columnconfigure(0, weight=1)
        parameter_controls.columnconfigure(1, weight=1)
        parameter_controls.columnconfigure(2, weight=1)
        parameter_controls.columnconfigure(3, weight=1)
        parameter_entry = ttk.Combobox(
            parameter_controls, textvariable=parameter_var, state="normal"
        )
        select_column_var = tk.StringVar(value="All")
        select_operator_var = tk.StringVar(value="=")
        select_value_var = tk.StringVar()
        batch_direction_var = tk.StringVar(value="up")
        refine_z_threshold_var = tk.StringVar(value=self.refine_z_threshold_var.get())
        refine_max_iterations_var = tk.StringVar(value=self.refine_max_iterations_var.get())
        parameterless_actions = {
            "Initial values",
            "Fit current spectrum",
            "Fit selected",
            "Auto model (Hybrid DRT)",
        }

        def configure_parameter_input(action: str) -> None:
            for child in parameter_controls.winfo_children():
                child.grid_remove()
            if action == "Select spectra by metadata":
                ttk.Combobox(
                    parameter_controls,
                    textvariable=select_column_var,
                    values=[
                        "All",
                        *[
                            self._explorer_headings.get(column, column)
                            for column in self._explorer_columns()
                        ],
                    ],
                    state="readonly",
                ).grid(row=0, column=0, padx=(0, 3), sticky="ew")
                ttk.Combobox(
                    parameter_controls,
                    textvariable=select_operator_var,
                    values=("=", "<", ">", "<=", ">="),
                    state="readonly",
                    width=4,
                ).grid(row=0, column=1, padx=3, sticky="ew")
                ttk.Entry(parameter_controls, textvariable=select_value_var).grid(
                    row=0, column=2, padx=(3, 0), sticky="ew"
                )
            elif action == "Batch fit selected":
                ttk.Combobox(
                    parameter_controls,
                    textvariable=batch_direction_var,
                    values=("up", "down", "up and down"),
                    state="readonly",
                ).grid(row=0, column=0, columnspan=3, sticky="ew")
            elif action == "Refine fits":
                ttk.Label(parameter_controls, text="Z threshold").grid(
                    row=0, column=0, padx=(0, 3), sticky="w"
                )
                ttk.Entry(
                    parameter_controls, textvariable=refine_z_threshold_var, width=10
                ).grid(row=0, column=1, padx=3, sticky="ew")
                ttk.Label(parameter_controls, text="Max iterations").grid(
                    row=0, column=2, padx=(8, 3), sticky="w"
                )
                ttk.Entry(
                    parameter_controls, textvariable=refine_max_iterations_var, width=10
                ).grid(row=0, column=3, padx=(3, 0), sticky="ew")
            elif action in parameterless_actions:
                return
            else:
                parameter_entry.grid(row=0, column=0, columnspan=3, sticky="ew")

        def action_parameter() -> str:
            if action_var.get() == "Select spectra by metadata":
                return self._serialize_procedure_selection(
                    select_column_var.get(), select_operator_var.get(), select_value_var.get()
                )
            if action_var.get() == "Batch fit selected":
                return batch_direction_var.get()
            if action_var.get() == "Refine fits":
                return f"{refine_z_threshold_var.get().strip()},{refine_max_iterations_var.get().strip()}"
            if action_var.get() in parameterless_actions:
                return ""
            return parameter_var.get().strip()

        configure_parameter_input(action_var.get())
        help_var = tk.StringVar(value=action_help.get(action_names[0], ""))
        ttk.Label(add_frame, textvariable=help_var, width=38).grid(
            row=0, column=2, padx=(8, 0), sticky="w"
        )

        arguments_frame = ttk.LabelFrame(popup, text="Procedure arguments (edit directly)", padding=8)
        arguments_frame.grid(row=2, column=0, padx=10, pady=(8, 0), sticky="ew")

        procedure_entries: list[dict[str, object]] = []
        procedure_name_var = tk.StringVar(value="Procedure 1")
        procedure_block_var = tk.StringVar()
        procedure_var = tk.StringVar()
        procedure_frame = ttk.LabelFrame(
            popup, text="Build procedure from blocks", padding=8
        )
        procedure_frame.grid(row=3, column=0, padx=10, pady=(8, 0), sticky="ew")
        procedure_frame.columnconfigure(1, weight=1)
        procedure_frame.columnconfigure(3, weight=1)
        procedure_frame.columnconfigure(5, weight=1)
        ttk.Label(procedure_frame, text="Procedure name").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(procedure_frame, textvariable=procedure_name_var, width=18).grid(
            row=0, column=1, padx=(6, 12), sticky="ew"
        )
        ttk.Label(procedure_frame, text="Block").grid(row=0, column=2, sticky="w")
        procedure_block_box = ttk.Combobox(
            procedure_frame, textvariable=procedure_block_var, state="readonly", width=20
        )
        procedure_block_box.grid(row=0, column=3, padx=6, sticky="ew")
        ttk.Label(procedure_frame, text="Saved procedure").grid(
            row=0, column=4, padx=(12, 4), sticky="w"
        )
        procedure_box = ttk.Combobox(
            procedure_frame,
            textvariable=procedure_var,
            values=sorted(self.procedures),
            state="readonly",
            width=20,
        )
        procedure_box.grid(row=0, column=5, sticky="ew")
        procedure_list = tk.Listbox(
            procedure_frame, height=4, exportselection=False, activestyle="dotbox"
        )
        procedure_list.grid(row=1, column=0, columnspan=4, pady=(6, 0), sticky="ew")

        def render_procedure_entries() -> None:
            procedure_list.delete(0, tk.END)
            for index, entry in enumerate(procedure_entries, 1):
                block_name = str(entry.get("block", ""))
                entry_steps = entry.get("steps", [])
                procedure_list.insert(
                    tk.END, f"{index}. {block_name} ({len(entry_steps)} actions)"
                )

        def add_procedure_block() -> None:
            block_name = procedure_block_var.get().strip()
            if block_name not in self.procedure_blocks:
                messagebox.showwarning(
                    "No block selected", "Save a block and select it first.", parent=popup
                )
                return
            procedure_entries.append(
                {
                    "block": block_name,
                    "steps": copy.deepcopy(self.procedure_blocks[block_name]),
                }
            )
            render_procedure_entries()
            procedure_list.selection_set(tk.END)

        def remove_procedure_block() -> None:
            selection = procedure_list.curselection()
            if selection:
                procedure_entries.pop(int(selection[0]))
                render_procedure_entries()

        def edit_procedure_block() -> None:
            selection = procedure_list.curselection()
            if not selection:
                return
            entry = procedure_entries[int(selection[0])]
            name_var.set(str(entry.get("block", "")))
            steps[:] = copy.deepcopy(entry.get("steps", []))
            render_steps()

        def update_procedure_block() -> None:
            selection = procedure_list.curselection()
            if not selection:
                return
            entry = procedure_entries[int(selection[0])]
            entry["steps"] = copy.deepcopy(steps)
            render_procedure_entries()
            procedure_list.selection_set(int(selection[0]))
            self._autosave_procedures_to_project()

        procedure_buttons = ttk.Frame(procedure_frame)
        procedure_buttons.grid(row=2, column=0, columnspan=4, pady=(6, 0), sticky="w")
        ttk.Button(procedure_buttons, text="Add block", command=add_procedure_block).pack(
            side=tk.LEFT
        )
        ttk.Button(
            procedure_buttons, text="Edit selected", command=edit_procedure_block
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            procedure_buttons, text="Update selected", command=update_procedure_block
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            procedure_buttons, text="Remove block", command=remove_procedure_block
        ).pack(side=tk.LEFT)
        ttk.Button(
            procedure_buttons, text="Save procedure", command=lambda: save_procedure()
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(
            procedure_buttons, text="New procedure", command=lambda: new_procedure()
        ).pack(side=tk.LEFT, padx=4)

        repeat_frame = ttk.LabelFrame(
            popup, text="Repeat block for input values (optional)", padding=8
        )
        repeat_frame.grid(row=4, column=0, padx=10, pady=(8, 0), sticky="ew")
        repeat_frame.columnconfigure(0, weight=1)
        ttk.Label(
            repeat_frame,
            text="Enter one value per line. Use {input} in an action parameter; the block runs once per value.",
        ).grid(row=0, column=0, sticky="w")
        repeat_values_text = tk.Text(repeat_frame, height=3, width=40, wrap=tk.NONE)
        repeat_values_text.grid(row=1, column=0, sticky="ew", pady=(3, 0))

        list_frame = ttk.LabelFrame(popup, text="Actions in procedure", padding=8)
        list_frame.grid(row=5, column=0, padx=10, pady=(8, 0), sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        action_list = tk.Listbox(list_frame, activestyle="dotbox", exportselection=False)
        action_list.grid(row=0, column=0, sticky="nsew")
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=action_list.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        action_list.configure(yscrollcommand=list_scroll.set)

        def render_steps() -> None:
            action_list.delete(0, tk.END)
            for child in arguments_frame.winfo_children():
                child.destroy()
            for index, step in enumerate(steps, 1):
                parameter = step["parameter"]
                suffix = f"  [{parameter}]" if parameter else ""
                action_list.insert(tk.END, f"{index}. {step['action']}{suffix}")
                arguments_frame.columnconfigure(index - 1, weight=1)
                ttk.Label(
                    arguments_frame,
                    text=f"{index}. {step['action']}",
                    wraplength=150,
                    justify=tk.LEFT,
                ).grid(row=0, column=index - 1, padx=4, sticky="w")
                argument_var = tk.StringVar(value=parameter)

                def update_argument(*_args, selected=index - 1, variable=argument_var) -> None:
                    if selected < len(steps):
                        steps[selected]["parameter"] = variable.get().strip()

                argument_var.trace_add("write", update_argument)
                if step["action"] == "Select spectra by metadata":
                    column_text, operator, value = self._parse_procedure_selection(parameter)
                    selection_frame = ttk.Frame(arguments_frame)
                    selection_frame.columnconfigure(0, weight=1)
                    selection_frame.columnconfigure(2, weight=1)
                    selection_column_var = tk.StringVar(value=column_text)
                    selection_operator_var = tk.StringVar(value=operator)
                    selection_value_var = tk.StringVar(value=value)
                    ttk.Combobox(
                        selection_frame,
                        textvariable=selection_column_var,
                        values=[
                            "All",
                            *[
                                self._explorer_headings.get(column, column)
                                for column in self._explorer_columns()
                            ],
                        ],
                        state="readonly",
                    ).grid(row=0, column=0, padx=(0, 2), sticky="ew")
                    ttk.Combobox(
                        selection_frame,
                        textvariable=selection_operator_var,
                        values=("=", "<", ">", "<=", ">="),
                        state="readonly",
                        width=4,
                    ).grid(row=0, column=1, padx=2, sticky="ew")
                    ttk.Entry(selection_frame, textvariable=selection_value_var).grid(
                        row=0, column=2, padx=(2, 0), sticky="ew"
                    )

                    def update_selection_argument(*_args, selected=index - 1) -> None:
                        if selected < len(steps):
                            steps[selected]["parameter"] = self._serialize_procedure_selection(
                                selection_column_var.get(),
                                selection_operator_var.get(),
                                selection_value_var.get(),
                            )

                    for variable in (
                        selection_column_var,
                        selection_operator_var,
                        selection_value_var,
                    ):
                        variable.trace_add("write", update_selection_argument)
                    argument_entry = selection_frame
                elif step["action"] == "Choose EEC model":
                    argument_entry = ttk.Combobox(
                        arguments_frame,
                        textvariable=argument_var,
                        values=self._model_presets,
                        state="normal",
                        width=18,
                    )
                elif step["action"] == "Go to spectrum…":
                    selected_count = len(self.explorer.selection())
                    choices = ["first", "middle", "last"]
                    choices.extend(str(position) for position in range(1, selected_count + 1))
                    argument_entry = ttk.Combobox(
                        arguments_frame,
                        textvariable=argument_var,
                        values=choices,
                        state="normal",
                        width=18,
                    )
                elif step["action"] == "Batch fit selected":
                    direction = parameter.casefold().strip()
                    if direction not in {"up", "down", "up and down"}:
                        direction = "up"
                    argument_var.set(direction)
                    argument_entry = ttk.Combobox(
                        arguments_frame,
                        textvariable=argument_var,
                        values=("up", "down", "up and down"),
                        state="readonly",
                        width=18,
                    )
                elif step["action"] == "Refine fits":
                    values = [part.strip() for part in parameter.split(",", 1)]
                    if len(values) == 1:
                        values.append("")
                    refinement_frame = ttk.Frame(arguments_frame)
                    refinement_frame.columnconfigure(1, weight=1)
                    refinement_frame.columnconfigure(3, weight=1)
                    ttk.Label(refinement_frame, text="Z threshold").grid(
                        row=0, column=0, padx=(0, 3), sticky="w"
                    )
                    refinement_z_var = tk.StringVar(value=values[0])
                    ttk.Entry(
                        refinement_frame, textvariable=refinement_z_var, width=8
                    ).grid(row=0, column=1, padx=(0, 5), sticky="ew")
                    ttk.Label(refinement_frame, text="Max iterations").grid(
                        row=0, column=2, padx=(0, 3), sticky="w"
                    )
                    refinement_iterations_var = tk.StringVar(value=values[1])
                    ttk.Entry(
                        refinement_frame,
                        textvariable=refinement_iterations_var,
                        width=8,
                    ).grid(row=0, column=3, sticky="ew")

                    def update_refinement_argument(
                        *_args, selected=index - 1
                    ) -> None:
                        if selected < len(steps):
                            steps[selected]["parameter"] = (
                                f"{refinement_z_var.get().strip()},"
                                f"{refinement_iterations_var.get().strip()}"
                            )

                    refinement_z_var.trace_add("write", update_refinement_argument)
                    refinement_iterations_var.trace_add("write", update_refinement_argument)
                    argument_entry = refinement_frame
                elif step["action"] in parameterless_actions:
                    argument_entry = None
                else:
                    argument_entry = ttk.Entry(arguments_frame, textvariable=argument_var, width=18)
                if step["action"] == "Sort spectra by metadata":
                    argument_entry.destroy()
                    argument_entry = ttk.Combobox(
                        arguments_frame,
                        textvariable=argument_var,
                        values=[
                            self._explorer_headings.get(column, column)
                            for column in self._explorer_columns()
                        ],
                        state="normal",
                        width=18,
                    )
                if argument_entry is not None:
                    argument_entry.grid(row=1, column=index - 1, padx=4, pady=(3, 0), sticky="ew")
                    argument_entry.bind("<FocusOut>", lambda _event: render_steps())

        def selected_index() -> int | None:
            selection = action_list.curselection()
            return int(selection[0]) if selection else None

        def add_step() -> None:
            steps.append({"action": action_var.get(), "parameter": action_parameter()})
            render_steps()
            action_list.selection_clear(0, tk.END)
            action_list.selection_set(tk.END)

        def remove_step() -> None:
            index = selected_index()
            if index is not None:
                steps.pop(index)
                render_steps()

        def move_step(direction: int) -> None:
            index = selected_index()
            target = index + direction if index is not None else None
            if index is None or target is None or not 0 <= target < len(steps):
                return
            steps[index], steps[target] = steps[target], steps[index]
            render_steps()
            action_list.selection_set(target)

        def new_block() -> None:
            steps.clear()
            name_var.set(f"Block {len(self.procedure_blocks) + 1}")
            render_steps()

        def save_block() -> None:
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Missing name", "Enter a procedure name.", parent=popup)
                return
            self.procedure_blocks[name] = copy.deepcopy(steps)
            block_box.configure(values=sorted(self.procedure_blocks))
            procedure_block_box.configure(values=sorted(self.procedure_blocks))
            block_var.set(name)
            self._autosave_procedures_to_project()
            self._update_status(f"block '{name}' saved in this session")

        def save_procedure() -> None:
            name = procedure_name_var.get().strip()
            if not name:
                messagebox.showerror("Missing name", "Enter a procedure name.", parent=popup)
                return
            if not procedure_entries:
                messagebox.showwarning(
                    "Empty procedure", "Add at least one block first.", parent=popup
                )
                return
            self.procedures[name] = copy.deepcopy(procedure_entries)
            procedure_box.configure(values=sorted(self.procedures))
            procedure_var.set(name)
            self._autosave_procedures_to_project()
            self._update_status(f"procedure '{name}' saved in this session")

        def load_procedure(_event=None) -> None:
            name = procedure_var.get()
            if name not in self.procedures:
                return
            procedure_entries[:] = copy.deepcopy(self.procedures[name])
            procedure_name_var.set(name)
            render_procedure_entries()

        def new_procedure() -> None:
            procedure_entries.clear()
            procedure_name_var.set(f"Procedure {len(self.procedures) + 1}")
            procedure_var.set("")
            render_procedure_entries()

        def load_procedures_from_preferences() -> None:
            self.procedure_blocks = copy.deepcopy(self._procedure_library_blocks)
            self.procedures = copy.deepcopy(self._procedure_library)
            block_box.configure(values=sorted(self.procedure_blocks))
            procedure_block_box.configure(values=sorted(self.procedure_blocks))
            procedure_box.configure(values=sorted(self.procedures))
            self._update_status("blocks and procedures loaded from Preferences")

        def run_procedure() -> None:
            if not procedure_entries:
                messagebox.showwarning(
                    "Empty procedure", "Add at least one block first.", parent=popup
                )
                return
            procedure_steps = [
                step
                for entry in procedure_entries
                for step in copy.deepcopy(entry.get("steps", []))
            ]
            if self.busy:
                self._update_status("wait for the current operation to finish")
                return
            input_values = [
                line.strip()
                for line in repeat_values_text.get("1.0", tk.END).splitlines()
                if line.strip()
            ]
            try:
                run_steps = self._expand_procedure_inputs(procedure_steps, input_values)
            except ValueError as error:
                messagebox.showerror("Invalid repeat inputs", str(error), parent=popup)
                return
            self._procedure_running = True
            self._run_procedure_steps(run_steps, 0)

        def save_to_file() -> None:
            name = name_var.get().strip()
            if name:
                self.procedure_blocks[name] = copy.deepcopy(steps)
            procedure_name = procedure_name_var.get().strip()
            if procedure_name and procedure_entries:
                self.procedures[procedure_name] = copy.deepcopy(procedure_entries)
            if not self.procedure_blocks and not self.procedures:
                messagebox.showwarning("No procedures", "Add and save a procedure first.", parent=popup)
                return
            path = filedialog.asksaveasfilename(
                parent=popup,
                title="Save procedures",
                defaultextension=".eisproc",
                filetypes=[("EIS procedures", "*.eisproc"), ("JSON files", "*.json")],
            )
            if not path:
                return
            payload = {
                "format": "eisfit-procedures-v1",
                "procedure_blocks": self.procedure_blocks,
                "procedures": self.procedures,
                "active_block": name,
                "active_procedure": procedure_name,
            }
            try:
                Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except OSError as error:
                messagebox.showerror("Save procedures failed", str(error), parent=popup)
                return
            block_box.configure(values=sorted(self.procedure_blocks))
            self._update_status(f"procedures saved to {Path(path).name}")

        def load_from_file() -> None:
            path = filedialog.askopenfilename(
                parent=popup,
                title="Load procedures",
                filetypes=[("EIS procedures", "*.eisproc"), ("JSON files", "*.json")],
            )
            if not path:
                return
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                loaded_blocks = payload.get("procedure_blocks")
                loaded_procedures = payload.get(
                    "procedures", payload.get("procedure_definitions")
                )
                # Earlier block-only files stored blocks under ``procedures``.
                if loaded_blocks is None and isinstance(loaded_procedures, dict):
                    loaded_blocks = loaded_procedures
                    loaded_procedures = {}
                validated, validated_procedures = self._validate_procedure_data(
                    loaded_blocks, loaded_procedures
                )
                if not validated and not validated_procedures:
                    raise ValueError("the file does not contain valid blocks or procedures")
            except (OSError, json.JSONDecodeError, ValueError) as error:
                messagebox.showerror("Load procedures failed", str(error), parent=popup)
                return
            self.procedure_blocks = validated
            block_box.configure(values=sorted(self.procedure_blocks))
            procedure_block_box.configure(values=sorted(self.procedure_blocks))
            self.procedures = validated_procedures
            procedure_box.configure(values=sorted(self.procedures))
            active_block = payload.get("active_block")
            if active_block not in self.procedure_blocks:
                active_block = next(iter(self.procedure_blocks), None)
            if active_block is not None:
                block_var.set(active_block)
                load_block()
            active_procedure = payload.get("active_procedure")
            if active_procedure in self.procedures:
                procedure_var.set(active_procedure)
                load_procedure()
            self._update_status(f"procedures loaded from {Path(path).name}")

        def load_block(_event=None) -> None:
            name = block_var.get()
            if name not in self.procedure_blocks:
                return
            steps[:] = copy.deepcopy(self.procedure_blocks[name])
            name_var.set(name)
            render_steps()

        def run_block() -> None:
            if not steps:
                messagebox.showwarning("Empty procedure", "Add at least one action.", parent=popup)
                return
            if self.busy:
                self._update_status("wait for the current operation to finish")
                return
            input_values = [
                line.strip()
                for line in repeat_values_text.get("1.0", tk.END).splitlines()
                if line.strip()
            ]
            try:
                run_steps = self._expand_procedure_inputs(
                    copy.deepcopy(steps), input_values
                )
            except ValueError as error:
                messagebox.showerror("Invalid repeat inputs", str(error), parent=popup)
                return
            self._procedure_running = True
            self._run_procedure_steps(run_steps, 0)

        def update_parameter_choices(_event=None) -> None:
            configure_parameter_input(action_var.get())
            if action_var.get() == "Go to spectrum…":
                selected_count = len(self.explorer.selection())
                choices = ["first", "middle", "last"]
                choices.extend(str(index) for index in range(1, selected_count + 1))
                parameter_entry.configure(values=choices)
            elif action_var.get() == "Choose EEC model":
                parameter_entry.configure(values=self._model_presets)
            elif action_var.get() == "Sort spectra by metadata":
                parameter_entry.configure(
                    values=[
                        self._explorer_headings.get(column, column)
                        for column in self._explorer_columns()
                    ]
                )
            else:
                parameter_entry.configure(values=())

        action_box.bind(
            "<<ComboboxSelected>>",
            lambda event: (
                help_var.set(action_help.get(action_var.get(), "")),
                update_parameter_choices(event),
            ),
        )
        block_box.bind("<<ComboboxSelected>>", load_block)
        procedure_box.bind("<<ComboboxSelected>>", load_procedure)
        ttk.Button(add_frame, text="Add", command=add_step).grid(
            row=0, column=3, padx=(8, 0), sticky="e"
        )
        buttons = ttk.Frame(popup, padding=10)
        buttons.grid(row=5, column=0, sticky="ew")
        ttk.Button(buttons, text="Remove", command=remove_step).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Move up", command=lambda: move_step(-1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Move down", command=lambda: move_step(1)).pack(side=tk.LEFT)
        ttk.Button(buttons, text="New", command=new_block).pack(side=tk.LEFT, padx=(18, 4))
        ttk.Button(buttons, text="Save block", command=save_block).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Save to file", command=save_to_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Load from file", command=load_from_file).pack(side=tk.LEFT)
        ttk.Button(
            buttons, text="Save to Preferences", command=self.save_procedures_to_preferences
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            buttons,
            text="Load from Preferences",
            command=load_procedures_from_preferences,
        ).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Run block", command=run_block).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Run procedure", command=run_procedure).pack(
            side=tk.RIGHT, padx=4
        )
        ttk.Button(buttons, text="Close", command=close_popup).pack(side=tk.RIGHT, padx=4)

    @staticmethod
    def _expand_procedure_inputs(
        steps: list[dict[str, str]], input_values: list[str]
    ) -> list[dict[str, str]]:
        """Expand a procedure block once for each optional ``{input}`` value."""
        if not input_values:
            return steps
        if not any("{input}" in step.get("parameter", "") for step in steps):
            raise ValueError(
                "repeat values require {input} in at least one action parameter"
            )
        return [
            {
                **step,
                "parameter": step.get("parameter", "").replace("{input}", value),
            }
            for value in input_values
            for step in steps
        ]

    def _procedure_column(self, value: str) -> str:
        candidate = value.strip()
        if candidate in self._explorer_columns():
            return candidate
        lowered = candidate.casefold()
        for column, heading in self._explorer_headings.items():
            if heading.casefold() == lowered:
                return column
        raise ValueError(f"unknown explorer column: {candidate}")

    @staticmethod
    def _serialize_procedure_selection(column: str, operator: str, value: str) -> str:
        return f"{column}|{operator}|{value}"

    @staticmethod
    def _parse_procedure_selection(parameter: str) -> tuple[str, str, str]:
        if "|" in parameter:
            column, operator, value = parameter.split("|", 2)
            return column.strip(), operator.strip(), value.strip()
        column, separator, value = parameter.partition("=")
        if separator:
            return column.strip(), "=", value.strip()
        return "All", "=", ""

    @staticmethod
    def _compare_procedure_value(actual, operator: str, expected: str, column: str) -> bool:
        if operator == "=":
            return (
                EISApplication._format_explorer_value(actual, column) == expected
                or str(actual) == expected
            )
        try:
            actual_number = float(actual)
            expected_number = float(expected)
        except (TypeError, ValueError):
            actual_text = str(actual)
            expected_text = expected
            comparisons = {
                "<": actual_text < expected_text,
                ">": actual_text > expected_text,
                "<=": actual_text <= expected_text,
                ">=": actual_text >= expected_text,
            }
        else:
            comparisons = {
                "<": actual_number < expected_number,
                ">": actual_number > expected_number,
                "<=": actual_number <= expected_number,
                ">=": actual_number >= expected_number,
            }
        return comparisons.get(operator, False)

    def _execute_procedure_step(self, step: dict[str, str]) -> None:
        action = step["action"]
        parameter = step["parameter"]
        if action == "Select spectra by metadata":
            column_text, operator, target = self._parse_procedure_selection(parameter)
            if column_text.casefold() == "all":
                self._set_explorer_selection(list(self._explorer_rows))
                return
            column = self._procedure_column(column_text)
            matching = []
            for item, (_dataset_id, loaded, spectrum) in self._explorer_rows.items():
                value = self._explorer_value(loaded, spectrum, column)
                if self._compare_procedure_value(value, operator, target, column):
                    matching.append(item)
            self._set_explorer_selection(matching)
            if matching:
                self._activate_explorer_item(matching[0])
            return
        if action == "Sort spectra by metadata":
            self._sort_explorer(self._procedure_column(parameter))
            return
        if action == "Set frequency limits for selected":
            values = [part.strip() for part in parameter.split(",")]
            if len(values) != 2:
                raise ValueError("frequency limits require minimum,maximum")
            self.minimum_frequency_var.set(values[0])
            self.maximum_frequency_var.set(values[1])
            if not self._capture_controls():
                raise ValueError("frequency limits are invalid")
            self.apply_frequency_window_to_selected()
            return
        if action == "Find outliers in selected":
            if parameter:
                self.threshold_var.set(parameter)
            self.find_outliers_for_selected()
            return
        if action == "Choose EEC model":
            self.analysis_mode_var.set("EEC")
            self._on_analysis_mode_selected()
            self.model_var.set(parameter)
            if self.explorer.selection():
                self.apply_model_to_selected()
            else:
                self.apply_model()
            return
        if action == "Go to spectrum…":
            items = list(self.explorer.get_children(""))
            selected = set(self.explorer.selection())
            candidates = [item for item in items if item in selected]
            if not candidates:
                raise ValueError("select spectra in the explorer first")
            choice = parameter.casefold()
            if choice == "first":
                target_index = 0
            elif choice == "last":
                target_index = len(candidates) - 1
            elif choice == "middle":
                target_index = (len(candidates) - 1) // 2
            else:
                try:
                    target_index = int(parameter) - 1
                except ValueError as error:
                    raise ValueError("use first, middle, last, or a spectrum number") from error
                if not 0 <= target_index < len(candidates):
                    raise ValueError(f"spectrum number must be between 1 and {len(candidates)}")
            target = candidates[target_index]
            self.explorer.focus(target)
            self.explorer.see(target)
            self._explorer_primary_item = target
            self._activate_explorer_item(target)
            return
        if action == "Initial values":
            self.initialize_from_ridge()
            return
        if action == "Auto model (Hybrid DRT)":
            self.auto_select_model()
            return
        if action == "Fit current spectrum":
            self.fit()
            return
        if action == "Fit selected":
            self.fit_selected()
            return
        if action == "Refine fits":
            if parameter:
                values = [part.strip() for part in parameter.split(",", 1)]
                if len(values) != 2 or not all(values):
                    raise ValueError("refine fits requires z threshold,max iterations")
                self.refine_z_threshold_var.set(values[0])
                self.refine_max_iterations_var.set(values[1])
            self.refine_fit_selected()
            return
        if action == "Find deterministic outliers":
            self.remove_deterministic_outliers()
            return
        if action == "Batch fit selected":
            direction = parameter.casefold().strip()
            if direction == "up":
                self.batch_fit_selected_down(-1)
            elif direction == "down":
                self.batch_fit_selected_down(1)
            elif direction == "up and down":
                self.batch_fit_selected_up_down()
            else:
                raise ValueError("batch-fit direction must be up, down, or up and down")
            return
        raise ValueError(f"unknown procedure action: {action}")

    def _run_procedure_steps(self, steps: list[dict[str, str]], index: int) -> None:
        if index >= len(steps):
            self._procedure_running = False
            self._update_status("procedure completed")
            return
        try:
            self._execute_procedure_step(steps[index])
        except Exception as error:
            self._procedure_running = False
            messagebox.showerror("Procedure failed", str(error), parent=self.root)
            return
        if self.busy:
            self.root.after(100, lambda: self._continue_procedure_steps(steps, index + 1))
        else:
            self.root.after(0, lambda: self._run_procedure_steps(steps, index + 1))

    def _continue_procedure_steps(self, steps: list[dict[str, str]], index: int) -> None:
        if self.busy or getattr(self, "_batch_fit_both_pending", False):
            self.root.after(100, lambda: self._continue_procedure_steps(steps, index))
            return
        if getattr(self, "_fit_cancel_requested", False):
            self._procedure_running = False
            self._update_status("procedure stopped")
            return
        self._run_procedure_steps(steps, index)

    def _on_alt_a(self, event):
        if self.analysis_mode_var.get() == "DRT":
            self.copy_neighbor_drt_peaks(-1)
            return "break"
        if event.state & 0x0001 or event.keysym == "A":
            self.copy_neighbor_fit_settings(-1)
        else:
            self.copy_neighbor_fit(-1)
        return "break"

    def _on_alt_d(self, event):
        if self.analysis_mode_var.get() == "DRT":
            self.copy_neighbor_drt_peaks(1)
            return "break"
        if event.state & 0x0001 or event.keysym == "D":
            self.copy_neighbor_fit_settings(1)
        else:
            self.copy_neighbor_fit(1)
        return "break"

    def _on_alt_s(self, event):
        if event.state & 0x0001 or event.keysym == "S":
            self.initialize_and_fit()
        else:
            self.fit()
        return "break"

    def _initial_values_key(self, _event=None):
        self.initialize_from_ridge()
        return "break"

    def _active_zoom_key(self, _event=None):
        self.reset_plot_view()
        return "break"

    def _open_batch_fit_key_menu(self, _event=None):
        if self.busy:
            return "break"
        popup = tk.Toplevel(self.root)
        popup.title("Batch fit")
        popup.transient(self.root)
        popup.resizable(False, False)
        ttk.Label(
            popup,
            text="Batch fit selected spectra\n\n↑  Up     ↓  Down     Enter  Up and Down\n\nPress another key to cancel.",
            padding=14,
            justify="center",
        ).pack()
        popup.grab_set()
        popup.focus_force()

        def choose(event):
            keysym = event.keysym
            popup.grab_release()
            popup.destroy()
            if keysym == "Up":
                self._start_drt_or_eec_batch(-1)
            elif keysym == "Down":
                self._start_drt_or_eec_batch(1)
            elif keysym in {"Return", "KP_Enter"}:
                self._start_drt_or_eec_batch(0)

        popup.bind("<KeyPress>", choose)
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)
        return "break"

    def _start_drt_or_eec_batch(self, direction: int) -> None:
        if self.analysis_mode_var.get() == "DRT":
            self._start_drt_peak_batch(direction)
        elif direction == 0:
            self.batch_fit_selected_up_down()
        else:
            self.batch_fit_selected_down(direction)

    def _toggle_legends_key(self, _event=None):
        self.hide_legends_var.set(not self.hide_legends_var.get())
        self._update_legend_visibility()
        self.canvas.draw_idle()
        return "break"

    def _open_plot_controls_key_menu(self, _event=None):
        popup = tk.Toplevel(self.root)
        popup.title("Plot controls")
        popup.transient(self.root)
        popup.resizable(False, False)
        ttk.Label(
            popup,
            text="Press a key for the plot action; another key cancels.",
            padding=(12, 10, 12, 6),
        ).pack()
        status = (
            f"e  Edit points: {'On' if self.point_toggle_mode else 'Off'}\n"
            f"E  Edit points and fit: {'On' if self.point_auto_fit else 'Off'}\n"
            f"h  Hide legends: {'On' if self.hide_legends_var.get() else 'Off'}\n"
            f"1  Show spectrum: {'On' if self.show_spectrum_var.get() else 'Off'}\n"
            f"2  Show KK residuals: {'On' if self.show_kk_var.get() else 'Off'}\n"
            f"3  Show DRT: {'On' if self.show_drt_var.get() else 'Off'}\n"
            f"4  Show EEC fit: {'On' if self.show_eec_fit_var.get() else 'Off'}\n"
            f"6  Show DRT fit: {'On' if self.show_drt_fit_var.get() else 'Off'}\n"
            f"9  Show DRT recovered: {'On' if self.show_drt_recovered_var.get() else 'Off'}\n"
            f"b  Nyquist/Bode: {self.plot_mode.title()}\n"
            "q  Active zoom"
        )
        ttk.Label(popup, text=status, justify="left", padding=(12, 4, 12, 12)).pack()
        popup.grab_set()
        popup.focus_force()

        def choose(event):
            key = event.keysym
            popup.grab_release()
            popup.destroy()
            actions = {
                "e": self.toggle_point_edit_mode,
                "E": self.toggle_auto_fit_points,
                "h": self._toggle_legends_key,
                "H": self._toggle_legends_key,
                "1": lambda: self._toggle_display_key(
                    "show_spectrum_var", "toggle_spectrum_view"
                ),
                "2": lambda: self._toggle_display_key(
                    "show_kk_var", "toggle_kk_view"
                ),
                "3": lambda: self._toggle_display_key(
                    "show_drt_var", "toggle_drt_view"
                ),
                "4": lambda: self._toggle_display_key(
                    "show_eec_fit_var", "toggle_fit_visibility"
                ),
                "6": lambda: self._toggle_display_key(
                    "show_drt_fit_var", "toggle_drt_fit_visibility"
                ),
                "9": lambda: self._toggle_display_key(
                    "show_drt_recovered_var", "toggle_drt_recovered_visibility"
                ),
                "b": self.toggle_plot_mode,
                "B": self.toggle_plot_mode,
                "q": self.reset_plot_view,
                "Q": self.reset_plot_view,
            }
            action = actions.get(key)
            if action is not None:
                action()

        popup.bind("<KeyPress>", choose)
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)
        return "break"

    def _toggle_analysis_mode_key(self, _event=None):
        self.analysis_mode_var.set(
            "DRT" if self.analysis_mode_var.get() == "EEC" else "EEC"
        )
        self._on_analysis_mode_selected()
        return "break"

    def _toggle_drt_method_key(self, _event=None):
        if self.analysis_mode_var.get() == "EEC":
            return self._next_eec_circuit_key()
        if self.analysis_mode_var.get() != "DRT":
            return "break"
        self.analysis_drt_mode_var.set(
            "Hybrid DRT"
            if self.analysis_drt_mode_var.get() == "Ridge DRT"
            else "Ridge DRT"
        )
        self._on_analysis_drt_mode_selected()
        return "break"

    def _next_eec_circuit_key(self, _event=None):
        if self.analysis_mode_var.get() != "EEC" or self.state is None:
            return "break"
        circuits = self._model_presets
        if not circuits:
            return "break"
        current = self.state.active.model(self.state.circuit)
        try:
            next_index = (circuits.index(current) + 1) % len(circuits)
        except ValueError:
            next_index = 0
        self.model_var.set(circuits[next_index])
        self.apply_model()
        return "break"

    def _open_circuit_picker_key(self, _event=None):
        if self.analysis_mode_var.get() != "EEC" or self.state is None:
            return "break"
        existing = getattr(self, "circuit_picker_popup", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return "break"

        popup = tk.Toplevel(self.root)
        self.circuit_picker_popup = popup
        popup.title("Choose EEC circuit")
        popup.transient(self.root)
        popup.resizable(False, False)
        ttk.Label(
            popup,
            text="Select a circuit and press Enter to apply it",
            padding=(12, 10, 12, 6),
        ).pack(anchor="w")
        circuit_list = tk.Listbox(
            popup,
            height=min(max(len(self._model_presets), 4), 12),
            width=max(28, max(map(len, self._model_presets), default=28) + 2),
            exportselection=False,
        )
        circuit_list.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
        for circuit in self._model_presets:
            circuit_list.insert(tk.END, circuit)

        current = self.state.active.model(self.state.circuit)
        try:
            current_index = self._model_presets.index(current)
        except ValueError:
            current_index = 0
        circuit_list.selection_set(current_index)
        circuit_list.activate(current_index)

        def close_popup() -> None:
            self.circuit_picker_popup = None
            popup.destroy()

        def apply_selected(_event=None):
            selection = circuit_list.curselection()
            if not selection:
                return "break"
            self.model_var.set(circuit_list.get(selection[0]))
            close_popup()
            if self._selected_spectrum_rows():
                self.apply_model_to_selected()
            else:
                self.apply_model()
            return "break"

        popup.protocol("WM_DELETE_WINDOW", close_popup)
        popup.bind("<Return>", apply_selected)
        popup.bind("<KP_Enter>", apply_selected)
        popup.bind("<Escape>", lambda _event: close_popup())
        circuit_list.bind("<Double-Button-1>", apply_selected)
        popup.grab_set()
        popup.focus_force()
        circuit_list.focus_set()
        return "break"

    def _calculate_current_drt_key(self, _event=None):
        if self.analysis_mode_var.get() == "DRT":
            self._fit_selected_drts()
        return "break"

    def _on_analysis_alt_s(self, event):
        if self.analysis_mode_var.get() == "DRT":
            self.fit_drt_peaks()
            return "break"
        return self._on_alt_s(event)

    def _on_drt_mode_selected(self, _event=None):
        if hasattr(self, "analysis_drt_mode_var"):
            self.analysis_drt_mode_var.set(self.drt_mode_var.get())
        if self.state is None:
            return
        self._ensure_current_drt_mode()

    def _on_explorer_click(self, event):
        if self.busy or self.state is None:
            return "break"
        if self.explorer.identify("region", event.x, event.y) not in {"tree", "cell"}:
            return None
        item = self.explorer.identify_row(event.y)
        if not item:
            return "break"
        visible_items = list(self.explorer.get_children(""))
        if item not in visible_items:
            return "break"

        selected = [row for row in visible_items if row in self.explorer.selection()]
        shift_pressed = bool(event.state & 0x0001)
        control_pressed = bool(event.state & 0x0004)

        if shift_pressed:
            anchor = self._explorer_anchor_item
            if anchor not in visible_items:
                anchor = (
                    self._explorer_primary_item
                    if self._explorer_primary_item in visible_items
                    else (selected[-1] if selected else item)
                )
            start = visible_items.index(anchor)
            end = visible_items.index(item)
            range_items = visible_items[min(start, end) : max(start, end) + 1]
            if control_pressed:
                selected_set = set(selected)
                if all(candidate in selected_set for candidate in range_items):
                    new_selection = [
                        candidate
                        for candidate in selected
                        if candidate not in range_items
                    ]
                else:
                    new_selection = list(dict.fromkeys([*selected, *range_items]))
            else:
                new_selection = range_items
        elif control_pressed:
            new_selection = (
                [candidate for candidate in selected if candidate != item]
                if item in selected
                else [*selected, item]
            )
        else:
            new_selection = [item]

        self._set_explorer_selection(new_selection, primary=item)
        self._activate_explorer_item(item)
        return "break"

    def _on_explorer_double_click(self, event):
        if self.busy or self.state is None:
            return "break"
        if self.explorer.identify("region", event.x, event.y) not in {"tree", "cell"}:
            return "break"
        item = self.explorer.identify_row(event.y)
        column_id = self.explorer.identify_column(event.x)
        columns = self._explorer_display_columns()
        if not item or not column_id.startswith("#"):
            return "break"
        column_index = int(column_id[1:]) - 1
        if column_index < 0 or column_index >= len(columns):
            return "break"
        row = self._explorer_rows.get(item)
        if row is None:
            return "break"
        column = columns[column_index]
        clicked_value = self._format_explorer_value(
            self._explorer_value(row[1], row[2], column), column
        )
        if bool(event.state & 0x0001) or self._explorer_shift_double_click:
            visible_items = list(self.explorer.get_children(""))
            clicked_index = visible_items.index(item)
            matching_items = [item]
            for direction in (-1, 1):
                index = clicked_index + direction
                while 0 <= index < len(visible_items):
                    candidate = visible_items[index]
                    candidate_row = self._explorer_rows.get(candidate)
                    if candidate_row is None:
                        break
                    candidate_value = self._format_explorer_value(
                        self._explorer_value(
                            candidate_row[1], candidate_row[2], column
                        ),
                        column,
                    )
                    if candidate_value != clicked_value:
                        break
                    if direction < 0:
                        matching_items.insert(0, candidate)
                    else:
                        matching_items.append(candidate)
                    index += direction
        else:
            matching_items = []
            for candidate, (_dataset_id, loaded, spectrum) in self._explorer_rows.items():
                value = self._format_explorer_value(
                    self._explorer_value(loaded, spectrum, column), column
                )
                if value == clicked_value:
                    matching_items.append(candidate)
        self._set_explorer_selection(matching_items, primary=item)
        self._activate_explorer_item(item)
        return "break"

    def _on_explorer_shift_double_click(self, event):
        self._explorer_shift_double_click = True
        self.root.after_idle(
            lambda: setattr(self, "_explorer_shift_double_click", False)
        )
        return self._on_explorer_double_click(event)

    def _set_explorer_selection(
        self,
        items: list[str],
        *,
        primary: str | None = None,
    ) -> None:
        valid_items = [item for item in items if self.explorer.exists(item)]
        self._suspend_explorer_select = True
        try:
            self.explorer.selection_set(valid_items)
        finally:
            self._suspend_explorer_select = False
        if not valid_items:
            self._explorer_anchor_item = None
            self._explorer_primary_item = None
            self._refresh_explorer_focus_tag()
            self._update_explorer_selection_status()
            return
        if primary not in valid_items:
            primary = valid_items[-1]
        assert primary is not None
        self.explorer.focus(primary)
        self.explorer.see(primary)
        self._explorer_anchor_item = primary
        self._explorer_primary_item = primary
        self._refresh_explorer_focus_tag()
        self._update_explorer_selection_status()

    def _update_explorer_selection_status(self) -> None:
        if not hasattr(self, "explorer_selection_var"):
            return
        count = len(self.explorer.selection()) if hasattr(self, "explorer") else 0
        label = "spectrum" if count == 1 else "spectra"
        self.explorer_selection_var.set(f"{count} {label} selected")

    def _refresh_explorer_focus_tag(self) -> None:
        if not hasattr(self, "explorer"):
            return
        for item in self._explorer_rows:
            if self.explorer.exists(item):
                self.explorer.item(item, tags=())
        current_item = None
        if self.current_dataset_id is not None and self.state is not None:
            current_item = self._explorer_lookup.get(
                (self.current_dataset_id, self.state.active_cycle)
            )
        if current_item is not None and self.explorer.exists(current_item):
            self.explorer.item(current_item, tags=("current_row",))

    def _activate_explorer_item(self, item: str) -> None:
        row = self._explorer_rows.get(item)
        if row is None:
            return
        dataset_id, loaded, spectrum = row
        if loaded is self.loaded and dataset_id == self.current_dataset_id:
            self._activate_cycle(
                spectrum.cycle,
                preserve_existing_selection=True,
            )
            return
        self._switch_dataset(dataset_id, loaded, spectrum.cycle)

    def _highlight_explorer_cycle(
        self,
        cycle_number: int,
        *,
        preserve_existing: bool = False,
        focus_only: bool = False,
    ) -> None:
        if self.current_dataset_id is None:
            return
        item = self._explorer_lookup.get((self.current_dataset_id, cycle_number))
        if item is not None and self.explorer.exists(item):
            if focus_only:
                self.explorer.focus(item)
                self.explorer.see(item)
                self._explorer_primary_item = item
                self._explorer_anchor_item = item
                self._refresh_explorer_focus_tag()
                self._update_explorer_selection_status()
            elif preserve_existing:
                selection = list(self.explorer.selection())
                if item not in selection:
                    selection.append(item)
                self._set_explorer_selection(selection, primary=item)
            else:
                self._set_explorer_selection([item], primary=item)

    def _build_controls(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        mode_frame = ttk.Frame(parent)
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        mode_frame.columnconfigure(1, weight=1)
        ttk.Label(mode_frame, text="Analysis mode").grid(
            row=0, column=0, padx=(0, 8), sticky="w"
        )
        self.analysis_mode_box = ttk.Combobox(
            mode_frame,
            textvariable=self.analysis_mode_var,
            values=("EEC", "DRT", "Spectra Simulator"),
            state="readonly",
            width=12,
        )
        self.analysis_mode_box.grid(row=0, column=1, sticky="ew")
        self.analysis_mode_box.bind(
            "<<ComboboxSelected>>", self._on_analysis_mode_selected
        )
        model_group = ttk.LabelFrame(parent, text="Fitting model", padding=8)
        self.model_group = model_group
        model_group.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        model_group.columnconfigure(0, weight=1)
        model_group.columnconfigure(1, weight=1)
        model_group.columnconfigure(2, weight=1)
        self.model_box = ttk.Combobox(
            model_group,
            textvariable=self.model_var,
            values=self._model_presets,
            state="normal",
        )
        self.model_box.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.model_box.bind("<Return>", lambda _event: self.apply_model())
        self.model_button = ttk.Button(
            model_group, text="Set model", command=self.apply_model
        )
        self.model_button.grid(row=0, column=1, padx=(5, 0), sticky="ew")
        self.model_selected_button = ttk.Button(
            model_group,
            text="Set model for selected",
            command=self.apply_model_to_selected,
        )
        self.model_selected_button.grid(row=0, column=2, padx=(5, 0), sticky="ew")
        self.sort_tau_selected_button = ttk.Button(
            model_group,
            text="Sort by tau",
            command=self.sort_selected_parameters_by_tau,
        )
        self.sort_tau_selected_button.grid(row=1, column=0, padx=(0, 3), pady=(5, 0), sticky="ew")
        self.switch_blocks_selected_button = ttk.Button(
            model_group,
            text="Switch blocks",
            command=self.switch_selected_parameter_blocks,
        )
        self.switch_blocks_selected_button.grid(
            row=1, column=1, columnspan=2, padx=(3, 0), pady=(5, 0), sticky="ew"
        )
        self.open_eec_analysis_button = ttk.Button(
            model_group,
            text="Open EEC analysis window",
            command=self.open_eec_analysis_window,
        )
        self.open_eec_analysis_button.grid(
            row=3, column=0, columnspan=3, pady=(6, 0), sticky="ew"
        )
        self.auto_model_button = ttk.Button(
            model_group,
            text="Auto model: selected (Hybrid DRT)",
            command=self.auto_select_model,
        )
        self.auto_model_button.grid(
            row=4, column=0, columnspan=3, pady=(5, 0), sticky="ew"
        )
        parameters_group = ttk.LabelFrame(parent, text="Circuit parameters", padding=8)
        self.parameters_group = parameters_group
        parameters_group.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        parameters_group.columnconfigure(0, weight=1)
        parameters_group.rowconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        parameter_canvas_frame = ttk.Frame(parameters_group)
        parameter_canvas_frame.grid(row=0, column=0, sticky="nsew")
        parameter_canvas_frame.columnconfigure(0, weight=1)
        parameter_canvas_frame.rowconfigure(0, weight=1)
        self.parameter_canvas = tk.Canvas(parameter_canvas_frame, highlightthickness=0)
        self.parameter_canvas.grid(row=0, column=0, sticky="nsew")
        self.parameter_scrollbar = ttk.Scrollbar(
            parameter_canvas_frame, orient=tk.VERTICAL, command=self.parameter_canvas.yview
        )
        self.parameter_scrollbar.grid(row=0, column=1, sticky="ns")
        self.parameter_canvas.configure(yscrollcommand=self.parameter_scrollbar.set)
        parameter_contents = ttk.Frame(self.parameter_canvas)
        self.parameter_contents = parameter_contents
        parameter_window = self.parameter_canvas.create_window(
            (0, 0), window=parameter_contents, anchor="nw"
        )
        parameter_contents.bind(
            "<Configure>",
            lambda _event: self._update_parameter_scroll_region(),
        )
        self.parameter_canvas.bind(
            "<Configure>",
            lambda event: self.parameter_canvas.itemconfigure(
                parameter_window, width=event.width
            ),
        )
        self.parameter_table = ParameterTable(parameter_contents)
        self.parameter_table.pack(fill=tk.BOTH, expand=True)
        parameter_actions = ttk.Frame(parameter_contents)
        parameter_actions.pack(fill=tk.X, pady=(8, 0))
        for column in range(2):
            parameter_actions.columnconfigure(column, weight=1)
        self.parameters_selected_button = ttk.Button(
            parameter_actions,
            text="Apply all to selected",
            command=self.apply_parameters_to_selected,
        )
        self.parameters_selected_button.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.apply_fix_selected_button = ttk.Button(
            parameter_actions,
            text="Apply Fix",
            command=lambda: self.apply_parameters_to_selected({"fixed"}),
        )
        self.apply_fix_selected_button.grid(row=1, column=0, padx=(0, 3), pady=(4, 0), sticky="ew")
        self.apply_initial_selected_button = ttk.Button(
            parameter_actions,
            text="Apply Initial",
            command=lambda: self.apply_parameters_to_selected({"initial"}),
        )
        self.apply_initial_selected_button.grid(row=1, column=1, padx=(3, 0), pady=(4, 0), sticky="ew")
        self.apply_lower_selected_button = ttk.Button(
            parameter_actions,
            text="Apply Lower",
            command=lambda: self.apply_parameters_to_selected({"lower"}),
        )
        self.apply_lower_selected_button.grid(row=2, column=0, padx=(0, 3), pady=(4, 0), sticky="ew")
        self.apply_upper_selected_button = ttk.Button(
            parameter_actions,
            text="Apply Upper",
            command=lambda: self.apply_parameters_to_selected({"upper"}),
        )
        self.apply_upper_selected_button.grid(row=2, column=1, padx=(3, 0), pady=(4, 0), sticky="ew")
        self.root.bind_all("<MouseWheel>", self._parameter_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._parameter_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._parameter_mousewheel, add="+")

        options_group = ttk.LabelFrame(parent, text="Selection", padding=8)
        self.options_group = options_group
        options_group.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        options_group.columnconfigure(1, weight=1)
        options_group.columnconfigure(3, weight=1)
        options_group.columnconfigure(4, weight=1)
        ttk.Label(options_group, text="Min. freq.").grid(
            row=0, column=0, sticky="w"
        )
        minimum_frequency_entry = ttk.Entry(
            options_group, textvariable=self.minimum_frequency_var
        )
        minimum_frequency_entry.grid(
            row=0, column=1, padx=(8, 0), pady=2, sticky="ew"
        )
        ttk.Label(options_group, text="Max. freq.").grid(
            row=0, column=2, padx=(12, 0), sticky="w"
        )
        self.maximum_frequency_entry = ttk.Entry(
            options_group, textvariable=self.maximum_frequency_var
        )
        self.maximum_frequency_entry.grid(
            row=0, column=3, padx=(8, 0), pady=2, sticky="ew"
        )
        ttk.Checkbutton(
            options_group,
            text="Auto",
            variable=self.auto_max_frequency_var,
            command=self._toggle_auto_max_frequency,
        ).grid(row=0, column=4, padx=(8, 0), pady=2, sticky="w")
        self.frequency_selected_button = ttk.Button(
            options_group,
            text="Apply to selected spectra",
            command=self.apply_frequency_window_to_selected,
        )
        self.frequency_selected_button.grid(
            row=1, column=0, columnspan=5, padx=0, pady=(6, 0), sticky="ew"
        )
        self.minimum_frequency_var.trace_add("write", self._schedule_frequency_application)
        self.maximum_frequency_var.trace_add("write", self._schedule_frequency_application)
        ttk.Label(options_group, text="Outlier threshold").grid(
            row=2, column=0, pady=(8, 2), sticky="w"
        )
        ttk.Entry(options_group, textvariable=self.threshold_var).grid(
            row=2, column=1, padx=(8, 0), pady=(8, 2), sticky="ew"
        )
        self.outlier_selected_button = ttk.Button(
            options_group,
            text="Outliers",
            command=self.find_outliers_for_selected,
        )
        self.outlier_selected_button.grid(
            row=2, column=2, columnspan=2, padx=(8, 0), pady=(8, 2), sticky="ew"
        )
        ttk.Label(options_group, text="Deterministic threshold").grid(
            row=3, column=0, pady=(8, 2), sticky="w"
        )
        ttk.Entry(
            options_group, textvariable=self.deterministic_threshold_var
        ).grid(row=3, column=1, padx=(8, 0), pady=(8, 2), sticky="ew")
        self.deterministic_outlier_button = ttk.Button(
            options_group,
            text="Remove deterministic outliers",
            command=self.remove_deterministic_outliers,
        )
        self.deterministic_outlier_button.grid(
            row=3, column=2, columnspan=2, padx=(8, 0), pady=(8, 2), sticky="ew"
        )
        self.reset_button = ttk.Button(
            options_group, text="Reset points", command=self.reset_points
        )
        self.reset_button.grid(
            row=4, column=0, columnspan=4, pady=(6, 0), sticky="ew"
        )

        actions = ttk.LabelFrame(parent, text="Actions", padding=8)
        self.actions_group = actions
        actions.grid(row=4, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.fit_button = ttk.Button(actions, text="Fit spectrum", command=self.fit)
        self.fit_button.grid(row=0, column=0, padx=(0, 4), pady=3, sticky="ew")
        self.fit_selected_button = ttk.Button(
            actions, text="Fit selected", command=self.fit_selected
        )
        self.fit_selected_button.grid(row=0, column=1, padx=(4, 0), pady=3, sticky="ew")
        self.initial_values_button = ttk.Button(
            actions, text="Initial values", command=self.initialize_from_ridge
        )
        self.initial_values_button.grid(
            row=1, column=0, columnspan=2, pady=3, sticky="ew"
        )
        ttk.Label(actions, text="Robust z threshold").grid(
            row=2, column=0, padx=(0, 4), pady=3, sticky="w"
        )
        ttk.Entry(actions, textvariable=self.refine_z_threshold_var).grid(
            row=2, column=1, padx=(4, 0), pady=3, sticky="ew"
        )
        ttk.Label(actions, text="Maximum refine iterations").grid(
            row=3, column=0, padx=(0, 4), pady=3, sticky="w"
        )
        ttk.Entry(actions, textvariable=self.refine_max_iterations_var).grid(
            row=3, column=1, padx=(4, 0), pady=3, sticky="ew"
        )
        self.refine_fit_button = ttk.Button(
            actions, text="Refine fit", command=self.refine_fit_selected
        )
        self.refine_fit_button.grid(
            row=4, column=0, columnspan=2, pady=3, sticky="ew"
        )
        self.stop_fit_button = ttk.Button(
            actions, text="Stop", command=self._cancel_fit, state="disabled"
        )
        self.stop_fit_button.grid(row=6, column=0, columnspan=2, pady=3, sticky="ew")
        self.drt_tools_group = ttk.LabelFrame(parent, text="DRT analysis", padding=8)
        self.drt_tools_group.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.drt_tools_group.columnconfigure(1, weight=1)
        self.drt_tools_group.columnconfigure(2, weight=1)
        ttk.Label(self.drt_tools_group, text="Method").grid(
            row=0, column=0, padx=(0, 8), sticky="w"
        )
        self.analysis_drt_mode_var = tk.StringVar(value="Ridge DRT")
        self.analysis_drt_mode_box = ttk.Combobox(
            self.drt_tools_group,
            textvariable=self.analysis_drt_mode_var,
            values=("Ridge DRT", "Hybrid DRT"),
            state="readonly",
        )
        self.analysis_drt_mode_box.grid(row=0, column=1, sticky="ew")
        self.analysis_drt_mode_box.bind(
            "<<ComboboxSelected>>", self._on_analysis_drt_mode_selected
        )
        self.drt_fit_button = ttk.Button(
            self.drt_tools_group,
            text="Fit selected",
            command=self._fit_selected_drts,
        )
        self.drt_fit_button.grid(
            row=1, column=0, columnspan=2, pady=(8, 0), sticky="ew"
        )
        self.add_gaussian_peak_button = ttk.Button(
            self.drt_tools_group,
            text="Add Gaussian peak",
            command=lambda: self.add_drt_peak("gaussian"),
        )
        self.add_gaussian_peak_button.grid(
            row=2, column=0, padx=(0, 3), pady=(6, 0), sticky="ew"
        )
        self.add_lorentzian_peak_button = ttk.Button(
            self.drt_tools_group,
            text="Add Lorentzian peak",
            command=lambda: self.add_drt_peak("lorentzian"),
        )
        self.add_lorentzian_peak_button.grid(
            row=2, column=1, padx=3, pady=(6, 0), sticky="ew"
        )
        self.add_voigt_peak_button = ttk.Button(
            self.drt_tools_group,
            text="Add Voigt peak",
            command=lambda: self.add_drt_peak("voigt"),
        )
        self.add_voigt_peak_button.grid(
            row=2, column=2, padx=(3, 0), pady=(6, 0), sticky="ew"
        )
        self.add_hn_peak_button = ttk.Button(
            self.drt_tools_group,
            text="Add HN peak",
            command=lambda: self.add_drt_peak("hn"),
        )
        self.add_hn_peak_button.grid(
            row=3, column=0, columnspan=3, pady=(6, 0), sticky="ew"
        )
        self.fit_peaks_button = ttk.Button(
            self.drt_tools_group,
            text="Fit peaks",
            command=self.fit_drt_peaks,
        )
        self.fit_peaks_button.grid(
            row=4, column=0, columnspan=3, pady=(6, 0), sticky="ew"
        )
        self.send_drt_initials_button = ttk.Button(
            self.drt_tools_group,
            text="Send initials",
            command=self.send_drt_initials,
        )
        self.send_drt_initials_button.grid(
            row=7, column=0, columnspan=3, pady=(6, 0), sticky="ew"
        )
        self.open_drt_analysis_button = ttk.Button(
            self.drt_tools_group,
            text="Open DRT analysis window",
            command=self.open_drt_analysis_window,
        )
        self.open_drt_analysis_button.grid(
            row=8, column=0, columnspan=3, pady=(6, 0), sticky="ew"
        )
        self.remove_all_peaks_button = ttk.Button(
            self.drt_tools_group,
            text="Remove all peaks",
            command=self.remove_all_drt_peaks,
        )
        self.remove_all_peaks_button.grid(
            row=9, column=0, columnspan=3, pady=(6, 0), sticky="ew"
        )
        self.drt_peak_table = ParameterTable(
            self.drt_tools_group,
            name_double_click=self._on_drt_parameter_double_click,
            display_name=self._drt_display_parameter_name,
        )
        self.drt_peak_table.grid(
            row=5, column=0, columnspan=3, pady=(6, 0), sticky="ew"
        )
        drt_parameter_actions = ttk.Frame(self.drt_tools_group)
        drt_parameter_actions.grid(
            row=6, column=0, columnspan=3, pady=(8, 0), sticky="ew"
        )
        for column in range(2):
            drt_parameter_actions.columnconfigure(column, weight=1)
        self.drt_parameters_selected_button = ttk.Button(
            drt_parameter_actions,
            text="Apply all to selected",
            command=self.apply_drt_parameters_to_selected,
        )
        self.drt_parameters_selected_button.grid(
            row=0, column=0, columnspan=2, sticky="ew"
        )
        self.drt_apply_fix_selected_button = ttk.Button(
            drt_parameter_actions,
            text="Apply Fix",
            command=lambda: self.apply_drt_parameters_to_selected({"fixed"}),
        )
        self.drt_apply_fix_selected_button.grid(
            row=1, column=0, padx=(0, 3), pady=(4, 0), sticky="ew"
        )
        self.drt_apply_initial_selected_button = ttk.Button(
            drt_parameter_actions,
            text="Apply Initial",
            command=lambda: self.apply_drt_parameters_to_selected({"initial"}),
        )
        self.drt_apply_initial_selected_button.grid(
            row=1, column=1, padx=(3, 0), pady=(4, 0), sticky="ew"
        )
        self.drt_apply_lower_selected_button = ttk.Button(
            drt_parameter_actions,
            text="Apply Lower",
            command=lambda: self.apply_drt_parameters_to_selected({"lower"}),
        )
        self.drt_apply_lower_selected_button.grid(
            row=2, column=0, padx=(0, 3), pady=(4, 0), sticky="ew"
        )
        self.drt_apply_upper_selected_button = ttk.Button(
            drt_parameter_actions,
            text="Apply Upper",
            command=lambda: self.apply_drt_parameters_to_selected({"upper"}),
        )
        self.drt_apply_upper_selected_button.grid(
            row=2, column=1, padx=(3, 0), pady=(4, 0), sticky="ew"
        )
        self.drt_tools_group.grid_remove()
        self._build_simulator_controls(parent)
        self.action_buttons = (
            self.fit_button,
            self.fit_selected_button,
            self.refine_fit_button,
            self.drt_fit_button,
            self.add_gaussian_peak_button,
            self.add_lorentzian_peak_button,
            self.add_voigt_peak_button,
            self.add_hn_peak_button,
            self.fit_peaks_button,
            self.remove_all_peaks_button,
            self.send_drt_initials_button,
            self.drt_parameters_selected_button,
            self.drt_apply_fix_selected_button,
            self.drt_apply_initial_selected_button,
            self.drt_apply_lower_selected_button,
            self.drt_apply_upper_selected_button,
            self.initial_values_button,
            self.outlier_selected_button,
            self.deterministic_outlier_button,
            self.reset_button,
            self.toggle_points_button,
            self.auto_fit_points_button,
            self.reset_view_button,
            self.toggle_plot_mode_button,
            self.delete_spectrum_button,
            self.plot_selected_button,
            self.plot_three_electrode_button,
            self.plot_fit_parameters_button,
            self.plot_drt_parameters_button,
            self.edit_metadata_button,
            self.frequency_selected_button,
            self.model_button,
            self.model_selected_button,
            self.sort_tau_selected_button,
            self.switch_blocks_selected_button,
            self.open_eec_analysis_button,
            self.auto_model_button,
            self.import_simulator_fit_button,
            self.open_drt_analysis_button,
            self.parameters_selected_button,
            self.apply_fix_selected_button,
            self.apply_initial_selected_button,
            self.apply_lower_selected_button,
            self.apply_upper_selected_button,
        )

    def _build_simulator_controls(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Spectra Simulator", padding=8)
        self.simulator_group = group
        group.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        group.columnconfigure(1, weight=1)
        group.columnconfigure(3, weight=1)
        self.simulator_circuit_var = tk.StringVar(value=self.model_var.get())
        self.simulator_min_frequency_var = tk.StringVar(value="1")
        self.simulator_max_frequency_var = tk.StringVar(value="1e6")
        self.simulator_points_var = tk.StringVar(value="10")
        self.simulator_noise_var = tk.BooleanVar(value=False)
        self.simulator_noise_level_var = tk.StringVar(value="1")
        self.simulator_seed_var = tk.StringVar(value="")
        ttk.Label(group, text="Circuit").grid(row=0, column=0, sticky="w")
        circuit_box = ttk.Combobox(group, textvariable=self.simulator_circuit_var,
                                   values=self._model_presets, state="normal")
        circuit_box.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(8, 0))
        circuit_box.bind("<<ComboboxSelected>>", lambda _event: self._ensure_simulator_parameters())
        self.import_simulator_fit_button = ttk.Button(
            group, text="Import Fit Parameters", command=self.import_simulator_fit_parameters,
            state="disabled",
        )
        self.import_simulator_fit_button.grid(
            row=0, column=3, sticky="ew", padx=(5, 0)
        )
        fields = (("Min. freq.", self.simulator_min_frequency_var),
                  ("Max. freq.", self.simulator_max_frequency_var),
                  ("Points/decade", self.simulator_points_var),
                  ("Noise (%)", self.simulator_noise_level_var),
                  ("Seed", self.simulator_seed_var))
        for row, (label, variable) in enumerate(fields, start=1):
            ttk.Label(group, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(group, textvariable=variable).grid(row=row, column=1, columnspan=3,
                                                         sticky="ew", padx=(8, 0), pady=2)
        ttk.Checkbutton(group, text="Add noise", variable=self.simulator_noise_var).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(3, 0))
        ttk.Button(group, text="Calculate spectrum", command=self.calculate_simulator_spectrum).grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=(7, 0), padx=(0, 3))
        ttk.Button(group, text="Add to Spectra Explorer", command=self.add_simulator_to_explorer).grid(
            row=7, column=2, columnspan=2, sticky="ew", pady=(7, 0), padx=(3, 0))
        ttk.Combobox(group, textvariable=self.simulator_drt_mode_var,
                     values=("Ridge DRT", "Hybrid DRT"), state="readonly").grid(
                         row=9, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Button(group, text="Calculate DRT", command=self.calculate_simulator_drt).grid(
            row=9, column=2, columnspan=2, sticky="ew", padx=(3, 0), pady=(7, 0))
        self.simulator_parameter_table = ParameterTable(group)
        self.simulator_parameter_table.grid(row=8, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        group.grid_remove()
        self._ensure_simulator_parameters()

    def _update_parameter_scroll_region(self) -> None:
        if not hasattr(self, "parameter_canvas"):
            return
        self.parameter_canvas.configure(
            scrollregion=self.parameter_canvas.bbox("all") or (0, 0, 0, 0)
        )
        content_height = self.parameter_contents.winfo_reqheight()
        canvas_height = self.parameter_canvas.winfo_height()
        if content_height > canvas_height + 1:
            self.parameter_scrollbar.grid()
        else:
            self.parameter_scrollbar.grid_remove()
            self.parameter_canvas.yview_moveto(0)

    def _parameter_mousewheel(self, event):
        try:
            widget = self.root.winfo_containing(event.x_root, event.y_root)
        except (KeyError, tk.TclError):
            return None
        inside = False
        while widget is not None:
            if widget is self.parameters_group:
                inside = True
                break
            widget = getattr(widget, "master", None)
        if not inside:
            return None
        if getattr(event, "num", None) == 4:
            direction = -1
        elif getattr(event, "num", None) == 5:
            direction = 1
        else:
            direction = -1 if event.delta > 0 else 1
        self.parameter_canvas.yview_scroll(direction, "units")
        return "break"

    def _ensure_simulator_parameters(self) -> None:
        circuit = self.simulator_circuit_var.get().strip()
        if not circuit:
            return
        try:
            parameters = circuit_parameters(circuit)
        except (TypeError, ValueError, ImportError):
            return
        if [p.name for p in parameters] != [p.name for p in self.simulator_parameters]:
            self.simulator_parameters = parameters
            self.simulator_parameter_table.set_parameters(parameters)

    def calculate_simulator_spectrum(self) -> None:
        try:
            self._ensure_simulator_parameters()
            parameters = self.simulator_parameter_table.values()
            seed_text = self.simulator_seed_var.get().strip()
            seed = int(seed_text) if seed_text else None
            frequencies = logarithmic_frequencies(float(self.simulator_min_frequency_var.get()),
                                                   float(self.simulator_max_frequency_var.get()),
                                                   int(self.simulator_points_var.get()))
            self.simulator_spectrum = simulate_spectrum(
                self.simulator_circuit_var.get().strip(), frequencies,
                [p.initial for p in parameters], noise_enabled=self.simulator_noise_var.get(),
                noise_level_percent=float(self.simulator_noise_level_var.get()), seed=seed)
        except (TypeError, ValueError, ImportError) as error:
            messagebox.showerror("Simulator", str(error), parent=self.root)
            return
        self._update_status("simulated spectrum calculated")
        self._refresh_plot(rescale=True)

    @staticmethod
    def _normalized_circuit(circuit: str | None) -> str:
        return re.sub(r"\s+", "", str(circuit or "")).casefold()

    def import_simulator_fit_parameters(self) -> None:
        if self.state is None:
            messagebox.showinfo(
                "Import Fit Parameters",
                "Select a spectrum in Spectra Explorer first.",
                parent=self.root,
            )
            return
        cycle = self.state.active
        current_model = cycle.model(self.state.circuit)
        simulator_model = self.simulator_circuit_var.get().strip()
        if not circuits_equivalent(current_model, simulator_model):
            messagebox.showerror(
                "Different EEC models",
                "Current spectrum and simulator EEC models are different. "
                "Fit parameters cannot be imported.",
                parent=self.root,
            )
            return
        if cycle.fit_parameters is None or len(cycle.fit_parameters) != len(cycle.parameters):
            messagebox.showinfo(
                "Import Fit Parameters",
                "The current spectrum has no complete EEC fit to import.",
                parent=self.root,
            )
            return
        self._ensure_simulator_parameters()
        fitted_by_name = {
            parameter.name: float(value)
            for parameter, value in zip(cycle.parameters, cycle.fit_parameters)
            if np.isfinite(value)
        }
        mapping = parameter_name_mapping(current_model, simulator_model)
        if mapping is not None:
            fitted_by_name = {
                mapped_name: value
                for name, value in fitted_by_name.items()
                if (mapped_name := map_parameter_name(name, mapping)) is not None
            }
        simulator_parameters = self.simulator_parameter_table.values()
        missing = [parameter.name for parameter in simulator_parameters
                   if parameter.name not in fitted_by_name]
        if missing:
            messagebox.showwarning(
                "Incomplete fit parameters",
                "The following simulator parameters were not found in the fit: "
                + ", ".join(missing) + ". Existing simulator values were preserved.",
                parent=self.root,
            )
        for parameter in simulator_parameters:
            if parameter.name in fitted_by_name:
                parameter.initial = self._clamp_parameter_value(
                    fitted_by_name[parameter.name], parameter.lower, parameter.upper
                )
        self.simulator_parameters = simulator_parameters
        self.simulator_parameter_table.set_parameters(simulator_parameters)
        frequency_window = cycle.frequency_window
        if frequency_window is None:
            frequency_window = (
                float(np.nanmin(cycle.frequency_hz)),
                float(np.nanmax(cycle.frequency_hz)),
            )
        self.simulator_min_frequency_var.set(f"{min(frequency_window):g}")
        self.simulator_max_frequency_var.set(f"{max(frequency_window):g}")
        self.calculate_simulator_spectrum()
        self._update_status("fit parameters and frequency limits imported to simulator")

    def _simulator_cycle(self) -> CycleState:
        if self.simulator_spectrum is None:
            raise ValueError("Calculate a simulated spectrum first")
        cycle = CycleState(1, self.simulator_spectrum.frequency_hz,
                           self.simulator_spectrum.impedance)
        cycle.parameters = copy.deepcopy(self.simulator_parameters)
        cycle.circuit = self.simulator_circuit_var.get().strip()
        return cycle

    def calculate_simulator_drt(self) -> None:
        try:
            cycle = self._simulator_cycle()
        except ValueError as error:
            messagebox.showerror("Simulator DRT", str(error), parent=self.root)
            return
        method = self.simulator_drt_mode_var.get()
        self._submit(
            (lambda: calculate_hybrid_drt(cycle)) if method == "Hybrid DRT"
            else (lambda: analyze_outliers(cycle, float(self.threshold_var.get()), cycle.parameters)),
            self._finish_simulator_drt,
            "Simulator DRT calculation failed",
        )

    def _finish_simulator_drt(self, result) -> None:
        if isinstance(result, RidgeInitialization):
            self.simulator_drt_result = DRTComputation(
                result.ridge_tau_s, result.ridge_gamma_ohm, result.ohmic_resistance
            )
        else:
            self.simulator_drt_result = result
        self.show_drt_var.set(True)
        self._refresh_plot(rescale=True)
        self._update_status(f"{self.simulator_drt_mode_var.get()} calculated for simulated spectrum")

    def add_simulator_to_explorer(self) -> None:
        if self.simulator_spectrum is None:
            self.calculate_simulator_spectrum()
        if self.simulator_spectrum is None:
            return
        import pandas as pd
        spectrum = self.simulator_spectrum
        source = Path.cwd() / f"simulated_{len(self._dataset_order) + 1}.eisfit"
        dataframe = pd.DataFrame({"cycle_number": np.ones(spectrum.frequency_hz.size, dtype=int),
                                  "freq_hz": spectrum.frequency_hz,
                                  "re_zwe_ce_ohm": spectrum.impedance.real,
                                  "minus_im_zwe_ce_ohm": -spectrum.impedance.imag,
                                  "ewe_ece_v": np.zeros(spectrum.frequency_hz.size),
                                  "time_s": np.zeros(spectrum.frequency_hz.size),
                                  "current_ma": np.zeros(spectrum.frequency_hz.size)})
        loaded = load_project_from_dataframe(dataframe, source, 1, "cell",
                                             self.simulator_circuit_var.get().strip(), "Simulated")
        loaded.dataset_id = f"simulator::{source.stem}"
        loaded.dataset_label = f"{source.stem} [Cell · simulated]"
        loaded.state.active.custom_metadata.update({"Source": "Simulator",
                                                     "Circuit": self.simulator_circuit_var.get().strip()})
        self._register_dataset(loaded.dataset_id, loaded)
        self._populate_explorer()
        self._switch_dataset(loaded.dataset_id, loaded, 1, capture_current=False)

    def _on_analysis_mode_selected(self, _event=None) -> None:
        simulator_mode = self.analysis_mode_var.get() == "Spectra Simulator"
        drt_mode = self.analysis_mode_var.get() == "DRT"
        if hasattr(self, "drt_fit_button"):
            self.drt_fit_button.configure(
                text="Calculate DRT" if drt_mode else "Fit selected"
            )
        if simulator_mode:
            self.model_group.grid_remove()
            self.parameters_group.grid_remove()
            self.refine_fit_button.grid_remove()
            self.drt_tools_group.grid_remove()
            self.simulator_group.grid()
            self._update_status("Spectra Simulator mode")
            self._refresh_plot(rescale=True)
        elif drt_mode:
            self.show_drt_var.set(True)
            self.show_drt_recovered_var.set(True)
            self.model_group.grid_remove()
            self.parameters_group.grid_remove()
            self.refine_fit_button.grid_remove()
            self.drt_tools_group.grid()
            self.simulator_group.grid_remove()
            self.toggle_drt_view()
            self.toggle_drt_recovered_visibility()
            self._update_status("DRT analysis mode")
        else:
            self.model_group.grid()
            self.parameters_group.grid()
            self.refine_fit_button.grid()
            self.drt_tools_group.grid_remove()
            self.simulator_group.grid_remove()
            self._update_status("EEC fitting mode")

    def _capture_detached_eec_parameters(self) -> bool:
        table = getattr(self, "detached_eec_parameter_table", None)
        if table is None or self.state is None:
            return self.state is not None
        try:
            detached = {parameter.name: parameter for parameter in table.values()}
        except (TypeError, ValueError):
            return False
        current = self.state.parameters_for(self.state.active_cycle)
        for parameter in current:
            source = detached.get(parameter.name)
            if source is None:
                continue
            parameter.initial = source.initial
            parameter.lower = source.lower
            parameter.upper = source.upper
            parameter.fixed = source.fixed
        self.state.active.parameters = current
        self.parameter_table.set_parameters(current)
        return True

    def _capture_detached_drt_parameters(self) -> bool:
        table = getattr(self, "detached_drt_peak_table", None)
        if table is None:
            return True
        try:
            parameters = table.values()
        except (TypeError, ValueError):
            return False
        self.drt_peak_table.set_parameters(parameters)
        return self._sync_drt_peak_parameters_from_table()

    def open_eec_analysis_window(self) -> None:
        if self.state is None or self.busy:
            return
        existing = getattr(self, "eec_analysis_popup", None)
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            existing.focus_force()
            return
        popup = tk.Toplevel(self.root)
        self.eec_analysis_popup = popup
        popup.title("EEC Analysis")
        popup.geometry("430x650")
        popup.minsize(360, 420)
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(2, weight=1)

        def close_popup() -> None:
            self.eec_analysis_popup = None
            self.detached_eec_parameter_table = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", close_popup)
        body = ttk.Frame(popup, padding=10)
        body.grid(row=0, column=0, rowspan=3, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)
        ttk.Label(body, text="Fitting model").grid(row=0, column=0, sticky="w")
        model_var = tk.StringVar(value=self.model_var.get())
        model_box = ttk.Combobox(
            body,
            textvariable=model_var,
            values=self._model_presets,
            state="normal",
        )
        model_box.grid(row=1, column=0, sticky="ew", pady=(3, 8))
        table_group = ttk.LabelFrame(body, text="Circuit parameters", padding=6)
        table_group.grid(row=2, column=0, sticky="nsew")
        table_group.columnconfigure(0, weight=1)
        table_group.rowconfigure(0, weight=1)
        table = ParameterTable(table_group)
        table.grid(row=0, column=0, sticky="nsew")
        self.detached_eec_parameter_table = table
        table.set_parameters(self.state.parameters_for(self.state.active_cycle))

        buttons = ttk.Frame(body)
        buttons.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)

        def apply_model() -> None:
            if not self._capture_detached_eec_parameters():
                return
            self.model_var.set(model_var.get())
            self.apply_model()

        def fit_current() -> None:
            if self._capture_detached_eec_parameters():
                self.fit()

        ttk.Button(buttons, text="Apply model", command=apply_model).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4)
        )
        ttk.Button(buttons, text="Fit spectrum", command=fit_current).grid(
            row=1, column=0, padx=(0, 3), sticky="ew"
        )
        ttk.Button(
            buttons,
            text="Initial values",
            command=lambda: self.initialize_from_ridge(),
        ).grid(row=1, column=1, padx=(3, 0), sticky="ew")

    def open_drt_analysis_window(self) -> None:
        if self.state is None or self.busy:
            return
        existing = getattr(self, "drt_analysis_popup", None)
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            existing.focus_force()
            return
        popup = tk.Toplevel(self.root)
        self.drt_analysis_popup = popup
        popup.title("DRT Analysis")
        popup.geometry("430x650")
        popup.minsize(360, 420)
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(1, weight=1)

        def close_popup() -> None:
            self.drt_analysis_popup = None
            self.detached_drt_peak_table = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", close_popup)
        body = ttk.Frame(popup, padding=10)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        ttk.Label(body, text="Method").grid(row=0, column=0, sticky="w")
        mode_var = tk.StringVar(value=self.analysis_drt_mode_var.get())
        mode_box = ttk.Combobox(
            body,
            textvariable=mode_var,
            values=("Ridge DRT", "Hybrid DRT"),
            state="readonly",
        )
        mode_box.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        table_group = ttk.LabelFrame(body, text="Peak parameters", padding=6)
        table_group.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        table_group.columnconfigure(0, weight=1)
        table_group.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        table = ParameterTable(
            table_group,
            name_double_click=self._on_drt_parameter_double_click,
            display_name=self._drt_display_parameter_name,
        )
        table.grid(row=0, column=0, sticky="nsew")
        self.detached_drt_peak_table = table
        self._update_drt_peak_table()
        table.set_parameters(self.drt_peak_table.values())

        def change_mode(_event=None) -> None:
            self.analysis_drt_mode_var.set(mode_var.get())
            self.drt_mode_var.set(mode_var.get())
            self._refresh_plot(rescale=True)
            self._update_drt_peak_table()
            table.set_parameters(self.drt_peak_table.values())

        mode_box.bind("<<ComboboxSelected>>", change_mode)
        buttons = ttk.Frame(popup, padding=(10, 0, 10, 10))
        buttons.grid(row=1, column=0, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        buttons.columnconfigure(2, weight=1)

        def run_action(action: Callable[[], object]) -> None:
            if self._capture_detached_drt_parameters():
                action()

        ttk.Button(
            buttons,
            text="Fit DRT",
            command=lambda: run_action(self._fit_selected_drts),
        ).grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        ttk.Button(
            buttons,
            text="Add Gaussian peak",
            command=lambda: run_action(lambda: self.add_drt_peak("gaussian")),
        ).grid(row=1, column=0, padx=(0, 3), sticky="ew")
        ttk.Button(
            buttons,
            text="Add Lorentzian peak",
            command=lambda: run_action(lambda: self.add_drt_peak("lorentzian")),
        ).grid(row=1, column=1, padx=3, sticky="ew")
        ttk.Button(
            buttons,
            text="Add Voigt peak",
            command=lambda: run_action(lambda: self.add_drt_peak("voigt")),
        ).grid(row=1, column=2, padx=(3, 0), sticky="ew")
        ttk.Button(
            buttons,
            text="Add HN peak",
            command=lambda: run_action(lambda: self.add_drt_peak("hn")),
        ).grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(
            buttons,
            text="Fit peaks",
            command=lambda: run_action(self.fit_drt_peaks),
        ).grid(row=3, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(
            buttons,
            text="Send initials",
            command=lambda: run_action(self.send_drt_initials),
        ).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(
            buttons,
            text="Remove all peaks",
            command=lambda: run_action(self.remove_all_drt_peaks),
        ).grid(row=5, column=0, columnspan=3, sticky="ew", pady=(4, 0))

    def _poll_analysis_windows(self) -> None:
        if not self.root.winfo_exists():
            return
        current_key = (
            self.current_dataset_id,
            self.state.active_cycle if self.state is not None else None,
        )
        previous_busy = getattr(self, "_analysis_windows_last_busy", False)
        needs_refresh = current_key != getattr(self, "_analysis_windows_cycle_key", None)
        needs_refresh |= previous_busy and not self.busy
        if self.state is not None and needs_refresh:
            table = getattr(self, "detached_eec_parameter_table", None)
            if table is not None and self.eec_analysis_popup.winfo_exists():
                table.set_parameters(self.state.parameters_for(self.state.active_cycle))
            table = getattr(self, "detached_drt_peak_table", None)
            if table is not None and self.drt_analysis_popup.winfo_exists():
                self._update_drt_peak_table()
                table.set_parameters(self.drt_peak_table.values())
            self._analysis_windows_cycle_key = current_key
        self._analysis_windows_last_busy = self.busy
        self._update_window_title()
        self.root.after(300, self._poll_analysis_windows)

    def _fit_selected_drts(self) -> None:
        if self.analysis_drt_mode_var.get() == "Hybrid DRT":
            self.calculate_selected_hybrid_drts()
        else:
            self.calculate_selected_ridge_drts()
        self._set_controls_enabled(False)

    def _on_analysis_drt_mode_selected(self, _event=None) -> None:
        if hasattr(self, "drt_mode_var"):
            self.drt_mode_var.set(self.analysis_drt_mode_var.get())
        if self.state is not None:
            self._refresh_plot(rescale=True)

    def _begin_loading(self) -> None:
        if self.path is None:
            self._set_controls_enabled(False)
            return
        self.status_var.set(f"Loading {self.path.name}…")
        self._submit(
            lambda: [(self.path, inspect_eis_file_spectrum_kinds(self.path))],
            lambda inspections: self._finish_initial_inspection(inspections),
            "Could not open the spectrum",
        )

    def _finish_initial_inspection(
        self, inspections: list[tuple[Path, list[str]]]
    ) -> None:
        selected_kinds = self._select_import_spectrum_kinds(inspections)
        if selected_kinds is None:
            self._update_status("data import cancelled")
            return
        self._submit(
            lambda: load_projects(
                [self.path],
                self.control,
                self.circuit,
                self.requested_cycle,
                spectrum_kinds_by_path=selected_kinds,
            ),
            self._finish_loading,
            "Could not open the spectrum",
        )

    def _finish_loading(self, report: ProjectImportReport) -> None:
        if not report.loaded:
            detail = report.errors[0][1] if report.errors else "No spectra were loaded."
            raise ValueError(detail)
        for dataset_id, loaded in report.loaded:
            self._register_dataset(dataset_id, loaded)
        self._populate_explorer()
        dataset_id, loaded = report.loaded[0]
        self._switch_dataset(
            dataset_id,
            loaded,
            loaded.state.active_cycle,
            capture_current=False,
        )
        self._set_controls_enabled(True)
        self._update_status()

    def _register_dataset(self, dataset_id: str, loaded: LoadedProject) -> None:
        loaded.state.source_path = loaded.state.source_path.resolve()
        if dataset_id not in self.loaded_projects:
            self._dataset_order.append(dataset_id)
        self.loaded_projects[dataset_id] = loaded

    def _switch_dataset(
        self,
        dataset_id: str,
        loaded: LoadedProject,
        cycle_number: int,
        *,
        capture_current: bool = True,
        preserve_existing_selection: bool = True,
        focus_only: bool = False,
    ) -> None:
        if capture_current and self.state is not None and not self._capture_controls():
            self._highlight_explorer_cycle(
                self.state.active_cycle,
                preserve_existing=False,
            )
            return
        state = loaded.state
        if cycle_number not in state.cycles:
            cycle = load_cycle(loaded.dataframe, cycle_number, state.control)
            if state.all_frequency_window is not None:
                cycle.frequency_window = state.all_frequency_window
            cycle.parameters = state.parameters_for(cycle_number)
            cycle.circuit = state.circuit
            state.cycles[cycle_number] = cycle
        state.active_cycle = cycle_number
        self.path = state.source_path.resolve()
        self.current_dataset_id = dataset_id
        self.loaded = loaded
        self.state = state
        self.control = state.control
        self.circuit = state.circuit
        self.root.title(f"EIS Fitting — {loaded.dataset_label}")
        self.cycle_var.set(str(self.state.active_cycle))
        self._highlight_explorer_cycle(
            self.state.active_cycle,
            preserve_existing=preserve_existing_selection,
            focus_only=focus_only,
        )
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._update_status(f"source: {loaded.dataset_label}")
        if self.show_kk_var.get():
            self._ensure_kk_residuals()
        self._set_controls_enabled(self.state is not None)

    def _submit(
        self,
        work: Callable[[], object],
        success: Callable[[object], None],
        error_title: str,
        *,
        operation_labels: list[str] | None = None,
        operation_name: str = "operation",
    ) -> None:
        if self.busy:
            return
        self._fit_cancel_requested = False
        self._stop_event.clear()
        self._operation_labels = list(operation_labels or [])
        self._operation_name = operation_name
        self.busy = True
        if hasattr(self, "stop_fit_button"):
            self.stop_fit_button.configure(state="normal")
        self._set_controls_enabled(False)
        future = self.executor.submit(work)
        self.root.after(40, lambda: self._poll_future(future, success, error_title))

    def _poll_future(
        self,
        future: Future,
        success: Callable[[object], None],
        error_title: str,
    ) -> None:
        if not self.root.winfo_exists():
            return
        if not future.done():
            self.root.after(40, lambda: self._poll_future(future, success, error_title))
            return
        self.busy = False
        try:
            result = future.result()
        except Exception as error:
            if isinstance(error, FitTimeoutError):
                self._restore_fit_initial_parameters()
                self.status_var.set(f"Error: {error}")
                messagebox.showerror("Fit timed out", str(error), parent=self.root)
            else:
                self.status_var.set(f"Error: {error}")
                messagebox.showerror(
                    error_title, f"{type(error).__name__}: {error}", parent=self.root
                )
            self._operation_labels = []
            self._set_controls_enabled(self.state is not None)
            self._fit_parameter_snapshot = None
            return
        success(result)
        self._fit_parameter_snapshot = None
        self._operation_labels = []
        self._set_controls_enabled(self.state is not None)

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled and not self.busy else tk.DISABLED
        for button in getattr(self, "action_buttons", ()):
            button.configure(state=state)
        if hasattr(self, "stop_fit_button"):
            self.stop_fit_button.configure(
                state=tk.NORMAL if self.busy else tk.DISABLED
            )
        if hasattr(self, "drt_mode_box"):
            self.drt_mode_box.configure(
                state="readonly" if enabled and not self.busy else "disabled"
            )
        if hasattr(self, "model_box"):
            self.model_box.configure(
                state="normal" if enabled and not self.busy else "disabled"
            )
        if hasattr(self, "explorer"):
            self.explorer.state(
                ("!disabled",) if enabled and not self.busy else ("disabled",)
            )
        menu_state = tk.NORMAL if enabled and not self.busy else tk.DISABLED
        if hasattr(self, "file_menu"):
            self.file_menu.entryconfigure(
                "Import data…",
                state=tk.DISABLED if self.busy else tk.NORMAL,
            )
            self.file_menu.entryconfigure(
                "Load project…",
                state=tk.DISABLED if self.busy else tk.NORMAL,
            )
            for label in self._project_menu_actions:
                self.file_menu.entryconfigure(label, state=menu_state)
        if hasattr(self, "fit_menu"):
            for label in self._fit_menu_actions:
                self.fit_menu.entryconfigure(label, state=menu_state)
        if hasattr(self, "export_menu"):
            for label in self._export_menu_actions:
                self.export_menu.entryconfigure(label, state=menu_state)

    @staticmethod
    def _automatic_max_frequency(cycle) -> float | None:
        frequency = np.asarray(cycle.frequency_hz, dtype=float).reshape(-1)
        impedance = np.asarray(cycle.impedance, dtype=complex).reshape(-1)
        valid = (
            np.isfinite(frequency)
            & (frequency > 0)
            & np.isfinite(impedance.real)
            & np.isfinite(impedance.imag)
        )
        if np.count_nonzero(valid) < 7:
            return None
        order = np.argsort(frequency[valid])[::-1]
        frequencies = frequency[valid][order]
        impedance = impedance[valid][order]
        real = impedance.real
        negative_imaginary = -impedance.imag

        def rolling_median(values: np.ndarray, width: int) -> np.ndarray:
            half_width = width // 2
            return np.asarray(
                [
                    np.median(values[max(0, index - half_width): index + half_width + 1])
                    for index in range(values.size)
                ],
                dtype=float,
            )

        window = min(5, frequencies.size if frequencies.size % 2 else frequencies.size - 1)
        smoothed_real = rolling_median(real, window)
        smoothed_negative_imaginary = rolling_median(negative_imaginary, window)
        smoothed_real = rolling_median(smoothed_real, 3)
        smoothed_negative_imaginary = rolling_median(smoothed_negative_imaginary, 3)
        delta_real = np.diff(real)
        delta_smoothed_real = np.diff(smoothed_real)
        delta_smoothed_negative_imaginary = np.diff(smoothed_negative_imaginary)
        noise_count = max(5, min(12, delta_smoothed_real.size))
        scale_y = max(
            float(np.ptp(smoothed_negative_imaginary[:noise_count + 1])),
            np.finfo(float).eps,
        )
        y_noise = np.diff(smoothed_negative_imaginary)[:noise_count]
        y_mad = float(np.median(np.abs(y_noise - np.median(y_noise))))
        upward_limit = max(0.01 * scale_y, 1.5 * 1.4826 * y_mad)
        angles = np.arctan2(
            delta_smoothed_negative_imaginary,
            np.abs(delta_smoothed_real) + np.finfo(float).eps,
        )
        angles = rolling_median(angles, 3)
        x_changes = delta_smoothed_real
        x_scale = max(float(np.ptp(smoothed_real)), np.finfo(float).eps)
        x_noise = x_changes[:noise_count]
        x_tolerance = max(
            0.02 * float(np.median(np.abs(x_noise))),
            1e-6 * x_scale,
        )
        search_limit = max(5, int(np.ceil(0.35 * angles.size)))
        for index in range(1, min(search_limit, angles.size)):
            turned_right = (
                x_changes[index - 1] < -x_tolerance
                and x_changes[index] >= -x_tolerance
            )
            if not turned_right or delta_smoothed_negative_imaginary[index] <= upward_limit:
                continue
            if index >= 1 and delta_real[index - 1] < 0 and delta_real[index] < 0:
                continue
            return float(frequencies[index + 1])
        candidates = [
            index
            for index in range(min(search_limit, angles.size))
            if delta_smoothed_negative_imaginary[index] > upward_limit
            and np.isfinite(angles[index])
        ]
        for index in sorted(candidates, key=lambda candidate: angles[candidate], reverse=True):
            if index >= 1 and delta_real[index - 1] < 0 and delta_real[index] < 0:
                continue
            return float(frequencies[index + 1])
        return float(frequencies[0])

    def _update_automatic_max_frequency(self, *, apply: bool = False) -> float | None:
        if not self.auto_max_frequency_var.get() or self.state is None:
            return None
        maximum = self._automatic_max_frequency(self.state.active)
        if maximum is None:
            return None
        try:
            minimum = float(self.minimum_frequency_var.get())
        except ValueError:
            minimum = float(np.nanmin(self.state.active.frequency_hz))
        if minimum > maximum:
            minimum = maximum
            self._frequency_control_guard = True
            try:
                self.minimum_frequency_var.set(f"{minimum:g}")
            finally:
                self._frequency_control_guard = False
        self._frequency_control_guard = True
        try:
            self.maximum_frequency_var.set(f"{maximum:g}")
        finally:
            self._frequency_control_guard = False
        if apply:
            self.state.active.frequency_window = (minimum, maximum)
            self.state.active.invalidate_drt_cache()
        return maximum

    def _schedule_frequency_application(self, *_args) -> None:
        if self._frequency_control_guard or self.busy or self.state is None:
            return
        if self._frequency_apply_after_id is not None:
            self.root.after_cancel(self._frequency_apply_after_id)
        self._frequency_apply_after_id = self.root.after(
            300, self._apply_frequency_controls
        )

    def _apply_frequency_controls(self) -> None:
        self._frequency_apply_after_id = None
        if self._frequency_control_guard or self.busy or self.state is None:
            return
        try:
            minimum = float(self.minimum_frequency_var.get())
            maximum = float(self.maximum_frequency_var.get())
        except ValueError:
            return
        if minimum > maximum:
            return
        if not self._capture_controls():
            return
        self.state.active.clear_fit()
        self._refresh_plot(rescale=True)
        self._update_status("frequency range applied")

    def _toggle_auto_max_frequency(self) -> None:
        self.maximum_frequency_entry.configure(
            state=tk.DISABLED if self.auto_max_frequency_var.get() else tk.NORMAL
        )
        if self.state is None:
            return
        self.state.active.auto_max_frequency = self.auto_max_frequency_var.get()
        if self.auto_max_frequency_var.get():
            maximum = self._update_automatic_max_frequency(apply=True)
        else:
            frequency = np.asarray(self.state.active.frequency_hz, dtype=float)
            valid = frequency[np.isfinite(frequency) & (frequency > 0)]
            maximum = float(np.max(valid)) if valid.size else None
            if maximum is not None:
                self._frequency_control_guard = True
                try:
                    self.maximum_frequency_var.set(f"{maximum:g}")
                finally:
                    self._frequency_control_guard = False
                self._apply_frequency_controls()
        if maximum is None:
            self._update_status("no valid measured frequency is available")
            return
        self._refresh_plot(rescale=True)
        self._update_status(
            f"{'automatic' if self.auto_max_frequency_var.get() else 'manual'} maximum frequency set to {maximum:g} Hz"
        )

    def _restore_controls(self) -> None:
        if self.state is None:
            return
        cycle = self.state.active
        self.parameter_table.set_parameters(self.state.parameters_for(cycle.cycle))
        self.model_var.set(cycle.model(self.state.circuit))
        self.auto_max_frequency_var.set(bool(cycle.auto_max_frequency))
        self.maximum_frequency_entry.configure(
            state=tk.DISABLED if self.auto_max_frequency_var.get() else tk.NORMAL
        )
        self._frequency_control_guard = True
        try:
            if cycle.frequency_window is not None:
                self.minimum_frequency_var.set(f"{cycle.frequency_window[0]:g}")
                self.maximum_frequency_var.set(f"{cycle.frequency_window[1]:g}")
            self._update_automatic_max_frequency(apply=self.auto_max_frequency_var.get())
        finally:
            self._frequency_control_guard = False

    @staticmethod
    def _clamp_parameter_value(value, lower, upper):
        try:
            if lower is not None and value < lower:
                return lower
            if upper is not None and value > upper:
                return upper
        except (TypeError, ValueError):
            pass
        return value

    def _clamp_state_parameter_values(self) -> None:
        state_values = getattr(self.state, "parameters", None)
        if state_values is None:
            return
        if isinstance(state_values, dict):
            parameter_values = state_values.values()
        else:
            parameter_values = state_values
        for parameter in parameter_values:
            if isinstance(parameter, dict):
                value_key = "initial" if "initial" in parameter else "value"
                if value_key in parameter:
                    parameter[value_key] = self._clamp_parameter_value(
                        parameter[value_key],
                        parameter.get("lower"),
                        parameter.get("upper"),
                    )
            else:
                value_name = "initial" if hasattr(parameter, "initial") else "value"
                if hasattr(parameter, value_name):
                    current_value = getattr(parameter, value_name)
                    setattr(
                        parameter,
                        value_name,
                        self._clamp_parameter_value(
                            current_value,
                            getattr(parameter, "lower", None),
                            getattr(parameter, "upper", None),
                        ),
                    )

    def _capture_controls(self) -> bool:
        self._clamp_state_parameter_values()
        if self.state is None:
            return False
        self.state.active.auto_max_frequency = self.auto_max_frequency_var.get()
        self._update_automatic_max_frequency()
        try:
            parameters = self.parameter_table.values()
            minimum = float(self.minimum_frequency_var.get())
            maximum = float(self.maximum_frequency_var.get())
            for parameter in parameters:
                if parameter.lower > parameter.upper:
                    raise ValueError(
                        f"{parameter.name}: lower bound exceeds upper bound"
                    )
                parameter.initial = self._clamp_parameter_value(
                    parameter.initial,
                    parameter.lower,
                    parameter.upper,
                )
            self.parameter_table.set_parameters(parameters)
        except ValueError as error:
            messagebox.showerror("Invalid value", str(error), parent=self.root)
            return False
        self.state.remember_parameters(parameters)
        previous_window = self.state.active.frequency_window
        self.state.active.frequency_window = (minimum, maximum)
        if previous_window != self.state.active.frequency_window:
            self.state.active.invalidate_drt_cache()
        return True

    def apply_parameters_to_selected(
        self,
        fields: set[str] | None = None,
    ) -> None:
        if self.busy or self.state is None:
            return
        if not self._capture_controls():
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        try:
            source_parameters = self.parameter_table.values()
            for parameter in source_parameters:
                if parameter.lower > parameter.upper:
                    raise ValueError(
                        f"{parameter.name}: lower bound exceeds upper bound"
                    )
                if not parameter.lower <= parameter.initial <= parameter.upper:
                    raise ValueError(
                        f"{parameter.name}: initial value is outside its bounds"
                    )
        except ValueError as error:
            messagebox.showerror("Invalid parameter value", str(error), parent=self.root)
            return
        fields = fields or {"fixed", "initial", "lower", "upper"}
        source_by_name = {parameter.name: parameter for parameter in source_parameters}
        updated = 0
        skipped = 0
        for _dataset_id, loaded, spectrum in selected_rows:
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            target_parameters = loaded.state.parameters_for(spectrum.cycle)
            if not target_parameters:
                skipped += 1
                continue
            copied = []
            for target in target_parameters:
                source = source_by_name.get(target.name)
                if source is None:
                    copied.append(target)
                    continue
                initial = source.initial if "initial" in fields else target.initial
                lower = source.lower if "lower" in fields else target.lower
                upper = source.upper if "upper" in fields else target.upper
                fixed = source.fixed if "fixed" in fields else target.fixed
                copied.append(
                    ParameterValue(
                        target.name,
                        target.unit,
                        initial,
                        lower,
                        upper,
                        target.error_percent,
                        fixed,
                    )
                )
            cycle.parameters = copied
            cycle.clear_fit()
            updated += 1
        self._refresh_explorer_values()
        self._restore_controls()
        self._refresh_plot(rescale=True)
        suffix = f", {skipped} skipped" if skipped else ""
        if fields == {"fixed", "initial", "lower", "upper"}:
            action = "parameter settings"
        else:
            action = ", ".join(
                label
                for field, label in (
                    ("fixed", "Fix"),
                    ("initial", "Initial values"),
                    ("lower", "Lower limits"),
                    ("upper", "Upper limits"),
                )
                if field in fields
            )
        self._update_status(f"{action} applied to {updated} spectra{suffix}")

    def _drt_peak_parameter_values(self, peaks) -> list[dict[str, float]]:
        values = []
        for peak in peaks:
            tau, area, fwhm = self._peak_summary(peak)
            shape = self._peak_shape(peak)
            item = {"tau": float(tau), "area": float(area)}
            if shape == "voigt":
                item["sigma"] = float(peak.get("sigma_log10", 0.12))
                item["gamma"] = float(
                    peak.get("gamma_log10", item["sigma"])
                )
            elif shape == "hn":
                item["alpha"] = float(peak.get("alpha", 0.8))
                item["beta"] = float(peak.get("beta", 0.8))
            else:
                item["fwhm"] = float(fwhm)
            values.append(item)
        return values

    def _set_drt_peak_initial_value(
        self,
        peak: dict,
        name: str,
        value: float,
    ) -> None:
        tau, area, fwhm = self._peak_summary(peak)
        shape = self._peak_shape(peak)
        value = float(value)
        if name == "tau":
            tau = max(value, 1e-300)
        elif name == "area":
            area = value
        elif name == "fwhm" and shape not in {"voigt", "hn"}:
            fwhm = max(value, 1e-300)
            peak["sigma_log10"] = self._peak_width_from_fwhm(
                shape, tau, fwhm
            )
        elif name in {"sigma", "gamma"} and shape == "voigt":
            peak[f"{name}_log10"] = max(value, 1e-6)
        elif name in {"alpha", "beta"} and shape == "hn":
            peak[name] = value
        else:
            return
        peak["center_log10"] = float(np.log10(max(tau, 1e-300)))
        if name == "area":
            peak["area"] = area
        peak["height"] = self._peak_height_from_area(
            shape,
            area,
            peak.get("sigma_log10", 0.12),
            peak.get("gamma_log10"),
            peak.get("alpha"),
            peak.get("beta"),
        )

    def apply_drt_parameters_to_selected(
        self,
        fields: set[str] | None = None,
    ) -> None:
        if self.busy or self.state is None:
            return
        if not self._sync_drt_peak_parameters_from_table():
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        fields = fields or {"fixed", "initial", "lower", "upper"}
        source_peaks = copy.deepcopy(self.drt_peak_parameters)
        source_values = self._drt_peak_parameter_values(source_peaks)
        if not source_peaks:
            self._update_status("add at least one DRT peak first")
            return
        mode = self.analysis_drt_mode_var.get()
        peak_attribute = (
            "saved_hybrid_peak_parameters"
            if mode == "Hybrid DRT"
            else "saved_ridge_peak_parameters"
        )
        updated = 0
        skipped = 0
        all_fields = {"fixed", "initial", "lower", "upper"}
        for _dataset_id, loaded, spectrum in selected_rows:
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            target_peaks = copy.deepcopy(getattr(cycle, peak_attribute, None) or [])
            if fields == all_fields:
                target_peaks = copy.deepcopy(source_peaks)
            elif not target_peaks:
                skipped += 1
                continue
            target_values = self._drt_peak_parameter_values(target_peaks)
            for index, target_peak in enumerate(target_peaks):
                if index >= len(source_values) or index >= len(target_values):
                    continue
                source_peak = source_peaks[index]
                source_item = source_values[index]
                target_item = target_values[index]
                target_shape = self._peak_shape(target_peak)
                source_shape = self._peak_shape(source_peak)
                if source_shape != target_shape and fields != all_fields:
                    continue
                for name in target_item:
                    if name not in source_item:
                        continue
                    if "initial" in fields:
                        self._set_drt_peak_initial_value(
                            target_peak, name, source_item[name]
                        )
                    if "lower" in fields:
                        target_peak[f"{name}_lower"] = source_peak.get(
                            f"{name}_lower", target_peak.get(f"{name}_lower")
                        )
                    if "upper" in fields:
                        target_peak[f"{name}_upper"] = source_peak.get(
                            f"{name}_upper", target_peak.get(f"{name}_upper")
                        )
                    if "fixed" in fields:
                        target_peak[f"{name}_fixed"] = bool(
                            source_peak.get(
                                f"{name}_fixed",
                                target_peak.get(f"{name}_fixed", False),
                            )
                        )
            cycle.store_drt_peak_parameters(mode, target_peaks)
            updated += 1
        self._refresh_explorer_values()
        self._restore_controls()
        self._refresh_plot(rescale=True)
        suffix = f", {skipped} skipped" if skipped else ""
        if fields == all_fields:
            action = "DRT peak settings"
        else:
            action = ", ".join(
                label
                for field, label in (
                    ("fixed", "DRT Fix"),
                    ("initial", "DRT Initial values"),
                    ("lower", "DRT Lower limits"),
                    ("upper", "DRT Upper limits"),
                )
                if field in fields
            )
        self._update_status(f"{action} applied to {updated} spectra{suffix}")

    def _update_window_title(self) -> None:
        project_path = next(
            (
                getattr(self, name, None)
                for name in (
                "project_path",
                    "_project_title_path",
                    "_project_path",
                    "current_project_path",
                    "loaded_project_path",
                    "project_file",
                )
                if getattr(self, name, None)
            ),
            None,
        )
        project_name = Path(project_path).name if project_path else "Untitled"
        self.root.title(f"EIS Fitting — {project_name}")

    def load_ml_results(self) -> None:
        directory = filedialog.askdirectory(
            parent=self.root,
            title="Select ML results directory",
            initialdir=str(self._current_directory()),
        )
        if not directory:
            return
        try:
            results = load_ml_results(Path(directory))
        except (OSError, ValueError, TypeError) as error:
            messagebox.showerror("ML results", str(error), parent=self.root)
            return
        self.ml_results = results
        self.ml_results_directory = Path(directory).resolve()
        self._refresh_ml_visuals()
        if results:
            self.ml_results_status_var.set(f"Loaded {len(results)} ML results")
        else:
            self.ml_results_status_var.set("No ML results found")

    def open_ml_processing(self) -> None:
        """Configure and run an ordered ML/EEC pipeline on selected spectra."""
        if self.busy or self.state is None:
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        existing = getattr(self, "ml_processing_popup", None)
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            return

        popup = tk.Toplevel(self.root)
        self.ml_processing_popup = popup
        popup.title("ML processing")
        popup.transient(self.root)
        popup.grab_set()
        popup.resizable(False, False)
        frame = ttk.Frame(popup, padding=12)
        frame.grid(sticky="nsew")
        ttk.Label(frame, text=f"Selected spectra: {len(selected_rows)}").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text="Trained model").grid(row=1, column=0, sticky="w", pady=(8, 2))
        model_var = tk.StringVar(value="Sputtered cathode")
        ttk.Combobox(frame, textvariable=model_var, values=tuple(ML_TRAINED_MODELS), state="readonly", width=28).grid(row=1, column=1, sticky="ew", pady=(8, 2))
        ttk.Label(frame, text="Pipeline actions (top to bottom)").grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 2))
        action_labels = {
            "frequency": "ML frequency range",
            "outliers": "Deterministic outliers removal",
            "model": "ML EEC model selection",
            "initial_parameters": "ML initial parameters",
            "fit": "Fit selected spectra",
            "refine": "Refine fit",
        }
        action_values = tuple(action_labels)
        actions = ["frequency", "outliers", "model", "initial_parameters", "fit", "refine"]
        action_var = tk.StringVar(value="frequency")
        action_box = ttk.Combobox(frame, textvariable=action_var, values=action_values, state="readonly", width=22)
        action_box.grid(row=3, column=0, sticky="ew")
        action_list = tk.Listbox(frame, height=6, width=38, exportselection=False)
        action_list.grid(row=3, column=1, rowspan=3, sticky="ew", padx=(8, 0))
        def render_actions():
            action_list.delete(0, tk.END)
            for action in actions:
                action_list.insert(tk.END, action_labels[action])
        def add_action():
            actions.append(action_var.get()); render_actions()
        def remove_action():
            selection = action_list.curselection()
            if selection: actions.pop(selection[0]); render_actions()
        def move_action(direction):
            selection = action_list.curselection()
            if not selection: return
            index = selection[0]; target = index + direction
            if 0 <= target < len(actions):
                actions[index], actions[target] = actions[target], actions[index]
                render_actions(); action_list.selection_set(target)
        ttk.Button(frame, text="Add", command=add_action).grid(row=4, column=0, sticky="w", pady=3)
        control_row = ttk.Frame(frame); control_row.grid(row=5, column=0, sticky="w")
        ttk.Button(control_row, text="Remove", command=remove_action).pack(side=tk.LEFT)
        ttk.Button(control_row, text="Up", command=lambda: move_action(-1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(control_row, text="Down", command=lambda: move_action(1)).pack(side=tk.LEFT)
        settings = ttk.LabelFrame(frame, text="Deterministic/refinement settings", padding=6)
        settings.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        ttk.Label(settings, text="Outlier threshold").grid(row=0, column=0, sticky="w")
        threshold_var = tk.StringVar(value=self.deterministic_threshold_var.get())
        ttk.Entry(settings, textvariable=threshold_var, width=10).grid(row=0, column=1, padx=5)
        ttk.Label(settings, text="Refine Z threshold").grid(row=0, column=2, sticky="w")
        refine_z_var = tk.StringVar(value=self.refine_z_threshold_var.get())
        ttk.Entry(settings, textvariable=refine_z_var, width=10).grid(row=0, column=3, padx=5)
        ttk.Label(settings, text="Refine iterations").grid(row=0, column=4, sticky="w")
        refine_iterations_var = tk.StringVar(value=self.refine_max_iterations_var.get())
        ttk.Entry(settings, textvariable=refine_iterations_var, width=8).grid(row=0, column=5, padx=5)
        render_actions()
        save_results_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Save ML results sidecar", variable=save_results_var).grid(row=7, column=0, columnspan=2, sticky="w", pady=(6, 2))

        def close() -> None:
            try:
                popup.grab_release()
            except tk.TclError:
                pass
            self.ml_processing_popup = None
            popup.destroy()

        def run() -> None:
            if not actions:
                messagebox.showerror("ML processing", "Select at least one operation.", parent=popup)
                return
            destination = None
            if save_results_var.get():
                destination = filedialog.asksaveasfilename(
                    parent=popup, title="Save calculated ML results",
                    initialdir=str(self._current_directory()),
                    initialfile=f"{self._current_stem()}_ml_results.json",
                    defaultextension=".json",
                    filetypes=(("ML results JSON", "*_ml_results.json"), ("JSON", "*.json")),
                )
                if not destination:
                    return
            close()
            self._start_named_ml_pipeline(model_var.get(), actions, threshold_var.get(), refine_z_var.get(), refine_iterations_var.get(), selected_rows, Path(destination) if destination else None)

        buttons = ttk.Frame(frame)
        buttons.grid(row=8, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="Cancel", command=close).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Run", command=run).pack(side=tk.RIGHT, padx=(0, 8))
        popup.protocol("WM_DELETE_WINDOW", close)

    def _start_named_ml_pipeline(self, model_name: str, actions: list[str], outlier_threshold: str, refine_z: str, refine_iterations: str, selected_rows, destination: Path | None) -> None:
        model_path = ML_TRAINED_MODELS.get(model_name)
        if model_path is None or not model_path.exists():
            messagebox.showerror("ML model unavailable", f"Trained model '{model_name}' is not available:\n{model_path}", parent=self.root)
            return
        try:
            threshold = float(outlier_threshold)
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError("outlier threshold must be positive")
            float(refine_z); iterations = int(refine_iterations)
            if iterations < 1:
                raise ValueError("refinement iterations must be at least 1")
        except ValueError as error:
            messagebox.showerror("Invalid ML pipeline settings", str(error), parent=self.root)
            return
        selected_keys = {(dataset_id, int(spectrum.cycle)) for dataset_id, _loaded, spectrum in selected_rows}
        needs_ml = bool(set(actions) & {"frequency", "model", "initial_parameters"})
        targets = []
        target_labels = []
        failures = []
        for dataset_id in self._dataset_order:
            loaded = self.loaded_projects[dataset_id]
            for spectrum in loaded.spectra:
                if (dataset_id, int(spectrum.cycle)) not in selected_keys:
                    continue
                label = f"{loaded.dataset_label}, cycle {spectrum.cycle}"
                try:
                    cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
                    targets.append(make_runtime_spectrum(f"{dataset_id}::{loaded.state.control}::{spectrum.cycle}", cycle, cycle.model(loaded.state.circuit)))
                    target_labels.append(label)
                except Exception as error:
                    failures.append((label, f"{type(error).__name__}: {error}"))
        if not targets:
            self._show_ml_spectrum_failures(failures, 0, len(selected_rows))
            self._update_status("no valid selected spectra available for ML processing")
            return
        self.status_var.set(f"Preparing {model_name} pipeline for {len(targets)} spectra…")
        def work():
            bundle = load_pipeline_bundle(model_path) if needs_ml else None
            predictions = []
            inference_failures = list(failures)
            if bundle is not None:
                for target, label in zip(targets, target_labels):
                    try:
                        predictions.extend(infer_bundle_records(bundle, [target.record], threshold=threshold))
                    except Exception as error:
                        inference_failures.append((label, f"{type(error).__name__}: {error}"))
            if destination is not None:
                write_ml_results(destination, predictions, source_project=str(self._current_stem()), pipeline={"name": model_name, "actions": list(actions), "outlier_threshold": threshold, "refine_z_threshold": float(refine_z), "refine_max_iterations": iterations})
            return destination, predictions, inference_failures, len(targets)
        self._submit(work, lambda result: self._finish_named_ml_pipeline(result, actions, threshold, float(refine_z), iterations, selected_rows), "ML pipeline failed", operation_labels=[f"{loaded.dataset_label}, cycle {spectrum.cycle}" for _id, loaded, spectrum in selected_rows], operation_name="ML pipeline")

    def _finish_named_ml_pipeline(self, result, actions: list[str], threshold: float, refine_z: float, refine_iterations: int, selected_rows) -> None:
        destination, predictions, failures, processed = result
        self._show_ml_spectrum_failures(failures, processed, len(selected_rows))
        if destination is not None:
            self.ml_results = load_ml_results(destination)
            self.ml_results_directory = destination.resolve()
        else:
            payload = {"format": "eis-fitting-ml-results", "version": 1, "source_project": str(self._current_stem()), "spectra": predictions}
            self.ml_results = load_ml_results_payload(payload)
            self.ml_results_directory = None
        self._attach_ml_initial_results_to_projects()
        self._refresh_ml_visuals()
        self._run_named_ml_pipeline_steps(actions, 0, threshold, refine_z, refine_iterations, selected_rows)

    def _show_ml_spectrum_failures(self, failures, processed: int, total: int) -> None:
        if not failures:
            return
        details = "\n".join(f"• {label}: {error}" for label, error in failures)
        messagebox.showwarning(
            "ML pipeline: spectra skipped",
            f"Processed {processed} of {total} selected spectra.\n\n{details}",
            parent=self.root,
        )

    def _run_named_ml_pipeline_steps(self, actions: list[str], index: int, threshold: float, refine_z: float, refine_iterations: int, selected_rows) -> None:
        if index >= len(actions):
            self._update_status("ML pipeline completed")
            return
        action = actions[index]
        if action == "frequency":
            self.apply_ml_frequency_to_selected()
        elif action == "outliers":
            for _dataset_id, loaded, spectrum in selected_rows:
                cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
                indices, _diagnostics = detect_outliers_in_active_points(cycle.frequency_hz, cycle.impedance, cycle.included, threshold=threshold)
                cycle.apply_outliers(indices)
            self._refresh_explorer_values(); self._refresh_plot(rescale=True)
        elif action == "model":
            self._run_ml_processing({"model"}, selected_rows)
        elif action == "initial_parameters":
            self._apply_ml_initial_parameters_to_selected()
        elif action == "fit":
            self._ml_pipeline_pending = (actions, index + 1, threshold, refine_z, refine_iterations, selected_rows)
            self.fit_selected()
            return
        elif action == "refine":
            self.refine_z_threshold_var.set(str(refine_z)); self.refine_max_iterations_var.set(str(refine_iterations))
            self._ml_pipeline_pending = (actions, index + 1, threshold, refine_z, refine_iterations, selected_rows)
            self.refine_fit_selected()
            return
        self.root.after(0, lambda: self._run_named_ml_pipeline_steps(actions, index + 1, threshold, refine_z, refine_iterations, selected_rows))

    def _continue_named_ml_pipeline(self) -> bool:
        pending = getattr(self, "_ml_pipeline_pending", None)
        if pending is None:
            return False
        self._ml_pipeline_pending = None
        actions, index, threshold, refine_z, refine_iterations, selected_rows = pending
        self.root.after(0, lambda: self._run_named_ml_pipeline_steps(actions, index, threshold, refine_z, refine_iterations, selected_rows))
        return True

    def _start_runtime_ml_processing(self, operations: set[str], selected_rows, destination: Path) -> None:
        """Load the pretrained bundle and infer the selected open spectra."""
        selected_keys = {(dataset_id, int(spectrum.cycle)) for dataset_id, _loaded, spectrum in selected_rows}
        artifacts, missing_artifacts = discover_pretrained_artifacts()
        if artifacts is None:
            command = (
                ".\\.venv\\Scripts\\python.exe -m ml.run_stage4a_parameters <six training projects>\n"
                ".\\.venv\\Scripts\\python.exe -m ml.run_stage4b_parameters <six training projects>\n"
                "Then regenerate the incompatible HGB artifacts with the current environment."
            )
            messagebox.showerror(
                "ML model bundle unavailable",
                "The pretrained ML bundle is incomplete:\n\n"
                + "\n".join(missing_artifacts)
                + "\n\nRequired training commands:\n"
                + command,
                parent=self.root,
            )
            return
        targets = []
        target_labels = []
        failures = []
        for dataset_id in self._dataset_order:
            loaded = self.loaded_projects[dataset_id]
            for spectrum in loaded.spectra:
                cycle_number = int(spectrum.cycle)
                key = (dataset_id, cycle_number)
                if key not in selected_keys:
                    continue
                label = f"{loaded.dataset_label}, cycle {cycle_number}"
                try:
                    cycle = self._loaded_cycle_for_popup(loaded, cycle_number)
                    runtime = make_runtime_spectrum(
                        f"{dataset_id}::{loaded.state.control}::{cycle_number}",
                        cycle,
                        cycle.model(loaded.state.circuit),
                    )
                    targets.append(runtime)
                    target_labels.append(label)
                except Exception as error:
                    failures.append((label, f"{type(error).__name__}: {error}"))
        if not targets:
            self._show_ml_spectrum_failures(failures, 0, len(selected_rows))
            self._update_status("no valid selected spectra available for ML processing")
            return
        self.status_var.set(f"Calculating ML predictions for {len(targets)} selected spectra…")

        def work():
            predictions = []
            inference_failures = list(failures)
            for target, label in zip(targets, target_labels):
                try:
                    predictions.extend(infer_pretrained(artifacts, [target], operations=operations))
                except Exception as error:
                    inference_failures.append((label, f"{type(error).__name__}: {error}"))
            save_runtime_results(destination, predictions, training_count=6, operations=operations)
            return destination, predictions, inference_failures, len(targets)

        self._submit(
            work,
            lambda result: self._finish_runtime_ml_processing(result, operations, selected_rows),
            "ML processing failed",
            operation_labels=[f"{loaded.dataset_label}, cycle {spectrum.cycle}" for _id, loaded, spectrum in selected_rows],
            operation_name="ML inference",
        )

    def _finish_runtime_ml_processing(self, result, operations: set[str], selected_rows) -> None:
        destination, predictions, failures, processed = result
        self._show_ml_spectrum_failures(failures, processed, len(selected_rows))
        self.ml_results = load_ml_results(destination)
        self.ml_results_directory = Path(destination).resolve()
        self.ml_results_status_var.set(f"Calculated and saved {len(self.ml_results)} ML results")
        self._run_ml_processing(operations, selected_rows)

    def _run_ml_processing(self, operations: set[str], selected_rows=None) -> None:
        """Apply loaded ML predictions, then optionally start the normal EEC fit."""
        if self.busy or self.state is None:
            return
        selected_rows = selected_rows or self._selected_spectrum_rows()
        if bool(operations & {"frequency", "active_points", "model", "initial_parameters"}) and any(
            self._loaded_cycle_for_popup(loaded, spectrum.cycle).fit_parameters is not None
            for _dataset_id, loaded, spectrum in selected_rows
        ) and not messagebox.askyesno(
            "Replace existing fits?",
            "The selected ML operations will clear existing fits before recalculation. Continue?",
            parent=self.root,
        ):
            return
        assignments = []
        missing: list[str] = []
        for dataset_id, loaded, spectrum in selected_rows:
            result = self._ml_result_for_spectrum(dataset_id, loaded, spectrum)
            label = f"{loaded.dataset_label}, cycle {spectrum.cycle}"
            if result is None:
                missing.append(label)
                continue
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            if "frequency" in operations:
                if not result.frequency_ranges:
                    missing.append(f"{label} (frequency range unavailable)")
                    continue
                self._preserve_ml_original_selection(cycle)
                cycle.frequency_window = tuple(result.frequency_ranges[0])
                cycle.invalidate_drt_cache()
                cycle.clear_fit()
            if "active_points" in operations:
                if result.active_mask is None or result.active_mask.size != cycle.frequency_hz.size:
                    missing.append(f"{label} (active-point mask unavailable or misaligned)")
                    continue
                self._preserve_ml_original_selection(cycle)
                cycle.manually_included = result.active_mask.copy()
                if result.outlier_mask is not None and result.outlier_mask.size == cycle.frequency_hz.size:
                    cycle.outliers = result.outlier_mask.copy()
                else:
                    cycle.outliers = ~cycle.manually_included
                cycle.invalidate_drt_cache()
                cycle.clear_fit()
            if "model" in operations or "initial_parameters" in operations:
                circuit = suggested_eec(result)
                if not circuit:
                    missing.append(f"{label} (EEC model unavailable)")
                    continue
                try:
                    parameters = circuit_parameters(circuit, self._eec_parameter_bounds)
                except Exception:
                    missing.append(f"{label} (invalid EEC model)")
                    continue
                if "model" in operations:
                    self._configure_cycle_model(cycle, circuit, parameters, loaded.state.circuit)
                if "initial_parameters" in operations:
                    by_name = {parameter.name: parameter for parameter in cycle.parameters}
                    current_model = cycle.model(loaded.state.circuit)
                    mapping = parameter_name_mapping(circuit, current_model) if circuits_equivalent(circuit, current_model) else {}
                    for name, value in result.model_parameters.items():
                        target_name = map_parameter_name(name, mapping) or name
                        parameter = by_name.get(target_name)
                        if parameter is not None:
                            limits = result.parameter_limits.get(name)
                            if limits is not None:
                                parameter.lower, parameter.upper = limits
                            parameter.initial = self._clamp_parameter_value(value, parameter.lower, parameter.upper)
                    cycle.clear_fit()
            assignments.append((dataset_id, loaded, spectrum, cycle))

        if assignments and assignments[0][1] is self.loaded and assignments[0][3].cycle == self.state.active_cycle:
            self.model_var.set(self.state.active.model(self.state.circuit))
            self.parameter_table.set_parameters(self.state.parameters_for(self.state.active_cycle))
        self._refresh_explorer_values()
        self._refresh_plot(rescale=True)
        if "fit" not in operations:
            self._update_status(f"ML processing applied to {len(assignments)} spectra; unavailable: {len(missing)}")
            return
        targets = [
            SpectrumFitTarget(loaded=loaded, cycle=spectrum.cycle, label=f"{loaded.dataset_label}, cycle {spectrum.cycle}")
            for _dataset_id, loaded, spectrum, _cycle in assignments
        ]
        if not targets:
            self._update_status(f"ML processing unavailable for {len(missing)} selected spectra")
            return
        self.status_var.set(f"ML processing and fitting {len(targets)} selected spectra…")
        initial_parameters = assignments[0][3].parameters
        self._submit(
            lambda: batch_fit_spectra(
                targets,
                initial_parameters,
                use_target_initial_parameters=True,
                stop_event=self._stop_event,
                fit_timeout_seconds=self._fit_timeout_seconds,
            ),
            lambda report: self._finish_ml_processing(report, len(missing)),
            "ML EEC fit failed",
            operation_labels=[target.label for target in targets],
            operation_name="ML processing and fit",
        )

    def _finish_ml_processing(self, report: SpectrumBatchReport, unavailable: int) -> None:
        self._finish_explorer_batch_fit(report)
        self._update_status(
            f"ML processing and fit completed for {len(report.fits)} spectra; unavailable: {unavailable}"
        )

    def load_ml_results_file(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load ML results file",
            initialdir=str(self._current_directory()),
            filetypes=(("ML results JSON", "*_ml_results.json"), ("JSON", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            results = load_ml_results(Path(path))
        except (OSError, ValueError, TypeError) as error:
            messagebox.showerror("ML results", str(error), parent=self.root)
            return
        self.ml_results = results
        self.ml_results_directory = Path(path).resolve()
        self._attach_ml_initial_results_to_projects()
        self._refresh_ml_visuals()
        self.ml_results_status_var.set(
            f"Loaded {len(results)} ML results" if results else "No ML results found"
        )

    def load_ml_initial_parameters(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load ML initial parameters",
            initialdir=str(self._current_directory()),
            filetypes=(("EIS-fit JSON", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            results = load_ml_results(Path(path))
        except (OSError, ValueError, TypeError) as error:
            messagebox.showerror("ML initial parameters", str(error), parent=self.root)
            return
        self.ml_results = results
        self.ml_results_directory = Path(path).resolve()
        self.ml_results_status_var.set(
            f"Loaded {len(results)} ML initial parameter results"
            if results else "No ML initial parameters found"
        )
        self._attach_ml_initial_results_to_projects()
        self._refresh_ml_visuals()

    def load_and_apply_ml_initial_parameters(self) -> None:
        """Apply parameters from the already loaded ML sidecar."""
        if not self.ml_results:
            self._update_status("load an ML results file first")
            return
        self._apply_ml_initial_parameters_to_selected()

    def _ml_result_for_spectrum(
        self, dataset_id: str, loaded: LoadedProject, spectrum: SpectrumMetadata
    ) -> MLResult | None:
        cycle = int(spectrum.cycle)
        control = str(loaded.state.control).casefold()
        source_path = str(loaded.state.source_path.resolve()).casefold()
        dataset_text = str(dataset_id).casefold()
        try:
            spectrum_key = spectrum_identifier(
                (loaded_cycle := self._loaded_cycle_for_popup(loaded, cycle)).frequency_hz,
                loaded_cycle.impedance.real,
                loaded_cycle.impedance.imag,
                cycle,
                loaded.state.control,
            )
        except (TypeError, ValueError):
            spectrum_key = ""
        candidates = []
        for result in self.ml_results.values():
            if result.cycle != cycle:
                continue
            if result.control and result.control.casefold() != control:
                continue
            if spectrum_key and result.spectrum_key == spectrum_key:
                return result
            result_source = (result.source_name or "").casefold()
            result_id = result.spectrum_id.casefold()
            exact_source = result_source in {source_path, dataset_text}
            source_in_id = source_path in result_id or dataset_text in result_id
            if exact_source or source_in_id:
                candidates.append(result)
        if len(candidates) == 1:
            return candidates[0]
        # Legacy ML files include the project path before the dataset id.
        basename = loaded.state.source_path.name.casefold()
        candidates = [
            result for result in self.ml_results.values()
            if result.cycle == cycle
            and (not result.control or result.control.casefold() == control)
            and (
                basename in (result.source_name or "").casefold()
                or basename in result.spectrum_id.casefold()
            )
        ]
        if len(candidates) == 1:
            return candidates[0]
        stored = self._loaded_cycle_for_popup(loaded, cycle).custom_metadata.get(
            "_ml_initial_eec"
        )
        if not isinstance(stored, dict):
            return None
        return MLResult(
            spectrum_id=spectrum_key or f"{source_path}::{control}::{cycle}",
            source_name=str(loaded.state.source_path),
            cycle=cycle,
            control=control,
            model_circuit=stored.get("model_circuit"),
            suggested_eec=stored.get("model_circuit"),
            model_parameters=dict(stored.get("model_parameters", {})),
            initial_sources=dict(stored.get("initial_sources", {})),
            frequency_ranges=[tuple(value) for value in stored.get("frequency_ranges", [])],
        )

    def _preserve_ml_original_selection(self, cycle) -> None:
        metadata = cycle.custom_metadata
        metadata.setdefault(
            "_ml_original_frequency_window",
            list(cycle.frequency_window) if cycle.frequency_window is not None else None,
        )
        metadata.setdefault(
            "_ml_original_manually_included",
            cycle.manually_included.astype(bool).tolist(),
        )
        metadata.setdefault(
            "_ml_original_outliers",
            cycle.outliers.astype(bool).tolist(),
        )

    def apply_ml_frequency_to_selected(self) -> None:
        if self.busy or self.state is None:
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        updated = 0
        missing = 0
        for dataset_id, loaded, spectrum in selected_rows:
            result = self._ml_result_for_spectrum(dataset_id, loaded, spectrum)
            if result is None or not result.frequency_ranges:
                missing += 1
                continue
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            self._preserve_ml_original_selection(cycle)
            cycle.frequency_window = tuple(result.frequency_ranges[0])
            cycle.clear_fit()
            updated += 1
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._update_status(f"ML frequency selection applied to {updated} spectra; unavailable: {missing}")

    def apply_ml_active_points_to_selected(self) -> None:
        if self.busy or self.state is None:
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        updated = 0
        missing = 0
        for dataset_id, loaded, spectrum in selected_rows:
            result = self._ml_result_for_spectrum(dataset_id, loaded, spectrum)
            if result is None or result.active_mask is None:
                missing += 1
                continue
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            if result.active_mask.size != cycle.frequency_hz.size:
                missing += 1
                continue
            self._preserve_ml_original_selection(cycle)
            cycle.manually_included = result.active_mask.copy()
            if result.outlier_mask is not None and result.outlier_mask.size == cycle.frequency_hz.size:
                cycle.outliers = result.outlier_mask.copy()
            else:
                cycle.outliers = ~cycle.manually_included
            cycle.clear_fit()
            cycle.invalidate_drt_cache()
            updated += 1
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._update_status(f"ML active-point selection applied to {updated} spectra; unavailable: {missing}")

    def restore_ml_original_selection(self) -> None:
        if self.busy or self.state is None:
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        restored = 0
        for _dataset_id, loaded, spectrum in selected_rows:
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            metadata = cycle.custom_metadata
            included = metadata.get("_ml_original_manually_included")
            outliers = metadata.get("_ml_original_outliers")
            if not isinstance(included, list) or not isinstance(outliers, list):
                continue
            if len(included) != cycle.frequency_hz.size or len(outliers) != cycle.frequency_hz.size:
                continue
            cycle.manually_included = np.asarray(included, dtype=bool)
            cycle.outliers = np.asarray(outliers, dtype=bool)
            window = metadata.get("_ml_original_frequency_window")
            cycle.frequency_window = (
                tuple(float(value) for value in window)
                if isinstance(window, list) and len(window) == 2
                else None
            )
            cycle.clear_fit()
            cycle.invalidate_drt_cache()
            restored += 1
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._update_status(f"original selection restored for {restored} spectra")

    def _apply_ml_initial_parameters_to_selected(self) -> None:
        if self.busy or self.state is None:
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        updated = 0
        missing_results = []
        missing_parameters = []
        for dataset_id, loaded, spectrum in selected_rows:
            result = self._ml_result_for_spectrum(dataset_id, loaded, spectrum)
            if result is None or not result.model_parameters:
                missing_results.append(f"{dataset_id}:{spectrum.cycle}")
                continue
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            circuit = suggested_eec(result) or result.model_circuit
            if not circuit:
                missing_results.append(f"{dataset_id}:{spectrum.cycle}")
                continue
            current_model = cycle.model(loaded.state.circuit)
            equivalent = circuits_equivalent(current_model, circuit)
            if equivalent:
                parameters = cycle.parameters
            elif cycle.fit_parameters is not None:
                missing_results.append(f"{dataset_id}:{spectrum.cycle} (model already fitted)")
                continue
            else:
                parameters = circuit_parameters(circuit, self._eec_parameter_bounds)
                cycle.circuit = circuit
            by_name = {parameter.name: parameter for parameter in parameters}
            for name, value in result.model_parameters.items():
                target_name = name
                if equivalent:
                    mapping = parameter_name_mapping(circuit, current_model)
                    target_name = map_parameter_name(name, mapping or {}) or name
                parameter = by_name.get(target_name)
                if parameter is not None:
                    limits = result.parameter_limits.get(name)
                    if limits is not None:
                        parameter.lower, parameter.upper = limits
                    parameter.initial = self._clamp_parameter_value(
                        value, parameter.lower, parameter.upper
                    )
            missing = [name for name in by_name if name not in result.model_parameters]
            if missing:
                missing_parameters.extend(f"{spectrum.cycle}: {name}" for name in missing)
            cycle.parameters = parameters
            updated += 1
            if loaded is self.loaded and cycle.cycle == self.state.active_cycle:
                self.model_var.set(cycle.model(self.state.circuit))
                self.parameter_table.set_parameters(parameters)
        self._refresh_explorer_values()
        self._refresh_plot(rescale=True)
        message = f"ML initial parameters loaded for {updated} selected spectra; no fit was started"
        if missing_results:
            message += f"; unavailable: {', '.join(missing_results)}"
        if missing_parameters:
            message += f"; missing parameters: {', '.join(missing_parameters)}"
        self._update_status(message)

    def _attach_ml_initial_results_to_projects(self) -> None:
        for loaded in self.loaded_projects.values():
            source_name = loaded.state.source_path.name.casefold()
            for spectrum in loaded.spectra:
                candidates = [
                    result for result in self.ml_results.values()
                    if result.cycle == spectrum.cycle
                    and (
                        source_name in (result.source_name or "").casefold()
                        or source_name in result.spectrum_id.casefold()
                    )
                    and (
                        not result.control
                        or result.control.casefold() == loaded.state.control.casefold()
                    )
                ]
                if len(candidates) != 1:
                    continue
                result = candidates[0]
                cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
                cycle.custom_metadata["_ml_initial_eec"] = {
                    "model_circuit": result.model_circuit,
                    "model_parameters": dict(result.model_parameters),
                    "initial_sources": dict(result.initial_sources),
                    "frequency_ranges": [list(value) for value in result.frequency_ranges],
                }

    def _current_ml_result(self) -> MLResult | None:
        if not self.ml_results or self.state is None or self.loaded is None:
            return None
        cycle = self.state.active_cycle
        control = self.state.control
        current_spectrum = next(
            (item for item in self.loaded.spectra if item.cycle == cycle), None
        )
        if current_spectrum is not None:
            matched = self._ml_result_for_spectrum(
                self.current_dataset_id or self.loaded.dataset_id,
                self.loaded,
                current_spectrum,
            )
            if matched is not None:
                return matched
        exact_ids = []
        if self.project_path is not None:
            exact_ids.append(
                f"{self.project_path.resolve()}::{self.loaded.dataset_id}::{control}::{cycle}"
            )
        exact_ids.append(
            f"{self.loaded.state.source_path.resolve()}::{control}::{cycle}"
        )
        for identifier in exact_ids:
            for result_id, result in self.ml_results.items():
                if result_id.casefold() == identifier.casefold():
                    return result
        candidates = []
        if self.loaded.dataset_id.startswith("ml-results::"):
            source_name = self.loaded.dataset_id.removeprefix("ml-results::").casefold()
            candidates = [
                result
                for result in self.ml_results.values()
                if result.cycle == cycle
                and (result.source_name or "").casefold() == source_name
            ]
            if len(candidates) == 1:
                return candidates[0]
        project_names = {
            path.name.casefold()
            for path in (self.project_path, self.loaded.state.source_path)
            if path is not None
        }
        for result in self.ml_results.values():
            if result.cycle != cycle:
                continue
            identifier = result.spectrum_id.casefold()
            if control.casefold() not in identifier:
                continue
            source_name = (
                Path(result.source_project).name.casefold()
                if result.source_project
                else ""
            )
            if source_name in project_names or any(
                name in identifier for name in project_names if name
            ):
                candidates.append(result)
        if len(candidates) == 1:
            return candidates[0]
        stored = self.state.active.custom_metadata.get("_ml_initial_eec")
        if isinstance(stored, dict):
            return MLResult(
                spectrum_id=f"{self.loaded.state.source_path.resolve()}::{control}::{cycle}",
                source_name=str(self.loaded.state.source_path),
                cycle=cycle,
                source_project=str(self.loaded.state.source_path),
                control=control,
                model_circuit=stored.get("model_circuit"),
                suggested_eec=stored.get("model_circuit"),
                model_parameters=dict(stored.get("model_parameters", {})),
                initial_sources=dict(stored.get("initial_sources", {})),
                frequency_ranges=[tuple(value) for value in stored.get("frequency_ranges", [])],
            )
        return None

    @staticmethod
    def _ml_model_curve(result: MLResult, frequency: np.ndarray):
        if not result.model_circuit:
            return None
        try:
            from impedance.models.circuits import CustomCircuit

            circuit = circuit_parameters(result.model_circuit)
            values = [result.model_parameters[p.name] for p in circuit]
            model = CustomCircuit(result.model_circuit, initial_guess=values)
            model.parameters_ = np.asarray(values, dtype=float)
            return np.asarray(model.predict(frequency), dtype=complex)
        except (KeyError, TypeError, ValueError, ImportError):
            return None

    def _refresh_ml_visuals(self) -> None:
        if not hasattr(self, "ml_model_artist"):
            return
        for patch in getattr(self, "_ml_range_patches", []):
            patch.remove()
        self._ml_range_patches = []
        self.ml_range_artist.set_data([], [])
        self.ml_active_artist.set_data([], [])
        self.ml_rejected_artist.set_data([], [])
        self.ml_model_artist.set_data([], [])
        self.ml_residual_artist.set_segments([])
        for artist in (
            self.ml_phase_active_artist,
            self.ml_phase_rejected_artist,
            self.ml_phase_model_artist,
        ):
            if artist is not None:
                artist.set_data([], [])
        if self.ml_phase_residual_artist is not None:
            self.ml_phase_residual_artist.set_segments([])
        result = self._current_ml_result()
        if result is None or self.state is None:
            if self.ml_results:
                self.ml_results_status_var.set("No ML results available")
            return
        cycle = self.state.active
        frequency = np.asarray(cycle.frequency_hz, dtype=float)
        impedance = np.asarray(cycle.impedance, dtype=complex)
        range_mask = np.zeros(frequency.size, dtype=bool)
        for minimum, maximum in result.frequency_ranges:
            range_mask |= (frequency >= minimum) & (frequency <= maximum)
            if self.show_ml_frequency_ranges_var.get() and self.plot_mode == "bode":
                self._ml_range_patches.append(
                    self.axes.axvspan(
                        minimum, maximum, color="#80cbc4", alpha=0.12,
                        label="_nolegend_",
                    )
                )
                if self.phase_axes is not None:
                    self._ml_range_patches.append(
                        self.phase_axes.axvspan(
                            minimum, maximum, color="#80cbc4", alpha=0.08,
                        )
                    )
        if self.show_ml_frequency_ranges_var.get():
            self.ml_range_artist.set_data(
                impedance.real[range_mask], -impedance.imag[range_mask]
            )
        mask = None
        if result.active_mask is not None and result.active_mask.size == frequency.size:
            mask = result.active_mask
        elif result.frequency_ranges:
            mask = range_mask
        if self.show_ml_active_points_var.get() and mask is not None:
            self.ml_active_artist.set_data(
                impedance.real[mask], -impedance.imag[mask]
            )
            self.ml_rejected_artist.set_data(
                impedance.real[~mask], -impedance.imag[~mask]
            )
            if self.phase_axes is not None:
                phase = self._phase_degrees(impedance)
                self.ml_phase_active_artist.set_data(frequency[mask], phase[mask])
                self.ml_phase_rejected_artist.set_data(frequency[~mask], phase[~mask])
        model_at_data = self._ml_model_curve(result, frequency)
        if self.show_ml_model_var.get() and model_at_data is not None:
            smooth_frequency = np.geomspace(
                float(np.min(frequency[frequency > 0])),
                float(np.max(frequency[frequency > 0])),
                300,
            )
            smooth_model = self._ml_model_curve(result, smooth_frequency)
            if smooth_model is not None:
                if self.plot_mode == "bode":
                    self.ml_model_artist.set_data(smooth_frequency, np.abs(smooth_model))
                    if self.ml_phase_model_artist is not None:
                        self.ml_phase_model_artist.set_data(
                            smooth_frequency, self._phase_degrees(smooth_model)
                        )
                else:
                    self.ml_model_artist.set_data(smooth_model.real, -smooth_model.imag)
            if self.show_ml_residuals_var.get():
                if self.plot_mode == "bode":
                    magnitude_segments = np.stack(
                        (
                            np.column_stack((frequency, np.abs(impedance))),
                            np.column_stack((frequency, np.abs(model_at_data))),
                        ),
                        axis=1,
                    )
                    self.ml_residual_artist.set_segments(magnitude_segments)
                    if self.ml_phase_residual_artist is not None:
                        phase_segments = np.stack(
                            (
                                np.column_stack((frequency, self._phase_degrees(impedance))),
                                np.column_stack((frequency, self._phase_degrees(model_at_data))),
                            ),
                            axis=1,
                        )
                        self.ml_phase_residual_artist.set_segments(phase_segments)
                else:
                    residual_segments = np.stack(
                        (
                            np.column_stack((impedance.real, -impedance.imag)),
                            np.column_stack((model_at_data.real, -model_at_data.imag)),
                        ),
                        axis=1,
                    )
                    self.ml_residual_artist.set_segments(residual_segments)
        self.ml_range_artist.set_visible(self.show_ml_frequency_ranges_var.get())
        self.ml_active_artist.set_visible(self.show_ml_active_points_var.get())
        self.ml_rejected_artist.set_visible(self.show_ml_active_points_var.get())
        self.ml_model_artist.set_visible(
            self.show_ml_model_var.get() and model_at_data is not None
        )
        if self.ml_phase_active_artist is not None:
            self.ml_phase_active_artist.set_visible(self.show_ml_active_points_var.get())
            self.ml_phase_rejected_artist.set_visible(self.show_ml_active_points_var.get())
            self.ml_phase_model_artist.set_visible(
                self.show_ml_model_var.get() and model_at_data is not None
            )
        self.ml_residual_artist.set_visible(self.show_ml_residuals_var.get())
        if self.ml_phase_residual_artist is not None:
            self.ml_phase_residual_artist.set_visible(self.show_ml_residuals_var.get())
        details = []
        if result.topology_prediction:
            details.append(f"model={result.topology_prediction}")
        if result.frequency_ranges:
            details.append(f"{len(result.frequency_ranges)} range(s)")
        if result.active_mask is not None:
            details.append("active points")
        if not result.has_eec_model:
            details.append("no ML EEC parameters")
        self.ml_results_status_var.set("ML: " + ", ".join(details) if details else "ML result available")
        self._update_legend_visibility()
        self.canvas.draw_idle()

    def _refresh_simulator_plot(self, rescale: bool = False) -> None:
        spectrum = self.simulator_spectrum
        if spectrum is None:
            self.included_artist.set_data([], [])
            self.excluded_artist.set_data([], [])
            self.fit_artist.set_data([], [])
            self.axes.set_title("Spectra Simulator")
            self.canvas.draw_idle()
            return
        frequency = spectrum.frequency_hz
        impedance = spectrum.impedance
        ideal = spectrum.ideal_impedance
        if self.plot_mode == "bode":
            self.included_artist.set_data(frequency, np.abs(impedance))
            self.excluded_artist.set_data([], [])
            if self.phase_included_artist is not None:
                self.phase_included_artist.set_data(frequency, self._phase_degrees(impedance))
                self.phase_excluded_artist.set_data([], [])
            self.fit_artist.set_data(frequency, np.abs(ideal))
            if self.phase_fit_artist is not None:
                self.phase_fit_artist.set_data(frequency, self._phase_degrees(ideal))
            self.axes.set_title("Spectra Simulator · Bode")
        else:
            self.included_artist.set_data(impedance.real, -impedance.imag)
            self.excluded_artist.set_data([], [])
            self.fit_artist.set_data(ideal.real, -ideal.imag)
            self.axes.set_title("Spectra Simulator · Nyquist")
        self.fit_artist.set_visible(self.show_eec_fit_var.get())
        self.included_artist.set_visible(True)
        self.excluded_artist.set_visible(False)
        if self.phase_fit_artist is not None:
            self.phase_fit_artist.set_visible(self.show_eec_fit_var.get())
        if self.drt_artist is not None and self.drt_axes is not None:
            result = self.simulator_drt_result
            if self.show_drt_var.get() and result is not None:
                self.drt_artist.set_data(result.tau_s, result.gamma_ohm)
                self.drt_artist.set_visible(True)
                self.drt_axes.set_title(self.simulator_drt_mode_var.get())
            else:
                self.drt_artist.set_data([], [])
                self.drt_artist.set_visible(False)
        if rescale:
            if self.plot_mode == "bode":
                self.axes.relim()
                self.axes.autoscale_view()
                if self.phase_axes is not None:
                    self.phase_axes.relim()
                    self.phase_axes.autoscale_view()
            else:
                self.axes.relim()
                self.axes.autoscale_view()
            if self.drt_axes is not None and self.drt_artist is not None:
                self.drt_axes.relim()
                self.drt_axes.autoscale_view()
        self.canvas.draw_idle()

    def _refresh_plot(self, rescale: bool = False) -> None:
        if self.analysis_mode_var.get() == "Spectra Simulator":
            self._refresh_simulator_plot(rescale)
            return
        if self.state is None:
            return
        cycle = self.state.active
        drt_key = (
            self.current_dataset_id,
            cycle.cycle,
            self.analysis_drt_mode_var.get()
            if hasattr(self, "analysis_drt_mode_var")
            else "Ridge DRT",
        )
        if (
            drt_key[2] == "Hybrid DRT"
            and cycle.saved_hybrid_tau_s is not None
            and cycle.saved_hybrid_inductance is None
        ):
            cycle.saved_hybrid_inductance = self._estimate_drt_inductance(cycle)
        if drt_key != self._drt_peak_cycle_key:
            saved_name = (
                "saved_hybrid_peak_parameters"
                if drt_key[2] == "Hybrid DRT"
                else "saved_ridge_peak_parameters"
            )
            self.drt_peak_parameters = [
                dict(peak)
                for peak in getattr(cycle, saved_name, [])
            ]
            if (
                drt_key[2] == "Hybrid DRT"
                and cycle.saved_hybrid_tau_s is not None
                and cycle.saved_hybrid_inductance is None
            ):
                cycle.saved_hybrid_inductance = self._estimate_drt_inductance(cycle)
            self._drt_aux_parameter_limits = {}
            self._selected_drt_peak_index = None
            self._clamp_drt_peak_parameters_to_limits()
            self._drt_peak_cycle_key = drt_key
        included = cycle.included
        if self.plot_mode == "bode":
            frequency = cycle.frequency_hz
            magnitude = np.abs(cycle.impedance)
            phase = self._phase_degrees(cycle.impedance)
            self.included_artist.set_data(frequency[included], magnitude[included])
            self.excluded_artist.set_data(frequency[~included], magnitude[~included])
            assert self.phase_included_artist is not None
            assert self.phase_excluded_artist is not None
            assert self.phase_fit_artist is not None
            assert self.phase_fit_points_included_artist is not None
            assert self.phase_fit_points_excluded_artist is not None
            assert self.phase_residual_artist is not None
            assert self.phase_excluded_residual_artist is not None
            self.phase_included_artist.set_data(frequency[included], phase[included])
            self.phase_excluded_artist.set_data(frequency[~included], phase[~included])
            if cycle.fit_impedance is None or cycle.fit_frequency_hz is None:
                self.fit_artist.set_data([], [])
                self.phase_fit_artist.set_data([], [])
            else:
                self.fit_artist.set_data(
                    cycle.fit_frequency_hz,
                    np.abs(cycle.fit_impedance),
                )
                self.phase_fit_artist.set_data(
                    cycle.fit_frequency_hz,
                    self._phase_degrees(cycle.fit_impedance),
                )
            if cycle.fit_at_data_impedance is None:
                self.fit_points_included_artist.set_data([], [])
                self.fit_points_excluded_artist.set_data([], [])
                self.phase_fit_points_included_artist.set_data([], [])
                self.phase_fit_points_excluded_artist.set_data([], [])
                self.residual_artist.set_segments([])
                self.excluded_residual_artist.set_segments([])
                self.phase_residual_artist.set_segments([])
                self.phase_excluded_residual_artist.set_segments([])
            else:
                fitted = cycle.fit_at_data_impedance
                fitted_magnitude = np.abs(fitted)
                fitted_phase = self._phase_degrees(fitted)
                self.fit_points_included_artist.set_data(
                    frequency[included], fitted_magnitude[included]
                )
                self.fit_points_excluded_artist.set_data(
                    frequency[~included], fitted_magnitude[~included]
                )
                self.phase_fit_points_included_artist.set_data(
                    frequency[included], fitted_phase[included]
                )
                self.phase_fit_points_excluded_artist.set_data(
                    frequency[~included], fitted_phase[~included]
                )
                measured_magnitude_points = np.column_stack((frequency, magnitude))
                fitted_magnitude_points = np.column_stack((frequency, fitted_magnitude))
                magnitude_residuals = np.stack(
                    (measured_magnitude_points, fitted_magnitude_points), axis=1
                )
                self.residual_artist.set_segments(magnitude_residuals[included])
                self.excluded_residual_artist.set_segments(
                    magnitude_residuals[~included]
                )
                measured_phase_points = np.column_stack((frequency, phase))
                fitted_phase_points = np.column_stack((frequency, fitted_phase))
                phase_residuals = np.stack(
                    (measured_phase_points, fitted_phase_points), axis=1
                )
                self.phase_residual_artist.set_segments(phase_residuals[included])
                self.phase_excluded_residual_artist.set_segments(
                    phase_residuals[~included]
                )
        else:
            real = cycle.impedance.real
            negative_imaginary = -cycle.impedance.imag
            self.included_artist.set_data(real[included], negative_imaginary[included])
            self.excluded_artist.set_data(real[~included], negative_imaginary[~included])
            if cycle.fit_impedance is None:
                self.fit_artist.set_data([], [])
            else:
                self.fit_artist.set_data(
                    cycle.fit_impedance.real, -cycle.fit_impedance.imag
                )
            if cycle.fit_at_data_impedance is None:
                self.fit_points_included_artist.set_data([], [])
                self.fit_points_excluded_artist.set_data([], [])
                self.residual_artist.set_segments([])
                self.excluded_residual_artist.set_segments([])
            else:
                fitted = cycle.fit_at_data_impedance
                self.fit_points_included_artist.set_data(
                    fitted.real[included], -fitted.imag[included]
                )
                self.fit_points_excluded_artist.set_data(
                    fitted.real[~included], -fitted.imag[~included]
                )
                measured_points = np.column_stack((real, negative_imaginary))
                fitted_points = np.column_stack((fitted.real, -fitted.imag))
                residuals = np.stack((measured_points, fitted_points), axis=1)
                self.residual_artist.set_segments(residuals[included])
                self.excluded_residual_artist.set_segments(residuals[~included])
        if not self.show_eec_fit_var.get():
            self.fit_artist.set_data([], [])
            self.fit_points_included_artist.set_data([], [])
            self.fit_points_excluded_artist.set_data([], [])
            self.residual_artist.set_segments([])
            self.excluded_residual_artist.set_segments([])
            if self.phase_fit_artist is not None:
                self.phase_fit_artist.set_data([], [])
            if self.phase_fit_points_included_artist is not None:
                self.phase_fit_points_included_artist.set_data([], [])
            if self.phase_fit_points_excluded_artist is not None:
                self.phase_fit_points_excluded_artist.set_data([], [])
            if self.phase_residual_artist is not None:
                self.phase_residual_artist.set_segments([])
            if self.phase_excluded_residual_artist is not None:
                self.phase_excluded_residual_artist.set_segments([])
        eec_visible = self.show_eec_fit_var.get()
        for artist in (
            self.fit_artist,
            self.fit_points_included_artist,
            self.fit_points_excluded_artist,
            self.residual_artist,
            self.excluded_residual_artist,
            self.phase_fit_artist,
            self.phase_fit_points_included_artist,
            self.phase_fit_points_excluded_artist,
            self.phase_residual_artist,
            self.phase_excluded_residual_artist,
        ):
            if artist is not None:
                artist.set_visible(eec_visible)
        self._refresh_drt_fit_artists(cycle)
        if self.kk_axes is not None and self.kk_real_artist is not None and self.kk_imag_artist is not None:
            if (
                cycle.kk_cache_matches()
                and cycle.kk_residual_real is not None
                and cycle.kk_residual_imag is not None
            ):
                x_values = cycle.frequency_hz[cycle.included]
                self.kk_real_artist.set_data(x_values, 100.0 * cycle.kk_residual_real)
                self.kk_imag_artist.set_data(x_values, 100.0 * cycle.kk_residual_imag)
            else:
                self.kk_real_artist.set_data([], [])
                self.kk_imag_artist.set_data([], [])
        self.axes.set_title(
            (
                f"{self.loaded.dataset_label if self.loaded is not None else self._current_name()}\n"
                f"Cycle {cycle.cycle} · {cycle.model(self.state.circuit)}"
            )
        )
        if self.drt_artist is not None and self.drt_axes is not None:
            drt_tau_s, drt_gamma_ohm, drt_label = self._apply_saved_drt_mode(cycle)
            if drt_tau_s is None or drt_gamma_ohm is None:
                self.drt_artist.set_data([], [])
            else:
                self.drt_artist.set_data(drt_tau_s, drt_gamma_ohm)
            self.drt_axes.set_title(drt_label)
            self._refresh_drt_peak_artists()
        if rescale:
            self._autoscale_to_included(cycle)
            if self.kk_axes is not None:
                self._autoscale_kk(cycle)
            if self.drt_artist is not None and self.drt_axes is not None:
                self._autoscale_drt(cycle)
        self._refresh_ml_visuals()
        self._refresh_explorer_values()
        self.canvas.draw_idle()

    def _autoscale_to_included(self, cycle) -> None:
        if self.show_all_points_var.get():
            included = np.ones(cycle.frequency_hz.size, dtype=bool)
        else:
            included = cycle.included
            if not np.any(included):
                included = np.ones(cycle.frequency_hz.size, dtype=bool)
        if self.plot_mode == "bode":
            frequency = cycle.frequency_hz[included]
            magnitude = np.abs(cycle.impedance[included])
            phase = self._phase_degrees(cycle.impedance[included])
            finite_frequency = np.isfinite(frequency) & (frequency > 0)
            finite_magnitude = np.isfinite(magnitude)
            finite_phase = np.isfinite(phase)
            finite = finite_frequency & finite_magnitude & finite_phase
            if not np.any(finite):
                return
            frequency = frequency[finite]
            magnitude = magnitude[finite]
            phase = phase[finite]
            x_min = float(np.min(frequency))
            x_max = float(np.max(frequency))
            if x_min == x_max:
                x_min /= 1.3
                x_max *= 1.3
            else:
                span_decades = np.log10(x_max) - np.log10(x_min)
                x_padding_factor = 10 ** (0.04 * span_decades)
                x_min /= x_padding_factor
                x_max *= x_padding_factor
            magnitude_min = float(np.min(magnitude))
            magnitude_max = float(np.max(magnitude))
            magnitude_span = magnitude_max - magnitude_min
            magnitude_padding = 0.06 * (
                magnitude_span if magnitude_span > 0 else max(abs(magnitude_min), 1.0)
            )
            phase_min = float(np.min(phase))
            phase_max = float(np.max(phase))
            phase_span = phase_max - phase_min
            phase_padding = 0.08 * (
                phase_span if phase_span > 0 else max(abs(phase_min), 1.0)
            )
            self.axes.set_xlim(x_min, x_max)
            self.axes.set_ylim(
                magnitude_min - magnitude_padding,
                magnitude_max + magnitude_padding,
            )
            if self.phase_axes is not None:
                self.phase_axes.set_ylim(
                    phase_min - phase_padding,
                    phase_max + phase_padding,
                )
            return
        x_values = cycle.impedance.real[included]
        y_values = -cycle.impedance.imag[included]
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(finite):
            return
        x_values = x_values[finite]
        y_values = y_values[finite]
        x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
        y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
        x_span = x_max - x_min
        y_span = y_max - y_min
        x_padding = 0.06 * (x_span if x_span > 0 else max(abs(x_min), 1.0))
        y_padding = 0.06 * (y_span if y_span > 0 else max(abs(y_min), 1.0))
        self.axes.set_xlim(x_min - x_padding, x_max + x_padding)
        self.axes.set_ylim(y_min - y_padding, y_max + y_padding)

    def _autoscale_drt(self, cycle) -> None:
        if (
            self.drt_axes is None
            or cycle.ridge_tau_s is None
            or cycle.ridge_gamma_ohm is None
        ):
            return
        tau = cycle.ridge_tau_s
        gamma = cycle.ridge_gamma_ohm
        finite = np.isfinite(tau) & np.isfinite(gamma) & (tau > 0)
        if not np.any(finite):
            return
        tau = tau[finite]
        gamma = gamma[finite]
        x_min, x_max = float(np.min(tau)), float(np.max(tau))
        y_min, y_max = float(np.min(gamma)), float(np.max(gamma))
        y_span = y_max - y_min
        y_padding = 0.08 * (y_span if y_span > 0 else max(abs(y_max), 1.0))
        if x_min == x_max:
            x_min /= 1.3
            x_max *= 1.3
        self.drt_axes.set_xlim(x_min, x_max)
        self.drt_axes.set_ylim(y_min - y_padding, y_max + y_padding)

    def _current_drt_arrays(self):
        if self.state is None:
            return None, None
        cycle = self.state.active
        if self.analysis_drt_mode_var.get() == "Hybrid DRT":
            tau = cycle.saved_hybrid_tau_s
            gamma = cycle.saved_hybrid_gamma_ohm
        else:
            tau = cycle.saved_ridge_tau_s
            gamma = cycle.saved_ridge_gamma_ohm
        if tau is None or gamma is None:
            return None, None
        tau = np.asarray(tau, dtype=float)
        gamma = np.asarray(gamma, dtype=float)
        finite = np.isfinite(tau) & np.isfinite(gamma) & (tau > 0)
        if not np.any(finite):
            return None, None
        order = np.argsort(tau[finite])
        return tau[finite][order], gamma[finite][order]

    @staticmethod
    def _estimate_drt_inductance(cycle) -> float | None:
        frequency = np.asarray(cycle.frequency_hz, dtype=float)
        impedance = np.asarray(cycle.impedance, dtype=complex)
        finite = np.isfinite(frequency) & (frequency > 0) & np.isfinite(impedance)
        if not np.any(finite):
            return None
        frequency = frequency[finite]
        impedance = impedance[finite]
        order = np.argsort(frequency)[::-1]
        count = max(3, min(20, frequency.size // 5 or frequency.size))
        values = impedance[order[:count]].imag / (2.0 * np.pi * frequency[order[:count]])
        values = values[np.isfinite(values) & (values >= 0)]
        return float(np.median(values)) if values.size else None

    @staticmethod
    def _peak_shape(peak) -> str:
        return str(peak.get("shape", "gaussian")).lower()

    @classmethod
    def _peak_values(cls, log_tau, peak):
        shape = cls._peak_shape(peak)
        distance = np.asarray(log_tau, dtype=float) - peak["center_log10"]
        width = max(float(peak["sigma_log10"]), 1e-6)
        if shape == "lorentzian":
            return peak["height"] / (1.0 + (distance / width) ** 2)
        if shape == "voigt":
            gamma = max(float(peak.get("gamma_log10", width)), 1e-6)
            values = voigt_profile(distance, width, gamma)
            peak_value = float(voigt_profile(0.0, width, gamma))
            return peak["height"] * values / max(peak_value, 1e-300)
        if shape == "hn":
            alpha = float(np.clip(peak.get("alpha", 0.8), 1e-3, 0.999))
            beta = float(np.clip(peak.get("beta", 0.8), 1e-3, 0.999))
            ratio = 10.0**distance
            alpha_ratio = ratio**alpha
            angle = np.arctan2(
                np.sin(np.pi * alpha), alpha_ratio + np.cos(np.pi * alpha)
            )
            profile = (
                alpha_ratio * np.sin(beta * angle)
                / (alpha_ratio**2 + 2.0 * alpha_ratio * np.cos(np.pi * alpha) + 1.0)
                ** (beta / 2.0)
            )
            center_profile = np.sin(beta * np.arctan2(
                np.sin(np.pi * alpha), 1.0 + np.cos(np.pi * alpha)
            )) / (2.0 + 2.0 * np.cos(np.pi * alpha)) ** (beta / 2.0)
            return peak["height"] * profile / max(float(center_profile), 1e-300)
        return peak["height"] * np.exp(-0.5 * (distance / width) ** 2)

    @classmethod
    def _gaussian_peak_values(cls, log_tau, peak):
        return cls._peak_values(log_tau, {**peak, "shape": "gaussian"})

    @staticmethod
    def _peak_summary(peak):
        tau = 10.0 ** peak["center_log10"]
        sigma = max(peak["sigma_log10"], 1e-6)
        shape = str(peak.get("shape", "gaussian")).lower()
        if shape == "lorentzian":
            area = peak["height"] * np.pi * sigma * np.log(10.0)
            half_width = sigma
        elif shape == "voigt":
            gamma = max(float(peak.get("gamma_log10", sigma)), 1e-6)
            area = peak["height"] / max(
                float(voigt_profile(0.0, sigma, gamma)), 1e-300
            ) * np.log(10.0)
            fwhm_log10 = 0.5346 * 2.0 * gamma + np.sqrt(
                0.2166 * (2.0 * gamma) ** 2 + (2.35482 * sigma) ** 2
            )
            half_width = fwhm_log10 / 2.0
        elif shape == "hn":
            alpha = float(np.clip(peak.get("alpha", 0.8), 1e-3, 0.999))
            beta = float(np.clip(peak.get("beta", 0.8), 1e-3, 0.999))
            grid = np.linspace(-5.0, 5.0, 2001)
            profile = EISApplication._peak_values(
                grid,
                {**peak, "height": 1.0, "alpha": alpha, "beta": beta},
            )
            area = peak.get(
                "area",
                float(np.trapezoid(profile, grid) * np.log(10.0) * peak["height"]),
            )
            half = np.flatnonzero(profile >= 0.5 * np.max(profile))
            if half.size:
                half_width = max(
                    abs(grid[half[-1]] - grid[half[0]]) / 2.0,
                    1e-6,
                )
            else:
                half_width = 0.1
            return tau, area, tau * (10.0**half_width - 10.0 ** (-half_width))
        else:
            area = peak["height"] * sigma * np.sqrt(2.0 * np.pi) * np.log(10.0)
            half_width = np.sqrt(2.0 * np.log(2.0)) * sigma
        fwhm = tau * (10.0**half_width - 10.0 ** (-half_width))
        return tau, area, fwhm

    @staticmethod
    def _peak_width_from_fwhm(shape, tau, fwhm):
        ratio = max(float(fwhm) / max(float(tau), 1e-300), 1e-12)
        half_width = np.arcsinh(ratio / 2.0) / np.log(10.0)
        if shape == "lorentzian":
            return max(half_width, 1e-6)
        if shape == "voigt":
            voigt_fwhm_factor = 0.5346 * 2.0 + np.sqrt(0.2166 * 4.0 + 2.35482**2)
            return max(2.0 * half_width / voigt_fwhm_factor, 1e-6)
        return max(
            half_width / np.sqrt(2.0 * np.log(2.0)),
            1e-6,
        )

    @staticmethod
    def _peak_height_from_area(shape, area, sigma, gamma=None, alpha=None, beta=None):
        if shape == "lorentzian":
            normalization = np.pi * sigma * np.log(10.0)
        elif shape == "voigt":
            gamma = max(float(gamma if gamma is not None else sigma), 1e-6)
            normalization = np.log(10.0) / max(
                float(voigt_profile(0.0, sigma, gamma)), 1e-300
            )
        elif shape == "hn":
            profile = EISApplication._peak_values(
                np.linspace(-5.0, 5.0, 2001),
                {
                    "shape": "hn",
                    "center_log10": 0.0,
                    "height": 1.0,
                    "sigma_log10": 0.12,
                    "alpha": alpha if alpha is not None else 0.8,
                    "beta": beta if beta is not None else 0.8,
                },
            )
            normalization = float(np.trapezoid(profile, np.linspace(-5.0, 5.0, 2001)) * np.log(10.0))
        else:
            normalization = sigma * np.sqrt(2.0 * np.pi) * np.log(10.0)
        return float(area) / max(normalization, 1e-300)

    @classmethod
    def _peak_half_width_log10(cls, peak):
        shape = cls._peak_shape(peak)
        sigma = max(float(peak["sigma_log10"]), 1e-6)
        if shape == "lorentzian":
            return sigma
        if shape == "voigt":
            gamma = max(float(peak.get("gamma_log10", sigma)), 1e-6)
            return (0.5346 * 2.0 * gamma + np.sqrt(
                0.2166 * (2.0 * gamma) ** 2 + (2.35482 * sigma) ** 2
            )) / 2.0
        if shape == "hn":
            tau, _area, fwhm = cls._peak_summary(peak)
            return max(np.arcsinh(fwhm / max(2.0 * tau, 1e-300)) / np.log(10.0), 1e-6)
        return np.sqrt(2.0 * np.log(2.0)) * sigma

    def _drt_display_parameter_name(self, name: str) -> str:
        match = re.fullmatch(r"Peak(\d+)_(.+)", name)
        if match is None:
            return _external_parameter_name(name)
        index = int(match.group(1)) - 1
        shape = (
            self._peak_shape(self.drt_peak_parameters[index]).title()
            if 0 <= index < len(self.drt_peak_parameters)
            else "Peak"
        )
        prefix = {"Gaussian": "Gauss", "Lorentzian": "Lor", "Voigt": "Voigt", "Hn": "HN"}.get(
            shape, shape
        )
        return f"{prefix}_{match.group(1)}_{match.group(2)}"

    def _drt_peak_parameter_names(self, index: int) -> set[str]:
        prefix = f"Peak{index + 1}_"
        return {
            parameter.name
            for parameter in self.drt_peak_table.values()
            if parameter.name.startswith(prefix)
        }

    def _select_drt_peak(self, index: int | None) -> None:
        if index is None or not (0 <= index < len(self.drt_peak_parameters)):
            self._selected_drt_peak_index = None
            names = set()
        else:
            self._selected_drt_peak_index = index
            names = self._drt_peak_parameter_names(index)
        if hasattr(self, "drt_peak_table"):
            self.drt_peak_table.set_highlighted_names(names)
        detached = getattr(self, "detached_drt_peak_table", None)
        if detached is not None:
            detached.set_highlighted_names(names)
        self._refresh_drt_peak_artists()
        self.canvas.draw_idle()

    def _on_drt_parameter_double_click(self, name: str) -> None:
        match = re.fullmatch(r"Peak(\d+)_.*", name)
        if match is not None:
            index = int(match.group(1)) - 1
            self._select_drt_peak(
                None if index == self._selected_drt_peak_index else index
            )

    def _select_drt_peak_from_event(self, event) -> bool:
        if event.inaxes is not self.drt_axes or event.xdata is None or event.ydata is None:
            return False
        best = None
        for index, peak in enumerate(self.drt_peak_parameters):
            x_value = max(float(event.xdata), 1e-300)
            y_value = float(self._peak_values([np.log10(x_value)], peak)[0])
            display = self.drt_axes.transData.transform((x_value, y_value))
            distance = float(np.hypot(display[0] - event.x, display[1] - event.y))
            if best is None or distance < best[0]:
                best = (distance, index)
        if best is None or best[0] > 18.0:
            return False
        self._select_drt_peak(
            None if best[1] == self._selected_drt_peak_index else best[1]
        )
        return True

    def _on_delete_key(self, _event=None):
        focus = self.root.focus_get()
        if focus is not None and str(focus).startswith(str(self.explorer)):
            return "break"
        if self.analysis_mode_var.get() == "DRT" and self._selected_drt_peak_index is not None:
            self.remove_selected_drt_peak()
            return "break"
        return None

    def remove_selected_drt_peak(self) -> None:
        index = self._selected_drt_peak_index
        if index is None or not (0 <= index < len(self.drt_peak_parameters)):
            return
        self.drt_peak_parameters.pop(index)
        self._selected_drt_peak_index = (
            min(index, len(self.drt_peak_parameters) - 1)
            if self.drt_peak_parameters
            else None
        )
        self._store_current_drt_peaks()
        self._refresh_drt_peak_artists()
        self.canvas.draw_idle()

    def remove_all_drt_peaks(self) -> None:
        if not self.drt_peak_parameters:
            return
        self.drt_peak_parameters.clear()
        self._selected_drt_peak_index = None
        self._store_current_drt_peaks()
        self._refresh_drt_peak_artists()
        self.canvas.draw_idle()
        self._update_status("removed all DRT peaks")

    def _update_drt_peak_table(self) -> None:
        if not hasattr(self, "drt_peak_table"):
            return
        parameters = []
        if self.state is not None:
            mode = self.analysis_drt_mode_var.get()
            cycle = self.state.active
            resistance = (
                cycle.saved_hybrid_ohmic_resistance
                if mode == "Hybrid DRT"
                else cycle.saved_ridge_ohmic_resistance
            )
            inductance = (
                cycle.saved_ridge_inductance
                if mode == "Ridge DRT"
                else cycle.saved_hybrid_inductance
            )
            auxiliary_parameters = (
                (("R_inf", resistance, "Ohm"), ("inductance", inductance, "H"))
                if mode == "Hybrid DRT"
                else (("R0", resistance, "Ohm"), ("L0", inductance, "H"))
            )
            for name, value, unit in auxiliary_parameters:
                if value is None:
                    continue
                value = float(value)
                lower, upper = self._drt_aux_parameter_limits.get(
                    name,
                    (0.0, max(abs(value) * 10.0, 1.0 if name == "R0" else 1e-6)),
                )
                parameters.append(
                    ParameterValue(name, unit, value, lower, upper, None)
                )

        for index, peak in enumerate(self.drt_peak_parameters, 1):
            tau, area, fwhm = self._peak_summary(peak)
            shape = self._peak_shape(peak)
            if shape in {"voigt", "hn"}:
                if shape == "hn":
                    peak_parameters = (
                        ("area", area, "area_error_percent", "Ohm"),
                        ("tau", tau, "tau_error_percent", "s"),
                        ("alpha", peak.get("alpha", 0.8), "alpha_error_percent", ""),
                        ("beta", peak.get("beta", 0.8), "beta_error_percent", ""),
                    )
                else:
                    peak_parameters = (
                        ("area", area, "area_error_percent", "Ohm"),
                        ("tau", tau, "tau_error_percent", "s"),
                        ("sigma", peak.get("sigma_log10", 0.12), "sigma_error_percent", "log10(s)"),
                        ("gamma", peak.get("gamma_log10", peak.get("sigma_log10", 0.12)), "gamma_error_percent", "log10(s)"),
                    )
            else:
                peak_parameters = (
                    ("area", area, "area_error_percent", "Ohm"),
                    ("tau", tau, "tau_error_percent", "s"),
                    ("fwhm", fwhm, "fwhm_error_percent", "s"),
                )
            for suffix, value, error_key, unit in peak_parameters:
                lower_key = f"{suffix}_lower"
                upper_key = f"{suffix}_upper"
                magnitude = max(abs(value), np.finfo(float).eps)
                default_limits = {
                    "tau": (1e-5, 10.0),
                    "area": (0.0, 1e3),
                    "fwhm": (0.0, 1.0),
                    "sigma": (1e-6, 1.0),
                    "gamma": (1e-6, 1.0),
                    "alpha": (1e-3, 1.0),
                    "beta": (1e-3, 1.0),
                }
                default_lower, default_upper = default_limits[suffix]
                lower = peak.get(lower_key, default_lower)
                upper = peak.get(upper_key, default_upper)
                parameters.append(
                    ParameterValue(
                        f"Peak{index}_{suffix}",
                        unit,
                        value,
                        lower,
                        upper,
                        peak.get(error_key),
                        bool(peak.get(f"{suffix}_fixed", False)),
                    )
                )
        self.drt_peak_table.set_parameters(parameters)
        self.drt_peak_table.set_highlighted_names(
            self._drt_peak_parameter_names(self._selected_drt_peak_index)
            if self._selected_drt_peak_index is not None
            else set()
        )

    def _clamp_drt_peak_parameters_to_limits(self) -> None:
        for peak in self.drt_peak_parameters:
            tau, area, fwhm = self._peak_summary(peak)
            shape = self._peak_shape(peak)
            values = {"tau": tau, "area": area}
            if shape == "voigt":
                values.update({
                    "sigma": peak.get("sigma_log10", 0.12),
                    "gamma": peak.get("gamma_log10", peak.get("sigma_log10", 0.12)),
                })
            elif shape == "hn":
                values.update({
                    "alpha": peak.get("alpha", 0.8),
                    "beta": peak.get("beta", 0.8),
                })
            else:
                values["fwhm"] = fwhm
            for suffix, value in values.items():
                defaults = {
                    "tau": (1e-5, 10.0), "area": (0.0, 1e3),
                    "fwhm": (0.0, 1.0), "sigma": (1e-6, 1.0),
                    "gamma": (1e-6, 1.0),
                    "alpha": (1e-3, 1.0), "beta": (1e-3, 1.0),
                }
                lower = peak.get(f"{suffix}_lower", defaults[suffix][0])
                upper = peak.get(f"{suffix}_upper", defaults[suffix][1])
                values[suffix] = self._clamp_parameter_value(value, lower, upper)
            tau = max(values["tau"], 1e-300)
            peak["center_log10"] = float(np.log10(tau))
            if shape == "voigt":
                peak["sigma_log10"] = values["sigma"]
                peak["gamma_log10"] = values["gamma"]
            elif shape == "hn":
                peak["alpha"] = values["alpha"]
                peak["beta"] = values["beta"]
                peak["area"] = values["area"]
            else:
                peak["sigma_log10"] = self._peak_width_from_fwhm(
                    shape, tau, max(values["fwhm"], 1e-300)
                )
            peak["height"] = self._peak_height_from_area(
                shape, values["area"], peak["sigma_log10"],
                peak.get("gamma_log10"),
                peak.get("alpha"), peak.get("beta"),
            )
            if shape == "hn":
                peak["area"] = values["area"]
            if self._peak_shape(peak) == "hn":
                peak["area"] = peak_area
        self._store_current_drt_peaks()

    def _sync_drt_peak_parameters_from_table(self) -> bool:
        if not hasattr(self, "drt_peak_table"):
            return True
        try:
            values = self.drt_peak_table.values()
            for parameter in values:
                if parameter.lower > parameter.upper:
                    raise ValueError(f"{parameter.name}: lower bound exceeds upper bound")
                parameter.initial = self._clamp_parameter_value(
                    parameter.initial,
                    parameter.lower,
                    parameter.upper,
                )
            self.drt_peak_table.set_parameters(values)
            by_name = {parameter.name: parameter for parameter in values}
            for index, peak in enumerate(self.drt_peak_parameters, 1):
                tau = by_name[f"Peak{index}_tau"]
                area = by_name[f"Peak{index}_area"]
                shape = self._peak_shape(peak)
                width_names = (
                    ("sigma", "gamma") if shape == "voigt"
                    else ("alpha", "beta") if shape == "hn"
                    else ("fwhm",)
                )
                widths = [by_name[f"Peak{index}_{name}"] for name in width_names]
                if tau.initial <= 0 or any(width.initial <= 0 for width in widths):
                    raise ValueError(f"Peak {index}: tau and width parameters must be positive")
                peak["center_log10"] = float(np.log10(tau.initial))
                if shape == "voigt":
                    peak["sigma_log10"] = widths[0].initial
                    peak["gamma_log10"] = widths[1].initial
                elif shape == "hn":
                    peak["alpha"] = widths[0].initial
                    peak["beta"] = widths[1].initial
                else:
                    peak["sigma_log10"] = self._peak_width_from_fwhm(
                        shape, tau.initial, widths[0].initial
                    )
                peak["height"] = self._peak_height_from_area(
                    shape, area.initial, peak.get("sigma_log10", 0.12),
                    peak.get("gamma_log10"),
                    peak.get("alpha"), peak.get("beta"),
                )
                sources = [(tau, "tau"), (area, "area")]
                sources.extend(zip(widths, width_names))
                for source, target in sources:
                    peak[f"{target}_lower"] = source.lower
                    peak[f"{target}_upper"] = source.upper
                    peak[f"{target}_fixed"] = source.fixed
            self._drt_aux_parameter_limits = {
                parameter.name: (parameter.lower, parameter.upper)
                for parameter in values
                if parameter.name in {"R0", "L0"}
            }
        except (KeyError, TypeError, ValueError) as error:
            messagebox.showerror("Invalid DRT peak parameter", str(error), parent=self.root)
            return False
        self._store_current_drt_peaks()
        return True

    def _store_current_drt_peaks(self) -> None:
        if self.state is None:
            return
        self.state.active.store_drt_peak_parameters(
            self.analysis_drt_mode_var.get(), self.drt_peak_parameters
        )

    def _refresh_drt_peak_artists(self) -> None:
        if self.drt_axes is None:
            return
        for artist in self._drt_peak_artists:
            artist.remove()
        self._drt_peak_artists = []
        if self._drt_peak_sum_artist is not None:
            self._drt_peak_sum_artist.remove()
            self._drt_peak_sum_artist = None
        tau, _gamma = self._current_drt_arrays()
        if tau is None or not self.drt_peak_parameters:
            self._update_drt_peak_table()
            return
        tau_grid = np.geomspace(float(np.min(tau)), float(np.max(tau)), 300)
        log_tau_grid = np.log10(tau_grid)
        total = np.zeros_like(tau_grid)
        for index, peak in enumerate(self.drt_peak_parameters):
            values = self._peak_values(log_tau_grid, peak)
            total += values
            selected = index == self._selected_drt_peak_index
            line, = self.drt_axes.plot(
                tau_grid,
                values,
                "--",
                linewidth=2.6 if selected else 1.2,
                alpha=1.0 if selected else 0.85,
                label=f"{self._peak_shape(peak).title()} {index + 1}",
            )
            center_tau = 10.0 ** peak["center_log10"]
            _tau, _area, fwhm = self._peak_summary(peak)
            half_width = np.log10(
                (fwhm + np.sqrt(fwhm**2 + 4.0 * center_tau**2))
                / (2.0 * center_tau)
            )
            left_tau = 10.0 ** (peak["center_log10"] - half_width)
            right_tau = 10.0 ** (peak["center_log10"] + half_width)
            top, = self.drt_axes.plot(
                [center_tau], [peak["height"]], "o", color=line.get_color(),
                ms=9 if selected else 6,
                markeredgewidth=2.0 if selected else 0.8,
            )
            self._drt_peak_artists.extend((line, top))
            if self._peak_shape(peak) != "hn":
                left, = self.drt_axes.plot(
                    [left_tau], [peak["height"] / 2.0], "s", color=line.get_color(),
                    ms=7 if selected else 5,
                )
                right, = self.drt_axes.plot(
                    [right_tau], [peak["height"] / 2.0], "s", color=line.get_color(),
                    ms=7 if selected else 5,
                )
                self._drt_peak_artists.extend((left, right))
        self._drt_peak_sum_artist, = self.drt_axes.plot(
            tau_grid,
            total,
            "-",
            color="#222222",
            linewidth=1.8,
            alpha=0.95,
            zorder=4,
            label="Peak sum",
        )
        self._update_drt_peak_table()
        self.drt_axes.legend(loc="best")

    def add_drt_peak(self, shape: str = "gaussian") -> None:
        tau, gamma = self._current_drt_arrays()
        if tau is None:
            self._update_status("calculate a DRT before adding peaks")
            return
        peak_index = int(np.nanargmax(gamma))
        center = float(np.log10(tau[peak_index]))
        if self.drt_peak_parameters:
            center = float(np.mean(np.log10(tau)))
        height = max(float(gamma[peak_index]), 0.0)
        if height == 0.0:
            height = float(np.nanmax(np.abs(gamma)))
        self.drt_peak_parameters.append(
            {
                "shape": shape,
                "center_log10": center,
                "height": height,
                "sigma_log10": 0.12,
                **({"gamma_log10": 0.12} if shape == "voigt" else {}),
                **({"alpha": 0.8, "beta": 0.8} if shape == "hn" else {}),
            }
        )
        self._selected_drt_peak_index = len(self.drt_peak_parameters) - 1
        self._store_current_drt_peaks()
        self._refresh_drt_peak_artists()
        self.canvas.draw_idle()

    def add_gaussian_peak(self) -> None:
        self.add_drt_peak("gaussian")

    def fit_drt_peaks(self) -> None:
        if self.busy or self.state is None:
            return
        tau, gamma = self._current_drt_arrays()
        if tau is None or not self.drt_peak_parameters:
            self._update_status("add at least one Gaussian peak first")
            return
        if not self._sync_drt_peak_parameters_from_table():
            return
        log_tau = np.log10(tau)
        count = len(self.drt_peak_parameters)
        specs = []
        initial = []
        lower = []
        upper = []
        fixed = []
        for index, peak in enumerate(self.drt_peak_parameters, 1):
            peak_tau, peak_area, peak_fwhm = self._peak_summary(peak)
            if self._peak_shape(peak) == "voigt":
                names_values = (
                    ("tau", peak_tau, peak_tau / 10.0, peak_tau * 10.0),
                    ("area", peak_area, min(0.0, peak_area), max(abs(peak_area) * 10.0, 1e-12)),
                    ("sigma", peak.get("sigma_log10", 0.12), peak.get("sigma_lower", 1e-6), peak.get("sigma_upper", 1.0)),
                    ("gamma", peak.get("gamma_log10", 0.12), peak.get("gamma_lower", 1e-6), peak.get("gamma_upper", 1.0)),
                )
            elif self._peak_shape(peak) == "hn":
                names_values = (
                    ("tau", peak_tau, peak_tau / 10.0, peak_tau * 10.0),
                    ("area", peak_area, min(0.0, peak_area), max(abs(peak_area) * 10.0, 1e-12)),
                    ("alpha", peak.get("alpha", 0.8), peak.get("alpha_lower", 1e-3), peak.get("alpha_upper", 1.0)),
                    ("beta", peak.get("beta", 0.8), peak.get("beta_lower", 1e-3), peak.get("beta_upper", 1.0)),
                )
            else:
                names_values = (
                    ("tau", peak_tau, peak_tau / 10.0, peak_tau * 10.0),
                    ("area", peak_area, min(0.0, peak_area), max(abs(peak_area) * 10.0, 1e-12)),
                    ("fwhm", peak_fwhm, peak_fwhm / 10.0, peak_fwhm * 10.0),
                )
            for name, value, default_lower, default_upper in names_values:
                specs.append((index - 1, name))
                initial.append(value)
                lower.append(peak.get(f"{name}_lower", default_lower))
                upper.append(peak.get(f"{name}_upper", default_upper))
                fixed.append(bool(peak.get(f"{name}_fixed", False)))

        initial = np.asarray(initial, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if np.any(upper <= lower):
            messagebox.showerror(
                "Invalid DRT peak limits",
                "Every lower limit must be smaller than its upper limit.",
                parent=self.root,
            )
            return
        free_indices = [index for index, is_fixed in enumerate(fixed) if not is_fixed]

        def model(values, *parameters):
            all_parameters = initial.copy()
            all_parameters[free_indices] = parameters
            result = np.zeros_like(values, dtype=float)
            for peak_index in range(count):
                peak = self.drt_peak_parameters[peak_index]
                values_by_name = {
                    name: all_parameters[offset]
                    for offset, (owner, name) in enumerate(specs)
                    if owner == peak_index
                }
                peak_tau = values_by_name["tau"]
                area = values_by_name["area"]
                if self._peak_shape(peak) == "voigt":
                    sigma = values_by_name["sigma"]
                    gamma_width = values_by_name["gamma"]
                    alpha = beta = None
                elif self._peak_shape(peak) == "hn":
                    sigma = 0.12
                    gamma_width = None
                    alpha = values_by_name["alpha"]
                    beta = values_by_name["beta"]
                else:
                    sigma = self._peak_width_from_fwhm(
                        self._peak_shape(peak), peak_tau, values_by_name["fwhm"]
                    )
                    gamma_width = None
                    alpha = beta = None
                height = self._peak_height_from_area(
                    self._peak_shape(peak), area, sigma, gamma_width, alpha, beta
                )
                model_peak = {
                    "shape": self._peak_shape(peak),
                    "center_log10": np.log10(peak_tau),
                    "sigma_log10": sigma,
                    "height": height,
                }
                if gamma_width is not None:
                    model_peak["gamma_log10"] = gamma_width
                if alpha is not None:
                    model_peak["alpha"] = alpha
                    model_peak["beta"] = beta
                result += self._peak_values(values, model_peak)
            return result
        fit_initial = initial[free_indices]
        fit_lower = lower[free_indices].copy()
        fit_upper = upper[free_indices].copy()
        for index in range(fit_initial.size):
            parameter_index = free_indices[index]
            if fit_lower[index] <= 0 and specs[parameter_index][1] in {"tau", "fwhm", "sigma", "gamma", "alpha", "beta"}:
                fit_lower[index] = 1e-12
            margin = max(abs(fit_initial[index]) * 1e-9, 1e-12)
            fit_initial[index] = np.clip(
                fit_initial[index],
                fit_lower[index] + margin,
                fit_upper[index] - margin,
            )
        if not free_indices:
            fitted = initial.copy()
            errors = np.zeros_like(initial)
        else:
            try:
                fitted_free, covariance = curve_fit(
                    model,
                    log_tau,
                    gamma,
                    p0=fit_initial,
                    bounds=(fit_lower, fit_upper),
                    maxfev=50000,
                )
            except (RuntimeError, ValueError) as error:
                messagebox.showerror("DRT peak fit failed", str(error), parent=self.root)
                return
            fitted = initial.copy()
            fitted[free_indices] = fitted_free
            errors = np.zeros_like(initial)
            errors[free_indices] = np.sqrt(
                np.maximum(np.diag(covariance), 0.0)
            )
        for peak_index, peak in enumerate(self.drt_peak_parameters):
            fitted_values = {
                name: fitted[offset]
                for offset, (owner, name) in enumerate(specs)
                if owner == peak_index
            }
            peak_tau = fitted_values["tau"]
            peak_area = fitted_values["area"]
            peak["center_log10"] = float(np.log10(peak_tau))
            if self._peak_shape(peak) == "voigt":
                peak["sigma_log10"] = fitted_values["sigma"]
                peak["gamma_log10"] = fitted_values["gamma"]
            elif self._peak_shape(peak) == "hn":
                peak["alpha"] = fitted_values["alpha"]
                peak["beta"] = fitted_values["beta"]
            else:
                peak_fwhm = fitted_values["fwhm"]
                peak["sigma_log10"] = self._peak_width_from_fwhm(
                    self._peak_shape(peak), peak_tau, peak_fwhm
                )
            peak["height"] = self._peak_height_from_area(
                self._peak_shape(peak), peak_area, peak["sigma_log10"],
                peak.get("gamma_log10"), peak.get("alpha"), peak.get("beta")
            )
            for offset, (owner, key) in enumerate(specs):
                if owner != peak_index:
                    continue
                value = fitted[offset]
                peak[f"{key}_error_percent"] = float(
                    100.0 * errors[offset] / max(abs(value), 1e-300)
                )
        self._store_current_drt_peaks()
        self._refresh_drt_peak_artists()
        self.canvas.draw_idle()
        self._update_status(f"fitted {count} DRT peaks")

        if getattr(self, "_drt_peak_batch_queue", None) is not None:
            self._drt_peak_batch_template = [
                copy.deepcopy(peak) for peak in self.drt_peak_parameters
            ]
            self.root.after(0, self._drt_peak_batch_next)
        else:
            self._refresh_open_parameter_explorers()

    def _select_drt_peaks_for_eec(self, peaks, maximum: int):
        window = tk.Toplevel(self.root)
        window.title("Select DRT peaks")
        window.transient(self.root)
        window.grab_set()
        ttk.Label(
            window,
            text=f"Select up to {maximum} DRT peaks to send to the EEC model:",
            padding=8,
        ).pack(anchor="w")
        variables = []
        for index, peak in enumerate(peaks):
            tau, area, fwhm = self._peak_summary(peak)
            variable = tk.BooleanVar(value=index < maximum)
            variables.append(variable)
            ttk.Checkbutton(
                window,
                text=f"Peak {index + 1}: tau={tau:.5g}, area={area:.5g}, FWHM={fwhm:.5g}",
                variable=variable,
            ).pack(anchor="w", padx=8, pady=2)
        result = [None]

        def accept() -> None:
            selected = [
                index for index, variable in enumerate(variables) if variable.get()
            ]
            if len(selected) > maximum:
                messagebox.showerror(
                    "Too many peaks",
                    f"Select no more than {maximum} peaks.",
                    parent=window,
                )
                return
            result[0] = selected
            window.destroy()

        buttons = ttk.Frame(window, padding=8)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="OK", command=accept).pack(side=tk.RIGHT)
        ttk.Button(
            buttons, text="Cancel", command=window.destroy
        ).pack(side=tk.RIGHT, padx=(0, 6))
        window.protocol("WM_DELETE_WINDOW", window.destroy)
        self.root.wait_window(window)
        return result[0]

    def send_drt_initials(self) -> None:
        if self.busy or self.state is None:
            return
        cycle = self.state.active
        mode = self.analysis_drt_mode_var.get()
        peaks = [dict(peak) for peak in self.drt_peak_parameters]
        if not peaks:
            saved_name = (
                "saved_hybrid_peak_parameters"
                if mode == "Hybrid DRT"
                else "saved_ridge_peak_parameters"
            )
            peaks = [dict(peak) for peak in getattr(cycle, saved_name, [])]
        if not peaks:
            self._update_status("add and fit at least one DRT peak first")
            return
        if not self._sync_drt_peak_parameters_from_table():
            return
        peaks = [dict(peak) for peak in self.drt_peak_parameters]
        table_parameters = {
            parameter.name: parameter
            for parameter in self.drt_peak_table.values()
        }

        circuit = re.sub(r"\s+", "", cycle.model(self.state.circuit))
        branch_ids = sorted(
            {
                match.group(1)
                for match in re.finditer(r"p\(R(\d+),CPE\1\)", circuit)
            },
            key=int,
        )
        selected_indices = list(range(len(peaks)))
        if len(peaks) > len(branch_ids) and branch_ids:
            selected_indices = self._select_drt_peaks_for_eec(peaks, len(branch_ids))
            if selected_indices is None:
                return
        elif not branch_ids:
            selected_indices = []
        selected_peaks = [peaks[index] for index in selected_indices]

        parameters = [
            ParameterValue(
                parameter.name,
                parameter.unit,
                parameter.initial,
                parameter.lower,
                parameter.upper,
                parameter.error_percent,
                parameter.fixed,
            )
            for parameter in cycle.parameters
        ]
        by_name = {parameter.name: parameter for parameter in parameters}
        if "R0" in by_name:
            resistance = table_parameters.get("R0")
            resistance = resistance.initial if resistance is not None else None
            if resistance is not None:
                by_name["R0"].initial = self._clamp_parameter_value(
                    resistance, by_name["R0"].lower, by_name["R0"].upper
                )
        inductance_value = table_parameters.get("L0")
        if inductance_value is not None:
            inductance_value = inductance_value.initial
        elif mode == "Ridge DRT":
            inductance_value = cycle.saved_ridge_inductance
        if inductance_value is not None:
            for parameter in parameters:
                if re.fullmatch(r"L\d+", parameter.name):
                    parameter.initial = self._clamp_parameter_value(
                        inductance_value,
                        parameter.lower,
                        parameter.upper,
                    )

        for branch_id, peak in zip(branch_ids, selected_peaks):
            tau, area, _fwhm = self._peak_summary(peak)
            resistance = max(abs(area), np.finfo(float).eps)
            resistance_parameter = by_name.get(f"R{branch_id}")
            if resistance_parameter is not None:
                resistance_parameter.initial = self._clamp_parameter_value(
                    resistance,
                    resistance_parameter.lower,
                    resistance_parameter.upper,
                )
            exponent_parameter = by_name.get(f"CPE{branch_id}_1")
            exponent = (
                exponent_parameter.initial if exponent_parameter is not None else 1.0
            )
            exponent = float(np.clip(exponent, 0.05, 1.0))
            q_parameter = by_name.get(f"CPE{branch_id}_0")
            if q_parameter is not None:
                q_value = tau**exponent / resistance
                q_parameter.initial = self._clamp_parameter_value(
                    q_value, q_parameter.lower, q_parameter.upper
                )
            capacitance_parameter = by_name.get(f"C{branch_id}")
            if capacitance_parameter is not None:
                capacitance_parameter.initial = self._clamp_parameter_value(
                    tau / resistance,
                    capacitance_parameter.lower,
                    capacitance_parameter.upper,
                )

        cycle.parameters = parameters
        self.parameter_table.set_parameters(parameters)
        self.analysis_mode_var.set("EEC")
        self._on_analysis_mode_selected()
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._update_status(
            f"sent {len(selected_peaks)} DRT peak initials to the EEC model"
        )

    def _autoscale_kk(self, cycle) -> None:
        if (
            self.kk_axes is None
            or cycle.kk_residual_real is None
            or cycle.kk_residual_imag is None
            or not cycle.kk_cache_matches()
        ):
            return
        y_values = 100.0 * np.concatenate(
            (cycle.kk_residual_real, cycle.kk_residual_imag)
        )
        finite = np.isfinite(y_values)
        if not np.any(finite):
            return
        y_values = y_values[finite]
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        y_span = y_max - y_min
        y_padding = 0.1 * (y_span if y_span > 0 else max(abs(y_max), 0.01))
        self.kk_axes.set_ylim(y_min - y_padding, y_max + y_padding)
        frequency = cycle.frequency_hz[cycle.included]
        finite_frequency = np.isfinite(frequency) & (frequency > 0)
        if np.any(finite_frequency):
            frequency = frequency[finite_frequency]
            self.kk_axes.set_xlim(float(np.min(frequency)), float(np.max(frequency)))

    def reset_plot_view(self) -> None:
        if self.analysis_mode_var.get() == "Spectra Simulator":
            self._refresh_plot(rescale=True)
            self._update_status("zoom reset to simulated spectrum")
            return
        if self.state is None:
            return
        self._refresh_plot(rescale=True)
        self._update_status(
            "zoom reset to all measured points"
            if self.show_all_points_var.get()
            else "zoom reset to active points"
        )

    def toggle_show_all_points(self) -> None:
        self._refresh_plot(rescale=True)
        self._update_status(
            "zooming to all measured points"
            if self.show_all_points_var.get()
            else "zooming to active points"
        )

    def toggle_plot_mode(self) -> None:
        self.plot_mode = "bode" if self.plot_mode == "nyquist" else "nyquist"
        self._configure_plot_layout()
        if self.state is None and self.analysis_mode_var.get() == "Spectra Simulator":
            self._refresh_plot(rescale=True)
            self._update_status(f"{self.plot_mode.title()} view")
            return
        if self.state is None:
            self.axes.set_title("No spectrum loaded")
            if self.drt_axes is not None:
                self.drt_axes.set_title("Ridge DRT")
            self.canvas.draw_idle()
            return
        self._refresh_plot(rescale=True)
        self._update_status(f"{self.plot_mode.title()} view")

    def _toggle_plot_mode_key(self, _event=None):
        self.toggle_plot_mode()
        return "break"

    def _toggle_display_key(self, variable_name: str, callback_name: str):
        variable = getattr(self, variable_name)
        variable.set(not variable.get())
        getattr(self, callback_name)()

    def _handle_alt_keypad(self, event):
        if getattr(event, "keysym", "").casefold() == "m":
            self.edit_metadata_column_from_clipboard()
            return "break"
        if getattr(event, "keycode", None) == 51 or getattr(event, "keysym", "") in {
            "3", "scaron"
        }:
            return self._open_plot_controls_key_menu(event)
        keysym_options = {
            "KP_1": ("show_spectrum_var", self.toggle_spectrum_view),
            "KP_End": ("show_spectrum_var", self.toggle_spectrum_view),
            "KP_2": ("show_kk_var", self.toggle_kk_view),
            "KP_Down": ("show_kk_var", self.toggle_kk_view),
            "KP_3": ("show_drt_var", self.toggle_drt_view),
            "KP_Next": ("show_drt_var", self.toggle_drt_view),
            "KP_4": ("show_eec_fit_var", self.toggle_fit_visibility),
            "KP_Left": ("show_eec_fit_var", self.toggle_fit_visibility),
            "KP_6": ("show_drt_fit_var", self.toggle_drt_fit_visibility),
            "KP_Right": ("show_drt_fit_var", self.toggle_drt_fit_visibility),
            "KP_9": ("show_drt_recovered_var", self.toggle_drt_recovered_visibility),
            "KP_Prior": ("show_drt_recovered_var", self.toggle_drt_recovered_visibility),
        }
        option = keysym_options.get(getattr(event, "keysym", ""))
        if option is None:
            keycode_options = {
                97: ("show_spectrum_var", self.toggle_spectrum_view),
                98: ("show_kk_var", self.toggle_kk_view),
                99: ("show_drt_var", self.toggle_drt_view),
                100: ("show_eec_fit_var", self.toggle_fit_visibility),
                102: ("show_drt_fit_var", self.toggle_drt_fit_visibility),
                105: ("show_drt_recovered_var", self.toggle_drt_recovered_visibility),
            }
            option = keycode_options.get(getattr(event, "keycode", None))
        if option is None:
            return None
        variable_name, callback = option
        variable = getattr(self, variable_name)
        variable.set(not variable.get())
        callback()
        return "break"

    def toggle_drt_view(self) -> None:
        self._configure_plot_layout()
        if self.analysis_mode_var.get() == "Spectra Simulator":
            self._refresh_plot(rescale=True)
            self.canvas.draw_idle()
            return
        if self.state is None:
            self.axes.set_title("No spectrum loaded")
            if self.drt_axes is not None:
                self.drt_axes.set_title("Ridge DRT")
        else:
            self._refresh_plot(rescale=True)
        self.canvas.draw_idle()

    def toggle_spectrum_view(self) -> None:
        self._configure_plot_layout()
        if self.analysis_mode_var.get() == "Spectra Simulator":
            self._refresh_plot(rescale=True)
            self.canvas.draw_idle()
            return
        if self.state is None:
            self.axes.set_title("No spectrum loaded")
            if self.drt_axes is not None:
                self.drt_axes.set_title("Ridge DRT")
        else:
            self._refresh_plot(rescale=True)
        self.canvas.draw_idle()

    def _drt_peak_impedance(self, cycle, frequency=None):
        if not self.drt_peak_parameters:
            return None
        if frequency is None:
            frequency = cycle.frequency_hz
        frequency = np.asarray(frequency, dtype=float)
        mode = self.analysis_drt_mode_var.get()
        resistance = (
            cycle.saved_hybrid_ohmic_resistance
            if mode == "Hybrid DRT"
            else cycle.saved_ridge_ohmic_resistance
        )
        inductance = (
            cycle.saved_ridge_inductance
            if mode == "Ridge DRT"
            else cycle.saved_hybrid_inductance
        )
        omega = 2.0 * np.pi * frequency
        impedance = np.full(frequency.size, float(resistance or 0.0), dtype=complex)
        if inductance is not None:
            impedance += 1j * omega * float(inductance)
        distributed_peaks = []
        for peak in self.drt_peak_parameters:
            tau, area, _fwhm = self._peak_summary(peak)
            if self._peak_shape(peak) == "hn":
                alpha = float(np.clip(peak.get("alpha", 0.8), 1e-3, 0.999))
                beta = float(np.clip(peak.get("beta", 0.8), 1e-3, 0.999))
                impedance += area / (1.0 + (1j * omega * tau) ** alpha) ** beta
            else:
                distributed_peaks.append(peak)
        if distributed_peaks:
            finite_frequency = frequency[np.isfinite(frequency) & (frequency > 0)]
            if finite_frequency.size:
                log_tau_min = np.log10(
                    1.0 / (2.0 * np.pi * np.max(finite_frequency))
                ) - 1.0
                log_tau_max = np.log10(
                    1.0 / (2.0 * np.pi * np.min(finite_frequency))
                ) + 1.0
            else:
                log_tau_min, log_tau_max = -8.0, 8.0
            for peak in distributed_peaks:
                center = float(peak["center_log10"])
                width = self._peak_half_width_log10(peak)
                log_tau_min = min(log_tau_min, center - 6.0 * width)
                log_tau_max = max(log_tau_max, center + 6.0 * width)
            log_tau_grid = np.linspace(log_tau_min, log_tau_max, 1600)
            tau_grid = 10.0**log_tau_grid
            gamma_grid = np.zeros_like(log_tau_grid)
            for peak in distributed_peaks:
                gamma_grid += self._peak_values(log_tau_grid, peak)
            for index, angular_frequency in enumerate(omega):
                kernel = 1.0 / (1.0 + 1j * angular_frequency * tau_grid)
                impedance[index] += np.trapezoid(
                    gamma_grid * kernel, log_tau_grid
                ) * np.log(10.0)
        return impedance

    def _calculate_drt_fit_impedance(self, cycle):
        if not self.drt_peak_parameters:
            return None, None
        measured_frequency = np.asarray(cycle.frequency_hz, dtype=float)
        finite = np.isfinite(measured_frequency) & (measured_frequency > 0)
        if not np.any(finite):
            return None, None
        frequency = np.geomspace(
            float(np.min(measured_frequency[finite])),
            float(np.max(measured_frequency[finite])),
            300,
        )
        mode = self.analysis_drt_mode_var.get()
        table_values = {
            parameter.name: parameter
            for parameter in self.drt_peak_table.values()
        }
        resistance = (
            cycle.saved_hybrid_ohmic_resistance
            if mode == "Hybrid DRT"
            else cycle.saved_ridge_ohmic_resistance
        )
        inductance = (
            cycle.saved_ridge_inductance if mode == "Ridge DRT" else None
        )
        if "R0" in table_values:
            resistance = table_values["R0"].initial
        if "R_inf" in table_values:
            resistance = table_values["R_inf"].initial
        if "L0" in table_values:
            inductance = table_values["L0"].initial
        if "inductance" in table_values:
            inductance = table_values["inductance"].initial
        log_tau_min = np.log10(1.0 / (2.0 * np.pi * np.max(frequency))) - 1.0
        log_tau_max = np.log10(1.0 / (2.0 * np.pi * np.min(frequency))) + 1.0
        for peak in self.drt_peak_parameters:
            center = float(peak["center_log10"])
            width = self._peak_half_width_log10(peak)
            log_tau_min = min(log_tau_min, center - 6.0 * width)
            log_tau_max = max(log_tau_max, center + 6.0 * width)
        log_tau_grid = np.linspace(log_tau_min, log_tau_max, 1600)
        tau_grid = 10.0**log_tau_grid
        gamma_grid = np.zeros_like(log_tau_grid)
        impedance = np.full(frequency.size, float(resistance or 0.0), dtype=complex)
        omega = 2.0 * np.pi * frequency
        if inductance is not None:
            impedance += 1j * omega * float(inductance)
        distributed_peaks = []
        for peak in self.drt_peak_parameters:
            if self._peak_shape(peak) == "hn":
                tau, area, _fwhm = self._peak_summary(peak)
                alpha = float(np.clip(peak.get("alpha", 0.8), 1e-3, 0.999))
                beta = float(np.clip(peak.get("beta", 0.8), 1e-3, 0.999))
                impedance += area / (1.0 + (1j * omega * tau) ** alpha) ** beta
            else:
                distributed_peaks.append(peak)
        for peak in distributed_peaks:
            gamma_grid += self._peak_values(log_tau_grid, peak)
        if distributed_peaks:
            for index, angular_frequency in enumerate(omega):
                kernel = 1.0 / (1.0 + 1j * angular_frequency * tau_grid)
                impedance[index] += np.trapezoid(
                    gamma_grid * kernel, log_tau_grid
                ) * np.log(10.0)
        return frequency, impedance

    def _refresh_drt_fit_artists(self, cycle) -> None:
        if not hasattr(self, "drt_fit_artist"):
            return
        self.drt_fit_artist.set_visible(self.show_drt_fit_var.get())
        if self.drt_phase_fit_artist is not None:
            self.drt_phase_fit_artist.set_visible(self.show_drt_fit_var.get())
        if not self.show_drt_fit_var.get():
            self.drt_fit_artist.set_data([], [])
            if self.drt_phase_fit_artist is not None:
                self.drt_phase_fit_artist.set_data([], [])
            return
        frequency, impedance = self._calculate_drt_fit_impedance(cycle)
        if frequency is None:
            self.drt_fit_artist.set_data([], [])
            if self.drt_phase_fit_artist is not None:
                self.drt_phase_fit_artist.set_data([], [])
            return
        if self.plot_mode == "bode":
            self.drt_fit_artist.set_data(frequency, np.abs(impedance))
            if self.drt_phase_fit_artist is not None:
                self.drt_phase_fit_artist.set_data(
                    frequency, self._phase_degrees(impedance)
                )
        else:
            self.drt_fit_artist.set_data(impedance.real, -impedance.imag)
            if self.drt_phase_fit_artist is not None:
                self.drt_phase_fit_artist.set_data([], [])

    def toggle_fit_visibility(self) -> None:
        self._refresh_plot(rescale=False)

    def toggle_drt_fit_visibility(self) -> None:
        self._refresh_plot(rescale=False)

    def toggle_kk_view(self) -> None:
        self._configure_plot_layout()
        if self.state is None:
            self.axes.set_title("No spectrum loaded")
        else:
            self._refresh_plot(rescale=True)
            if self.show_kk_var.get():
                self._ensure_kk_residuals()
                return
        self.canvas.draw_idle()

    @staticmethod
    def _popup_active_limits(cycles) -> tuple[float, float, float, float] | None:
        x_segments = []
        y_segments = []
        fallback_x_segments = []
        fallback_y_segments = []
        for _loaded, cycle in cycles:
            included = cycle.included
            real = cycle.impedance.real
            negative_imaginary = -cycle.impedance.imag
            if real.size:
                fallback_x_segments.append(real)
                fallback_y_segments.append(negative_imaginary)
            if np.any(included):
                x_segments.append(real[included])
                y_segments.append(negative_imaginary[included])
        if not x_segments:
            x_segments = fallback_x_segments
            y_segments = fallback_y_segments
        if not x_segments:
            return None
        x_values = np.concatenate(x_segments)
        y_values = np.concatenate(y_segments)
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(finite):
            return None
        x_values = x_values[finite]
        y_values = y_values[finite]
        x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
        y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
        x_span = x_max - x_min
        y_span = y_max - y_min
        x_padding = 0.06 * (x_span if x_span > 0 else max(abs(x_min), 1.0))
        y_padding = 0.06 * (y_span if y_span > 0 else max(abs(y_min), 1.0))
        return (
            x_min - x_padding,
            x_max + x_padding,
            y_min - y_padding,
            y_max + y_padding,
        )

    def _popup_bode_limits(
        self, cycles
    ) -> tuple[float, float, float, float, float, float] | None:
        frequency_segments = []
        magnitude_segments = []
        phase_segments = []
        fallback_frequency_segments = []
        fallback_magnitude_segments = []
        fallback_phase_segments = []
        for _loaded, cycle in cycles:
            included = cycle.included
            frequency = cycle.frequency_hz
            magnitude = np.abs(cycle.impedance)
            phase = self._phase_degrees(cycle.impedance)
            valid_frequency = np.isfinite(frequency) & (frequency > 0)
            valid_magnitude = np.isfinite(magnitude)
            valid_phase = np.isfinite(phase)
            valid = valid_frequency & valid_magnitude & valid_phase
            if np.any(valid):
                fallback_frequency_segments.append(frequency[valid])
                fallback_magnitude_segments.append(magnitude[valid])
                fallback_phase_segments.append(phase[valid])
            active = valid & included
            if np.any(active):
                frequency_segments.append(frequency[active])
                magnitude_segments.append(magnitude[active])
                phase_segments.append(phase[active])
        if not frequency_segments:
            frequency_segments = fallback_frequency_segments
            magnitude_segments = fallback_magnitude_segments
            phase_segments = fallback_phase_segments
        if not frequency_segments:
            return None
        frequency = np.concatenate(frequency_segments)
        magnitude = np.concatenate(magnitude_segments)
        phase = np.concatenate(phase_segments)
        x_min = float(np.min(frequency))
        x_max = float(np.max(frequency))
        if x_min == x_max:
            x_min /= 1.3
            x_max *= 1.3
        else:
            span_decades = np.log10(x_max) - np.log10(x_min)
            x_padding_factor = 10 ** (0.04 * span_decades)
            x_min /= x_padding_factor
            x_max *= x_padding_factor
        magnitude_min = float(np.min(magnitude))
        magnitude_max = float(np.max(magnitude))
        magnitude_span = magnitude_max - magnitude_min
        magnitude_padding = 0.06 * (
            magnitude_span if magnitude_span > 0 else max(abs(magnitude_min), 1.0)
        )
        phase_min = float(np.min(phase))
        phase_max = float(np.max(phase))
        phase_span = phase_max - phase_min
        phase_padding = 0.08 * (
            phase_span if phase_span > 0 else max(abs(phase_min), 1.0)
        )
        return (
            x_min,
            x_max,
            magnitude_min - magnitude_padding,
            magnitude_max + magnitude_padding,
            phase_min - phase_padding,
            phase_max + phase_padding,
        )

    def _update_status(self, suffix: str = "") -> None:
        if self.state is None:
            return
        cycle = self.state.active
        total = cycle.frequency_hz.size
        included = int(np.count_nonzero(cycle.included))
        outliers = int(np.count_nonzero(cycle.outliers))
        status = (
            f"Cycle {cycle.cycle} · {included}/{total} points included · "
            f"{outliers} detected outliers · click a point to toggle it"
        )
        self.status_var.set(f"{status} · {suffix}" if suffix else status)

    def _clear_loaded_view(self, message: str) -> None:
        self.loaded = None
        self.state = None
        self.current_dataset_id = None
        self.loaded_projects.clear()
        self._dataset_order.clear()
        self._custom_metadata_columns.clear()
        self._explorer_rows.clear()
        self._explorer_lookup.clear()
        self._explorer_anchor_item = None
        self._explorer_primary_item = None
        self.explorer.delete(*self.explorer.get_children())
        self.cycle_var.set("")
        self.parameter_table.set_parameters([])
        self.included_artist.set_data([], [])
        self.excluded_artist.set_data([], [])
        self.fit_artist.set_data([], [])
        if hasattr(self, "drt_fit_artist"):
            self.drt_fit_artist.set_data([], [])
        if getattr(self, "drt_phase_fit_artist", None) is not None:
            self.drt_phase_fit_artist.set_data([], [])
        self.fit_points_included_artist.set_data([], [])
        self.fit_points_excluded_artist.set_data([], [])
        self.residual_artist.set_segments([])
        self.excluded_residual_artist.set_segments([])
        if self.kk_real_artist is not None:
            self.kk_real_artist.set_data([], [])
        if self.kk_imag_artist is not None:
            self.kk_imag_artist.set_data([], [])
        if self.phase_included_artist is not None:
            self.phase_included_artist.set_data([], [])
        if self.phase_excluded_artist is not None:
            self.phase_excluded_artist.set_data([], [])
        if self.phase_fit_artist is not None:
            self.phase_fit_artist.set_data([], [])
        if self.phase_fit_points_included_artist is not None:
            self.phase_fit_points_included_artist.set_data([], [])
        if self.phase_fit_points_excluded_artist is not None:
            self.phase_fit_points_excluded_artist.set_data([], [])
        if self.phase_residual_artist is not None:
            self.phase_residual_artist.set_segments([])
        if self.phase_excluded_residual_artist is not None:
            self.phase_excluded_residual_artist.set_segments([])
        if self.drt_artist is not None:
            self.drt_artist.set_data([], [])
        if self.drt_axes is not None:
            self.drt_axes.set_title("Ridge DRT")
        if self.kk_axes is not None:
            self.kk_axes.set_title("Lin-KK residuals")
        self.axes.set_title("No spectrum loaded")
        self.canvas.draw_idle()
        self.status_var.set(message)
        self._update_window_title()
        self._set_controls_enabled(False)

    def _start_drt_peak_drag(self, event) -> bool:
        if self.busy or self.state is None or not self.drt_peak_parameters:
            return False
        if event.x is None or event.y is None:
            return False
        best = None
        for index, peak in enumerate(self.drt_peak_parameters):
            center_tau = 10.0 ** peak["center_log10"]
            half_width = self._peak_half_width_log10(peak)
            points = [("top", center_tau, peak["height"])]
            if self._peak_shape(peak) != "hn":
                points.extend(
                    (
                        ("left", 10.0 ** (peak["center_log10"] - half_width), peak["height"] / 2.0),
                        ("right", 10.0 ** (peak["center_log10"] + half_width), peak["height"] / 2.0),
                    )
                )
            for action, x_value, y_value in points:
                display = self.drt_axes.transData.transform((x_value, y_value))
                distance = float(np.hypot(display[0] - event.x, display[1] - event.y))
                if distance <= 12.0 and (best is None or distance < best[0]):
                    best = (distance, index, action)
        if best is None:
            return False
        self._drt_peak_drag = {"index": best[1], "action": best[2]}
        return True

    def _on_plot_click(self, event) -> None:
        if (
            self.busy
            or self.state is None
            or event.button not in (1, 3)
            or event.inaxes not in {self.axes, getattr(self, "phase_axes", None)}
            or not self.point_toggle_mode
        ):
            return
        if event.x is None or event.y is None:
            return
        cycle = self.state.active
        if self.plot_mode == "bode" and self.phase_axes is not None:
            if event.inaxes is self.phase_axes:
                display_points = self.phase_axes.transData.transform(
                    np.column_stack(
                        (cycle.frequency_hz, self._phase_degrees(cycle.impedance))
                    )
                )
            else:
                display_points = self.axes.transData.transform(
                    np.column_stack((cycle.frequency_hz, np.abs(cycle.impedance)))
                )
        else:
            display_points = self.axes.transData.transform(
                np.column_stack((cycle.impedance.real, -cycle.impedance.imag))
            )
        distances = np.hypot(
            display_points[:, 0] - event.x, display_points[:, 1] - event.y
        )
        if distances.size == 0:
            return
        index = int(np.argmin(distances))
        if distances[index] > 10:
            return
        cycle.toggle_point(index)
        self._refresh_plot(rescale=True)
        self._update_status()
        self._fit_after_point_edit()

    def _on_plot_button_press(self, event) -> None:
        if (
            event.button == 1
            and event.inaxes is getattr(self, "drt_axes", None)
            and self._start_drt_peak_drag(event)
        ):
            self._drt_peak_drag_moved = False
            return
        if event.button == 1 and event.inaxes is getattr(self, "drt_axes", None):
            self._select_drt_peak_from_event(event)
            return
        if self.busy or self.state is None or event.button != 2 or event.inaxes is None:
            return
        axes = event.inaxes
        if axes not in self._active_plot_axes():
            return
        if event.xdata is None or event.ydata is None:
            return
        self._pan_state = {
            "axes": axes,
            "xdata": event.xdata,
            "ydata": event.ydata,
            "xlim": axes.get_xlim(),
            "ylim": axes.get_ylim(),
        }
        if hasattr(self, "zoom_selector"):
            self.zoom_selector.set_active(False)

    def _on_plot_button_release(self, event) -> None:
        if event.button == 1 and self._drt_peak_drag is not None:
            if not self._drt_peak_drag_moved:
                index = self._drt_peak_drag["index"]
                self._select_drt_peak(
                    None if index == self._selected_drt_peak_index else index
                )
            self._drt_peak_drag = None
            self._drt_peak_drag_moved = False
            self._update_drt_peak_table()
            self.canvas.draw_idle()
            return
        if event.button != 2 or self._pan_state is None:
            return
        self._pan_state = None
        if hasattr(self, "zoom_selector"):
            self.zoom_selector.set_active(not self.point_toggle_mode)

    def _on_plot_motion(self, event) -> None:
        if self._drt_peak_drag is not None:
            if event.inaxes is not self.drt_axes or event.xdata is None or event.ydata is None:
                return
            self._drt_peak_drag_moved = True
            peak = self.drt_peak_parameters[self._drt_peak_drag["index"]]
            action = self._drt_peak_drag["action"]
            if action == "top":
                peak["center_log10"] = float(np.log10(max(event.xdata, 1e-300)))
                peak["height"] = float(event.ydata)
            else:
                distance = abs(
                    float(np.log10(max(event.xdata, 1e-300)))
                    - peak["center_log10"]
                )
                peak["sigma_log10"] = max(
                    distance / (
                        1.0
                        if self._peak_shape(peak) == "lorentzian"
                        else self._peak_half_width_log10(peak)
                        / max(peak["sigma_log10"], 1e-6)
                    ),
                    1e-4,
                )
            self._store_current_drt_peaks()
            self._refresh_drt_peak_artists()
            self.canvas.draw_idle()
            return
        if self._pan_state is None:
            self._update_point_hover(event)
            return
        self._hide_point_hover()
        axes = self._pan_state["axes"]
        if event.inaxes is not axes or event.xdata is None or event.ydata is None:
            return
        delta_x = event.xdata - self._pan_state["xdata"]
        delta_y = event.ydata - self._pan_state["ydata"]
        x_min, x_max = self._pan_state["xlim"]
        y_min, y_max = self._pan_state["ylim"]
        axes.set_xlim(x_min - delta_x, x_max - delta_x)
        axes.set_ylim(y_min - delta_y, y_max - delta_y)
        self.canvas.draw_idle()

    def _on_plot_scroll(self, event) -> None:
        if self.busy or self.state is None or event.inaxes is None:
            return
        axes = event.inaxes
        if axes not in self._active_plot_axes():
            return
        if event.xdata is None or event.ydata is None:
            return
        scale = 1 / 1.2 if event.button == "up" else 1.2
        x_min, x_max = axes.get_xlim()
        y_min, y_max = axes.get_ylim()
        new_x_min = event.xdata - (event.xdata - x_min) * scale
        new_x_max = event.xdata + (x_max - event.xdata) * scale
        new_y_min = event.ydata - (event.ydata - y_min) * scale
        new_y_max = event.ydata + (y_max - event.ydata) * scale
        axes.set_xlim(new_x_min, new_x_max)
        axes.set_ylim(new_y_min, new_y_max)
        self.canvas.draw_idle()

    def _on_zoom_select(self, press_event, release_event) -> None:
        if (
            self.point_toggle_mode
            or self._pan_state is not None
            or press_event.inaxes is not self.axes
        ):
            return
        if (
            press_event.xdata is None
            or press_event.ydata is None
            or release_event.xdata is None
            or release_event.ydata is None
        ):
            return
        x0, x1 = sorted((press_event.xdata, release_event.xdata))
        y0, y1 = sorted((press_event.ydata, release_event.ydata))
        if abs(x1 - x0) <= np.finfo(float).eps or abs(y1 - y0) <= np.finfo(float).eps:
            return
        self.axes.set_xlim(x0, x1)
        self.axes.set_ylim(y0, y1)
        self.canvas.draw_idle()

    def _on_edit_area_select(self, press_event, release_event) -> None:
        if (
            not self.point_toggle_mode
            or self.busy
            or self.state is None
            or press_event.inaxes is not self.axes
            or release_event.inaxes is not self.axes
            or press_event.xdata is None
            or press_event.ydata is None
            or release_event.xdata is None
            or release_event.ydata is None
        ):
            return
        x0, x1 = sorted((press_event.xdata, release_event.xdata))
        y0, y1 = sorted((press_event.ydata, release_event.ydata))
        cycle = self.state.active
        if self.plot_mode == "bode":
            x_values = cycle.frequency_hz
            y_values = np.abs(cycle.impedance)
        else:
            x_values = cycle.impedance.real
            y_values = -cycle.impedance.imag
        selected = (
            np.isfinite(x_values)
            & np.isfinite(y_values)
            & (x_values >= x0)
            & (x_values <= x1)
            & (y_values >= y0)
            & (y_values <= y1)
        )
        if cycle.frequency_window is not None:
            minimum, maximum = sorted(cycle.frequency_window)
            selected &= (cycle.frequency_hz >= minimum) & (cycle.frequency_hz <= maximum)
        indices = np.flatnonzero(selected)
        if indices.size == 0:
            return
        cycle.manually_included[indices] = ~cycle.manually_included[indices]
        cycle.outliers[indices[cycle.manually_included[indices]]] = False
        cycle.invalidate_drt_cache()
        cycle.clear_fit()
        self._refresh_plot(rescale=True)
        self._update_status(f"toggled {indices.size} points in selected area")
        self._fit_after_point_edit()

    def toggle_point_edit_mode(self, _event=None) -> str | None:
        self.point_toggle_mode = not self.point_toggle_mode
        if self.point_toggle_mode:
            self._hide_point_hover()
        self.toggle_points_button.configure(
            text=f"Edit points: {'On' if self.point_toggle_mode else 'Off'}"
        )
        if hasattr(self, "zoom_selector"):
            self.zoom_selector.set_active(not self.point_toggle_mode)
        if hasattr(self, "edit_selector"):
            self.edit_selector.set_active(self.point_toggle_mode)
        self._update_status()
        return "break" if _event is not None else None

    def toggle_auto_fit_points(self) -> None:
        self.point_auto_fit = not self.point_auto_fit
        if self.point_auto_fit:
            self._hide_point_hover()
        if self.point_auto_fit and not self.point_toggle_mode:
            self.point_toggle_mode = True
            self.toggle_points_button.configure(text="Edit points: On")
            if hasattr(self, "zoom_selector"):
                self.zoom_selector.set_active(False)
            if hasattr(self, "edit_selector"):
                self.edit_selector.set_active(True)
        self.auto_fit_points_button.configure(
            text=f"Edit points and fit: {'On' if self.point_auto_fit else 'Off'}"
        )
        self._update_status()

    def _toggle_auto_fit_points_key(self, _event=None):
        self.toggle_auto_fit_points()
        return "break"

    def _fit_after_point_edit(self) -> None:
        if not self.point_auto_fit or self.busy or self.state is None:
            return
        if self.analysis_mode_var.get() == "DRT":
            self._fit_drt_peaks_after_point_edit = True
            if self.analysis_drt_mode_var.get() == "Hybrid DRT":
                self.calculate_hybrid_drt()
            else:
                self.calculate_ridge_drt()
            return
        self.fit()

    def _update_status(self, suffix: str = "") -> None:
        if self.state is None:
            return
        cycle = self.state.active
        total = cycle.frequency_hz.size
        included = int(np.count_nonzero(cycle.included))
        outliers = int(np.count_nonzero(cycle.outliers))
        mode_text = (
            "point edit mode on"
            if self.point_toggle_mode
            else "wheel/drag zoom active"
        )
        status = (
            f"Cycle {cycle.cycle} · {included}/{total} points included · "
            f"{outliers} detected outliers · {mode_text}"
        )
        self.status_var.set(f"{status} · {suffix}" if suffix else status)

    def change_cycle(
        self,
        direction: int,
        preserve_selection: bool = False,
        focus_only: bool = False,
    ) -> None:
        if self.busy or self.state is None:
            return
        visible_items = list(self.explorer.get_children(""))
        if not visible_items:
            return
        current_item = self._explorer_lookup.get(
            (self.current_dataset_id, self.state.active_cycle)
        )
        if current_item not in visible_items:
            current_item = self._explorer_primary_item
        if current_item in visible_items:
            index = visible_items.index(current_item)
        else:
            index = 0 if direction > 0 else len(visible_items) - 1
        next_index = max(0, min(len(visible_items) - 1, index + direction))
        next_item = visible_items[next_index]
        dataset_id, loaded, spectrum = self._explorer_rows[next_item]
        if dataset_id == self.current_dataset_id:
            self._activate_cycle(
                spectrum.cycle,
                preserve_existing_selection=preserve_selection,
                focus_only=focus_only,
            )
        else:
            self._switch_dataset(
                dataset_id,
                loaded,
                spectrum.cycle,
                preserve_existing_selection=preserve_selection,
                focus_only=focus_only,
            )

    def _activate_cycle(
        self,
        cycle_number: int,
        *,
        preserve_existing_selection: bool = False,
        focus_only: bool = False,
    ) -> None:
        if self.state is None or cycle_number == self.state.active_cycle:
            return
        if not self._capture_controls():
            self.cycle_var.set(str(self.state.active_cycle))
            self._highlight_explorer_cycle(
                self.state.active_cycle,
                preserve_existing=False,
            )
            return
        if cycle_number not in self.state.cycles:
            assert self.loaded is not None
            new_cycle = load_cycle(
                self.loaded.dataframe, cycle_number, self.state.control
            )
            if self.state.all_frequency_window is not None:
                new_cycle.frequency_window = self.state.all_frequency_window
            new_cycle.parameters = self.state.parameters_for(cycle_number)
            new_cycle.circuit = self.state.circuit
            self.state.cycles[cycle_number] = new_cycle
        self.state.active_cycle = cycle_number
        self.cycle_var.set(str(cycle_number))
        self._highlight_explorer_cycle(
            cycle_number,
            preserve_existing=preserve_existing_selection,
            focus_only=focus_only,
        )
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._update_status()
        if self.show_kk_var.get():
            self._ensure_kk_residuals()

    def apply_frequency_window(self) -> None:
        if self._capture_controls():
            self.state.active.clear_fit()
            self._refresh_plot(rescale=True)
            self._update_status("frequency range applied")

    def apply_frequency_window_to_selected(self) -> None:
        if self.state is None or not self._capture_controls():
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        window = self.state.active.frequency_window
        assert window is not None
        minimum, maximum = window
        updated = 0
        for _dataset_id, loaded, spectrum in selected_rows:
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            cycle.auto_max_frequency = self.auto_max_frequency_var.get()
            selected_window = window
            if self.auto_max_frequency_var.get():
                detected_maximum = self._automatic_max_frequency(cycle)
                if detected_maximum is not None:
                    selected_window = (
                        min(minimum, detected_maximum),
                        detected_maximum,
                    )
            cycle.frequency_window = selected_window
            cycle.invalidate_drt_cache()
            cycle.clear_fit()
            updated += 1
        self._refresh_plot(rescale=True)
        self._update_status(
            f"frequency range applied to {updated} selected spectra"
        )

    def _configure_cycle_model(
        self,
        cycle,
        circuit: str,
        parameters=None,
        fallback_circuit: str | None = None,
    ) -> None:
        if parameters is None:
            parameters = circuit_parameters(circuit, self._eec_parameter_bounds)
        old_circuit = cycle.model(fallback_circuit or circuit)
        equivalent = circuits_equivalent(old_circuit, circuit)
        element_mapping = (
            parameter_name_mapping(old_circuit, circuit) if equivalent else None
        )
        old_parameters = list(cycle.parameters)
        old_fit = (
            np.asarray(cycle.fit_parameters, dtype=float).copy()
            if cycle.fit_parameters is not None
            else None
        )
        old_by_target_name = {}
        if element_mapping is not None:
            for index, old_parameter in enumerate(old_parameters):
                target_name = map_parameter_name(old_parameter.name, element_mapping)
                if target_name is not None:
                    old_by_target_name[target_name] = (
                        old_parameter,
                        old_fit[index] if old_fit is not None and index < old_fit.size else None,
                    )
        cycle.circuit = circuit
        cycle.parameters = [
            ParameterValue(
                parameter.name,
                parameter.unit,
                parameter.initial,
                parameter.lower,
                parameter.upper,
                parameter.error_percent,
                parameter.fixed,
            )
            for parameter in parameters
        ]
        if equivalent and old_by_target_name:
            for parameter in cycle.parameters:
                previous = old_by_target_name.get(parameter.name)
                if previous is None:
                    continue
                old_parameter, fitted_value = previous
                parameter.initial = old_parameter.initial
                parameter.lower = old_parameter.lower
                parameter.upper = old_parameter.upper
                parameter.error_percent = old_parameter.error_percent
                parameter.fixed = old_parameter.fixed
            fitted_values = [
                old_by_target_name[parameter.name][1]
                for parameter in cycle.parameters
            ] if all(parameter.name in old_by_target_name for parameter in cycle.parameters) else []
            if old_fit is not None and fitted_values and all(
                value is not None for value in fitted_values
            ):
                cycle.fit_parameters = np.asarray(
                    fitted_values,
                    dtype=float,
                )
            elif old_fit is not None:
                cycle.fit_parameters = None
                cycle.fit_frequency_hz = None
                cycle.fit_impedance = None
                cycle.fit_at_data_impedance = None
            return
        cycle.clear_fit()
        cycle.invalidate_drt_cache()

    def apply_model(self) -> None:
        if self.state is None or self.busy:
            return
        circuit = self.model_var.get().strip()
        if not circuit:
            messagebox.showerror(
                "Invalid model", "Enter a circuit model", parent=self.root
            )
            return
        try:
            parameters = circuit_parameters(circuit, self._eec_parameter_bounds)
        except Exception as error:
            messagebox.showerror(
                "Invalid model",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            self.model_var.set(self.state.active.model(self.state.circuit))
            return
        self._configure_cycle_model(
            self.state.active, circuit, parameters, self.state.circuit
        )
        self.circuit = circuit
        self.parameter_table.set_parameters(
            self.state.parameters_for(self.state.active_cycle)
        )
        self._refresh_plot(rescale=True)
        self._update_status("fitting model changed")

    def apply_model_to_selected(self) -> None:
        if self.state is None or self.busy or not self._capture_controls():
            return
        circuit = self.model_var.get().strip()
        if not circuit:
            messagebox.showerror(
                "Invalid model", "Enter a circuit model", parent=self.root
            )
            return
        try:
            parameters = circuit_parameters(circuit, self._eec_parameter_bounds)
        except Exception as error:
            messagebox.showerror(
                "Invalid model",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        updated = 0
        for _dataset_id, loaded, spectrum in selected_rows:
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            self._configure_cycle_model(cycle, circuit, parameters, loaded.state.circuit)
            updated += 1
        self._restore_controls()
        self._refresh_explorer_values()
        self._refresh_plot(rescale=True)
        self._update_status(f"model applied to {updated} selected spectra")

    def apply_ml_eec_to_selected(self) -> None:
        if self.busy or self.state is None:
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        if self.analysis_mode_var.get() != "EEC":
            self.analysis_mode_var.set("EEC")
            self._on_analysis_mode_selected()
        self._run_ml_processing({"model"}, selected_rows)

    def auto_select_model(self) -> None:
        if self.state is None or self.busy:
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        if not self._capture_controls():
            return
        targets = [
            (
                dataset_id,
                spectrum.cycle,
                copy.deepcopy(self._loaded_cycle_for_popup(loaded, spectrum.cycle)),
            )
            for dataset_id, loaded, spectrum in selected_rows
        ]
        self.status_var.set(
            f"Selecting EEC models with Hybrid DRT for {len(targets)} spectra..."
        )

        def select_models():
            results = []
            for dataset_id, cycle_number, cycle in targets:
                if self._stop_event.is_set():
                    break
                results.append(
                    (
                        dataset_id,
                        cycle_number,
                        select_eec_model_from_hybrid_drt(
                            cycle,
                            settings=copy.deepcopy(self._auto_model_settings),
                        ),
                    )
                )
            return results

        self._submit(
            select_models,
            self._finish_auto_model_selection,
            "Automatic model selection failed",
            operation_labels=[f"cycle {cycle_number}" for _dataset_id, cycle_number, _cycle in targets],
            operation_name="Automatic EEC model selection",
        )

    def _finish_auto_model_selection(
        self,
        results: list[tuple[str, int, AutomaticEECModel]],
    ) -> None:
        if self.state is None:
            return
        current_result = None
        updated = 0
        for dataset_id, cycle_number, result in results:
            loaded = self.loaded_projects.get(dataset_id)
            if loaded is None:
                continue
            cycle = self._loaded_cycle_for_popup(loaded, cycle_number)
            parameters = circuit_parameters(
                result.circuit, self._eec_parameter_bounds
            )
            for parameter in parameters:
                if parameter.name in result.initials:
                    parameter.initial = self._clamp_parameter_value(
                        result.initials[parameter.name],
                        parameter.lower,
                        parameter.upper,
                    )
            cycle.circuit = result.circuit
            cycle.parameters = parameters
            cycle.clear_fit()
            cycle.invalidate_drt_cache()
            updated += 1
            if (
                loaded is self.loaded
                and cycle_number == self.state.active_cycle
            ):
                current_result = result
        if current_result is not None:
            self.circuit = current_result.circuit
            self.model_var.set(current_result.circuit)
            self.parameter_table.set_parameters(
                self.state.parameters_for(self.state.active_cycle)
            )
        self._restore_controls()
        self._refresh_explorer_values()
        self._refresh_plot(rescale=True)
        self._update_status(
            f"Hybrid DRT selected models for {updated} spectra"
        )

    @staticmethod
    def _rcpe_block_ids(circuit: str) -> tuple[str, ...]:
        """Return R/CPE branch suffixes in circuit order."""
        try:
            root = parse_circuit(circuit)
        except ValueError:
            return ()
        block_ids: list[str] = []

        def visit(node) -> None:
            if node.kind == "parallel":
                elements = [
                    child.value
                    for child in node.children
                    if child.kind == "element" and child.value
                ]
                if len(elements) == 2:
                    resistance = next(
                        (
                            match
                            for value in elements
                            if (match := re.fullmatch(r"R(\d+)", value)) is not None
                        ),
                        None,
                    )
                    cpe = next(
                        (
                            match
                            for value in elements
                            if (match := re.fullmatch(r"CPE(\d+)", value)) is not None
                        ),
                        None,
                    )
                    if resistance is not None and cpe is not None:
                        if resistance.group(1) == cpe.group(1):
                            block_ids.append(resistance.group(1))
            for child in node.children:
                visit(child)

        visit(root)
        return tuple(dict.fromkeys(block_ids))

    @staticmethod
    def _switch_parameter_blocks(
        state: ProjectState,
        cycle,
        first_id: str = "1",
        second_id: str = "2",
    ) -> bool:
        group_ids = EISApplication._rcpe_block_ids(cycle.model(state.circuit))
        if len(group_ids) < 2 or first_id == second_id:
            return False
        if first_id not in group_ids or second_id not in group_ids:
            return False
        parameters = {parameter.name: parameter for parameter in cycle.parameters}
        first_names = (f"R{first_id}", f"CPE{first_id}_0", f"CPE{first_id}_1")
        second_names = (f"R{second_id}", f"CPE{second_id}_0", f"CPE{second_id}_1")
        names = (*first_names, *second_names)
        if any(name not in parameters for name in names):
            return False
        for first_name, second_name in zip(first_names, second_names):
            first = parameters[first_name]
            second = parameters[second_name]
            first.initial, second.initial = second.initial, first.initial
            first.error_percent, second.error_percent = (
                second.error_percent,
                first.error_percent,
            )
            first.fixed, second.fixed = second.fixed, first.fixed
        if cycle.fit_parameters is not None:
            fitted = {
                parameter.name: float(value)
                for parameter, value in zip(cycle.parameters, cycle.fit_parameters)
            }
            for first_name, second_name in zip(first_names, second_names):
                fitted[first_name], fitted[second_name] = (
                    fitted[second_name],
                    fitted[first_name],
                )
            cycle.fit_parameters = np.asarray(
                [fitted[parameter.name] for parameter in cycle.parameters],
                dtype=float,
            )
        cycle.invalidate_drt_cache()
        return True

    def _choose_parameter_blocks(self, group_ids: tuple[str, ...]) -> tuple[str, str] | None:
        popup = tk.Toplevel(self.root)
        popup.title("Switch parameter blocks")
        popup.transient(self.root)
        popup.resizable(False, False)
        popup.columnconfigure(1, weight=1)
        result: list[tuple[str, str] | None] = [None]
        block_values = tuple(f"Block {group_id} (R{group_id}, CPE{group_id})" for group_id in group_ids)
        display_to_id = dict(zip(block_values, group_ids))
        first_var = tk.StringVar(value=block_values[0])
        second_var = tk.StringVar(value=block_values[1])

        ttk.Label(
            popup,
            text="Select the two R/CPE blocks whose complete parameter groups should be exchanged.",
            wraplength=430,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 12), sticky="w")
        ttk.Label(popup, text="First block").grid(row=1, column=0, padx=(14, 8), pady=4, sticky="w")
        ttk.Combobox(
            popup, textvariable=first_var, values=block_values, state="readonly", width=30
        ).grid(row=1, column=1, padx=(0, 14), pady=4, sticky="ew")
        ttk.Label(popup, text="Second block").grid(row=2, column=0, padx=(14, 8), pady=4, sticky="w")
        ttk.Combobox(
            popup, textvariable=second_var, values=block_values, state="readonly", width=30
        ).grid(row=2, column=1, padx=(0, 14), pady=4, sticky="ew")

        buttons = ttk.Frame(popup)
        buttons.grid(row=3, column=0, columnspan=2, padx=14, pady=(12, 14), sticky="e")

        def close() -> None:
            popup.grab_release()
            popup.destroy()

        def confirm() -> None:
            first_id = display_to_id[first_var.get()]
            second_id = display_to_id[second_var.get()]
            if first_id == second_id:
                messagebox.showerror(
                    "Choose two different blocks",
                    "Select two different R/CPE blocks to switch.",
                    parent=popup,
                )
                return
            result[0] = (first_id, second_id)
            close()

        ttk.Button(buttons, text="Switch", command=confirm).pack(side=tk.LEFT, padx=3)
        ttk.Button(buttons, text="Cancel", command=close).pack(side=tk.LEFT, padx=3)
        popup.protocol("WM_DELETE_WINDOW", close)
        popup.grab_set()
        popup.focus_force()
        popup.wait_window()
        return result[0]

    def switch_selected_parameter_blocks(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        available_blocks: tuple[str, ...] = ()
        selected_cycles = []
        for _dataset_id, loaded, spectrum in selected_rows:
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            group_ids = self._rcpe_block_ids(cycle.model(loaded.state.circuit))
            if len(group_ids) > len(available_blocks):
                available_blocks = group_ids
            selected_cycles.append((loaded, cycle))
        if len(available_blocks) < 2:
            self._update_status("switch blocks requires at least two R/CPE blocks")
            return
        block_pair = (available_blocks[0], available_blocks[1])
        if len(available_blocks) > 2:
            block_pair = self._choose_parameter_blocks(available_blocks)
            if block_pair is None:
                return
        changed = 0
        for loaded, cycle in selected_cycles:
            if self._switch_parameter_blocks(loaded.state, cycle, *block_pair):
                changed += 1
        self._restore_controls()
        self._refresh_explorer_values()
        self._refresh_plot(rescale=True)
        self._update_status(
            f"parameter blocks switched for {changed} selected spectra"
            if changed
            else "selected spectra do not contain the chosen R/CPE blocks"
        )

    def _sort_cycle_parameters_by_tau(
        self,
        state: ProjectState,
        cycle,
    ) -> bool:
        circuit = cycle.model(state.circuit)
        group_ids = re.findall(r"p\(\s*R(\d+)\s*,\s*CPE\1\s*\)", circuit)
        if len(group_ids) < 2:
            return False
        parameters_by_name = {parameter.name: parameter for parameter in cycle.parameters}
        groups = []
        for group_id in group_ids:
            names = (f"R{group_id}", f"CPE{group_id}_0", f"CPE{group_id}_1")
            if not all(name in parameters_by_name for name in names):
                continue
            try:
                tau_value = float(
                    cpe_tau(
                        parameters_by_name[names[0]].initial,
                        parameters_by_name[names[1]].initial,
                        parameters_by_name[names[2]].initial,
                    )
                )
            except (TypeError, ValueError):
                tau_value = float("inf")
            groups.append((group_id, tau_value))
        if len(groups) < 2:
            return False
        original_parameter_names = [parameter.name for parameter in cycle.parameters]
        ordered_groups = sorted(groups, key=lambda item: item[1])
        if [group_id for group_id, _tau in groups] == [
            group_id for group_id, _tau in ordered_groups
        ]:
            return False
        original_parameters = dict(parameters_by_name)
        for destination, (destination_id, _tau) in enumerate(ordered_groups):
            source_id = groups[destination][0]
            for suffix in ("", "_0", "_1"):
                destination_name = (
                    f"R{destination_id}" if suffix == "" else f"CPE{destination_id}{suffix}"
                )
                source_name = (
                    f"R{source_id}" if suffix == "" else f"CPE{source_id}{suffix}"
                )
                source = original_parameters[source_name]
                parameters_by_name[destination_name] = ParameterValue(
                    destination_name,
                    source.unit,
                    source.initial,
                    source.lower,
                    source.upper,
                    source.error_percent,
                    source.fixed,
                )
        cycle.parameters = [parameters_by_name[parameter.name] for parameter in cycle.parameters]
        if cycle.fit_parameters is not None:
            fitted_by_name = {
                name: float(value)
                for name, value in zip(original_parameter_names, cycle.fit_parameters)
            }
            original_fitted = dict(fitted_by_name)
            for destination, (destination_id, _tau) in enumerate(ordered_groups):
                source_id = groups[destination][0]
                for suffix in ("", "_0", "_1"):
                    destination_name = (
                        f"R{destination_id}" if suffix == "" else f"CPE{destination_id}{suffix}"
                    )
                    source_name = (
                        f"R{source_id}" if suffix == "" else f"CPE{source_id}{suffix}"
                    )
                    fitted_by_name[destination_name] = original_fitted[source_name]
            cycle.fit_parameters = np.asarray(
                [fitted_by_name[parameter.name] for parameter in cycle.parameters],
                dtype=float,
            )
        return True

    def sort_selected_parameters_by_tau(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        changed = 0
        for _dataset_id, loaded, spectrum in selected_rows:
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            if self._sort_cycle_parameters_by_tau(loaded.state, cycle):
                changed += 1
        self._restore_controls()
        self._refresh_explorer_values()
        self._refresh_plot(rescale=True)
        self._update_status(f"parameters sorted by tau for {changed} selected spectra")

    def find_outliers(self) -> None:
        if self.state is None or not self._capture_controls():
            return
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid threshold", "Enter a numeric threshold", parent=self.root
            )
            return
        cycle_number = self.state.active_cycle
        cycle = self.state.active
        parameters = self.state.parameters_for(cycle_number)
        self.status_var.set(f"Cycle {cycle_number} · finding outliers…")
        self._submit(
            lambda: self._cached_ridge_analysis(cycle, threshold, parameters)
            or analyze_outliers(cycle, threshold, parameters),
            lambda analysis: self._finish_outliers(cycle_number, analysis),
            "Outlier search failed",
        )

        if getattr(self, "_drt_peak_batch_queue", None) is not None:
            self.root.after(0, self._continue_drt_peak_batch_after_calculation)

    def _finish_outliers(
        self,
        cycle_number: int,
        analysis: RidgeInitialization,
    ) -> None:
        if self.state is None:
            return
        cycle = self.state.cycles[cycle_number]
        cycle.apply_outliers(analysis.outlier_indices)
        cycle.parameters = analysis.parameters
        if self.state.active_cycle == cycle_number:
            self.parameter_table.set_parameters(analysis.parameters)
            self._refresh_plot(rescale=True)
            self._update_status(
                f"outlier search complete; ridge initialized "
                f"R∞={analysis.ohmic_resistance:.3g} Ω, "
                f"L={analysis.inductance:.3g} H, "
                f"{analysis.peak_count} peaks"
            )

    def find_outliers_for_selected(self) -> None:
        if self.state is None or self.loaded is None or not self._capture_controls():
            return
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid threshold", "Enter a numeric threshold", parent=self.root
            )
            return
        cycle_count = len(self.state.available_cycles)
        self.status_var.set(f"Finding outliers in all {cycle_count} cycles…")
        self._submit(
            lambda: find_outliers_for_all_cycles(
                self.loaded.dataframe,
                self.state,
                threshold,
                self._stop_event,
            ),
            self._finish_all_outliers,
            "File-wide outlier search failed",
            operation_labels=[f"cycle {cycle}" for cycle in self.state.available_cycles],
            operation_name="Outlier search",
        )

    def _finish_all_outliers(self, results) -> None:
        if self.state is None:
            return
        peak_count = 0
        for cycle_number, (loaded_cycle, analysis) in results.items():
            if cycle_number in self.state.cycles:
                cycle = self.state.cycles[cycle_number]
            else:
                cycle = loaded_cycle
                cycle.parameters = self.state.parameters_for(cycle_number)
                self.state.cycles[cycle_number] = cycle
            if self.state.all_frequency_window is not None:
                cycle.frequency_window = self.state.all_frequency_window
            cycle.apply_outliers(analysis.outlier_indices)
            cycle.parameters = analysis.parameters
            peak_count += analysis.peak_count
        self._restore_controls()
        self._refresh_plot(rescale=True)
        if self._stop_event.is_set():
            self._update_status(
                f"outlier search stopped: processed {len(results)}, "
                f"skipped {max(len(self._operation_labels) - len(results), 0)} cycles"
            )
        else:
            self._update_status(
                f"outliers and ridge initial values calculated for {len(results)} cycles "
                f"({peak_count} peaks)"
            )

    def find_outliers_for_selected(self) -> None:
        if self.state is None or self.loaded is None or not self._capture_controls():
            return
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid threshold", "Enter a numeric threshold", parent=self.root
            )
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        grouped_projects: dict[str, tuple[LoadedProject, ProjectState]] = {}
        grouped_cycles: dict[str, list[int]] = {}
        for dataset_id, loaded, spectrum in selected_rows:
            grouped_cycles.setdefault(dataset_id, []).append(spectrum.cycle)
            if dataset_id not in grouped_projects:
                grouped_projects[dataset_id] = (loaded, loaded.state)
        selected_projects: dict[str, tuple[LoadedProject, ProjectState]] = {}
        for dataset_id, cycle_numbers in grouped_cycles.items():
            loaded, _state = grouped_projects[dataset_id]
            unique_cycles = list(dict.fromkeys(cycle_numbers))
            selected_projects[dataset_id] = (
                loaded,
                ProjectState(
                    source_path=loaded.state.source_path,
                    circuit=loaded.state.circuit,
                    control=loaded.state.control,
                    available_cycles=unique_cycles,
                    active_cycle=unique_cycles[0],
                    default_parameters=loaded.state.default_parameters,
                    cycles={
                        cycle_number: self._loaded_cycle_for_popup(loaded, cycle_number)
                        for cycle_number in unique_cycles
                    },
                    all_frequency_window=loaded.state.all_frequency_window,
                ),
            )
        spectrum_count = len(selected_rows)
        self.status_var.set(
            f"Finding outliers in {spectrum_count} selected spectra..."
        )
        self._submit(
            lambda: {
                dataset_id: find_outliers_for_all_cycles(
                    loaded.dataframe,
                    project,
                    threshold,
                    self._stop_event,
                )
                for dataset_id, (loaded, project) in selected_projects.items()
            },
            self._finish_all_outliers,
            "Selected-spectra outlier search failed",
            operation_labels=[
                f"{loaded.dataset_label}, cycle {spectrum.cycle}"
                for _dataset_id, loaded, spectrum in selected_rows
            ],
            operation_name="Selected outlier search",
        )

    def _finish_all_outliers(self, results) -> None:
        if self.state is None:
            return
        peak_count = 0
        spectra_count = 0
        outlier_count = 0
        for dataset_id, dataset_results in results.items():
            loaded = self.loaded_projects.get(dataset_id)
            if loaded is None:
                continue
            for cycle_number, (loaded_cycle, analysis) in dataset_results.items():
                if cycle_number in loaded.state.cycles:
                    cycle = loaded.state.cycles[cycle_number]
                else:
                    cycle = loaded_cycle
                    cycle.parameters = loaded.state.parameters_for(cycle_number)
                    loaded.state.cycles[cycle_number] = cycle
                if loaded.state.all_frequency_window is not None:
                    cycle.frequency_window = loaded.state.all_frequency_window
                cycle.apply_outliers(analysis.outlier_indices)
                cycle.parameters = analysis.parameters
                peak_count += analysis.peak_count
                outlier_count += int(np.count_nonzero(analysis.outlier_indices))
                spectra_count += 1
        self._restore_controls()
        self._refresh_plot(rescale=True)
        if self._stop_event.is_set():
            self._update_status(
                f"outlier search stopped: processed {spectra_count}, "
                f"skipped {max(len(self._operation_labels) - spectra_count, 0)} spectra"
            )
        else:
            self._update_status(
                f"outlier search complete for {spectra_count} spectra "
                f"({outlier_count} points excluded)"
            )

    def initialize_from_ridge(self) -> None:
        if self.state is None or not self._capture_controls():
            return
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid threshold", "Enter a numeric threshold", parent=self.root
            )
            return
        cycle_number = self.state.active_cycle
        cycle = self.state.active
        parameters = self.state.parameters_for(cycle_number)
        self.status_var.set(f"Cycle {cycle_number} · estimating initial values...")
        self._submit(
            lambda: self._cached_ridge_analysis(cycle, threshold, parameters)
            or analyze_outliers(cycle, threshold, parameters),
            lambda analysis: self._finish_initial_values(cycle_number, analysis),
            "Initial-value estimation failed",
        )

    def initialize_and_fit(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid threshold", "Enter a numeric threshold", parent=self.root
            )
            return
        cycle_number = self.state.active_cycle
        cycle = self.state.active
        parameters = self.state.parameters_for(cycle_number)
        self.status_var.set(
            f"Cycle {cycle_number} · estimating initial values before fitting..."
        )
        self._submit(
            lambda: self._cached_ridge_analysis(cycle, threshold, parameters)
            or analyze_outliers(cycle, threshold, parameters),
            lambda analysis: self._finish_initial_values_and_fit(cycle_number, analysis),
            "Initial-value estimation failed",
        )

    def _finish_initial_values_and_fit(
        self,
        cycle_number: int,
        analysis: RidgeInitialization,
    ) -> None:
        self._finish_initial_values(cycle_number, analysis)
        self.root.after(0, self.fit)

    def calculate_ridge_drt(self) -> None:
        if self.state is None or not self._capture_controls():
            return
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid threshold", "Enter a numeric threshold", parent=self.root
            )
            return
        cycle_number = self.state.active_cycle
        cycle = self.state.active
        parameters = self.state.parameters_for(cycle_number)
        self.status_var.set(f"Cycle {cycle_number} · calculating ridge DRT...")
        self._submit(
            lambda: self._cached_ridge_analysis(cycle, threshold, parameters)
            or analyze_outliers(cycle, threshold, parameters),
            lambda analysis: self._finish_ridge_drt(cycle_number, analysis),
            "Ridge DRT calculation failed",
        )

    def calculate_hybrid_drt(self) -> None:
        if self.state is None or not self._capture_controls():
            return
        cycle_number = self.state.active_cycle
        cycle = self.state.active
        self.status_var.set(f"Cycle {cycle_number} · calculating hybrid DRT...")
        self._submit(
            lambda: calculate_hybrid_drt(cycle),
            lambda result: self._finish_hybrid_drt(cycle_number, result),
            "Hybrid DRT calculation failed",
        )

    def _finish_initial_values(
        self,
        cycle_number: int,
        analysis: RidgeInitialization,
    ) -> None:
        if self.state is None:
            return
        cycle = self.state.cycles[cycle_number]
        existing_parameters = {parameter.name: parameter for parameter in cycle.parameters}
        updated_parameters = []
        for proposed in analysis.parameters:
            existing = existing_parameters.get(proposed.name)
            if existing is None:
                updated_parameters.append(proposed)
                continue
            updated_parameters.append(
                ParameterValue(
                    proposed.name,
                    proposed.unit,
                    self._clamp_parameter_value(
                        proposed.initial,
                        existing.lower,
                        existing.upper,
                    ),
                    existing.lower,
                    existing.upper,
                    existing.error_percent,
                    existing.fixed,
                )
            )
        cycle.parameters = updated_parameters
        cycle.store_ridge_analysis(
            self._require_threshold_value(),
            analysis.outlier_indices,
            updated_parameters,
            analysis.peak_count,
            analysis.ohmic_resistance,
            analysis.inductance,
            analysis.ridge_tau_s,
            analysis.ridge_gamma_ohm,
        )
        if self.state.active_cycle == cycle_number:
            self.parameter_table.set_parameters(updated_parameters)
            self._refresh_plot(rescale=True)
            self._update_status(
                f"initial values set from ridge: "
                f"R∞={analysis.ohmic_resistance:.3g} Ω, "
                f"L={analysis.inductance:.3g} H, "
                f"{analysis.peak_count} peaks"
            )

    def _finish_ridge_drt(
        self,
        cycle_number: int,
        analysis: RidgeInitialization,
    ) -> None:
        if self.state is None:
            return
        cycle = self.state.cycles[cycle_number]
        cycle.store_ridge_analysis(
            self._require_threshold_value(),
            analysis.outlier_indices,
            analysis.parameters,
            analysis.peak_count,
            analysis.ohmic_resistance,
            analysis.inductance,
            analysis.ridge_tau_s,
            analysis.ridge_gamma_ohm,
        )
        auto_fit_peaks = getattr(
            self, "_fit_drt_peaks_after_point_edit", False
        )
        self._fit_drt_peaks_after_point_edit = False
        if self.state.active_cycle == cycle_number:
            self._refresh_plot(rescale=True)
            self._update_status(
                f"ridge DRT calculated: "
                f"R∞={analysis.ohmic_resistance:.3g} Ω, "
                f"{analysis.peak_count} peaks"
            )

        if auto_fit_peaks and self.state.active_cycle == cycle_number and self.drt_peak_parameters:
            self.root.after(0, self.fit_drt_peaks)

        if getattr(self, "_drt_peak_batch_queue", None) is not None:
            self.root.after(0, self._continue_drt_peak_batch_after_calculation)

    def _finish_hybrid_drt(
        self,
        cycle_number: int,
        result: DRTComputation,
    ) -> None:
        if self.state is None:
            return
        cycle = self.state.cycles[cycle_number]
        cycle.store_hybrid_drt(
            result.tau_s,
            result.gamma_ohm,
            result.ohmic_resistance,
            getattr(result, "inductance", None),
        )
        auto_fit_peaks = getattr(
            self, "_fit_drt_peaks_after_point_edit", False
        )
        self._fit_drt_peaks_after_point_edit = False
        if self.state.active_cycle == cycle_number:
            self._refresh_plot(rescale=True)
            if auto_fit_peaks and self.drt_peak_parameters:
                self.root.after(0, self.fit_drt_peaks)
            if result.ohmic_resistance is None:
                self._update_status("hybrid DRT calculated")
            else:
                self._update_status(
                    f"hybrid DRT calculated: R∞={result.ohmic_resistance:.3g} Ω"
                )

    def _finish_outliers(
        self,
        cycle_number: int,
        analysis: RidgeInitialization,
    ) -> None:
        if self.state is None:
            return
        cycle = self.state.cycles[cycle_number]
        cycle.apply_outliers(analysis.outlier_indices)
        cycle.store_ridge_analysis(
            self._require_threshold_value(),
            analysis.outlier_indices,
            analysis.parameters,
            analysis.peak_count,
            analysis.ohmic_resistance,
            analysis.inductance,
            analysis.ridge_tau_s,
            analysis.ridge_gamma_ohm,
        )
        if self.state.active_cycle == cycle_number:
            self._refresh_plot(rescale=True)
            self._update_status(
                f"outlier search complete; {int(np.count_nonzero(cycle.outliers))} "
                f"points excluded"
            )

    def find_outliers_for_selected(self) -> None:
        if self.state is None or self.loaded is None or not self._capture_controls():
            return
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid threshold", "Enter a numeric threshold", parent=self.root
            )
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        grouped_projects: dict[str, tuple[LoadedProject, ProjectState]] = {}
        grouped_cycles: dict[str, list[int]] = {}
        for dataset_id, loaded, spectrum in selected_rows:
            grouped_cycles.setdefault(dataset_id, []).append(spectrum.cycle)
            if dataset_id not in grouped_projects:
                grouped_projects[dataset_id] = (loaded, loaded.state)
        selected_projects: dict[str, tuple[LoadedProject, ProjectState]] = {}
        for dataset_id, cycle_numbers in grouped_cycles.items():
            loaded, _state = grouped_projects[dataset_id]
            unique_cycles = list(dict.fromkeys(cycle_numbers))
            selected_projects[dataset_id] = (
                loaded,
                ProjectState(
                    source_path=loaded.state.source_path,
                    circuit=loaded.state.circuit,
                    control=loaded.state.control,
                    available_cycles=unique_cycles,
                    active_cycle=unique_cycles[0],
                    default_parameters=loaded.state.default_parameters,
                    cycles={
                        cycle_number: self._loaded_cycle_for_popup(loaded, cycle_number)
                        for cycle_number in unique_cycles
                    },
                    all_frequency_window=loaded.state.all_frequency_window,
                ),
            )
        spectrum_count = len(selected_rows)
        self.status_var.set(
            f"Finding outliers in {spectrum_count} selected spectra..."
        )
        def calculate():
            results = {}
            for dataset_id, (_loaded, project) in selected_projects.items():
                dataset_results = {}
                results[dataset_id] = dataset_results
                for cycle_number in project.available_cycles:
                    if self._stop_event.is_set():
                        return results
                    cycle = project.cycles[cycle_number]
                    dataset_results[cycle_number] = (
                        cycle,
                        self._cached_ridge_analysis(
                            cycle, threshold, project.parameters_for(cycle_number)
                        )
                        or analyze_outliers(
                            cycle, threshold, project.parameters_for(cycle_number)
                        ),
                    )
            return results

        self._submit(
            calculate,
            self._finish_selected_outliers,
            "Selected-spectra outlier search failed",
            operation_labels=[
                f"{loaded.dataset_label}, cycle {spectrum.cycle}"
                for _dataset_id, loaded, spectrum in selected_rows
            ],
            operation_name="Selected outlier search",
        )

    def _finish_selected_outliers(self, results) -> None:
        if self.state is None:
            return
        spectra_count = 0
        outlier_count = 0
        for dataset_id, dataset_results in results.items():
            loaded = self.loaded_projects.get(dataset_id)
            if loaded is None:
                continue
            for cycle_number, (loaded_cycle, analysis) in dataset_results.items():
                if cycle_number in loaded.state.cycles:
                    cycle = loaded.state.cycles[cycle_number]
                else:
                    cycle = loaded_cycle
                    cycle.parameters = loaded.state.parameters_for(cycle_number)
                    loaded.state.cycles[cycle_number] = cycle
                if loaded.state.all_frequency_window is not None:
                    cycle.frequency_window = loaded.state.all_frequency_window
                cycle.apply_outliers(analysis.outlier_indices)
                cycle.store_ridge_analysis(
                    self._require_threshold_value(),
                    analysis.outlier_indices,
                    analysis.parameters,
                    analysis.peak_count,
                    analysis.ohmic_resistance,
                    analysis.inductance,
                    analysis.ridge_tau_s,
                    analysis.ridge_gamma_ohm,
                )
                outlier_count += int(np.count_nonzero(analysis.outlier_indices))
                spectra_count += 1
        self._restore_controls()
        self._refresh_plot(rescale=True)
        if self._stop_event.is_set():
            self._update_status(
                f"outlier search stopped: processed {spectra_count}, "
                f"skipped {max(len(self._operation_labels) - spectra_count, 0)} spectra"
            )
        else:
            self._update_status(
                f"outlier search complete for {spectra_count} spectra "
                f"({outlier_count} points excluded)"
            )

    def calculate_selected_ridge_drts(self) -> None:
        if self.state is None or not self._capture_controls():
            return
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid threshold", "Enter a numeric threshold", parent=self.root
            )
            return
        batches = self._selected_project_batches()
        if not batches:
            self._update_status("select one or more spectra in the explorer first")
            return
        spectrum_count = sum(len(project.available_cycles) for _loaded, project in batches.values())
        self.status_var.set(f"Calculating ridge DRT for {spectrum_count} selected spectra...")

        def calculate() -> dict[str, dict[int, RidgeInitialization]]:
            results: dict[str, dict[int, RidgeInitialization]] = {}
            for dataset_id, (_loaded, project) in batches.items():
                dataset_results: dict[int, RidgeInitialization] = {}
                results[dataset_id] = dataset_results
                for cycle_number in project.available_cycles:
                    if self._stop_event.is_set():
                        return results
                    dataset_results[cycle_number] = analyze_outliers(
                        project.cycles[cycle_number],
                        threshold,
                        project.parameters_for(cycle_number),
                    )
            return results

        self._submit(
            calculate,
            self._finish_selected_ridge_drts,
            "Selected ridge DRT calculation failed",
            operation_labels=[
                f"{loaded.dataset_label}, cycle {cycle}"
                for loaded, project in batches.values()
                for cycle in project.available_cycles
            ],
            operation_name="Selected Ridge DRT",
        )

    def _finish_selected_ridge_drts(self, results) -> None:
        if self.state is None:
            return
        spectra_count = 0
        for dataset_id, dataset_results in results.items():
            loaded = self.loaded_projects.get(dataset_id)
            if loaded is None:
                continue
            for cycle_number, analysis in dataset_results.items():
                cycle = loaded.state.cycles.get(cycle_number)
                if cycle is None:
                    cycle = self._loaded_cycle_for_popup(loaded, cycle_number)
                    loaded.state.cycles[cycle_number] = cycle
                cycle.store_ridge_analysis(
                    self._require_threshold_value(),
                    analysis.outlier_indices,
                    analysis.parameters,
                    analysis.peak_count,
                    analysis.ohmic_resistance,
                    analysis.inductance,
                    analysis.ridge_tau_s,
                    analysis.ridge_gamma_ohm,
                )
                spectra_count += 1
        self._refresh_plot(rescale=True)
        if self._stop_event.is_set():
            self._update_status(
                f"Ridge DRT stopped: processed {spectra_count}, "
                f"skipped {max(len(self._operation_labels) - spectra_count, 0)} spectra"
            )
        else:
            self._update_status(f"ridge DRT recalculated for {spectra_count} selected spectra")

    def calculate_selected_hybrid_drts(self) -> None:
        if self.state is None or not self._capture_controls():
            return
        batches = self._selected_project_batches()
        if not batches:
            self._update_status("select one or more spectra in the explorer first")
            return
        spectrum_count = sum(len(project.available_cycles) for _loaded, project in batches.values())
        self.status_var.set(f"Calculating hybrid DRT for {spectrum_count} selected spectra...")

        def calculate() -> dict[str, dict[int, DRTComputation]]:
            results: dict[str, dict[int, DRTComputation]] = {}
            for dataset_id, (_loaded, project) in batches.items():
                dataset_results: dict[int, DRTComputation] = {}
                results[dataset_id] = dataset_results
                for cycle_number in project.available_cycles:
                    if self._stop_event.is_set():
                        return results
                    dataset_results[cycle_number] = calculate_hybrid_drt(
                        project.cycles[cycle_number]
                    )
            return results

        self._submit(
            calculate,
            self._finish_selected_hybrid_drts,
            "Selected hybrid DRT calculation failed",
            operation_labels=[
                f"{loaded.dataset_label}, cycle {cycle}"
                for loaded, project in batches.values()
                for cycle in project.available_cycles
            ],
            operation_name="Selected Hybrid DRT",
        )

    def _finish_selected_hybrid_drts(self, results) -> None:
        if self.state is None:
            return
        spectra_count = 0
        for dataset_id, dataset_results in results.items():
            loaded = self.loaded_projects.get(dataset_id)
            if loaded is None:
                continue
            for cycle_number, result in dataset_results.items():
                cycle = loaded.state.cycles.get(cycle_number)
                if cycle is None:
                    cycle = self._loaded_cycle_for_popup(loaded, cycle_number)
                    loaded.state.cycles[cycle_number] = cycle
                cycle.store_hybrid_drt(
                    result.tau_s,
                    result.gamma_ohm,
                    result.ohmic_resistance,
                )
                spectra_count += 1
        self._refresh_plot(rescale=True)
        if self._stop_event.is_set():
            self._update_status(
                f"Hybrid DRT stopped: processed {spectra_count}, "
                f"skipped {max(len(self._operation_labels) - spectra_count, 0)} spectra"
            )
        else:
            self._update_status(f"hybrid DRT recalculated for {spectra_count} selected spectra")

    def _fit_options_from_controls(self) -> FitOptions:
        names = {
            "local only": ("least_squares",),
            "PSO → local": ("pso", "least_squares"),
            "GA → local": ("ga", "least_squares"),
            "PSO only": ("pso",),
            "GA only": ("ga",),
        }
        seed_text = self.fit_seed_var.get().strip()
        return FitOptions(
            pipeline=names.get(self.fit_pipeline_var.get(), ("least_squares",)),
            seed=int(seed_text) if seed_text else None,
            population_size=int(self.fit_population_var.get()),
            iterations=int(self.fit_iterations_var.get()),
            weight_by_modulus=bool(self.fit_weight_modulus_var.get()),
            jacobian_mode={
                "Numerical only": "numerical",
                "Automatic": "automatic",
                "Analytical when supported": "analytical",
            }.get(self.fit_jacobian_mode_var.get(), "numerical"),
        ).validated()

    def _show_fit_diagnostics(self) -> None:
        result = self._last_fit_result
        if result is None:
            self._update_status("no fit diagnostics are available yet")
            return
        stages = "\n".join(
            f"{stage['method']}: objective={stage['objective']:.6g}, "
            f"converged={stage['converged']}"
            for stage in result.stages
        ) or "least_squares: completed"
        messagebox.showinfo(
            "EEC fit diagnostics",
            f"Pipeline: {' → '.join(result.options.stages())}\n"
            f"RMSE: {result.rmse:.6g}\n"
            f"Objective: {result.objective:.6g}\n"
            f"Jacobian: {result.jacobian_mode}"
            + (f" (fallback: {result.jacobian_fallback_reason})" if result.jacobian_fallback_reason else "")
            + "\n"
            f"Elapsed: {result.elapsed_seconds:.3f} s\n\n{stages}",
            parent=self.root,
        )

    def fit(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        cycle_number = self.state.active_cycle
        cycle = self.state.active
        parameters = self.state.parameters_for(cycle_number)
        try:
            fit_options = self._fit_options_from_controls()
        except (TypeError, ValueError) as error:
            messagebox.showerror("Invalid optimizer settings", str(error), parent=self.root)
            return
        self._fit_parameter_snapshot = (
            parameters,
            [parameter.initial for parameter in parameters],
        )
        self.status_var.set(f"Cycle {cycle_number} · fitting…")
        self._submit(
            lambda: fit_cycle_with_timeout(
                cycle,
                cycle.model(self.state.circuit),
                parameters,
                self._fit_timeout_seconds,
                fit_options,
            ),
            lambda result: self._finish_fit(cycle_number, parameters, result),
            "Fit failed",
        )

    def _restore_fit_initial_parameters(self) -> None:
        snapshot = self._fit_parameter_snapshot
        if snapshot is None:
            return
        parameters, initial_values = snapshot
        for parameter, initial in zip(parameters, initial_values):
            parameter.initial = initial
        if self.state is not None and self.state.active.parameters is parameters:
            self.parameter_table.set_parameters(parameters)

    def _finish_fit(self, cycle_number, parameters, result) -> None:
        if self.state is None:
            return
        self._last_fit_result = result
        (
            fitted_parameters,
            errors_percent,
            fit_frequency,
            fit_impedance,
            fit_at_data,
        ) = result
        cycle = self.state.cycles[cycle_number]
        cycle.fit_parameters = fitted_parameters
        cycle.fit_frequency_hz = fit_frequency
        cycle.fit_impedance = fit_impedance
        cycle.fit_at_data_impedance = fit_at_data
        if hasattr(result, "options"):
            cycle.fit_provenance = {
                "pipeline": list(result.options.stages()),
                "seed": result.options.seed,
                "objective": result.objective,
                "rmse": result.rmse,
                "converged": result.converged,
                "elapsed_seconds": result.elapsed_seconds,
                "stages": result.stages,
            }
        for parameter, fitted, error_percent in zip(
            parameters, fitted_parameters, errors_percent
        ):
            parameter.initial = float(fitted)
            parameter.error_percent = float(error_percent)
        cycle.parameters = parameters
        if self.state.active_cycle == cycle_number:
            self.parameter_table.set_parameters(parameters)
            self._refresh_plot(rescale=True)
            self._update_status("fit complete")
        self._refresh_open_parameter_explorers()

    def fit_selected(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        targets = [
            SpectrumFitTarget(
                loaded=loaded,
                cycle=spectrum.cycle,
                label=f"{loaded.dataset_label}, cycle {spectrum.cycle}",
            )
            for _dataset_id, loaded, spectrum in selected_rows
        ]
        parameters = self.state.parameters_for(self.state.active_cycle)
        try:
            fit_options = self._fit_options_from_controls()
        except (TypeError, ValueError) as error:
            messagebox.showerror("Invalid optimizer settings", str(error), parent=self.root)
            return
        self.status_var.set(f"Fitting {len(targets)} selected spectra…")
        self._submit(
            lambda: batch_fit_spectra(
                targets,
                parameters,
                use_target_initial_parameters=True,
                stop_event=self._stop_event,
                fit_timeout_seconds=self._fit_timeout_seconds,
                fit_options=fit_options,
            ),
            self._finish_explorer_batch_fit,
            "Selected fit failed",
            operation_labels=[target.label for target in targets],
            operation_name="Selected fit",
        )

    def refine_fit_selected(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        if self.analysis_mode_var.get() != "EEC":
            return
        try:
            z_threshold = float(self.refine_z_threshold_var.get())
            max_iterations = int(self.refine_max_iterations_var.get())
            if not np.isfinite(z_threshold) or z_threshold <= 0:
                raise ValueError("the robust z threshold must be positive")
            if max_iterations < 1:
                raise ValueError("the maximum iteration count must be at least 1")
        except ValueError as error:
            messagebox.showerror("Invalid refinement settings", str(error), parent=self.root)
            return

        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            messagebox.showerror(
                "No spectra selected",
                "Select one or more spectra in the Spectra Explorer first.",
                parent=self.root,
            )
            return
        targets = []
        seen: set[tuple[str, int]] = set()
        missing_fit = []
        for dataset_id, loaded, spectrum in selected_rows:
            key = (dataset_id, spectrum.cycle)
            if key in seen:
                continue
            seen.add(key)
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            if cycle.fit_parameters is None or cycle.fit_at_data_impedance is None:
                missing_fit.append(f"{loaded.dataset_label}, cycle {spectrum.cycle}")
                continue
            targets.append(
                (
                    dataset_id,
                    loaded,
                    spectrum.cycle,
                    copy.deepcopy(cycle),
                    copy.deepcopy(cycle.parameters),
                    cycle.model(loaded.state.circuit),
                )
            )
        if missing_fit:
            messagebox.showerror(
                "Refine fit unavailable",
                "Every selected spectrum must have an existing fit. Missing fit:\n"
                + "\n".join(missing_fit),
                parent=self.root,
            )
            return

        self.status_var.set(f"Refining fit for {len(targets)} selected spectra…")
        def refine():
            results = []
            for dataset_id, _loaded, cycle_number, cycle, parameters, circuit in targets:
                if self._stop_event.is_set():
                    break
                results.append(
                    (
                        dataset_id,
                        cycle_number,
                        refine_fit_cycle(
                            cycle,
                            circuit,
                            parameters,
                            z_threshold,
                            max_iterations,
                            self._fit_timeout_seconds,
                        ),
                    )
                )
            return results

        self._submit(
            refine,
            self._finish_refine_fit,
            "Refine fit failed",
            operation_labels=[
                f"{loaded.dataset_label}, cycle {cycle_number}"
                for _dataset_id, loaded, cycle_number, *_rest in targets
            ],
            operation_name="Refine fit",
        )

    def _finish_refine_fit(self, results) -> None:
        if self.state is None:
            return
        removed_count = 0
        iteration_count = 0
        for dataset_id, cycle_number, refinement in results:
            loaded = self.loaded_projects.get(dataset_id)
            if loaded is None:
                continue
            cycle = self._loaded_cycle_for_popup(loaded, cycle_number)
            fit_result, removed_indices, iterations = refinement
            (
                fitted_parameters,
                errors_percent,
                fit_frequency,
                fit_impedance,
                fit_at_data,
            ) = fit_result
            valid_indices = removed_indices[
                (removed_indices >= 0) & (removed_indices < cycle.frequency_hz.size)
            ]
            cycle.outliers[valid_indices] = True
            cycle.manually_included[valid_indices] = False
            cycle.invalidate_drt_cache()
            cycle.fit_parameters = fitted_parameters
            cycle.fit_frequency_hz = fit_frequency
            cycle.fit_impedance = fit_impedance
            cycle.fit_at_data_impedance = fit_at_data
            parameters = cycle.parameters
            for parameter, fitted, error_percent in zip(
                parameters, fitted_parameters, errors_percent
            ):
                parameter.initial = float(fitted)
                parameter.error_percent = float(error_percent)
            cycle.parameters = parameters
            removed_count += len(valid_indices)
            iteration_count += iterations

        self._restore_controls()
        self._refresh_explorer_values()
        self.parameter_table.set_parameters(
            self.state.parameters_for(self.state.active_cycle)
        )
        self._refresh_plot(rescale=True)
        if self._continue_named_ml_pipeline():
            return
        if self._stop_event.is_set():
            self._update_status(
                f"Refine fit stopped: processed {len(results)}, "
                f"skipped {max(len(self._operation_labels) - len(results), 0)} spectra"
            )
        else:
            self._update_status(
                f"refine fit completed: {removed_count} points deactivated in "
                f"{iteration_count} iterations"
            )

    def batch_fit(self) -> None:
        if self.state is None or self.loaded is None or not self._capture_controls():
            return
        start_cycle = self.state.active_cycle
        parameters = self.state.parameters_for(start_cycle)
        cycle_count = len(
            self.state.available_cycles[
                self.state.available_cycles.index(start_cycle) :
            ]
        )
        self.status_var.set(
            f"Batch fitting {cycle_count} cycles from cycle {start_cycle}…"
        )
        self._submit(
            lambda: batch_fit_from_cycle(
                self.loaded.dataframe,
                self.state,
                start_cycle,
                parameters,
                self._stop_event,
                self.state.circuit,
                fit_timeout_seconds=self._fit_timeout_seconds,
                fit_options=self._fit_options_from_controls(),
            ),
            self._finish_batch_fit,
            "Batch fit failed",
            operation_labels=[
                f"cycle {cycle}"
                for cycle in self.state.available_cycles[
                    self.state.available_cycles.index(start_cycle) :
                ]
            ],
            operation_name="Batch fit",
        )

    def _finish_batch_fit(self, report: BatchFitReport) -> None:
        if self.state is None:
            return
        for result in report.fits:
            cycle = result.cycle
            cycle.parameters = result.parameters
            cycle.fit_parameters = result.fitted_parameters
            cycle.fit_frequency_hz = result.fit_frequency_hz
            cycle.fit_impedance = result.fit_impedance
            cycle.fit_at_data_impedance = result.fit_at_data_impedance
            cycle.fit_provenance = dict(result.fit_provenance)
            self.state.cycles[cycle.cycle] = cycle
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._refresh_open_parameter_explorers()
        if self._continue_named_ml_pipeline():
            return
        if report.stopped:
            self._update_status(
                f"batch fit stopped: processed {len(report.fits)}, "
                f"skipped {len(report.skipped_cycles)} cycles"
            )
            return
        if report.failed_cycle is None:
            self._update_status(f"batch fit completed for {len(report.fits)} cycles")
            return
        self._update_status(
            f"batch fit stopped at cycle {report.failed_cycle}; "
            f"{len(report.fits)} cycles completed, "
            f"{len(report.skipped_cycles)} cycles skipped"
        )
        messagebox.showwarning(
            "Batch fit stopped",
            f"Cycle {report.failed_cycle}: {report.error}\n\n"
            f"Completed: {len(report.fits)}\n"
            f"Failed: 1\n"
            f"Skipped: {len(report.skipped_cycles)}\n\n"
            f"The successful fits were retained.",
            parent=self.root,
        )

    def batch_fit_explorer(
        self,
        direction: int,
        *,
        to_metadata_value: bool = False,
    ) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        selected = self.explorer.selection()
        if not selected:
            self._update_status("select a spectrum in the explorer first")
            return
        visible_items = list(self.explorer.get_children(""))
        selected_index = visible_items.index(selected[0])
        if direction > 0:
            batch_items = visible_items[selected_index:]
            direction_name = "down"
        else:
            batch_items = list(reversed(visible_items[: selected_index + 1]))
            direction_name = "up"

        target_description = ""
        if to_metadata_value:
            column = self._explorer_selected_column
            numeric_values: list[float] = []
            try:
                for item in batch_items:
                    _dataset_id, loaded, spectrum = self._explorer_rows[item]
                    value = float(self._explorer_value(loaded, spectrum, column))
                    if not np.isfinite(value):
                        raise ValueError
                    numeric_values.append(value)
            except (TypeError, ValueError):
                messagebox.showerror(
                    "Select numeric metadata",
                    "Choose an explorer column containing numeric values for all spectra "
                    "in this batch.",
                    parent=self.root,
                )
                return
            label = self._explorer_headings.get(column, column)
            target_value = simpledialog.askfloat(
                "Batch fit limit",
                f"Stop near which {label} value?",
                parent=self.root,
            )
            if target_value is None:
                return
            nearest_index = min(
                range(len(batch_items)),
                key=lambda index: abs(numeric_values[index] - target_value),
            )
            batch_items = batch_items[: nearest_index + 1]
            target_description = f" toward {label}={target_value:g}"

        targets = []
        for item in batch_items:
            _dataset_id, loaded, spectrum = self._explorer_rows[item]
            targets.append(
                SpectrumFitTarget(
                    loaded=loaded,
                    cycle=spectrum.cycle,
                    label=f"{loaded.dataset_label}, cycle {spectrum.cycle}",
                )
            )
        parameters = self.state.parameters_for(self.state.active_cycle)
        self.status_var.set(
            f"Batch fitting {len(targets)} spectra {direction_name}"
            f"{target_description}…"
        )
        self._submit(
            lambda: batch_fit_spectra(
                targets,
                parameters,
                stop_event=self._stop_event,
                initial_circuit=self.state.active.model(self.state.circuit),
                fit_timeout_seconds=self._fit_timeout_seconds,
                fit_options=self._fit_options_from_controls(),
            ),
            self._finish_explorer_batch_fit,
            "Explorer batch fit failed",
            operation_labels=[target.label for target in targets],
            operation_name="Explorer batch fit",
        )

    def _start_drt_peak_batch(self, direction: int) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        visible_items = list(self.explorer.get_children(""))
        selected_items = set(self.explorer.selection())
        current_item = self._explorer_lookup.get(
            (self.current_dataset_id, self.state.active_cycle)
        )
        if current_item not in visible_items or current_item not in selected_items:
            self._update_status("include the displayed spectrum in the selection")
            return
        if len(selected_items) < 2:
            self._update_status("select at least two spectra in the explorer first")
            return
        self._drt_peak_batch_directions = [1, -1] if direction == 0 else [direction]
        self._drt_peak_batch_anchor = (
            self.current_dataset_id,
            self.state.active_cycle,
        )
        self._drt_peak_batch_template = [
            copy.deepcopy(peak) for peak in self.drt_peak_parameters
        ]
        self._drt_peak_batch_queue = None
        self._prepare_drt_peak_batch_direction()

    def _prepare_drt_peak_batch_direction(self) -> None:
        if not self._drt_peak_batch_directions:
            self._drt_peak_batch_template = None
            self._refresh_open_parameter_explorers()
            self._update_status("DRT peak batch fit completed")
            return
        direction = self._drt_peak_batch_directions.pop(0)
        visible_items = list(self.explorer.get_children(""))
        selected_items = set(self.explorer.selection())
        anchor_dataset, anchor_cycle = self._drt_peak_batch_anchor
        anchor_loaded = self.loaded_projects.get(anchor_dataset)
        if anchor_loaded is not None:
            self._switch_dataset(
                anchor_dataset,
                anchor_loaded,
                anchor_cycle,
                capture_current=False,
                preserve_existing_selection=True,
            )
            visible_items = list(self.explorer.get_children(""))
        current_item = self._explorer_lookup.get(
            (self.current_dataset_id, self.state.active_cycle)
        )
        if current_item not in visible_items:
            return
        start = visible_items.index(current_item)
        if direction > 0:
            ordered = visible_items[start:]
        else:
            ordered = list(reversed(visible_items[: start + 1]))
        self._drt_peak_batch_queue = [item for item in ordered if item in selected_items]
        if len(self._drt_peak_batch_queue) < 2:
            self._update_status("not enough selected spectra in that direction")
            self._drt_peak_batch_queue = None
            self._prepare_drt_peak_batch_direction()
            return
        self._drt_peak_batch_direction = direction
        self._drt_peak_batch_template = [
            copy.deepcopy(peak) for peak in self.drt_peak_parameters
        ]
        self._drt_peak_batch_next()

    def _drt_peak_batch_next(self) -> None:
        if self.busy:
            return
        if not self._drt_peak_batch_queue:
            self._prepare_drt_peak_batch_direction()
            return
        item = self._drt_peak_batch_queue.pop(0)
        dataset_id, loaded, spectrum = self._explorer_rows[item]
        self._switch_dataset(
            dataset_id,
            loaded,
            spectrum.cycle,
            capture_current=False,
            preserve_existing_selection=True,
        )
        self.drt_peak_parameters = [
            copy.deepcopy(peak) for peak in (self._drt_peak_batch_template or [])
        ]
        self._store_current_drt_peaks()
        self._refresh_drt_peak_artists()
        cycle = self.state.active
        mode = self.analysis_drt_mode_var.get()
        has_drt = (
            cycle.saved_hybrid_tau_s is not None
            if mode == "Hybrid DRT"
            else cycle.saved_ridge_tau_s is not None
        )
        if has_drt:
            self.fit_drt_peaks()
        elif mode == "Hybrid DRT":
            self.calculate_hybrid_drt()
        else:
            self.calculate_ridge_drt()

    def _continue_drt_peak_batch_after_calculation(self) -> None:
        if self.state is None:
            return
        self.drt_peak_parameters = [
            copy.deepcopy(peak) for peak in (self._drt_peak_batch_template or [])
        ]
        self._store_current_drt_peaks()
        self._refresh_drt_peak_artists()
        self.fit_drt_peaks()

    def batch_fit_selected_down(self, direction: int = 1) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        visible_items = list(self.explorer.get_children(""))
        selected_items = [
            item for item in visible_items if item in self.explorer.selection()
        ]
        if len(selected_items) < 2:
            self._update_status("select at least two spectra in the explorer first")
            return
        if self.current_dataset_id is None:
            self._update_status("no displayed spectrum is available")
            return
        current_item = self._explorer_lookup.get(
            (self.current_dataset_id, self.state.active_cycle)
        )
        if current_item is None or current_item not in selected_items:
            self._update_status("include the displayed spectrum in the selection")
            return

        start_index = visible_items.index(current_item)
        if direction > 0:
            batch_items = [
                item
                for item in visible_items[start_index:]
                if item in self.explorer.selection()
            ]
            direction_name = "downward"
        else:
            batch_items = [
                item
                for item in reversed(visible_items[: start_index + 1])
                if item in self.explorer.selection()
            ]
            direction_name = "upward"
        if len(batch_items) < 2:
            self._update_status(
                f"select spectra from the displayed one {direction_name} in the explorer"
            )
            return

        targets = []
        for item in batch_items:
            _dataset_id, loaded, spectrum = self._explorer_rows[item]
            targets.append(
                SpectrumFitTarget(
                    loaded=loaded,
                    cycle=spectrum.cycle,
                    label=f"{loaded.dataset_label}, cycle {spectrum.cycle}",
                )
            )
        parameters = self.state.parameters_for(self.state.active_cycle)
        self.status_var.set(
            f"Batch fitting {len(targets)} selected spectra {direction_name}..."
        )
        self._submit(
            lambda: batch_fit_spectra(
                targets,
                parameters,
                stop_event=self._stop_event,
                initial_circuit=self.state.active.model(self.state.circuit),
                fit_timeout_seconds=self._fit_timeout_seconds,
            ),
            self._finish_explorer_batch_fit,
            "Selected batch fit failed",
            operation_labels=[target.label for target in targets],
            operation_name="Selected batch fit",
        )

    def _finish_explorer_batch_fit(self, report: SpectrumBatchReport) -> None:
        if self.state is None:
            return
        for result in report.fits:
            cycle = result.fit.cycle
            cycle.parameters = result.fit.parameters
            cycle.fit_parameters = result.fit.fitted_parameters
            cycle.fit_frequency_hz = result.fit.fit_frequency_hz
            cycle.fit_impedance = result.fit.fit_impedance
            cycle.fit_at_data_impedance = result.fit.fit_at_data_impedance
            cycle.fit_provenance = dict(result.fit.fit_provenance)
            result.loaded.state.cycles[cycle.cycle] = cycle
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._refresh_open_parameter_explorers()
        if getattr(self, "_batch_fit_both_pending", False):
            if getattr(self, "_batch_fit_both_stage", "") == "up":
                self._batch_fit_both_up_completed = len(report.fits)
            else:
                self._batch_fit_both_down_completed = len(report.fits)
                self._batch_fit_both_pending = False
                if report.stopped or self._stop_event.is_set():
                    self._update_status(
                        "Batch fit up and down stopped by user. "
                        f"Up: {self._batch_fit_both_up_completed} completed. "
                        f"Down: {self._batch_fit_both_down_completed} completed."
                    )
                else:
                    self._update_status(
                        "Batch fit up and down completed. "
                        f"Up: {self._batch_fit_both_up_completed} completed. "
                        f"Down: {self._batch_fit_both_down_completed} completed."
                    )
                return
        if report.stopped:
            self._update_status(
                f"explorer batch fit stopped: processed {len(report.fits)}, "
                f"skipped {len(report.skipped_labels)} spectra"
            )
            return
        if report.failed_label is None:
            self._update_status(
                f"explorer batch fit completed for {len(report.fits)} spectra"
            )
            return
        self._update_status(
            f"explorer batch stopped at {report.failed_label}; "
            f"{len(report.fits)} completed, {len(report.skipped_labels)} skipped"
        )
        messagebox.showwarning(
            "Explorer batch fit stopped",
            f"{report.failed_label}: {report.error}\n\n"
            f"Completed: {len(report.fits)}\n"
            f"Failed: 1\n"
            f"Skipped: {len(report.skipped_labels)}\n\n"
            f"The successful fits were retained.",
            parent=self.root,
        )

    def batch_fit_selected_up_down(self) -> None:
        if self.busy or self.state is None:
            return
        selected = self._selected_spectrum_rows()
        if not selected:
            self._update_status("select one or more spectra in the explorer first")
            return
        self._batch_fit_both_pending = True
        self._batch_fit_both_stage = "up"
        self._batch_fit_both_up_completed = 0
        self._batch_fit_both_down_completed = 0
        self.batch_fit_selected_down(1)
        if not self.busy:
            self._batch_fit_both_pending = False
            return
        self.root.after(100, self._continue_batch_fit_selected_up_down)

    def _continue_batch_fit_selected_up_down(self) -> None:
        if not getattr(self, "_batch_fit_both_pending", False):
            return
        if self.busy:
            self.root.after(100, self._continue_batch_fit_selected_up_down)
            return
        if self._stop_event.is_set():
            self._batch_fit_both_pending = False
            self._update_status(
                "Batch fit up and down stopped by user. "
                f"Up: {self._batch_fit_both_up_completed} completed. "
                "Down: not started."
            )
            return
        self._batch_fit_both_stage = "down"
        self.batch_fit_selected_down(-1)
        if not self.busy:
            self._batch_fit_both_pending = False

    def copy_neighbor_drt_peaks(self, direction: int) -> None:
        if self.state is None or self.busy:
            return
        visible_items = list(self.explorer.get_children(""))
        current_item = self._explorer_lookup.get(
            (self.current_dataset_id, self.state.active_cycle)
        )
        if current_item not in visible_items:
            self._update_status("the displayed spectrum is not in the explorer")
            return
        source_index = visible_items.index(current_item) + direction
        if not 0 <= source_index < len(visible_items):
            self._update_status("no neighboring spectrum in that direction")
            return
        source_item = visible_items[source_index]
        _source_dataset_id, source_loaded, source_spectrum = self._explorer_rows[
            source_item
        ]
        source = self._loaded_cycle_for_popup(source_loaded, source_spectrum.cycle)
        peak_attribute = (
            "saved_hybrid_peak_parameters"
            if self.analysis_drt_mode_var.get() == "Hybrid DRT"
            else "saved_ridge_peak_parameters"
        )
        source_peaks = getattr(source, peak_attribute, [])
        self.drt_peak_parameters = [copy.deepcopy(peak) for peak in source_peaks]
        self._selected_drt_peak_index = None
        self._store_current_drt_peaks()
        self._refresh_plot(rescale=True)
        self._update_status(
            f"DRT peaks copied from spectrum {source_spectrum.cycle}"
        )

    def copy_neighbor_fit(self, direction: int) -> None:
        if self.state is None or self.busy or not self._capture_controls():
            return
        visible_items = list(self.explorer.get_children(""))
        current_item = self._explorer_lookup.get(
            (self.current_dataset_id, self.state.active_cycle)
        )
        if current_item not in visible_items:
            self._update_status("the displayed spectrum is not in the explorer")
            return
        source_index = visible_items.index(current_item) + direction
        if not 0 <= source_index < len(visible_items):
            self._update_status("no neighboring spectrum in that direction")
            return
        source_item = visible_items[source_index]
        _source_dataset_id, source_loaded, source_spectrum = self._explorer_rows[source_item]
        source = self._loaded_cycle_for_popup(source_loaded, source_spectrum.cycle)
        if source is None or source.fit_parameters is None:
            self._update_status(
                f"spectrum {source_spectrum.cycle} has no fit to copy"
            )
            return
        current_model = self.state.active.model(self.state.circuit)
        source_model = source.model(source_loaded.state.circuit)
        mapping = parameter_name_mapping(source_model, current_model)
        if mapping is None:
            self._update_status("neighboring spectrum uses a different fitting model")
            return
        current_parameters = self.state.parameters_for(self.state.active_cycle)
        source_parameters = source_loaded.state.parameters_for(source_spectrum.cycle)
        fitted = np.asarray(source.fit_parameters).reshape(-1)
        source_by_target = {
            map_parameter_name(parameter.name, mapping): (parameter, value)
            for parameter, value in zip(source_parameters, fitted)
        }
        if fitted.size != len(source_parameters) or any(
            parameter.name not in source_by_target for parameter in current_parameters
        ):
            self._update_status("neighboring fit uses incompatible parameters")
            return
        for parameter in current_parameters:
            _source_parameter, value = source_by_target[parameter.name]
            parameter.initial = float(value)
        self.state.active.parameters = current_parameters
        self.state.active.clear_fit()
        self.parameter_table.set_parameters(current_parameters)
        self._refresh_plot()
        self._update_status(
            f"initial parameters copied from spectrum {source_spectrum.cycle}"
        )

    def copy_neighbor_fit_settings(self, direction: int) -> None:
        if self.state is None or self.busy or not self._capture_controls():
            return
        visible_items = list(self.explorer.get_children(""))
        current_item = self._explorer_lookup.get(
            (self.current_dataset_id, self.state.active_cycle)
        )
        if current_item not in visible_items:
            self._update_status("the displayed spectrum is not in the explorer")
            return
        source_index = visible_items.index(current_item) + direction
        if not 0 <= source_index < len(visible_items):
            self._update_status("no neighboring spectrum in that direction")
            return
        source_item = visible_items[source_index]
        _source_dataset_id, source_loaded, source_spectrum = self._explorer_rows[source_item]
        source = self._loaded_cycle_for_popup(source_loaded, source_spectrum.cycle)
        if source.fit_parameters is None:
            self._update_status(
                f"spectrum {source_spectrum.cycle} has no fit to copy"
            )
            return
        current_model = self.state.active.model(self.state.circuit)
        source_model = source.model(source_loaded.state.circuit)
        mapping = parameter_name_mapping(source_model, current_model)
        if mapping is None:
            self._update_status("neighboring spectrum uses a different fitting model")
            return
        current_parameters = self.state.parameters_for(self.state.active_cycle)
        source_parameters = source_loaded.state.parameters_for(source_spectrum.cycle)
        fitted = np.asarray(source.fit_parameters).reshape(-1)
        source_by_target = {
            map_parameter_name(parameter.name, mapping): (parameter, value)
            for parameter, value in zip(source_parameters, fitted)
        }
        if fitted.size != len(source_parameters) or any(
            parameter.name not in source_by_target for parameter in current_parameters
        ):
            self._update_status("neighboring fit uses incompatible parameters")
            return
        copied_parameters = []
        for target in current_parameters:
            source_parameter, value = source_by_target[target.name]
            copied_parameters.append(
                ParameterValue(
                    target.name,
                    target.unit,
                    float(value),
                    target.lower,
                    target.upper,
                    target.error_percent,
                    source_parameter.fixed,
                )
            )
        self.state.active.parameters = copied_parameters
        if source.frequency_window is not None:
            self.state.active.frequency_window = tuple(source.frequency_window)
        self.state.active.invalidate_drt_cache()
        self.state.active.clear_fit()
        self.parameter_table.set_parameters(copied_parameters)
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._update_status(
            f"fit settings copied from spectrum {source_spectrum.cycle}"
        )

    def _collect_fit_parameter_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for dataset_id in self._dataset_order:
            loaded = self.loaded_projects[dataset_id]
            for spectrum in loaded.spectra:
                cycle = loaded.state.cycles.get(spectrum.cycle)
                if cycle is None or cycle.fit_parameters is None:
                    continue
                values = np.asarray(cycle.fit_parameters).reshape(-1)
                if values.size != len(cycle.parameters):
                    continue
                record: dict[str, object] = {
                    "source_file": loaded.state.source_path.name,
                    "cycle": spectrum.cycle,
                    "potential_V": cycle.potential_v,
                    "current_mA": cycle.current_ma,
                    "time_s": cycle.time_s,
                    "Spectrum": (
                        spectrum.custom_metadata.get(SPECTRUM_METADATA_COLUMN)
                        or {"working": "WE", "cell": "Cell", "counter": "CE"}.get(
                            loaded.state.control,
                            loaded.state.control,
                        )
                    ),
                    "circuit": cycle.model(loaded.state.circuit),
                }
                record.update(
                    {
                        parameter.name: float(value)
                        for parameter, value in zip(cycle.parameters, values)
                    }
                )
                record.update(
                    _derived_block_values(
                        cycle.model(loaded.state.circuit),
                        [parameter.name for parameter in cycle.parameters],
                        values,
                    )
                )
                metadata = dict(cycle.custom_metadata)
                metadata.update(spectrum.custom_metadata)
                for name, value in metadata.items():
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        if value is not None:
                            record[name] = value
                        continue
                    record[name] = numeric_value if np.isfinite(numeric_value) else value
                record.setdefault("Cycle mod 15", int(spectrum.cycle) % 15)
                records.append(_externalize_record(record))
        return records

    def _collect_drt_parameter_records(self, mode: str) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for dataset_id in self._dataset_order:
            loaded = self.loaded_projects[dataset_id]
            for spectrum in loaded.spectra:
                cycle = loaded.state.cycles.get(spectrum.cycle)
                if cycle is None:
                    continue
                if mode == "hybrid":
                    saved_peaks = cycle.saved_hybrid_peak_parameters
                    resistance = cycle.saved_hybrid_ohmic_resistance
                else:
                    saved_peaks = cycle.saved_ridge_peak_parameters
                    resistance = cycle.saved_ridge_ohmic_resistance
                if not saved_peaks and resistance is None:
                    continue
                record: dict[str, object] = {
                    "source_file": loaded.state.source_path.name,
                    "cycle": spectrum.cycle,
                    "potential_V": cycle.potential_v,
                    "current_mA": cycle.current_ma,
                    "time_s": cycle.time_s,
                    "Spectrum": (
                        spectrum.custom_metadata.get(SPECTRUM_METADATA_COLUMN)
                        or {"working": "WE", "cell": "Cell", "counter": "CE"}.get(
                            loaded.state.control,
                            loaded.state.control,
                        )
                    ),
                    "drt_mode": mode,
                }
                if resistance is not None:
                    record["R0"] = float(resistance)
                if cycle.saved_ridge_inductance is not None:
                    record["L0"] = float(cycle.saved_ridge_inductance)
                for index, peak in enumerate(saved_peaks, 1):
                    try:
                        tau, area, fwhm = self._peak_summary(peak)
                    except (KeyError, TypeError, ValueError):
                        continue
                    record[f"peak{index}_area"] = float(area)
                    record[f"peak{index}_tau"] = float(tau)
                    record[f"peak{index}_fwhm"] = float(fwhm)
                metadata = dict(cycle.custom_metadata)
                metadata.update(spectrum.custom_metadata)
                for name, value in metadata.items():
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        if value is not None:
                            record[name] = value
                        continue
                    if np.isfinite(numeric_value):
                        record[name] = numeric_value
                record.setdefault("Cycle mod 15", int(spectrum.cycle) % 15)
                records.append(_externalize_record(record))
        return records

    def _open_parameter_explorer_filter(
        self,
        title: str,
        records: list[dict[str, object]],
        definition: FilterDefinition,
        on_apply: Callable[[FilterDefinition], None],
    ) -> None:
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.transient(self.root)
        popup.geometry("680x360")
        popup.minsize(560, 260)
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(1, weight=1)
        fields = sorted(
            {key for record in records for key in record},
            key=lambda value: str(value).casefold(),
        )
        ttk.Label(
            popup,
            text="Keep records matching the following conditions:",
        ).grid(row=0, column=0, padx=10, pady=(10, 4), sticky="w")
        rows_frame = ttk.Frame(popup, padding=(10, 0))
        rows_frame.grid(row=1, column=0, sticky="nsew")
        rows_frame.columnconfigure(0, weight=1)
        rows_frame.rowconfigure(0, weight=1)
        canvas = tk.Canvas(rows_frame, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(rows_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        condition_frame = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=condition_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        condition_frame.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(canvas_window, width=event.width),
        )
        condition_frame.columnconfigure(0, weight=1)
        condition_frame.columnconfigure(1, weight=0)
        condition_frame.columnconfigure(2, weight=1)
        rows: list[tuple[tk.StringVar, tk.StringVar, tk.StringVar, ttk.Combobox, ttk.Combobox, ttk.Entry, ttk.Button]] = []

        def add_condition(condition: FilterCondition | None = None) -> None:
            row = len(rows)
            field_var = tk.StringVar(value=condition.field if condition else (fields[0] if fields else ""))
            operators = field_operators(records, field_var.get()) if fields else ("=", "!=")
            operator_var = tk.StringVar(
                value=condition.operator if condition and condition.operator in operators else operators[0]
            )
            value_var = tk.StringVar(value=condition.value if condition else "")
            field_box = ttk.Combobox(condition_frame, textvariable=field_var, values=fields, state="readonly")
            operator_box = ttk.Combobox(condition_frame, textvariable=operator_var, values=operators, state="readonly", width=16)
            value_entry = ttk.Entry(condition_frame, textvariable=value_var)
            remove_button = ttk.Button(condition_frame, text="Remove", width=9)

            def update_operators(_event=None) -> None:
                values = field_operators(records, field_var.get())
                operator_box.configure(values=values)
                if operator_var.get() not in values:
                    operator_var.set(values[0])

            def remove() -> None:
                index = next(index for index, item in enumerate(rows) if item[0] is field_var)
                rows.pop(index)
                for widget in (field_box, operator_box, value_entry, remove_button):
                    widget.destroy()
                for index, item in enumerate(rows):
                    for widget in item[3:]:
                        widget.grid_configure(row=index)

            field_box.bind("<<ComboboxSelected>>", update_operators)
            remove_button.configure(command=remove)
            field_box.grid(row=row, column=0, padx=(0, 5), pady=3, sticky="ew")
            operator_box.grid(row=row, column=1, padx=5, pady=3, sticky="ew")
            value_entry.grid(row=row, column=2, padx=5, pady=3, sticky="ew")
            remove_button.grid(row=row, column=3, padx=(5, 0), pady=3)
            rows.append((field_var, operator_var, value_var, field_box, operator_box, value_entry, remove_button))

        for condition in definition.conditions:
            if condition.field in fields:
                add_condition(condition)
        if not rows:
            add_condition()

        options = ttk.Frame(popup, padding=10)
        options.grid(row=2, column=0, sticky="ew")
        match_var = tk.StringVar(value=definition.match if definition.match in {"all", "any"} else "all")
        ttk.Label(options, text="Match:").pack(side=tk.LEFT)
        ttk.Radiobutton(options, text="All conditions", variable=match_var, value="all").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Radiobutton(options, text="Any condition", variable=match_var, value="any").pack(side=tk.LEFT, padx=(8, 0))

        buttons = ttk.Frame(popup, padding=(10, 0, 10, 10))
        buttons.grid(row=3, column=0, sticky="e")

        def collect() -> FilterDefinition:
            conditions = [
                FilterCondition(field.get(), operator.get(), value.get())
                for field, operator, value, *_widgets in rows
                if field.get() and value.get().strip() != ""
            ]
            return FilterDefinition(conditions, match_var.get())

        def apply_and_close() -> None:
            on_apply(collect())
            popup.destroy()

        def clear_and_close() -> None:
            on_apply(FilterDefinition())
            popup.destroy()

        ttk.Button(buttons, text="Add condition", command=add_condition).pack(side=tk.LEFT, padx=3)
        ttk.Button(buttons, text="Clear", command=clear_and_close).pack(side=tk.LEFT, padx=3)
        ttk.Button(buttons, text="Apply", command=apply_and_close).pack(side=tk.LEFT, padx=3)
        ttk.Button(buttons, text="Cancel", command=popup.destroy).pack(side=tk.LEFT, padx=3)
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)
        popup.grab_set()
        popup.focus_force()

    def _apply_drt_explorer_filter(
        self,
        definition: FilterDefinition,
        button_text: tk.StringVar,
        refresh: Callable[[], None],
    ) -> None:
        self._drt_explorer_filter = definition
        button_text.set(
            "Filter" if not definition.active else f"Filter ({len(definition.conditions)})"
        )
        refresh()

    def _apply_fit_explorer_filter(
        self,
        definition: FilterDefinition,
        button_text: tk.StringVar,
        refresh: Callable[[], None],
    ) -> None:
        self._fit_explorer_filter = definition
        button_text.set(
            "Filter" if not definition.active else f"Filter ({len(definition.conditions)})"
        )
        refresh()

    def open_drt_parameters_explorer(self) -> None:
        if self.busy or self.state is None:
            return
        self._sync_custom_metadata_columns()
        existing_popup = getattr(self, "drt_parameters_popup", None)
        if existing_popup is not None and existing_popup.winfo_exists():
            existing_popup.lift()
            existing_popup.focus_force()
            return
        mode = self._selected_drt_mode()
        records = self._collect_drt_parameter_records(mode)
        if not records:
            self._update_status(f"no {mode} parameters are available")
            return

        def filtered_records() -> list[dict[str, object]]:
            return apply_filters(records, self._drt_explorer_filter)

        def numeric_fields() -> list[str]:
            return [
                field
                for field in dict.fromkeys(
                    key
                    for record in records
                    for key in record
                    if key not in {"source_file", "drt_mode"}
                )
                if any(
                    isinstance(record.get(field), (int, float, np.integer, np.floating))
                    for record in records
                )
            ]

        fields = numeric_fields()
        manual_metadata_fields = set(self._custom_metadata_columns) | {
            "Spectrum",
            "Cycle mod 15",
        }
        parameter_fields = [
            field
            for field in fields
            if field not in manual_metadata_fields
            and (field == "R0" or field == "L0" or field.startswith("peak"))
        ]
        split_candidates = [
            field
            for field in dict.fromkeys(
                key for record in records for key in record
            )
            if field not in {"source_file", "drt_mode"}
            and field not in parameter_fields
        ]
        split_fields = ["None", *dict.fromkeys(["Spectrum", *split_candidates])]
        x_default = (
            self._drt_explorer_x_preference
            if self._drt_explorer_x_preference in fields
            else ("I_mA" if "I_mA" in fields else fields[0])
        )
        y_default = (
            self._drt_explorer_y_preference
            if self._drt_explorer_y_preference in fields
            else (parameter_fields[0] if parameter_fields else fields[0])
        )

        popup = tk.Toplevel(self.root)
        self.drt_parameters_popup = popup
        popup.title(f"DRT Parameters Explorer — {mode.title()}")
        popup.geometry("1180x760")
        popup.minsize(900, 600)
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(1, weight=1)

        def close_popup() -> None:
            self.drt_parameters_popup = None
            self._drt_parameters_refresh_callback = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", close_popup)
        controls = ttk.Frame(popup, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        for column in range(5):
            controls.columnconfigure(column, weight=1)
        x_var = tk.StringVar(value=x_default)
        y_vars: list[tk.StringVar] = []
        split_vars: list[tk.StringVar] = []
        x_equation = tk.StringVar(value="x")
        x_log = tk.BooleanVar(value=False)
        y_log = tk.BooleanVar(value=False)
        hide_legend = tk.BooleanVar(value=False)
        filter_text = tk.StringVar(
            value="Filter" if not self._drt_explorer_filter.active
            else f"Filter ({len(self._drt_explorer_filter.conditions)})"
        )
        y_rows_frame = ttk.Frame(controls)
        y_rows_frame.grid(row=1, column=1, columnspan=2, padx=3, sticky="ew")
        for column in range(3):
            y_rows_frame.columnconfigure(column, weight=1)
        y_rows: list[tuple[tk.StringVar, tk.StringVar, tk.StringVar, ttk.Combobox, ttk.Combobox, ttk.Entry]] = []

        def remove_y_row(
            y_value: tk.StringVar,
            split_value: tk.StringVar,
            y_box: ttk.Combobox,
            split_box: ttk.Combobox,
            equation_box: ttk.Entry,
            remove_button: ttk.Button,
        ) -> None:
            if len(y_rows) <= 1:
                return
            index = y_vars.index(y_value)
            y_vars.pop(index)
            split_vars.pop(index)
            y_rows.pop(index)
            for widget in (y_box, split_box, equation_box, remove_button):
                widget.destroy()
            for row, row_data in enumerate(y_rows):
                for widget in row_data[3:]:
                    widget.grid_configure(row=row)
            refresh_ranges()

        def add_y_row() -> None:
            row = len(y_rows)
            y_value = tk.StringVar(value=y_default)
            split_value = tk.StringVar(value="None")
            equation = tk.StringVar(value="y")
            y_box = ttk.Combobox(y_rows_frame, textvariable=y_value, state="readonly")
            split_box = ttk.Combobox(y_rows_frame, textvariable=split_value, state="readonly")
            equation_box = ttk.Entry(y_rows_frame, textvariable=equation)
            remove_button = ttk.Button(y_rows_frame, text="Remove", width=7)
            remove_button.configure(
                command=lambda: remove_y_row(
                    y_value, split_value, y_box, split_box, equation_box, remove_button
                )
            )
            y_box.configure(values=fields)
            split_box.configure(values=split_fields)
            y_box.grid(row=row, column=0, padx=(0, 4), sticky="ew")
            split_box.grid(row=row, column=1, padx=4, sticky="ew")
            equation_box.grid(row=row, column=2, padx=(4, 0), sticky="ew")
            remove_button.grid(row=row, column=3, padx=(4, 0), sticky="e")
            y_vars.append(y_value)
            split_vars.append(split_value)
            y_rows.append((y_value, split_value, equation, y_box, split_box, equation_box, remove_button))
            y_box.bind("<<ComboboxSelected>>", lambda _event: refresh_ranges())
            split_box.bind("<<ComboboxSelected>>", lambda _event: refresh_ranges())
            equation.trace_add("write", lambda *_args: refresh_plot())
            refresh_ranges()

        ttk.Label(controls, text="X axis").grid(row=0, column=0, sticky="w")
        ttk.Label(controls, text="Y axis / split / equation").grid(row=0, column=1, columnspan=2, sticky="w")
        ttk.Button(controls, text="Add y-variable", width=14, command=add_y_row).grid(row=0, column=4, sticky="e")
        x_box = ttk.Combobox(controls, textvariable=x_var, state="readonly")
        x_box.grid(row=1, column=0, padx=(0, 6), sticky="ew")
        ttk.Checkbutton(controls, text="Log X", variable=x_log).grid(row=2, column=3, sticky="w")
        ttk.Checkbutton(controls, text="Log Y", variable=y_log).grid(row=2, column=4, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Hide legend",
            variable=hide_legend,
            command=lambda: refresh_plot(),
        ).grid(row=4, column=2, sticky="w")
        ttk.Label(controls, text="X equation").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=x_equation).grid(row=3, column=0, padx=(0, 6), sticky="ew")
        ttk.Label(
            controls,
            text="Use x, y, column names, and np functions",
        ).grid(row=3, column=1, columnspan=4, padx=(6, 0), sticky="w")

        range_frame = ttk.LabelFrame(popup, text="Displayed value ranges", padding=6)
        range_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 6))
        range_controls = ttk.Frame(range_frame)
        range_controls.grid(row=0, column=0, columnspan=4, sticky="ew")
        range_controls.columnconfigure(1, weight=1)
        range_controls.columnconfigure(2, weight=1)
        chart_frame = ttk.Frame(range_frame)
        range_frame.rowconfigure(1, weight=1)
        range_frame.columnconfigure(0, weight=1)
        chart_frame.grid(row=1, column=0, columnspan=4, sticky="nsew")
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)

        from matplotlib import colormaps
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8.5, 5.8), dpi=100, constrained_layout=True)
        axes = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=chart_frame)
        self._attach_plot_export_menu(canvas, popup)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(canvas, chart_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        line_tool = _ExplorerLineTool(axes, canvas, controls, lambda: refresh_plot())

        range_state: dict[str, tuple[tk.DoubleVar, tk.DoubleVar, float, float]] = {}
        range_labels: dict[str, tk.StringVar] = {}
        range_widgets: dict[str, list[tk.Widget]] = {}

        def bounds(field: str) -> tuple[float, float]:
            values = []
            for record in filtered_records():
                try:
                    value = float(record[field])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values.append(value)
            if not values:
                return 0.0, 1.0
            minimum, maximum = min(values), max(values)
            return minimum, maximum if maximum > minimum else minimum + 1.0

        def value_at(field: str, position: float) -> float:
            _low, _high, minimum, maximum = range_state[field]
            return minimum + (maximum - minimum) * position / 100.0

        def range_text(field: str) -> str:
            low, high, _minimum, _maximum = range_state[field]
            return f"{value_at(field, low.get()):.5g} – {value_at(field, high.get()):.5g}"

        def update_range_label(field: str) -> None:
            low, high, _minimum, _maximum = range_state[field]
            if low.get() > high.get():
                high.set(low.get())
            range_labels[field].set(range_text(field))
            refresh_plot()

        def refresh_ranges() -> None:
            selected_fields = list(dict.fromkeys(
                [x_var.get()]
                + [y_var.get() for y_var in y_vars]
                + [
                    split_var.get() for split_var in split_vars
                    if split_var.get() not in {"None", "Spectrum"}
                    and split_var.get() in fields
                ]
            ))
            for widget_list in range_widgets.values():
                for widget in widget_list:
                    widget.grid_remove()
            for row, field in enumerate(selected_fields):
                if field not in range_state:
                    low = tk.DoubleVar(value=0.0)
                    high = tk.DoubleVar(value=100.0)
                    range_state[field] = (low, high, *bounds(field))
                    range_labels[field] = tk.StringVar(value=range_text(field))
                    range_widgets[field] = []
                    ttk.Label(range_controls, text=field).grid(row=row, column=0, sticky="w")
                    low_scale = ttk.Scale(
                        range_controls, from_=0, to=100, variable=low,
                        command=lambda _value, selected=field: update_range_label(selected),
                    )
                    high_scale = ttk.Scale(
                        range_controls, from_=0, to=100, variable=high,
                        command=lambda _value, selected=field: update_range_label(selected),
                    )
                    low_scale.grid(row=row, column=1, padx=6, sticky="ew")
                    high_scale.grid(row=row, column=2, padx=6, sticky="ew")
                    ttk.Label(range_controls, textvariable=range_labels[field], width=24).grid(row=row, column=3, sticky="w")
                    range_widgets[field].extend((low_scale, high_scale))
                else:
                    for widget in range_widgets[field]:
                        widget.grid()
                    range_controls.grid_slaves(row=row, column=0)[0].grid()
                    range_controls.grid_slaves(row=row, column=3)[0].grid()
            refresh_plot()

        def evaluate(
            record: dict[str, object],
            expression: str,
            fallback: str,
            x_field: str,
            y_field: str,
        ) -> float:
            variables = dict(record)
            variables["x"] = float(record[x_field])
            variables["y"] = float(record[y_field])
            return float(eval(expression.strip() or fallback, {"__builtins__": {}, "np": np}, variables))

        def refresh_plot(*_args) -> None:
            axes.clear()
            axes.grid(True, alpha=0.25)
            axes.axhline(0.0, color="#444444", linewidth=1.2, alpha=0.85, zorder=0)
            axes.axvline(0.0, color="#444444", linewidth=1.2, alpha=0.85, zorder=0)
            x_field = x_var.get()
            plotted_groups: list[tuple[str, list[tuple[float, float]]]] = []
            for y_value, split_value, equation, *_widgets in y_rows:
                groups: dict[object, list[tuple[float, float]]] = {}
                y_field = y_value.get()
                split_field = split_value.get()
                selected_fields = [x_field, y_field]
                if split_field not in {"None", "Spectrum"} and split_field in fields:
                    selected_fields.append(split_field)
                for record in filtered_records():
                    try:
                        if any(
                            field not in record
                            or not np.isfinite(float(record[field]))
                            or not value_at(field, range_state[field][0].get()) <= float(record[field]) <= value_at(field, range_state[field][1].get())
                            for field in selected_fields
                        ):
                            continue
                        x_value = evaluate(record, x_equation.get(), "x", x_field, x_field)
                        calculated = evaluate(record, equation.get(), "y", x_field, y_field)
                        if np.isfinite(x_value) and np.isfinite(calculated):
                            group = "DRT" if split_field == "None" else record.get(split_field)
                            groups.setdefault(group, []).append((x_value, calculated))
                    except (KeyError, TypeError, ValueError, SyntaxError, ZeroDivisionError):
                        continue
                natural_group_key = natsort_keygen(alg=ns.IGNORECASE)
                for group, values in sorted(
                    groups.items(),
                    key=lambda item: natural_group_key(str(item[0])),
                ):
                    plotted_groups.append((f"{y_field} = {equation.get() or 'y'} ({group})", values))
            color_scale = colormaps["rainbow"]
            for index, (label, values) in enumerate(plotted_groups):
                if not values:
                    continue
                color = color_scale(index / max(len(plotted_groups) - 1, 1))
                axes.plot(
                    [value[0] for value in values],
                    [value[1] for value in values],
                    "o-",
                    color=color,
                    linewidth=1.1,
                    markersize=4,
                    label=label,
                )
            axes.relim()
            axes.autoscale(enable=True, axis="both", tight=False)
            axes.autoscale_view()
            plotted_values = [
                value for _label, values in plotted_groups for value in values
            ]
            if plotted_values:
                plotted_x = np.asarray(
                    [value[0] for value in plotted_values], dtype=float
                )
                plotted_y = np.asarray(
                    [value[1] for value in plotted_values], dtype=float
                )
                x_min, x_max = float(np.min(plotted_x)), float(np.max(plotted_x))
                y_min, y_max = float(np.min(plotted_y)), float(np.max(plotted_y))
                if x_min == x_max:
                    padding = max(abs(x_min) * 0.05, 1.0)
                    x_min -= padding
                    x_max += padding
                elif x_log.get():
                    x_min /= 1.05
                    x_max *= 1.05
                else:
                    padding = 0.05 * (x_max - x_min)
                    x_min -= padding
                    x_max += padding
                if y_min == y_max:
                    padding = max(abs(y_min) * 0.05, 1.0)
                    y_min -= padding
                    y_max += padding
                elif y_log.get():
                    y_min /= 1.05
                    y_max *= 1.05
                else:
                    padding = 0.05 * (y_max - y_min)
                    y_min -= padding
                    y_max += padding
                axes.set_xlim(x_min, x_max)
                axes.set_ylim(y_min, y_max)
            x_expression = x_equation.get().strip()
            x_label = x_field if x_expression in {"", "x"} else x_expression
            y_labels = []
            for y_value, _split_value, equation, *_widgets in y_rows:
                expression = equation.get().strip()
                y_labels.append(y_value.get() if expression in {"", "y"} else expression)
            axes.set_xlabel(x_label)
            axes.set_ylabel(" / ".join(y_labels))
            axes.set_xscale("log" if x_log.get() else "linear")
            axes.set_yscale("log" if y_log.get() else "linear")
            line_tool.set_data(plotted_values)
            line_tool.redraw()
            if plotted_groups and not hide_legend.get():
                axes.legend(loc="best")
            canvas.draw_idle()

        def toggle_legend(_event=None):
            hide_legend.set(not hide_legend.get())
            refresh_plot()
            return "break"

        popup.bind("<Alt-h>", toggle_legend)
        popup.bind("<Alt-H>", toggle_legend)

        def active_view() -> None:
            axes.relim()
            axes.autoscale(enable=True, axis="both", tight=False)
            axes.autoscale_view()
            canvas.draw_idle()

        def refresh_data() -> None:
            nonlocal fields, split_fields
            current_mode = self._selected_drt_mode()
            records[:] = self._collect_drt_parameter_records(current_mode)
            fields = numeric_fields()
            if not fields:
                x_box.configure(values=[])
                for _y_value, _split_value, _equation, y_box, split_box, _entry, _remove_button in y_rows:
                    y_box.configure(values=[])
                    split_box.configure(values=["None"])
                axes.clear()
                line_tool.set_data([])
                canvas.draw_idle()
                self._update_status(f"no {current_mode} parameters are available")
                return
            manual_metadata_fields = set(self._custom_metadata_columns) | {
                "Spectrum",
                "Cycle mod 15",
            }
            parameter_fields = [
                field
                for field in fields
                if field not in manual_metadata_fields
                and (field == "R0" or field == "L0" or field.startswith("peak"))
            ]
            split_candidates = [
                field
                for field in dict.fromkeys(
                    key for record in records for key in record
                )
                if field not in {"source_file", "drt_mode"}
                and field not in parameter_fields
            ]
            split_fields = ["None", *dict.fromkeys(["Spectrum", *split_candidates])]
            x_values = [field for field in fields]
            x_box.configure(values=x_values)
            for _y_value, _split_value, _equation, y_box, split_box, _entry, _remove_button in y_rows:
                y_box.configure(values=x_values)
                split_box.configure(values=split_fields)
            if x_var.get() not in x_values:
                x_var.set(x_values[0])
            for y_value, split_value, _equation, _y_box, _split_box, _entry, _remove_button in y_rows:
                if y_value.get() not in x_values:
                    y_value.set(x_values[0])
                if split_value.get() not in split_fields:
                    split_value.set("None")
            for field, (low, high, _minimum, _maximum) in list(range_state.items()):
                if field in fields:
                    minimum, maximum = bounds(field)
                    range_state[field] = (low, high, minimum, maximum)
                    range_labels[field].set(range_text(field))
            refresh_ranges()

        def on_split_selected(_event=None) -> None:
            refresh_ranges()
            refresh_plot()

        x_box.bind("<<ComboboxSelected>>", lambda _event: refresh_ranges())
        x_box.configure(values=fields)
        for y_value, split_value, _equation, y_box, split_box, _entry, _remove_button in y_rows:
            y_box.configure(values=fields)
            split_box.configure(values=split_fields)
        add_y_row()
        x_equation.trace_add("write", lambda *_args: refresh_plot())
        x_log.trace_add("write", lambda *_args: refresh_plot())
        y_log.trace_add("write", lambda *_args: refresh_plot())
        self._drt_parameters_refresh_callback = refresh_data
        ttk.Button(controls, text="Refresh", command=refresh_data).grid(row=4, column=0, pady=(6, 0), sticky="w")
        ttk.Button(controls, text="Active view", command=active_view).grid(row=4, column=1, pady=(6, 0), sticky="w")
        ttk.Button(
            controls,
            textvariable=filter_text,
            command=lambda: self._open_parameter_explorer_filter(
                "DRT Parameters Filter",
                records,
                self._drt_explorer_filter,
                lambda definition: self._apply_drt_explorer_filter(definition, filter_text, refresh_ranges),
            ),
        ).grid(row=4, column=3, pady=(6, 0), sticky="w")
        refresh_ranges()

    def open_fit_parameters_explorer(self) -> None:
        if self.busy or self.state is None:
            return
        self._sync_custom_metadata_columns()
        existing_popup = getattr(self, "fit_parameters_popup", None)
        if existing_popup is not None and existing_popup.winfo_exists():
            existing_popup.lift()
            existing_popup.focus_force()
            return
        records: list[dict[str, object]] = []
        for dataset_id in self._dataset_order:
            loaded = self.loaded_projects[dataset_id]
            for spectrum in loaded.spectra:
                cycle = loaded.state.cycles.get(spectrum.cycle)
                if cycle is None or cycle.fit_parameters is None:
                    continue
                values = np.asarray(cycle.fit_parameters).reshape(-1)
                if values.size != len(cycle.parameters):
                    continue
                record: dict[str, object] = {
                    "source_file": loaded.state.source_path.name,
                    "cycle": spectrum.cycle,
                    "potential_V": cycle.potential_v,
                    "current_mA": cycle.current_ma,
                    "time_s": cycle.time_s,
                    "Spectrum": (
                        spectrum.custom_metadata.get(SPECTRUM_METADATA_COLUMN)
                        or {"working": "WE", "cell": "Cell", "counter": "CE"}.get(
                            loaded.state.control,
                            loaded.state.control,
                        )
                    ),
                    "circuit": cycle.model(loaded.state.circuit),
                }
                record.update(
                    {
                        parameter.name: float(value)
                        for parameter, value in zip(cycle.parameters, values)
                    }
                )
                record.update(
                    _derived_block_values(
                        cycle.model(loaded.state.circuit),
                        [parameter.name for parameter in cycle.parameters],
                        values,
                    )
                )
                metadata = dict(cycle.custom_metadata)
                metadata.update(spectrum.custom_metadata)
                for name, value in metadata.items():
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        if value is not None:
                            record[name] = value
                        continue
                    record[name] = numeric_value if np.isfinite(numeric_value) else value
                record.setdefault("Cycle mod 15", int(spectrum.cycle) % 15)
                records.append(_externalize_record(record))
        if not records:
            self._update_status("no fitted spectra are available")
            return

        def filtered_records() -> list[dict[str, object]]:
            return apply_filters(records, self._fit_explorer_filter)

        numeric_fields = []
        for field in dict.fromkeys(
            key for record in records for key in record if key != "circuit" and key != "source_file"
        ):
            values = [record.get(field) for record in records]
            if any(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
                numeric_fields.append(field)
        manual_metadata_fields = set(self._custom_metadata_columns) | {
            "Spectrum",
            "Cycle mod 15",
        }
        parameter_fields = [
            field
            for field in numeric_fields
            if field not in manual_metadata_fields
            and any(
                field.startswith(prefix)
                for prefix in ("R", "Q", "a", "C", "L", "W", "tau")
            )
        ]
        split_candidates = [
            field
            for field in dict.fromkeys(
                key for record in records for key in record
            )
            if field not in {"source_file", "circuit"}
            and field not in parameter_fields
        ]
        split_fields = ["None", *dict.fromkeys(["Spectrum", *split_candidates])]
        x_default = (
            self._fit_explorer_x_preference
            if self._fit_explorer_x_preference in numeric_fields
            else ("I_mA" if "I_mA" in numeric_fields else numeric_fields[0])
        )
        y_default = (
            self._fit_explorer_y_preference
            if self._fit_explorer_y_preference in numeric_fields
            else (parameter_fields[0] if parameter_fields else numeric_fields[0])
        )

        popup = tk.Toplevel(self.root)
        self.fit_parameters_popup = popup
        popup.title("Fit Parameters Explorer")
        popup.geometry("1180x760")
        popup.minsize(900, 600)
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(1, weight=1)

        def close_popup() -> None:
            self.fit_parameters_popup = None
            self._fit_parameters_refresh_callback = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", close_popup)

        controls = ttk.Frame(popup, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        for column in range(5):
            controls.columnconfigure(column, weight=1)

        x_var = tk.StringVar(value=x_default)
        y_vars: list[tk.StringVar] = []
        split_vars: list[tk.StringVar] = []
        x_equation = tk.StringVar(value="x")
        x_log = tk.BooleanVar(value=False)
        y_log = tk.BooleanVar(value=False)
        hide_legend = tk.BooleanVar(value=False)
        filter_text = tk.StringVar(
            value="Filter" if not self._fit_explorer_filter.active
            else f"Filter ({len(self._fit_explorer_filter.conditions)})"
        )
        y_rows_frame = ttk.Frame(controls)
        y_rows_frame.grid(row=1, column=1, columnspan=2, padx=3, sticky="ew")
        for column in range(3):
            y_rows_frame.columnconfigure(column, weight=1)
        y_rows: list[tuple[tk.StringVar, tk.StringVar, tk.StringVar, ttk.Combobox, ttk.Combobox, ttk.Entry]] = []

        def remove_y_row(
            y_value: tk.StringVar,
            split_value: tk.StringVar,
            y_box: ttk.Combobox,
            split_box: ttk.Combobox,
            equation_box: ttk.Entry,
            remove_button: ttk.Button,
        ) -> None:
            if len(y_rows) <= 1:
                return
            index = y_vars.index(y_value)
            y_vars.pop(index)
            split_vars.pop(index)
            y_rows.pop(index)
            for widget in (y_box, split_box, equation_box, remove_button):
                widget.destroy()
            for row, row_data in enumerate(y_rows):
                for widget in row_data[3:]:
                    widget.grid_configure(row=row)
            refresh_ranges()

        def add_y_row() -> None:
            row = len(y_rows)
            y_value = tk.StringVar(value=y_default)
            split_value = tk.StringVar(value="None")
            equation = tk.StringVar(value="y")
            y_box = ttk.Combobox(y_rows_frame, textvariable=y_value, state="readonly")
            split_box = ttk.Combobox(y_rows_frame, textvariable=split_value, state="readonly")
            equation_box = ttk.Entry(y_rows_frame, textvariable=equation)
            remove_button = ttk.Button(y_rows_frame, text="Remove", width=7)
            remove_button.configure(
                command=lambda: remove_y_row(
                    y_value, split_value, y_box, split_box, equation_box, remove_button
                )
            )
            y_box.configure(values=numeric_fields)
            split_box.configure(values=split_fields)
            y_box.grid(row=row, column=0, padx=(0, 4), sticky="ew")
            split_box.grid(row=row, column=1, padx=4, sticky="ew")
            equation_box.grid(row=row, column=2, padx=(4, 0), sticky="ew")
            remove_button.grid(row=row, column=3, padx=(4, 0), sticky="e")
            y_vars.append(y_value)
            split_vars.append(split_value)
            y_rows.append((y_value, split_value, equation, y_box, split_box, equation_box, remove_button))
            y_box.bind("<<ComboboxSelected>>", lambda _event: refresh_ranges())
            split_box.bind("<<ComboboxSelected>>", lambda _event: refresh_ranges())
            equation.trace_add("write", lambda *_args: refresh_plot())
            refresh_ranges()

        ttk.Label(controls, text="X axis").grid(row=0, column=0, sticky="w")
        ttk.Label(controls, text="Y axis / split / equation").grid(row=0, column=1, columnspan=2, sticky="w")
        ttk.Button(controls, text="Add y-variable", width=14, command=add_y_row).grid(row=0, column=4, sticky="e")
        x_box = ttk.Combobox(controls, textvariable=x_var, values=numeric_fields, state="readonly")
        x_box.grid(row=1, column=0, padx=(0, 6), sticky="ew")
        ttk.Checkbutton(controls, text="Log X", variable=x_log).grid(row=2, column=3, sticky="w")
        ttk.Checkbutton(controls, text="Log Y", variable=y_log).grid(row=2, column=4, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Hide legend",
            variable=hide_legend,
            command=lambda: refresh_plot(),
        ).grid(row=4, column=2, sticky="w")
        ttk.Label(controls, text="X equation").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=x_equation).grid(row=3, column=0, padx=(0, 6), sticky="ew")
        ttk.Label(
            controls,
            text="Use column names and np functions, e.g. 1/R1 or np.log10(I_mA)",
        ).grid(row=3, column=1, columnspan=4, padx=(6, 0), sticky="w")
        ttk.Button(
            controls,
            text="Refresh",
            command=lambda: refresh_data(),
        ).grid(row=4, column=0, pady=(6, 0), sticky="w")

        range_frame = ttk.LabelFrame(popup, text="Displayed value ranges", padding=6)
        range_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 6))
        range_controls = ttk.Frame(range_frame)
        range_controls.grid(row=0, column=0, columnspan=4, sticky="ew")
        range_controls.columnconfigure(1, weight=1)
        range_controls.columnconfigure(2, weight=1)
        chart_frame = ttk.Frame(range_frame)
        range_frame.rowconfigure(0, weight=0)
        range_frame.rowconfigure(1, weight=1)
        chart_frame.grid(row=1, column=0, columnspan=4, sticky="nsew")
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)

        from matplotlib import colormaps
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8.5, 5.8), dpi=100, constrained_layout=True)
        axes = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=chart_frame)
        self._attach_plot_export_menu(canvas, popup)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(canvas, chart_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        line_tool = _ExplorerLineTool(axes, canvas, controls, lambda: refresh_plot())

        range_state: dict[str, tuple[tk.DoubleVar, tk.DoubleVar, float, float]] = {}
        range_labels: dict[str, tk.StringVar] = {}
        range_widgets: dict[str, list[tk.Widget]] = {}

        def field_bounds(field: str) -> tuple[float, float]:
            values = np.asarray(
                [float(record[field]) for record in filtered_records() if field in record and np.isfinite(float(record[field]))],
                dtype=float,
            )
            if values.size == 0:
                return 0.0, 1.0
            return float(np.min(values)), float(np.max(values))

        def range_value(field: str, position: float) -> float:
            _low, _high, minimum, maximum = range_state[field]
            return minimum + (maximum - minimum) * position / 100.0

        def range_text(field: str) -> str:
            low, high, _minimum, _maximum = range_state[field]
            return f"{range_value(field, low.get()):.5g} – {range_value(field, high.get()):.5g}"

        def refresh_ranges() -> None:
            fields = [x_var.get(), *[y_var.get() for y_var in y_vars]]
            fields.extend(
                split_var.get()
                for split_var in split_vars
                if split_var.get() not in {"None", "Spectrum"}
                and split_var.get() in numeric_fields
            )
            for widget_list in range_widgets.values():
                for widget in widget_list:
                    widget.grid_remove()
            for row, field in enumerate(dict.fromkeys(fields)):
                if field not in range_state:
                    low = tk.DoubleVar(value=0.0)
                    high = tk.DoubleVar(value=100.0)
                    minimum, maximum = field_bounds(field)
                    range_state[field] = (low, high, minimum, maximum)
                    range_labels[field] = tk.StringVar(value=range_text(field))
                    range_widgets[field] = []
                    ttk.Label(range_controls, text=field).grid(row=row, column=0, sticky="w")
                    low_scale = ttk.Scale(
                        range_controls, from_=0, to=100, variable=low,
                        command=lambda _value, selected=field: update_range_label(selected),
                    )
                    high_scale = ttk.Scale(
                        range_controls, from_=0, to=100, variable=high,
                        command=lambda _value, selected=field: update_range_label(selected),
                    )
                    low_scale.grid(row=row, column=1, padx=6, sticky="ew")
                    high_scale.grid(row=row, column=2, padx=6, sticky="ew")
                    ttk.Label(range_controls, textvariable=range_labels[field], width=24).grid(
                        row=row, column=3, sticky="w"
                    )
                    range_widgets[field].extend((low_scale, high_scale))
                else:
                    for widget in range_widgets[field]:
                        widget.grid()
                    range_controls.grid_slaves(row=row, column=0)[0].grid()
                    range_controls.grid_slaves(row=row, column=3)[0].grid()
            refresh_plot()

        def update_range_label(field: str) -> None:
            low, high, _minimum, _maximum = range_state[field]
            if low.get() > high.get():
                if low.get() <= 100:
                    high.set(low.get())
            range_labels[field].set(range_text(field))
            refresh_plot()

        def evaluate(
            record: dict[str, object],
            expression: str,
            fallback: str,
            x_field: str,
            y_field: str,
        ) -> float:
            expression = expression.strip() or fallback
            variables = dict(record)
            variables["x"] = float(record[x_field])
            variables["y"] = float(record[y_field])
            value = eval(expression, {"__builtins__": {}, "np": np}, variables)
            return float(value)

        def refresh_plot(*_args) -> None:
            axes.clear()
            axes.grid(True, alpha=0.25)
            axes.axhline(0.0, color="#444444", linewidth=1.2, alpha=0.85, zorder=0)
            axes.axvline(0.0, color="#444444", linewidth=1.2, alpha=0.85, zorder=0)
            x_field = x_var.get()
            plotted_groups: list[tuple[str, list[tuple[float, float]]]] = []
            for y_var, split_var, equation, *_widgets in y_rows:
                groups: dict[object, list[tuple[float, float]]] = {}
                y_field = y_var.get()
                split_field = split_var.get()
                selected_fields = [x_field, y_field]
                if split_field not in {"None", "Spectrum"} and split_field in numeric_fields:
                    selected_fields.append(split_field)
                for record in filtered_records():
                    try:
                        if any(
                            field not in record
                            or not np.isfinite(float(record[field]))
                            or not (
                                range_value(field, range_state[field][0].get())
                                <= float(record[field])
                                <= range_value(field, range_state[field][1].get())
                            )
                            for field in selected_fields
                        ):
                            continue
                        x_value = evaluate(record, x_equation.get(), x_field, x_field, x_field)
                        y_value = evaluate(record, equation.get(), y_field, x_field, y_field)
                        if not np.isfinite(x_value) or not np.isfinite(y_value):
                            continue
                        if x_log.get() and x_value <= 0 or y_log.get() and y_value <= 0:
                            continue
                        group = "All spectra" if split_field == "None" else record.get(split_field)
                        groups.setdefault(group, []).append((x_value, y_value))
                    except Exception:
                        continue
                natural_group_key = natsort_keygen(alg=ns.IGNORECASE)
                for group, points in sorted(
                    groups.items(),
                    key=lambda item: natural_group_key(str(item[0])),
                ):
                    points.sort(key=lambda point: point[0])
                    plotted_groups.append((f"{y_field} = {equation.get() or 'y'} ({group})", points))
            color_scale = colormaps["rainbow"]
            for index, (label, points) in enumerate(plotted_groups):
                if not points:
                    continue
                x_values, y_values = zip(*points)
                axes.plot(
                    x_values,
                    y_values,
                    "o-",
                    color=color_scale(index / max(len(plotted_groups) - 1, 1)),
                    label=label,
                )
            axes.relim()
            axes.autoscale(enable=True, axis="both", tight=False)
            axes.autoscale_view()
            plotted_values = [point for _label, points in plotted_groups for point in points]
            if plotted_values:
                plotted_x = np.asarray([point[0] for point in plotted_values], dtype=float)
                plotted_y = np.asarray([point[1] for point in plotted_values], dtype=float)
                x_min, x_max = float(np.min(plotted_x)), float(np.max(plotted_x))
                y_min, y_max = float(np.min(plotted_y)), float(np.max(plotted_y))
                if x_min == x_max:
                    padding = max(abs(x_min) * 0.05, 1.0)
                    x_min -= padding
                    x_max += padding
                elif x_log.get():
                    x_min /= 1.05
                    x_max *= 1.05
                else:
                    padding = 0.05 * (x_max - x_min)
                    x_min -= padding
                    x_max += padding
                if y_min == y_max:
                    padding = max(abs(y_min) * 0.05, 1.0)
                    y_min -= padding
                    y_max += padding
                elif y_log.get():
                    y_min /= 1.05
                    y_max *= 1.05
                else:
                    padding = 0.05 * (y_max - y_min)
                    y_min -= padding
                    y_max += padding
                axes.set_xlim(x_min, x_max)
                axes.set_ylim(y_min, y_max)
            x_expression = x_equation.get().strip()
            x_label = x_field if x_expression in {"", "x"} else x_expression
            y_labels = []
            for y_var, _split_var, equation, *_widgets in y_rows:
                expression = equation.get().strip()
                y_labels.append(y_var.get() if expression in {"", "y"} else expression)
            axes.set_xlabel(x_label)
            axes.set_ylabel(" / ".join(y_labels))
            axes.set_xscale("log" if x_log.get() else "linear")
            axes.set_yscale("log" if y_log.get() else "linear")
            line_tool.set_data(plotted_values)
            line_tool.redraw()
            if plotted_groups and not hide_legend.get():
                axes.legend(loc="best")
            canvas.draw_idle()

        def toggle_legend(_event=None):
            hide_legend.set(not hide_legend.get())
            refresh_plot()
            return "break"

        popup.bind("<Alt-h>", toggle_legend)
        popup.bind("<Alt-H>", toggle_legend)

        def active_view() -> None:
            axes.relim()
            axes.autoscale(enable=True, axis="both", tight=False)
            axes.autoscale_view()
            canvas.draw_idle()

        def refresh_data() -> None:
            refreshed = self._collect_fit_parameter_records()
            if not refreshed:
                self._update_status("no fitted spectra are available")
                return
            records[:] = refreshed
            for field, (low, high, _minimum, _maximum) in list(range_state.items()):
                minimum, maximum = field_bounds(field)
                range_state[field] = (low, high, minimum, maximum)
                range_labels[field].set(range_text(field))
            refresh_plot()

        for variable in (x_var, x_equation):
            variable.trace_add("write", lambda *_args: refresh_ranges())
        x_log.trace_add("write", lambda *_args: refresh_plot())
        y_log.trace_add("write", lambda *_args: refresh_plot())

        def on_split_selected(_event=None) -> None:
            refresh_ranges()
            refresh_plot()

        x_box.bind("<<ComboboxSelected>>", lambda _event: refresh_ranges())
        add_y_row()
        self._fit_parameters_refresh_callback = refresh_data
        ttk.Button(controls, text="Active view", command=active_view).grid(row=4, column=1, pady=(6, 0), sticky="w")
        ttk.Button(
            controls,
            textvariable=filter_text,
            command=lambda: self._open_parameter_explorer_filter(
                "Fit Parameters Filter",
                records,
                self._fit_explorer_filter,
                lambda definition: self._apply_fit_explorer_filter(definition, filter_text, refresh_ranges),
            ),
        ).grid(row=4, column=3, pady=(6, 0), sticky="w")
        refresh_ranges()

    def refresh_fit_parameters_explorer(self, popup: tk.Toplevel) -> None:
        if popup.winfo_exists():
            popup.destroy()
        self.open_fit_parameters_explorer()

    def _refresh_open_parameter_explorers(self) -> None:
        for attribute, callback_attribute in (
            ("fit_parameters_popup", "_fit_parameters_refresh_callback"),
            ("drt_parameters_popup", "_drt_parameters_refresh_callback"),
        ):
            popup = getattr(self, attribute, None)
            callback = getattr(self, callback_attribute, None)
            if popup is None or not popup.winfo_exists() or callback is None:
                continue
            callback()

    def _metadata_edit_project_key(self) -> str:
        if self.project_path is not None:
            return str(self.project_path.resolve())
        if self.loaded is not None:
            return f"source::{self.loaded.state.source_path.resolve()}"
        return "session"

    def edit_metadata_column_from_clipboard(self) -> None:
        if self.busy or self.state is None:
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra first")
            return
        self._sync_custom_metadata_columns()
        editable_columns = ["Ecell_V", "I_mA", *self._custom_metadata_columns]
        if not editable_columns:
            self._update_status("no editable metadata columns are available")
            return
        dialog = MetadataEditDialog(
            self.root,
            len(selected_rows),
            editable_columns,
            self._last_metadata_edit_column.get(self._metadata_edit_project_key()),
            [loaded.state.source_path.name for _dataset_id, loaded, _spectrum in selected_rows],
            [
                loaded.state.source_path.name
                for loaded in self.loaded_projects.values()
            ],
        )
        self.root.wait_window(dialog)
        if dialog.result is None:
            return
        column_name, values, create_new = dialog.result
        if len(values) != len(selected_rows):
            messagebox.showerror(
                "Wrong number of values",
                f"Paste exactly {len(selected_rows)} values.",
                parent=self.root,
            )
            return
        project_key = self._metadata_edit_project_key()
        if create_new:
            reserved_names = {
                "source_file",
                "source_path",
                "circuit",
                "Ecell_V",
                "I_mA",
                "time/s",
                "included_points",
                "total_points",
                "outlier_points",
                "fmin_Hz",
                "fmax_Hz",
                "fmin_act_Hz",
                "fmax_act_Hz",
            }
            explorer_base_columns = self._explorer_base_columns()
            if column_name.casefold() == "time":
                explorer_base_columns = tuple(
                    name for name in explorer_base_columns if name != "time"
                )
            known_names = [
                *explorer_base_columns,
                *self._custom_metadata_columns,
                *self._explorer_headings.values(),
                *reserved_names,
            ]
            for loaded in self.loaded_projects.values():
                known_names.extend(str(name) for name in loaded.dataframe.columns)
            existing_names = {name.casefold(): name for name in known_names}
            if column_name.startswith("#"):
                messagebox.showerror(
                    "Invalid column name",
                    "Column names cannot start with '#'.",
                    parent=self.root,
                )
                return
            if column_name.casefold() in existing_names:
                messagebox.showerror(
                    "Column already exists",
                    f"A column named '{existing_names[column_name.casefold()]}' already exists.",
                    parent=self.root,
                )
                return
            for loaded in self.loaded_projects.values():
                if column_name not in loaded.dataframe.columns:
                    loaded.dataframe[column_name] = np.full(
                        len(loaded.dataframe), None, dtype=object
                    )
                for spectrum_item in loaded.spectra:
                    spectrum_item.custom_metadata.setdefault(column_name, None)
                    cycle_item = loaded.state.cycles.get(spectrum_item.cycle)
                    if cycle_item is None:
                        cycle_item = load_cycle(
                            loaded.dataframe,
                            spectrum_item.cycle,
                            loaded.state.control,
                        )
                        if loaded.state.all_frequency_window is not None:
                            cycle_item.frequency_window = loaded.state.all_frequency_window
                        cycle_item.circuit = loaded.state.circuit
                        loaded.state.cycles[spectrum_item.cycle] = cycle_item
                    cycle_item.custom_metadata.setdefault(column_name, None)
            if column_name not in self._custom_metadata_columns:
                self._custom_metadata_columns.append(column_name)
        for (_dataset_id, loaded, spectrum), value in zip(selected_rows, values):
            cycle = loaded.state.cycles.get(spectrum.cycle)
            if cycle is None:
                cycle = load_cycle(
                    loaded.dataframe,
                    spectrum.cycle,
                    loaded.state.control,
                )
                if loaded.state.all_frequency_window is not None:
                    cycle.frequency_window = loaded.state.all_frequency_window
                cycle.circuit = loaded.state.circuit
                loaded.state.cycles[spectrum.cycle] = cycle
            if column_name in {"Ecell_V", "I_mA"}:
                try:
                    numeric_value = float(value) if value is not None else np.nan
                except (TypeError, ValueError):
                    messagebox.showerror(
                        "Invalid metadata value",
                        f"{column_name} must contain numeric values.",
                        parent=self.root,
                    )
                    return
                if not np.isfinite(numeric_value):
                    messagebox.showerror(
                        "Invalid metadata value",
                        f"{column_name} must contain finite numeric values.",
                        parent=self.root,
                    )
                    return
                if column_name == "Ecell_V":
                    cycle.potential_v = numeric_value
                    raw_column = {
                        "working": "ewe_v",
                        "cell": "ewe_ece_v",
                        "counter": "ece_v",
                    }.get(loaded.state.control)
                else:
                    cycle.current_ma = numeric_value
                    raw_column = "i_ma"
                cycle.custom_metadata[column_name] = numeric_value
                if raw_column is not None and raw_column in loaded.dataframe.columns:
                    rows = (
                        loaded.dataframe["cycle_number"] == spectrum.cycle
                        if "cycle_number" in loaded.dataframe.columns
                        else np.ones(len(loaded.dataframe), dtype=bool)
                    )
                    loaded.dataframe.loc[rows, raw_column] = numeric_value
            else:
                cycle.custom_metadata[column_name] = value
                spectrum.custom_metadata[column_name] = value
                if column_name in loaded.dataframe.columns:
                    loaded.dataframe[column_name] = loaded.dataframe[column_name].astype(
                        object
                    )
                if "cycle_number" in loaded.dataframe.columns:
                    rows = loaded.dataframe["cycle_number"] == spectrum.cycle
                    loaded.dataframe.loc[rows, column_name] = value
                else:
                    loaded.dataframe[column_name] = value
        self._populate_explorer()
        self._last_metadata_edit_column[project_key] = column_name
        self._update_status(f"metadata column '{column_name}' updated")

    def paste_metadata_column_from_clipboard(self) -> None:
        if self.busy or self.state is None:
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra first")
            return

        dialog = MetadataColumnDialog(self.root, len(selected_rows))
        self.root.wait_window(dialog)
        if dialog.result is None:
            return
        column_name, values = dialog.result
        if len(values) != len(selected_rows):
            messagebox.showerror(
                "Wrong number of values",
                f"Paste exactly {len(selected_rows)} values.",
                parent=self.root,
            )
            return

        reserved_names = {
            "source_file",
            "source_path",
            "circuit",
            "Ecell_V",
            "I_mA",
            "included_points",
            "total_points",
            "outlier_points",
            "fmin_Hz",
            "fmax_Hz",
            "fmin_act_Hz",
            "fmax_act_Hz",
        }
        explorer_base_columns = self._explorer_base_columns()
        if column_name.casefold() == "time":
            explorer_base_columns = tuple(
                name for name in explorer_base_columns if name != "time"
            )
        known_names = [
            *explorer_base_columns,
            *self._custom_metadata_columns,
            *self._explorer_headings.values(),
            *reserved_names,
        ]
        for loaded in self.loaded_projects.values():
            known_names.extend(str(name) for name in loaded.dataframe.columns)
            for parameter in loaded.state.default_parameters:
                known_names.extend(
                    (
                        _external_parameter_name(parameter.name),
                        f"{_external_parameter_name(parameter.name)}_e",
                    )
                )
        existing_names = {
            name.casefold(): name for name in known_names
        }
        if column_name.startswith("#"):
            messagebox.showerror(
                "Invalid column name",
                "Column names cannot start with '#'.",
                parent=self.root,
            )
            return
        if column_name.casefold() in existing_names:
            messagebox.showerror(
                "Column already exists",
                f"A column named '{existing_names[column_name.casefold()]}' already exists.",
                parent=self.root,
            )
            return

        for loaded in self.loaded_projects.values():
            if column_name not in loaded.dataframe.columns:
                loaded.dataframe[column_name] = np.full(
                    len(loaded.dataframe), None, dtype=object
                )
            for spectrum in loaded.spectra:
                spectrum.custom_metadata.setdefault(column_name, None)
                cycle = loaded.state.cycles.get(spectrum.cycle)
                if cycle is None:
                    cycle = load_cycle(
                        loaded.dataframe,
                        spectrum.cycle,
                        loaded.state.control,
                    )
                    if loaded.state.all_frequency_window is not None:
                        cycle.frequency_window = loaded.state.all_frequency_window
                    cycle.circuit = loaded.state.circuit
                    loaded.state.cycles[spectrum.cycle] = cycle
                cycle.custom_metadata.setdefault(column_name, None)

        if column_name not in self._custom_metadata_columns:
            self._custom_metadata_columns.append(column_name)

        for (_dataset_id, loaded, spectrum), value in zip(selected_rows, values):
            cycle = loaded.state.cycles.get(spectrum.cycle)
            if cycle is None:
                cycle = load_cycle(
                    loaded.dataframe,
                    spectrum.cycle,
                    loaded.state.control,
                )
            if loaded.state.all_frequency_window is not None:
                cycle.frequency_window = loaded.state.all_frequency_window
            cycle.circuit = loaded.state.circuit
            loaded.state.cycles[spectrum.cycle] = cycle
            cycle.custom_metadata[column_name] = value
            spectrum.custom_metadata[column_name] = value

            if "cycle_number" in loaded.dataframe.columns:
                loaded.dataframe[column_name] = loaded.dataframe[column_name].astype(
                    object
                )
                rows = loaded.dataframe["cycle_number"] == spectrum.cycle
                loaded.dataframe.loc[rows, column_name] = value
            else:
                loaded.dataframe[column_name] = value

        self._populate_explorer()
        self._update_status(f"metadata column '{column_name}' added")

    def reset_points(self) -> None:
        if self.busy:
            return
        selected_rows = self._selected_spectrum_rows()
        if selected_rows:
            reset_cycles = {
                (dataset_id, spectrum.cycle): (loaded, spectrum.cycle)
                for dataset_id, loaded, spectrum in selected_rows
            }
            for loaded, cycle_number in reset_cycles.values():
                self._loaded_cycle_for_popup(loaded, cycle_number).reset_selection()
            status = f"selection reset for {len(reset_cycles)} selected spectra"
        else:
            if self.state is None:
                return
            self.state.active.reset_selection()
            status = "selection reset"
        self._refresh_plot(rescale=True)
        self._update_status(status)

    def remove_deterministic_outliers(self) -> None:
        if self.busy or self.state is None or self.loaded is None:
            return
        try:
            threshold = float(self.deterministic_threshold_var.get())
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid threshold",
                "Enter a positive finite deterministic-outlier threshold.",
                parent=self.root,
            )
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            current_spectrum = next(
                (
                    spectrum
                    for spectrum in self.loaded.spectra
                    if spectrum.cycle == self.state.active_cycle
                ),
                None,
            )
            if current_spectrum is None:
                self._update_status("no active spectrum is available")
                return
            selected_rows = [(self.loaded.dataset_id, self.loaded, current_spectrum)]
        targets = [
            self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            for _dataset_id, loaded, spectrum in selected_rows
        ]
        unique_targets = list(dict.fromkeys(id(cycle) for cycle in targets))
        targets = [
            next(cycle for cycle in targets if id(cycle) == cycle_id)
            for cycle_id in unique_targets
        ]
        self.status_var.set(
            f"Detecting deterministic outliers in {len(targets)} spectrum(s)…"
        )

        def detect() -> list[tuple[object, np.ndarray | None, str | None]]:
            results = []
            for cycle in targets:
                if self._stop_event.is_set():
                    break
                try:
                    indices, _diagnostics = detect_outliers_in_active_points(
                        cycle.frequency_hz,
                        cycle.impedance,
                        cycle.included,
                        threshold=threshold,
                    )
                    results.append((cycle, indices, None))
                except Exception as error:
                    results.append((cycle, None, f"{type(error).__name__}: {error}"))
            return results

        self._submit(
            detect,
            lambda results: self._finish_deterministic_outliers(results),
            "Deterministic outlier detection failed",
            operation_labels=[f"cycle {cycle.cycle}" for cycle in targets],
            operation_name="Deterministic outlier detection",
        )

    def _finish_deterministic_outliers(self, results) -> None:
        removed = 0
        processed = 0
        errors = []
        for cycle, indices, error in results:
            if error is not None:
                errors.append(f"Cycle {cycle.cycle}: {error}")
                continue
            cycle.apply_outliers(indices)
            removed += int(indices.size)
            processed += 1
        self._restore_controls()
        self._refresh_explorer_values()
        self._refresh_plot(rescale=True)
        message = f"deterministic outliers: processed {processed} spectrum(s), removed {removed} point(s)"
        if errors:
            message += "; " + " | ".join(errors)
        self._update_status(message)

    def delete_selected_spectrum(self) -> None:
        if self.busy or self.state is None:
            return
        selected_items = [
            item
            for item in self.explorer.get_children("")
            if item in self.explorer.selection()
        ]
        if not selected_items:
            self._update_status("select one or more spectra in the explorer first")
            return
        rows = [(item, *self._explorer_rows[item]) for item in selected_items]
        spectrum_count = len(rows)
        dataset_count = len(
            {
                loaded.state.source_path.resolve()
                for _item, _dataset_id, loaded, _spectrum in rows
            }
        )
        if not messagebox.askyesno(
            "Delete spectra",
            (
                f"Delete {spectrum_count} selected spectra "
                f"from {dataset_count} file(s)?\n\n"
                "This removes the spectra from the opened project."
            ),
            parent=self.root,
        ):
            return

        current_dataset_id = self.current_dataset_id
        active_deleted = any(
            loaded is self.loaded
            and dataset_id == current_dataset_id
            and spectrum.cycle == self.state.active_cycle
            for _item, dataset_id, loaded, spectrum in rows
        )
        first_selected_index = min(
            list(self.explorer.get_children("")).index(item) for item in selected_items
        )
        if active_deleted and self.state is not None:
            if not self._capture_controls():
                return

        deleted_cycles_by_dataset: dict[str, set[int]] = {}
        for _item, dataset_id, loaded, spectrum in rows:
            deleted_cycles_by_dataset.setdefault(dataset_id, set()).add(spectrum.cycle)

        for dataset_id, deleted_cycles in deleted_cycles_by_dataset.items():
            loaded = self.loaded_projects[dataset_id]
            for cycle in deleted_cycles:
                loaded.state.cycles.pop(cycle, None)
            loaded.state.available_cycles = [
                cycle
                for cycle in loaded.state.available_cycles
                if cycle not in deleted_cycles
            ]
            loaded.spectra = [
                entry for entry in loaded.spectra if entry.cycle not in deleted_cycles
            ]
            if "cycle_number" in loaded.dataframe.columns:
                loaded.dataframe = loaded.dataframe.loc[
                    ~loaded.dataframe["cycle_number"].isin(deleted_cycles)
                ].copy()

        for dataset_id in list(self._dataset_order):
            loaded = self.loaded_projects.get(dataset_id)
            if loaded is None or loaded.state.available_cycles:
                continue
            self.loaded_projects.pop(dataset_id, None)
            self._dataset_order.remove(dataset_id)

        self._populate_explorer()

        if not self._dataset_order:
            self._clear_loaded_view("all spectra were deleted from the project")
            return

        if active_deleted:
            remaining_items = list(self.explorer.get_children(""))
            fallback_item = remaining_items[
                min(first_selected_index, len(remaining_items) - 1)
            ]
            dataset_id, loaded, spectrum = self._explorer_rows[fallback_item]
            self._switch_dataset(
                dataset_id,
                loaded,
                spectrum.cycle,
                capture_current=False,
            )
            return

        current_loaded = (
            self.loaded_projects.get(current_dataset_id)
            if current_dataset_id is not None
            else None
        )
        if current_loaded is None:
            fallback_dataset_id = self._dataset_order[0]
            fallback_loaded = self.loaded_projects[fallback_dataset_id]
            self._switch_dataset(
                fallback_dataset_id,
                fallback_loaded,
                fallback_loaded.state.active_cycle,
                capture_current=False,
            )
            return

        self.loaded = current_loaded
        self.state = current_loaded.state
        self._highlight_explorer_cycle(self.state.active_cycle, preserve_existing=False)
        self._update_status(
            f"deleted {spectrum_count} spectrum"
            f"{'' if spectrum_count == 1 else 's'}"
        )

    def _loaded_cycle_for_popup(self, loaded: LoadedProject, cycle_number: int):
        cycle = loaded.state.cycles.get(cycle_number)
        if cycle is None:
            cycle = load_cycle(loaded.dataframe, cycle_number, loaded.state.control)
            if loaded.state.all_frequency_window is not None:
                cycle.frequency_window = loaded.state.all_frequency_window
            cycle.parameters = loaded.state.parameters_for(cycle_number)
            cycle.circuit = loaded.state.circuit
            loaded.state.cycles[cycle_number] = cycle
        return cycle

    def _ensure_saved_drt_for_cycles(
        self,
        plotted_cycles,
        mode: str,
        *,
        force: bool,
        on_ready: Callable[[], None],
    ) -> None:
        missing = []
        for loaded, cycle in plotted_cycles:
            has_saved = (
                cycle.saved_ridge_tau_s is not None
                if mode == "ridge"
                else cycle.saved_hybrid_tau_s is not None
            )
            if force or not has_saved:
                missing.append((loaded, cycle))
        if not missing:
            on_ready()
            return
        if mode == "ridge":
            threshold = self._require_threshold_value()
            self.status_var.set(f"Calculating ridge DRT for {len(missing)} spectra...")
            def calculate_ridge():
                results = []
                for loaded, cycle in missing:
                    if self._stop_event.is_set():
                        break
                    results.append(
                        (
                            loaded,
                            cycle.cycle,
                            analyze_outliers(
                                cycle,
                                threshold,
                                loaded.state.parameters_for(cycle.cycle),
                            ),
                        )
                    )
                return results

            self._submit(
                calculate_ridge,
                lambda results: self._finish_saved_ridge_batch(results, on_ready),
                "Ridge DRT calculation failed",
                operation_labels=[f"cycle {cycle.cycle}" for _loaded, cycle in missing],
                operation_name="Ridge DRT calculation",
            )
            return
        self.status_var.set(f"Calculating hybrid DRT for {len(missing)} spectra...")

        def calculate_hybrid():
            results = []
            for loaded, cycle in missing:
                if self._stop_event.is_set():
                    break
                results.append((loaded, cycle.cycle, calculate_hybrid_drt(cycle)))
            return results

        self._submit(
            calculate_hybrid,
            lambda results: self._finish_saved_hybrid_batch(results, on_ready),
            "Hybrid DRT calculation failed",
            operation_labels=[f"cycle {cycle.cycle}" for _loaded, cycle in missing],
            operation_name="Hybrid DRT calculation",
        )

    def _finish_saved_ridge_batch(self, results, on_ready: Callable[[], None]) -> None:
        for loaded, cycle_number, analysis in results:
            cycle = self._loaded_cycle_for_popup(loaded, cycle_number)
            cycle.store_ridge_analysis(
                self._require_threshold_value(),
                analysis.outlier_indices,
                analysis.parameters,
                analysis.peak_count,
                analysis.ohmic_resistance,
                analysis.inductance,
                analysis.ridge_tau_s,
                analysis.ridge_gamma_ohm,
            )
        self._refresh_plot(rescale=True)
        on_ready()

    def _finish_saved_hybrid_batch(self, results, on_ready: Callable[[], None]) -> None:
        for loaded, cycle_number, result in results:
            cycle = self._loaded_cycle_for_popup(loaded, cycle_number)
            cycle.store_hybrid_drt(
                result.tau_s,
                result.gamma_ohm,
                result.ohmic_resistance,
            )
        self._refresh_plot(rescale=True)
        on_ready()

    def _open_spectra_popup(
        self,
        title: str,
        plotted_cycles,
        status_message: str,
    ) -> None:
        from matplotlib import colormaps
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("920x700")
        popup.minsize(640, 480)

        mode_state = {"value": self.plot_mode}
        controls = ttk.Frame(popup, padding=(8, 8, 8, 0))
        controls.pack(side=tk.TOP, fill=tk.X)
        toggle_button = ttk.Button(controls)
        toggle_button.pack(side=tk.LEFT)

        figure = Figure(figsize=(8.2, 6.2), dpi=100, constrained_layout=True)
        canvas = FigureCanvasTkAgg(figure, master=popup)
        self._attach_plot_export_menu(canvas, popup)
        popup_axes: dict[str, object | None] = {"main": None, "phase": None}

        def _render_popup() -> None:
            figure.clear()
            color_scale = colormaps["rainbow"]
            if mode_state["value"] == "bode":
                axes = figure.add_subplot(111)
                phase_axes = axes.twinx()
                axes.set_xscale("log")
                axes.set_xlabel("Frequency / Hz")
                axes.set_ylabel("|Z| / Ohm")
                phase_axes.set_ylabel("-Phase / deg")
                axes.grid(True, alpha=0.25)
                for index, (loaded, cycle) in enumerate(plotted_cycles):
                    color = color_scale(index / max(len(plotted_cycles) - 1, 1))
                    included = cycle.included
                    frequency = cycle.frequency_hz
                    magnitude = np.abs(cycle.impedance)
                    phase = self._phase_degrees(cycle.impedance)
                    label = f"{loaded.dataset_label} - cycle {cycle.cycle}"
                    axes.plot(
                        frequency[included],
                        magnitude[included],
                        "o-",
                        color=color,
                        markersize=3,
                        linewidth=1.1,
                        alpha=0.9,
                        label=f"{label} |Z|",
                    )
                    phase_axes.plot(
                        frequency[included],
                        phase[included],
                        "s-",
                        color=color,
                        markersize=2.6,
                        linewidth=0.9,
                        alpha=0.75,
                        label=f"{label} phase",
                    )
                    if np.any(~included):
                        axes.plot(
                            frequency[~included],
                            magnitude[~included],
                            "o--",
                            color=color,
                            markersize=2,
                            linewidth=0.8,
                            alpha=0.22,
                        )
                        phase_axes.plot(
                            frequency[~included],
                            phase[~included],
                            "s--",
                            color=color,
                            markersize=1.8,
                            linewidth=0.7,
                            alpha=0.16,
                        )
                limits = self._popup_bode_limits(plotted_cycles)
                if limits is not None:
                    x_min, x_max, y_min, y_max, phase_min, phase_max = limits
                    axes.set_xlim(x_min, x_max)
                    axes.set_ylim(y_min, y_max)
                    phase_axes.set_ylim(phase_min, phase_max)
                magnitude_handles, magnitude_labels = axes.get_legend_handles_labels()
                phase_handles, phase_labels = phase_axes.get_legend_handles_labels()
                axes.legend(
                    magnitude_handles + phase_handles,
                    magnitude_labels + phase_labels,
                    loc="best",
                )
            else:
                axes = figure.add_subplot(111)
                phase_axes = None
                axes.set_xlabel("Re(Z) / Ohm")
                axes.set_ylabel("-Im(Z) / Ohm")
                axes.set_aspect("equal", adjustable="box")
                axes.grid(True, alpha=0.25)
                axes.axhline(0.0, color="#444444", linewidth=1.2, alpha=0.85, zorder=0)
                axes.axvline(0.0, color="#444444", linewidth=1.2, alpha=0.85, zorder=0)
                for index, (loaded, cycle) in enumerate(plotted_cycles):
                    color = color_scale(index / max(len(plotted_cycles) - 1, 1))
                    included = cycle.included
                    label = f"{loaded.dataset_label} - cycle {cycle.cycle}"
                    axes.plot(
                        cycle.impedance.real[included],
                        -cycle.impedance.imag[included],
                        "o-",
                        color=color,
                        markersize=3,
                        linewidth=1.1,
                        alpha=0.9,
                        label=label,
                    )
                    if np.any(~included):
                        axes.plot(
                            cycle.impedance.real[~included],
                            -cycle.impedance.imag[~included],
                            "o--",
                            color=color,
                            markersize=2,
                            linewidth=0.8,
                            alpha=0.22,
                        )
                limits = self._popup_active_limits(plotted_cycles)
                if limits is not None:
                    x_min, x_max, y_min, y_max = limits
                    axes.set_xlim(x_min, x_max)
                    axes.set_ylim(y_min, y_max)
                axes.legend(loc="best")
            popup_axes["main"] = axes
            popup_axes["phase"] = phase_axes
            toggle_button.configure(
                text="Show Nyquist" if mode_state["value"] == "bode" else "Show Bode"
            )
            canvas.draw_idle()

        def _reset_popup_view() -> None:
            axes = popup_axes["main"]
            if axes is None:
                return
            if mode_state["value"] == "bode":
                popup_limits = self._popup_bode_limits(plotted_cycles)
                if popup_limits is None:
                    return
                x_min, x_max, y_min, y_max, phase_min, phase_max = popup_limits
                axes.set_xlim(x_min, x_max)
                axes.set_ylim(y_min, y_max)
                phase_axes = popup_axes["phase"]
                if phase_axes is not None:
                    phase_axes.set_ylim(phase_min, phase_max)
            else:
                popup_limits = self._popup_active_limits(plotted_cycles)
                if popup_limits is None:
                    return
                x_min, x_max, y_min, y_max = popup_limits
                axes.set_xlim(x_min, x_max)
                axes.set_ylim(y_min, y_max)
            canvas.draw_idle()

        def _toggle_popup_mode() -> None:
            mode_state["value"] = "nyquist" if mode_state["value"] == "bode" else "bode"
            _render_popup()

        toggle_button.configure(command=_toggle_popup_mode)
        _render_popup()
        toolbar = self._create_toolbar(canvas, popup, _reset_popup_view)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._update_status(status_message)

    def _open_drt_popup(
        self,
        title: str,
        plotted_cycles,
        status_message: str,
    ) -> None:
        from matplotlib import colormaps
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("920x700")
        popup.minsize(640, 480)

        mode_state = {"value": self._selected_drt_mode()}
        controls = ttk.Frame(popup, padding=(8, 8, 8, 0))
        controls.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(controls, text="DRT mode").pack(side=tk.LEFT, padx=(0, 6))
        mode_var = tk.StringVar(
            value="Ridge DRT" if mode_state["value"] == "ridge" else "Hybrid DRT"
        )
        mode_box = ttk.Combobox(
            controls,
            textvariable=mode_var,
            values=("Ridge DRT", "Hybrid DRT"),
            state="readonly",
            width=12,
        )
        mode_box.pack(side=tk.LEFT)

        figure = Figure(figsize=(8.2, 6.2), dpi=100, constrained_layout=True)
        canvas = FigureCanvasTkAgg(figure, master=popup)
        self._attach_plot_export_menu(canvas, popup)
        popup_axes: dict[str, object | None] = {"main": None}

        def _drt_limits(selected_mode: str):
            tau_segments = []
            gamma_segments = []
            for _loaded, cycle in plotted_cycles:
                tau_values = (
                    cycle.saved_ridge_tau_s if selected_mode == "ridge" else cycle.saved_hybrid_tau_s
                )
                gamma_values = (
                    cycle.saved_ridge_gamma_ohm if selected_mode == "ridge" else cycle.saved_hybrid_gamma_ohm
                )
                if tau_values is None or gamma_values is None:
                    continue
                tau_values = np.asarray(tau_values, dtype=float)
                gamma_values = np.asarray(gamma_values, dtype=float)
                finite = np.isfinite(tau_values) & np.isfinite(gamma_values) & (tau_values > 0)
                if not np.any(finite):
                    continue
                tau_segments.append(tau_values[finite])
                gamma_segments.append(gamma_values[finite])
            if not tau_segments:
                return None
            tau_values = np.concatenate(tau_segments)
            gamma_values = np.concatenate(gamma_segments)
            x_min = float(np.min(tau_values))
            x_max = float(np.max(tau_values))
            y_min = float(np.min(gamma_values))
            y_max = float(np.max(gamma_values))
            y_span = y_max - y_min
            y_padding = 0.08 * (y_span if y_span > 0 else max(abs(y_max), 1.0))
            if x_min == x_max:
                x_min /= 1.3
                x_max *= 1.3
            return x_min, x_max, y_min - y_padding, y_max + y_padding

        def _render_popup() -> None:
            figure.clear()
            axes = figure.add_subplot(111)
            popup_axes["main"] = axes
            axes.set_xscale("log")
            axes.set_xlabel("Tau / s")
            axes.set_ylabel("Gamma / Ohm")
            axes.grid(True, alpha=0.25)
            axes.axhline(0.0, color="#444444", linewidth=1.2, alpha=0.85, zorder=0)
            color_scale = colormaps["rainbow"]
            for index, (loaded, cycle) in enumerate(plotted_cycles):
                color = color_scale(index / max(len(plotted_cycles) - 1, 1))
                tau_values = (
                    cycle.saved_ridge_tau_s if mode_state["value"] == "ridge" else cycle.saved_hybrid_tau_s
                )
                gamma_values = (
                    cycle.saved_ridge_gamma_ohm if mode_state["value"] == "ridge" else cycle.saved_hybrid_gamma_ohm
                )
                if tau_values is None or gamma_values is None:
                    continue
                axes.plot(
                    tau_values,
                    gamma_values,
                    "-",
                    color=color,
                    linewidth=1.4,
                    alpha=0.9,
                    label=f"{loaded.dataset_label} - cycle {cycle.cycle}",
                )
            limits = _drt_limits(mode_state["value"])
            if limits is not None:
                x_min, x_max, y_min, y_max = limits
                axes.set_xlim(x_min, x_max)
                axes.set_ylim(y_min, y_max)
            axes.set_title("Ridge DRT" if mode_state["value"] == "ridge" else "Hybrid DRT")
            axes.legend(loc="best")
            canvas.draw_idle()

        def _reset_popup_view() -> None:
            axes = popup_axes["main"]
            if axes is None:
                return
            limits = _drt_limits(mode_state["value"])
            if limits is None:
                return
            x_min, x_max, y_min, y_max = limits
            axes.set_xlim(x_min, x_max)
            axes.set_ylim(y_min, y_max)
            canvas.draw_idle()

        def _after_mode_ready(selected_mode: str) -> None:
            if not popup.winfo_exists():
                return
            mode_state["value"] = selected_mode
            mode_var.set("Ridge DRT" if selected_mode == "ridge" else "Hybrid DRT")
            _render_popup()

        def _on_mode_change(_event=None) -> None:
            selected_mode = "hybrid" if mode_var.get() == "Hybrid DRT" else "ridge"
            self._ensure_saved_drt_for_cycles(
                plotted_cycles,
                selected_mode,
                force=False,
                on_ready=lambda: _after_mode_ready(selected_mode),
            )

        mode_box.bind("<<ComboboxSelected>>", _on_mode_change)
        canvas.draw()
        toolbar = self._create_toolbar(canvas, popup, _reset_popup_view)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        _render_popup()
        self._update_status(status_message)

    def plot_selected_spectra(self) -> None:
        if self.busy or self.state is None:
            return
        selected_items = [
            item
            for item in self.explorer.get_children("")
            if item in self.explorer.selection()
        ]
        if not selected_items:
            self._update_status("select one or more spectra in the explorer first")
            return
        if not self._capture_controls():
            return

        selected_cycles = []
        for item in selected_items:
            _dataset_id, loaded, spectrum = self._explorer_rows[item]
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            selected_cycles.append((loaded, cycle))
        self._open_spectra_popup(
            f"Selected spectra ({len(selected_cycles)})",
            selected_cycles,
            f"opened comparison plot for {len(selected_cycles)} selected spectra",
        )

    def plot_selected_drts(self) -> None:
        if self.busy or self.state is None:
            return
        selected_items = [
            item
            for item in self.explorer.get_children("")
            if item in self.explorer.selection()
        ]
        if not selected_items:
            self._update_status("select one or more spectra in the explorer first")
            return
        if not self._capture_controls():
            return
        selected_cycles = []
        for item in selected_items:
            _dataset_id, loaded, spectrum = self._explorer_rows[item]
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            selected_cycles.append((loaded, cycle))
        selected_mode = self._selected_drt_mode()
        self._ensure_saved_drt_for_cycles(
            selected_cycles,
            selected_mode,
            force=False,
            on_ready=lambda: self._open_drt_popup(
                f"Selected DRTs ({len(selected_cycles)})",
                selected_cycles,
                f"opened DRT comparison for {len(selected_cycles)} selected spectra",
            ),
        )

    def plot_three_electrode_spectra(self) -> None:
        if self.busy or self.state is None or self.loaded is None:
            return
        selected = self.explorer.selection()
        if not selected:
            self._update_status("select a spectrum in the explorer first")
            return
        if not self._capture_controls():
            return
        item = self.explorer.focus()
        if item not in selected:
            item = selected[-1]
        row = self._explorer_rows.get(item)
        if row is None:
            self._update_status("select a spectrum in the explorer first")
            return
        _dataset_id, loaded, spectrum = row
        source_path = loaded.state.source_path.resolve()
        grouped_projects = [
            project
            for project in self.loaded_projects.values()
            if project.state.source_path.resolve() == source_path
            and spectrum.cycle in project.state.available_cycles
            and project.state.control in {"cell", "working", "counter"}
        ]
        if len(grouped_projects) < 2:
            self._update_status(
                "this source does not contain the cell/working/counter trio"
            )
            return
        order = {"cell": 0, "working": 1, "counter": 2}
        plotted_cycles = [
            (project, self._loaded_cycle_for_popup(project, spectrum.cycle))
            for project in sorted(
                grouped_projects,
                key=lambda project: order.get(project.state.control, 99),
            )
        ]
        role_names = {
            project.spectra[0].custom_metadata.get(SPECTRUM_METADATA_COLUMN)
            for project, _cycle in plotted_cycles
            if project.spectra
        }
        if not {"Cell", "WE", "CE"}.issubset(role_names):
            self._update_status(
                "this source does not contain the full cell/working/counter trio"
            )
            return
        self._open_spectra_popup(
            f"Cycle {spectrum.cycle} - cell, working, counter",
            plotted_cycles,
            f"opened cell/working/counter comparison for cycle {spectrum.cycle}",
        )

    def save_mask(self) -> None:
        if self.busy or self.state is None:
            return
        default_name = (
            f"{self._current_stem()}_cycle{self.state.active_cycle}_mask_included.npy"
        )
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save included-point mask",
            initialdir=str(self._current_directory()),
            initialfile=default_name,
            defaultextension=".npy",
            filetypes=[("NumPy mask", "*.npy")],
        )
        if not selected:
            return
        np.save(Path(selected), self.state.active.included.astype(bool))
        self._update_status(f"saved {Path(selected).name}")

    def import_data(self) -> None:
        if self.busy:
            return
        selected = filedialog.askopenfilenames(
            parent=self.root,
            title="Add BioLogic impedance data",
            initialdir=str(self._dialog_directory("last_import_directory")),
            filetypes=[
                ("BioLogic MPT", "*.mpt"),
                ("BioLogic MPR", "*.mpr"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        self._remember_dialog_directory("last_import_directory", selected[0])
        selected_paths = list(
            dict.fromkeys(Path(value).resolve() for value in selected)
        )
        imported_paths = {
            loaded.state.source_path.resolve()
            for loaded in self.loaded_projects.values()
        }
        new_paths = [
            path for path in selected_paths if path not in imported_paths
        ]
        if not new_paths:
            self._update_status("all selected files are already imported")
            return
        if self.state is not None and not self._capture_controls():
            return
        circuit = (
            self.state.active.model(self.state.circuit)
            if self.state is not None
            else self.circuit
        )
        control = self.state.control if self.state is not None else self.control
        self.status_var.set(f"Importing {len(new_paths)} data files…")
        self._submit(
            lambda: [
                (path, inspect_eis_file_spectrum_kinds(path)) for path in new_paths
            ],
            lambda inspections: self._finish_import_inspection(
                inspections, new_paths, control, circuit
            ),
            "Data import failed",
        )

    def _finish_import_inspection(
        self,
        inspections: list[tuple[Path, list[str]]],
        paths: list[Path],
        control: str,
        circuit: str,
    ) -> None:
        selected_kinds = self._select_import_spectrum_kinds(inspections)
        if selected_kinds is None:
            self._update_status("data import cancelled")
            return
        self.status_var.set(f"Importing {len(paths)} data files…")
        self._submit(
            lambda: load_projects(
                paths,
                control,
                circuit,
                spectrum_kinds_by_path=selected_kinds,
            ),
            self._finish_imports,
            "Data import failed",
        )

    def _select_import_spectrum_kinds(
        self, inspections: list[tuple[Path, list[str]]]
    ) -> dict[Path, list[str]] | None:
        selected_kinds: dict[Path, list[str]] = {}
        apply_to_all_selection: list[str] | None = None
        for path, available in inspections:
            if len(available) <= 1:
                continue
            if apply_to_all_selection is not None:
                compatible = _compatible_spectrum_selection(
                    apply_to_all_selection, available
                )
                if compatible is not None:
                    selected_kinds[path.resolve()] = compatible
                    continue
            dialog = ElectrodeSelectionDialog(self.root, path, available)
            self.root.wait_window(dialog)
            if dialog.result is None:
                return None
            selection, apply_to_all = dialog.result
            selected_kinds[path.resolve()] = selection
            if apply_to_all:
                apply_to_all_selection = selection
        return selected_kinds

    def _finish_imports(self, report: ProjectImportReport) -> None:
        for dataset_id, loaded in report.loaded:
            self._register_dataset(dataset_id, loaded)
        skipped_messages = [
            f"{loaded.dataset_label}: skipped cycles without impedance data: "
            f"{', '.join(str(cycle) for cycle in loaded.skipped_cycles)}"
            for _dataset_id, loaded in report.loaded
            if loaded.skipped_cycles
        ]
        self._populate_explorer()
        if report.loaded:
            dataset_id, loaded = report.loaded[0]
            self._switch_dataset(
                dataset_id,
                loaded,
                loaded.state.active_cycle,
                capture_current=False,
            )
        warning_details = [
            f"{path.name}: {error}" for path, error in report.errors
        ]
        warning_details.extend(skipped_messages)
        if warning_details:
            details = "\n".join(warning_details)
            messagebox.showwarning(
                "Some cycles were skipped" if skipped_messages and not report.errors else "Some data were not imported",
                details,
                parent=self.root,
            )
        self._update_status(
            f"added {len(report.loaded)} files; "
            f"{len(self._explorer_rows)} spectra loaded"
        )

    def _project_signature(self) -> str | None:
        if self.state is None:
            return None
        datasets = []
        for dataset_id in self._dataset_order:
            loaded = self.loaded_projects.get(dataset_id)
            if loaded is None:
                continue
            datasets.append(
                {
                    "dataset_id": dataset_id,
                    "state": _state_to_payload(loaded.state),
                    "dataframe": _dataframe_to_payload(loaded.dataframe),
                }
            )
        if not datasets:
            datasets.append(
                {
                    "dataset_id": self.current_dataset_id,
                    "state": _state_to_payload(self.state),
                    "dataframe": _dataframe_to_payload(self.loaded.dataframe),
                }
            )
        return json.dumps(datasets, sort_keys=True, default=str)

    def _project_has_unsaved_changes(self) -> bool:
        if self.state is None:
            return False
        current_signature = self._project_signature()
        return (
            self._saved_project_signature is None
            or current_signature != self._saved_project_signature
        )

    def _cancel_fit(self) -> None:
        if not self.busy:
            return
        self._fit_cancel_requested = True
        self._stop_event.set()
        self._update_status("Stop requested - finishing the current spectrum")

    def save_project(self, path: Path | None = None) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        if path is None:
            selected = filedialog.asksaveasfilename(
                parent=self.root,
                title="Save EIS fitting project",
                initialdir=str(self._current_directory()),
                initialfile=f"{self._current_stem()}.eisfit.json.gz",
                defaultextension=".eisfit.json.gz",
                filetypes=[
                    ("Compressed EIS fitting project", "*.eisfit.json.gz"),
                    ("EIS fitting project", "*.eisfit.json"),
                    ("JSON", "*.json"),
                ],
            )
            if not selected:
                return
            project_path = Path(selected)
        else:
            project_path = Path(path)
        try:
            ordered_dataset_ids = [
                self.current_dataset_id,
                *(
                    dataset_id
                    for dataset_id in self._dataset_order
                    if dataset_id != self.current_dataset_id
                ),
            ]
            datasets = [
                (
                    dataset_id,
                    self.loaded_projects[dataset_id].state,
                    self.loaded_projects[dataset_id].dataframe,
                )
                for dataset_id in ordered_dataset_ids
                if dataset_id is not None and dataset_id in self.loaded_projects
            ]
            save_project_file(
                self.state,
                project_path,
                datasets=datasets,
                procedure_blocks=self.procedure_blocks,
                procedures=self.procedures,
            )
        except Exception as error:
            messagebox.showerror(
                "Project save failed",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            return
        self.project_path = project_path.resolve()
        self._saved_project_signature = self._project_signature()
        self.root.title(
            f"EIS Fitting — {Path(project_path).name if project_path else 'Untitled'}"
        )
        self._update_status(f"project saved as {project_path.name}")

    def save_project_quick(self, _event=None):
        if self.project_path is not None:
            self.save_project(self.project_path)
        else:
            self.save_project()
        return "break"

    def save_project_as(self) -> None:
        self.save_project(None)

    def _on_control_s(self, event):
        if event.state & 0x0001 or event.keysym == "S":
            self.save_project_as()
        else:
            self.save_project_quick(event)
        return "break"

    def load_relaxis_project(self) -> None:
        if self.busy:
            return
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Load RelaxIS 3 project",
            initialdir=str(self._current_directory()),
            filetypes=[("RelaxIS 3 project", "*.eis3"), ("All files", "*.*")],
        )
        if not selected:
            return
        source_path = Path(selected).resolve()
        self.status_var.set(f"Loading RelaxIS 3 project {source_path.name}â€¦")
        self._submit(
            lambda: self._convert_relaxis_project(source_path),
            lambda result: self._finish_relaxis_project_load(result, source_path),
            "RelaxIS project load failed",
        )

    def _convert_relaxis_project(
        self, source_path: Path
    ) -> tuple[list[tuple[str, LoadedProject, ProjectState]], Path]:
        temporary_directory = Path(
            tempfile.mkdtemp(prefix="eis_fitting_relaxis_")
        )
        try:
            self._relaxis_model_override = None
            converted_path = export_to_eisfit_json(
                source_path,
                temporary_directory,
                model_conflict_handler=self._resolve_relaxis_model_conflict,
                unmapped_model_handler=self._resolve_unmapped_relaxis_model,
            )
            result = self._load_saved_project(converted_path)
            return result, temporary_directory
        except Exception:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise

    def _resolve_relaxis_model_conflict(
        self, datasource: str, instances: list[dict]
    ) -> list[dict] | int | str:
        override = getattr(self, "_relaxis_model_override", None)
        if override is not None:
            return dict(override)
        result: dict[str, object] = {}
        finished = threading.Event()

        def show_dialog() -> None:
            dialog = tk.Toplevel(self.root)
            dialog.title("Resolve duplicate RelaxIS spectrum")
            dialog.transient(self.root)
            dialog.grab_set()
            dialog.resizable(False, False)
            ttk.Label(
                dialog,
                text=f"Multiple EEC models were found for:\n{datasource}",
                justify=tk.LEFT,
            ).pack(anchor="w", padx=16, pady=(16, 10))
            choice = tk.StringVar(value="all")
            ttk.Radiobutton(
                dialog,
                text="Load all model instances",
                variable=choice,
                value="all",
            ).pack(anchor="w", padx=16, pady=3)
            ttk.Radiobutton(
                dialog,
                text="Load only this model:",
                variable=choice,
                value="one",
            ).pack(anchor="w", padx=16, pady=(8, 2))
            models = [
                f"{item.get('model', '')} (file {item.get('file_id', '')})"
                for item in instances
            ]
            selected_model = tk.StringVar(value=models[0] if models else "")
            ttk.Combobox(
                dialog,
                textvariable=selected_model,
                values=models,
                state="readonly",
                width=48,
            ).pack(anchor="w", padx=32, pady=(0, 12))
            apply_all = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                dialog,
                text="Apply selected model for all spectra in this import",
                variable=apply_all,
            ).pack(anchor="w", padx=16, pady=(0, 10))

            def accept() -> None:
                if choice.get() == "all":
                    result["value"] = "all"
                else:
                    index = next(
                        index
                        for index, label in enumerate(models)
                        if label == selected_model.get()
                    )
                    selected_instance = instances[index]
                    result["value"] = {"index": index}
                    if apply_all.get():
                        self._relaxis_model_override = {
                            "model": selected_instance.get("model"),
                            "circuit": selected_instance.get("circuit"),
                        }
                dialog.destroy()

            ttk.Button(dialog, text="Continue", command=accept).pack(
                anchor="e", padx=16, pady=(0, 14)
            )
            dialog.protocol("WM_DELETE_WINDOW", accept)
            dialog.wait_window()
            finished.set()

        self.root.after(0, show_dialog)
        finished.wait()
        return result.get("value", "all")

    def _resolve_unmapped_relaxis_model(self, model: str, instance: dict) -> dict:
        """Explicitly select a known EIS-fitting circuit for an unknown label."""
        override = getattr(self, "_relaxis_model_override", None)
        if override is not None:
            return dict(override)
        result: dict[str, object] = {}
        finished = threading.Event()

        def show_dialog() -> None:
            dialog = tk.Toplevel(self.root)
            dialog.title("Choose EEC model for RelaxIS import")
            dialog.transient(self.root)
            dialog.grab_set()
            ttk.Label(
                dialog,
                text=f"RelaxIS model {model!r} cannot be mapped automatically.\n"
                "Select an EIS-fitting circuit explicitly:",
                justify=tk.LEFT,
            ).pack(anchor="w", padx=16, pady=(16, 10))
            selected = tk.StringVar(value=self._model_presets[0] if self._model_presets else "")
            ttk.Combobox(
                dialog, textvariable=selected, values=self._model_presets,
                state="readonly", width=48,
            ).pack(anchor="w", padx=16, pady=(0, 12))
            apply_all = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                dialog,
                text="Apply selected model for all remaining unresolved spectra",
                variable=apply_all,
            ).pack(anchor="w", padx=16, pady=(0, 10))

            def accept() -> None:
                result["value"] = {"circuit": selected.get(), "model": selected.get()}
                if apply_all.get():
                    self._relaxis_model_override = dict(result["value"])
                dialog.destroy()

            def cancel() -> None:
                result["value"] = {"circuit": None}
                dialog.destroy()

            buttons = ttk.Frame(dialog)
            buttons.pack(anchor="e", padx=16, pady=(0, 14))
            ttk.Button(buttons, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=(0, 8))
            ttk.Button(buttons, text="Continue", command=accept).pack(side=tk.LEFT)
            dialog.protocol("WM_DELETE_WINDOW", cancel)
            dialog.wait_window()
            finished.set()

        self.root.after(0, show_dialog)
        finished.wait()
        selected = result.get("value", {"circuit": None})
        if not selected.get("circuit"):
            raise ValueError(f"No EIS-fitting circuit selected for RelaxIS model {model!r}")
        return selected

    def _finish_relaxis_project_load(
        self,
        result: tuple[list[tuple[str, LoadedProject, ProjectState]], Path],
        source_path: Path,
    ) -> None:
        loaded_result, temporary_directory = result
        converted_path = temporary_directory / f"{source_path.stem}.eisfit.json"
        try:
            self._finish_project_load(loaded_result, converted_path)
            # The converted file is temporary; the imported state is an unsaved
            # native project and can be saved through the normal Save command.
            self.project_path = None
            self._project_title_path = source_path
            self._saved_project_signature = self._project_signature()
            self._update_window_title()
            spectrum_count = sum(len(loaded.spectra) for _, loaded, _ in loaded_result)
            self._update_status(
                f"Loaded {spectrum_count} spectra from RelaxIS project {source_path.name}"
            )
        finally:
            shutil.rmtree(temporary_directory, ignore_errors=True)

    def load_project(self) -> None:
        if self.busy:
            return
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Load EIS fitting project",
            initialdir=str(self._dialog_directory("last_project_directory")),
            filetypes=[
                ("Compressed EIS fitting project", "*.eisfit.json.gz"),
                ("Compressed EIS fitting project", "*.eisfit.gz"),
                ("EIS fitting project", "*.eisfit.json"),
                ("JSON", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        self._remember_dialog_directory("last_project_directory", selected)
        project_path = Path(selected)
        self.status_var.set(f"Loading project {project_path.name}…")
        self._submit(
            lambda: self._load_saved_project(project_path),
                    lambda result: (
                        setattr(self, "project_path", Path(project_path)),
                        setattr(self, "_project_title_path", Path(project_path)),
                        self._finish_project_load(result, project_path),
                        self._update_window_title(),
                    )[1],
            "Project load failed",
        )

    @staticmethod
    def _load_ml_results_project(
        path: Path, payload: dict
    ) -> list[tuple[str, LoadedProject, ProjectState]]:
        import pandas as pd

        root = payload.get("ml_results", payload)
        spectra = root.get("spectra") if isinstance(root, dict) else None
        if not isinstance(spectra, list) or not spectra:
            raise ValueError("The ML-results file contains no spectra")
        grouped: dict[str, list[dict]] = {}
        for spectrum in spectra:
            if not isinstance(spectrum, dict):
                continue
            spectrum_id = str(spectrum.get("spectrum_id") or "")
            source_name = str(spectrum.get("source_name") or "").strip()
            if not source_name:
                parts = spectrum_id.split("::")
                source_name = parts[1] if len(parts) >= 4 else str(root.get("source_file") or path.name)
            grouped.setdefault(source_name, []).append(spectrum)
        restored = []
        for source_name, source_spectra in grouped.items():
            rows = []
            circuit = None
            metadata_by_cycle: dict[int, dict[str, object]] = {}
            for spectrum in source_spectra:
                frequency = np.asarray(spectrum.get("frequency", []), dtype=float)
                real = np.asarray(spectrum.get("z_real", []), dtype=float)
                imaginary = np.asarray(spectrum.get("z_imag", []), dtype=float)
                if frequency.size == 0 or frequency.size != real.size or real.size != imaginary.size:
                    continue
                cycle = int(spectrum.get("cycle", len(metadata_by_cycle) + 1))
                if cycle in metadata_by_cycle:
                    raise ValueError(f"Duplicate ML spectrum key ({source_name!r}, {cycle})")
                circuit = circuit or str(spectrum.get("existing_eec_topology") or "").strip()
                metadata = dict(spectrum.get("metadata") or {})
                metadata.setdefault("source_name", source_name)
                metadata_by_cycle[cycle] = metadata
                for frequency_value, real_value, imaginary_value in zip(frequency, real, imaginary):
                    rows.append({
                        "freq_hz": frequency_value,
                        "cycle_number": cycle,
                        "time_s": spectrum.get("time"),
                        "i_ma": spectrum.get("current"),
                        "ewe_v": spectrum.get("voltage"),
                        "re_z_ohm": real_value,
                        "minus_im_z_ohm": -imaginary_value,
                    })
            if not rows:
                continue
            circuit = circuit or "R0-L0-p(R1,CPE1)"
            dataframe = pd.DataFrame(rows)
            loaded = load_project_from_dataframe(
                dataframe,
                path,
                min(dataframe["cycle_number"].astype(int)),
                "working",
                circuit,
                technique="ML results",
            )
            loaded.dataset_id = f"ml-results::{source_name}"
            loaded.dataset_label = f"{source_name} [ML results]"
            for spectrum in source_spectra:
                cycle_number = int(spectrum.get("cycle", -1))
                if cycle_number not in loaded.state.available_cycles:
                    continue
                cycle = load_cycle(dataframe, cycle_number, "working")
                cycle.circuit = circuit
                cycle.parameters = loaded.state.parameters_for(cycle_number)
                cycle.custom_metadata.update(metadata_by_cycle.get(cycle_number, {}))
                for key in ("ml_envelope_mask", "stage2_active_mask", "stage1_active_mask"):
                    mask = spectrum.get(key)
                    if mask is not None and len(mask) == cycle.frequency_hz.size:
                        cycle.manually_included = np.asarray(mask, dtype=bool)
                        break
                minimum = spectrum.get("predicted_f_min")
                maximum = spectrum.get("predicted_f_max")
                try:
                    if float(minimum) > 0 and float(maximum) > 0:
                        cycle.frequency_window = tuple(sorted((float(minimum), float(maximum))))
                except (TypeError, ValueError):
                    pass
                loaded.state.cycles[cycle_number] = cycle
                for metadata_row in loaded.spectra:
                    if metadata_row.cycle == cycle_number:
                        metadata_row.custom_metadata.update(cycle.custom_metadata)
                        break
            loaded.state.active_cycle = min(loaded.state.available_cycles)
            restored.append((loaded.dataset_id, loaded, loaded.state))
        if not restored:
            raise ValueError("The ML-results file contains no usable impedance data")
        return restored

    @staticmethod
    def _load_saved_project(
        path: Path,
    ) -> list[tuple[str, LoadedProject, ProjectState]]:
        payload = load_json_payload(path)
        if payload.get("format") == "eis-fitting-ml-results":
            raise ValueError(
                "This is an ML-results sidecar, not an EIS-fit project. "
                "Open the associated .eisfit project and load the sidecar from the ML controls."
            )
        is_ml_results = (
            payload.get("format") != "eis-fitting-project"
            and "ml_results" in payload
        ) or (
            isinstance(payload.get("spectra"), list)
            and "source_path" not in payload
            and "datasets" not in payload
        )
        if is_ml_results:
            return EISApplication._load_ml_results_project(path, payload)
        embedded_datasets = payload.get("datasets")
        if embedded_datasets:
            restored_projects = []
            for dataset in embedded_datasets:
                state_payload = dataset["state"]
                source_path = Path(str(state_payload["source_path"]))
                dataframe = dataframe_from_payload(dataset["dataframe"])
                loaded = load_project_from_dataframe(
                    dataframe,
                    source_path,
                    int(state_payload.get("active_cycle", 1)),
                    str(state_payload.get("control", "working")),
                    str(state_payload["circuit"]),
                )
                restored = load_project_file(
                    loaded.state,
                    dataframe,
                    path,
                    payload=state_payload,
                )
                loaded.state = restored
                loaded.spectra = catalog_spectra(
                    dataframe,
                    restored.available_cycles,
                    restored.control,
                    {
                        cycle_number: {
                            **cycle.custom_metadata,
                            "time_s": cycle.time_s,
                        }
                        for cycle_number, cycle in restored.cycles.items()
                    },
                )
                restored_projects.append(
                    (str(dataset["dataset_id"]), loaded, restored)
                )
            if not restored_projects:
                raise ValueError("The project contains no embedded datasets")
            return restored_projects
        source_path = Path(str(payload["source_path"]))
        if not source_path.is_absolute():
            source_path = (path.parent / source_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"The source data file saved in this project was not found: {source_path}"
            )
        circuit = str(payload["circuit"])
        control = str(payload.get("control", "cell"))
        active_cycle = int(payload.get("active_cycle", 1))
        report = load_projects([source_path], control, circuit, active_cycle)
        if report.errors:
            raise ValueError(report.errors[0][1])
        if not report.loaded:
            raise ValueError(f"No spectra could be loaded from {source_path.name}")
        dataset_id, loaded = report.loaded[0]
        restored = load_project_file(loaded.state, loaded.dataframe, path)
        loaded.state = restored
        loaded.spectra = catalog_spectra(
            loaded.dataframe,
            restored.available_cycles,
            restored.control,
            {
                cycle_number: {
                    **cycle.custom_metadata,
                    "time_s": cycle.time_s,
                }
                for cycle_number, cycle in restored.cycles.items()
            },
        )
        return [(dataset_id, loaded, restored)]

    def _finish_project_load(
        self,
        result: list[tuple[str, LoadedProject, ProjectState]],
        path: Path,
    ) -> None:
        self.project_path = path.resolve()
        self.loaded_projects.clear()
        self._dataset_order.clear()
        for dataset_id, loaded, _restored in result:
            self._register_dataset(dataset_id, loaded)
        dataset_id, loaded, restored = result[0]
        self.current_dataset_id = dataset_id
        self.loaded = loaded
        self.state = restored
        payload = load_json_payload(path)
        self.procedure_blocks, self.procedures = self._validate_procedure_data(
            payload.get("procedure_blocks"), payload.get("procedures")
        )
        self._saved_project_signature = self._project_signature()
        self.control = restored.control
        self.circuit = restored.circuit
        self.model_var.set(restored.circuit)
        self.cycle_var.set(str(restored.active_cycle))
        associated_results = path
        if path.name.endswith(".eisfit.json.gz"):
            associated_results = path.with_name(
                path.name.removesuffix(".eisfit.json.gz") + "_ml_results.json"
            )
        elif path.name.endswith(".eisfit.gz"):
            associated_results = path.with_name(
                path.name.removesuffix(".eisfit.gz") + "_ml_results.json"
            )
        elif path.name.endswith(".eisfit.json"):
            associated_results = path.with_name(
                path.name.removesuffix(".eisfit.json") + "_ml_results.json"
            )
        if "ml_results" in payload:
            associated_results = path
        if associated_results.is_file():
            self.ml_results = load_ml_results(associated_results)
            self.ml_results_directory = associated_results.resolve()
            self.ml_results_status_var.set(
                f"Loaded {len(self.ml_results)} ML results"
                if self.ml_results
                else "No ML results found"
            )
        else:
            self.ml_results = {}
            self.ml_results_directory = None
            self.ml_results_status_var.set("No ML results loaded")
        self._populate_explorer()
        self._highlight_explorer_cycle(restored.active_cycle)
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._update_status(f"project loaded from {path.name}")

    def _all_export_states(self) -> list[ProjectState]:
        return [self.loaded_projects[dataset_id].state for dataset_id in self._dataset_order]

    def _selected_export_states(self) -> list[ProjectState]:
        if self.state is None:
            return []
        visible_items = list(self.explorer.get_children(""))
        selected_items = [
            item for item in visible_items if item in self.explorer.selection()
        ]
        if not selected_items:
            return []
        selected_cycles_by_dataset: dict[str, list[int]] = {}
        for item in selected_items:
            dataset_id, _loaded, spectrum = self._explorer_rows[item]
            selected_cycles_by_dataset.setdefault(dataset_id, []).append(spectrum.cycle)
        selected_states: list[ProjectState] = []
        for dataset_id in self._dataset_order:
            cycle_numbers = selected_cycles_by_dataset.get(dataset_id)
            if not cycle_numbers:
                continue
            loaded = self.loaded_projects[dataset_id]
            unique_cycles = list(dict.fromkeys(cycle_numbers))
            cycles = {
                cycle_number: self._loaded_cycle_for_popup(loaded, cycle_number)
                for cycle_number in unique_cycles
            }
            if not cycles:
                continue
            selected_states.append(
                ProjectState(
                    source_path=loaded.state.source_path,
                    circuit=loaded.state.circuit,
                    control=loaded.state.control,
                    available_cycles=unique_cycles,
                    active_cycle=unique_cycles[0],
                    default_parameters=loaded.state.parameters_for(unique_cycles[0]),
                    cycles=cycles,
                    all_frequency_window=loaded.state.all_frequency_window,
                )
            )
        return selected_states

    def _selected_spectrum_rows(self) -> list[tuple[str, LoadedProject, SpectrumMetadata]]:
        if self.state is None:
            return []
        visible_items = list(self.explorer.get_children(""))
        selected_items = [
            item for item in visible_items if item in self.explorer.selection()
        ]
        return [
            (dataset_id, loaded, spectrum)
            for item in selected_items
            for dataset_id, loaded, spectrum in [self._explorer_rows[item]]
        ]

    def _selected_project_batches(self) -> dict[str, tuple[LoadedProject, ProjectState]]:
        grouped_rows: dict[str, tuple[LoadedProject, list[int]]] = {}
        for dataset_id, loaded, spectrum in self._selected_spectrum_rows():
            if dataset_id not in grouped_rows:
                grouped_rows[dataset_id] = (loaded, [])
            grouped_rows[dataset_id][1].append(spectrum.cycle)
        batches: dict[str, tuple[LoadedProject, ProjectState]] = {}
        for dataset_id, (loaded, cycle_numbers) in grouped_rows.items():
            unique_cycles = list(dict.fromkeys(cycle_numbers))
            batches[dataset_id] = (
                loaded,
                ProjectState(
                    source_path=loaded.state.source_path,
                    circuit=loaded.state.circuit,
                    control=loaded.state.control,
                    available_cycles=unique_cycles,
                    active_cycle=unique_cycles[0],
                    default_parameters=loaded.state.default_parameters,
                    cycles={
                        cycle_number: self._loaded_cycle_for_popup(loaded, cycle_number)
                        for cycle_number in unique_cycles
                    },
                    all_frequency_window=loaded.state.all_frequency_window,
                ),
            )
        return batches

    def _cached_ridge_analysis(
        self,
        cycle,
        threshold: float,
        parameters: list[ParameterValue],
    ) -> RidgeInitialization | None:
        parameter_names = [parameter.name for parameter in parameters]
        if not cycle.ridge_cache_matches(threshold, parameter_names):
            return None
        return RidgeInitialization(
            outlier_indices=cycle.saved_ridge_outlier_indices.copy(),
            parameters=[
                ParameterValue(
                    value.name,
                    value.unit,
                    value.initial,
                    value.lower,
                    value.upper,
                    value.error_percent,
                    value.fixed,
                )
                for value in cycle.saved_ridge_parameters
            ],
            peak_count=int(cycle.saved_ridge_peak_count or 0),
            ohmic_resistance=float(cycle.saved_ridge_ohmic_resistance or 0.0),
            inductance=float(cycle.saved_ridge_inductance or 0.0),
            ridge_tau_s=cycle.saved_ridge_tau_s.copy(),
            ridge_gamma_ohm=cycle.saved_ridge_gamma_ohm.copy(),
        )

    def _require_threshold_value(self) -> float:
        return float(self.threshold_var.get())

    def _selected_drt_mode(self) -> str:
        return "hybrid" if self.drt_mode_var.get() == "Hybrid DRT" else "ridge"

    def _apply_saved_drt_mode(self, cycle) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        mode = self._selected_drt_mode()
        if mode == "hybrid":
            if cycle.saved_hybrid_tau_s is not None and cycle.saved_hybrid_gamma_ohm is not None:
                cycle.show_hybrid_drt()
                return cycle.ridge_tau_s, cycle.ridge_gamma_ohm, "Hybrid DRT"
            cycle.ridge_tau_s = None
            cycle.ridge_gamma_ohm = None
            return None, None, "Hybrid DRT (not saved)"
        if cycle.saved_ridge_tau_s is not None and cycle.saved_ridge_gamma_ohm is not None:
            cycle.show_ridge_drt()
            return cycle.ridge_tau_s, cycle.ridge_gamma_ohm, "Ridge DRT"
        cycle.ridge_tau_s = None
        cycle.ridge_gamma_ohm = None
        return None, None, "Ridge DRT (not saved)"

    def _ensure_current_drt_mode(self) -> None:
        if self.state is None or not self._capture_controls():
            return
        cycle = self.state.active
        mode = self._selected_drt_mode()
        if mode == "ridge":
            threshold = self._require_threshold_value()
            parameters = self.state.parameters_for(cycle.cycle)
            cached = self._cached_ridge_analysis(cycle, threshold, parameters)
            if cached is not None:
                cycle.store_ridge_analysis(
                    threshold,
                    cached.outlier_indices,
                    cached.parameters,
                    cached.peak_count,
                    cached.ohmic_resistance,
                    cached.inductance,
                    cached.ridge_tau_s,
                    cached.ridge_gamma_ohm,
                )
                self._refresh_plot(rescale=True)
                self._update_status("displaying saved ridge DRT")
                return
            cycle_number = cycle.cycle
            self.status_var.set(f"Cycle {cycle_number} · calculating ridge DRT...")
            self._submit(
                lambda: analyze_outliers(cycle, threshold, parameters),
                lambda analysis: self._finish_ridge_drt(cycle_number, analysis),
                "Ridge DRT calculation failed",
            )
            return
        if cycle.hybrid_cache_matches():
            cycle.show_hybrid_drt()
            self._refresh_plot(rescale=True)
            self._update_status("displaying saved hybrid DRT")
            return
        cycle_number = cycle.cycle
        self.status_var.set(f"Cycle {cycle_number} · calculating hybrid DRT...")
        self._submit(
            lambda: calculate_hybrid_drt(cycle),
            lambda result: self._finish_hybrid_drt(cycle_number, result),
            "Hybrid DRT calculation failed",
        )

    def _ensure_kk_residuals(self) -> None:
        if self.state is None or not self.show_kk_var.get():
            return
        cycle = self.state.active
        if cycle.kk_cache_matches():
            self._refresh_plot(rescale=True)
            self._update_status("displaying saved Lin-KK residuals")
            return
        cycle_number = cycle.cycle
        self.status_var.set(f"Cycle {cycle_number} · calculating Lin-KK residuals...")
        self._submit(
            lambda: calculate_lin_kk_residuals(cycle),
            lambda result: self._finish_kk_residuals(cycle_number, result),
            "Lin-KK calculation failed",
        )

    def _finish_kk_residuals(self, cycle_number: int, result: KKResiduals) -> None:
        if self.state is None:
            return
        cycle = self.state.cycles[cycle_number]
        cycle.store_kk_result(
            result.fit_impedance,
            result.residual_real,
            result.residual_imag,
        )
        if self.state.active_cycle == cycle_number:
            self._refresh_plot(rescale=True)
            self._update_status("Lin-KK residuals calculated")

    def export_fits(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export fitted parameters",
            initialdir=str(self._current_directory()),
            initialfile=f"{self._current_stem()}_fit_parameters.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not selected:
            return
        try:
            count = export_fit_parameters_for_states(
                self._all_export_states(),
                Path(selected),
            )
        except Exception as error:
            messagebox.showerror(
                "Fit export failed",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            return
        self._update_status(f"exported fit parameters for {count} spectra")

    def export_selected_fits(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        states = self._selected_export_states()
        if not states:
            self._update_status("select one or more spectra in the explorer first")
            return
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export fitted parameters for selected spectra",
            initialdir=str(self._current_directory()),
            initialfile=f"{self._current_stem()}_selected_fit_parameters.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not selected:
            return
        try:
            count = export_fit_parameters_for_states(states, Path(selected))
        except Exception as error:
            messagebox.showerror(
                "Selected fit export failed",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            return
        self._update_status(f"exported fit parameters for {count} selected spectra")

    def export_python_workspace(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export parameters and metadata for Python",
            initialdir=str(self._current_directory()),
            initialfile=f"{self._current_stem()}_analysis.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not selected:
            return
        export_path = Path(selected)
        script_path = export_path.with_suffix(".py")
        if script_path.exists() and not messagebox.askyesno(
            "Replace Python script?",
            f"{script_path.name} already exists. Replace it?",
            parent=self.root,
        ):
            return
        states = self._all_export_states()
        try:
            count, script_path = write_python_workspace(states, export_path)
            editor = self._open_python_script(script_path)
        except Exception as error:
            messagebox.showerror(
                "Python export failed",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            return
        self._update_status(
            f"exported {count} fitted spectra and opened "
            f"{script_path.name} in {editor}"
        )

    def export_selected_python_workspace(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        states = self._selected_export_states()
        if not states:
            self._update_status("select one or more spectra in the explorer first")
            return
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export selected parameters and metadata for Python",
            initialdir=str(self._current_directory()),
            initialfile=f"{self._current_stem()}_selected_analysis.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not selected:
            return
        export_path = Path(selected)
        script_path = export_path.with_suffix(".py")
        if script_path.exists() and not messagebox.askyesno(
            "Replace Python script?",
            f"{script_path.name} already exists. Replace it?",
            parent=self.root,
        ):
            return
        try:
            count, script_path = write_python_workspace(states, export_path)
            editor = self._open_python_script(script_path)
        except Exception as error:
            messagebox.showerror(
                "Selected Python export failed",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            return
        self._update_status(
            f"exported {count} selected fitted spectra and opened "
            f"{script_path.name} in {editor}"
        )

    def export_drts(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export saved DRTs",
            initialdir=str(self._current_directory()),
            initialfile=f"{self._current_stem()}_drts.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not selected:
            return
        try:
            count = export_drts_for_states(self._all_export_states(), Path(selected))
        except Exception as error:
            messagebox.showerror(
                "DRT export failed",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            return
        self._update_status(f"exported {count} saved DRT points")

    def export_selected_drts(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        states = self._selected_export_states()
        if not states:
            self._update_status("select one or more spectra in the explorer first")
            return
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export saved DRTs for selected spectra",
            initialdir=str(self._current_directory()),
            initialfile=f"{self._current_stem()}_selected_drts.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not selected:
            return
        try:
            count = export_drts_for_states(states, Path(selected))
        except Exception as error:
            messagebox.showerror(
                "Selected DRT export failed",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            return
        self._update_status(f"exported {count} saved DRT points for selected spectra")

    @staticmethod
    def _open_python_script(path: Path) -> str:
        local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
        program_files = Path(os.environ.get("ProgramFiles", ""))
        code_executables = (
            local_app_data / "Programs" / "Microsoft VS Code" / "Code.exe",
            program_files / "Microsoft VS Code" / "Code.exe",
        )
        for executable in code_executables:
            if executable.is_file():
                subprocess.Popen([str(executable), str(path)], cwd=path.parent)
                return "VS Code"
        for command, label in (
            ("code", "VS Code"),
            ("codium", "VSCodium"),
            ("pycharm64", "PyCharm"),
            ("spyder", "Spyder"),
        ):
            executable = shutil.which(command)
            if executable:
                subprocess.Popen([executable, str(path)], cwd=path.parent)
                return label
        if os.name == "nt":
            try:
                os.startfile(str(path), "edit")
                return "the configured editor"
            except OSError:
                subprocess.Popen(["notepad.exe", str(path)], cwd=path.parent)
                return "Notepad"
        raise RuntimeError("No Python editor was found")

    def close(self) -> None:
        if self._project_has_unsaved_changes():
            decision = messagebox.askyesnocancel(
                "Unsaved changes",
                "The project has unsaved changes. Save before closing?",
                parent=self.root,
            )
            if decision is None:
                return
            if decision:
                if self.project_path is not None:
                    self.save_project(self.project_path)
                else:
                    self.save_project()
                if self._project_has_unsaved_changes():
                    return
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def launch_nyquist_editor(
    *,
    mpt_path: Path | None = None,
    cycle: int = 1,
    control: str = "cell",
    outlier_threshold: float = 1.0,
    circuit: str = "R0-L0-p(R1,CPE1)",
) -> None:
    root = tk.Tk()
    root.withdraw()
    root.title("EIS Fitting")
    root.geometry("1220x760")
    root.deiconify()
    root.update()
    EISApplication(root, mpt_path, cycle, control, outlier_threshold, circuit)
    root.mainloop()
