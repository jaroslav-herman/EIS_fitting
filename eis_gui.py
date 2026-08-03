from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Callable

import numpy as np
from natsort import natsort_keygen, ns
from scipy.optimize import curve_fit
from wepy.eis import tau as cpe_tau

from eis_model import ParameterValue, ProjectState
from eis_project import (
    _dataframe_to_payload,
    _state_to_payload,
    _derived_block_values,
    _dataframe_to_payload,
    _state_to_payload,
    dataframe_from_payload,
    export_drts_for_states,
    export_fit_parameters,
    export_fit_parameters_for_states,
    export_python_workspace as write_python_workspace,
    load_project_file,
    save_project_file,
)
from eis_services import (
    BatchFitReport,
    DRTComputation,
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
    load_cycle,
    load_project_from_dataframe,
    load_project,
    load_projects,
)

MODEL_PRESETS = (
    "R0-L0-p(R1,CPE1)",
    "R0-p(R1,CPE1)",
    "R0-L0-p(R1,CPE1)-p(R2,CPE2)",
    "R0-p(R1,C1)",
    "R0-p(R1,CPE1)-W1",
)


class ParameterTable(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
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
        for row, parameter in enumerate(parameters, start=1):
            fixed = tk.BooleanVar(value=parameter.fixed)
            initial = tk.StringVar(value=f"{parameter.initial:g}")
            lower = tk.StringVar(value=f"{parameter.lower:g}")
            upper = tk.StringVar(value=f"{parameter.upper:g}")
            ttk.Label(self, text=parameter.name).grid(row=row, column=0, padx=3, pady=2)
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
                f"Paste one value per line for the {spectrum_count} spectra shown "
                "in the explorer. Blank lines create empty cells."
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

        self.bind("<Control-Return>", lambda _event: self._accept())
        self.grab_set()
        self.name_entry.focus_set()

    def _accept(self) -> None:
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror(
                "Missing column name", "Enter a name for the new column.", parent=self
            )
            return

        raw = self.values_text.get("1.0", "end-1c").replace("\r\n", "\n")
        lines = raw.split("\n")
        while len(lines) > self.spectrum_count and lines[-1] == "":
            lines.pop()
        if len(lines) != self.spectrum_count:
            messagebox.showerror(
                "Wrong number of values",
                f"Paste exactly {self.spectrum_count} values; {len(lines)} were found.",
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

        self.result = (name, values)
        self.destroy()


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
        self.analysis_mode_var = tk.StringVar(value="EEC")
        self.loaded: LoadedProject | None = None
        self.state: ProjectState | None = None
        self.loaded_projects: dict[str, LoadedProject] = {}
        self._dataset_order: list[str] = []
        self._custom_metadata_columns: list[str] = []
        self._explorer_rows: dict[str, tuple[str, LoadedProject, SpectrumMetadata]] = (
            {}
        )
        self._explorer_lookup: dict[tuple[str, int], str] = {}
        self._explorer_anchor_item: str | None = None
        self._explorer_primary_item: str | None = None
        self._suspend_explorer_select = False
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="eis-worker"
        )
        self.busy = False
        self._fit_cancel_requested = False
        self._fit_parameter_snapshot = None
        self.drt_peak_parameters: list[dict[str, float]] = []
        self._drt_peak_cycle_key = None
        self._drt_peak_artists = []
        self._drt_peak_sum_artist = None
        self._drt_peak_drag = None
        self._drt_aux_parameter_limits = {}
        self._plot_imports = None
        self.plot_mode = "nyquist"

        self.threshold_var = tk.StringVar(value=f"{threshold:g}")
        self.model_var = tk.StringVar(value=circuit)
        self.show_drt_var = tk.BooleanVar(value=False)
        self.show_kk_var = tk.BooleanVar(value=False)
        self.show_spectrum_var = tk.BooleanVar(value=True)
        self.show_eec_fit_var = tk.BooleanVar(value=True)
        self.show_drt_fit_var = tk.BooleanVar(value=False)
        self.show_drt_recovered_var = tk.BooleanVar(value=False)
        self.hide_legends_var = tk.BooleanVar(value=False)
        self.minimum_frequency_var = tk.StringVar()
        self.maximum_frequency_var = tk.StringVar()
        self.cycle_var = tk.StringVar(value=str(cycle))
        self.status_var = tk.StringVar(value="Opening application…")

        self._configure_window()
        self._build_menu()
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
        self.root.bind("<Control-e>", self.open_export_menu)
        self.root.bind("<Control-s>", self._on_control_s)
        self.root.bind("<Control-S>", self._on_control_s)
        self.root.bind("<Control-Shift-O>", lambda _event: self.load_project())
        self.root.bind("<Control-o>", lambda _event: self.import_data())
        self.root.bind("<Alt-a>", self._on_alt_a)
        self.root.bind("<Alt-A>", self._on_alt_a)
        self.root.bind("<Alt-d>", self._on_alt_d)
        self.root.bind("<Alt-D>", self._on_alt_d)
        self.root.bind("<Alt-e>", self.toggle_point_edit_mode)
        self.root.bind("<Alt-s>", self._on_alt_s)
        self.root.bind("<Alt-S>", self._on_alt_s)
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
        self.root.title("EIS Fitting")
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
            accelerator="Ctrl+O",
            command=self.import_data,
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(
            label="Load project…",
            accelerator="Ctrl+Shift+O",
            command=self.load_project,
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
        self.export_menu = tk.Menu(menu_bar, tearoff=False)
        self.export_menu.add_command(
            label="Export fit parameters - all spectra…",
            command=self.export_fits,
        )
        self.export_menu.add_command(
            label="Export fit parameters - selected spectra…",
            command=self.export_selected_fits,
        )
        self.export_menu.add_separator()
        self.export_menu.add_command(
            label="Export to Python - all spectra…",
            command=self.export_python_workspace,
        )
        self.export_menu.add_command(
            label="Export to Python - selected spectra…",
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
        ttk.Label(self.plot_controls, text="DRT").pack(side=tk.LEFT, padx=(10, 4))
        self.drt_mode_var = tk.StringVar(value="Ridge DRT")
        self.drt_mode_box = ttk.Combobox(
            self.plot_controls,
            textvariable=self.drt_mode_var,
            values=("Ridge DRT", "Hybrid DRT"),
            state="readonly",
            width=12,
        )
        self.drt_mode_box.pack(side=tk.LEFT)
        self.drt_mode_box.bind("<<ComboboxSelected>>", self._on_drt_mode_selected)
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

    @staticmethod
    def _phase_degrees(values: np.ndarray) -> np.ndarray:
        return -np.degrees(np.angle(values))

    def _update_point_hover(self, event) -> None:
        if self.state is None or self.plot_mode != "nyquist":
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

    def _drt_recovered_impedance(self, cycle) -> np.ndarray | None:
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
        frequencies = np.asarray(cycle.frequency_hz, dtype=float)
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
        inductance = cycle.saved_ridge_inductance
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
        active_frequency = np.asarray(cycle.frequency_hz[cycle.included], dtype=float)
        active_frequency = active_frequency[np.isfinite(active_frequency) & (active_frequency > 0)]
        if active_frequency.size < 2:
            self.drt_fit_artist.set_data([], [])
            return
        curve_frequency = np.unique(
            np.concatenate(
                (
                    np.geomspace(
                        float(np.min(active_frequency)),
                        float(np.max(active_frequency)),
                        500,
                    ),
                    active_frequency,
                )
            )
        )
        curve_impedance = self._drt_peak_impedance(cycle, curve_frequency)
        if curve_impedance is None:
            self.drt_fit_artist.set_data([], [])
            return
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
        if self.plot_mode == "nyquist":
            self.drt_recovered_artist.set_data(
                recovered.real[included], -recovered.imag[included]
            )
            if hasattr(self, "phase_drt_recovered_artist"):
                self.phase_drt_recovered_artist.set_data([], [])
        else:
            self.drt_recovered_artist.set_data(
                frequency[included], np.abs(recovered[included])
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
            self.phase_drt_recovered_artist.set_data(
                frequency[included], self._phase_degrees(recovered[included])
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
        self.axes.set_xlabel("Re(Z) / Ω")
        self.axes.set_ylabel("−Im(Z) / Ω")
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
        self.axes.set_ylabel("|Z| / Ω")
        self.axes.grid(True, alpha=0.25)
        self.phase_axes = self.axes.twinx()
        self.phase_axes.set_ylabel("−Phase / °")
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
            [], [], "s", color="#6a1b9a", markersize=4, alpha=0.85, label="−Phase included"
        )
        (self.phase_excluded_artist,) = self.phase_axes.plot(
            [], [], "x", color="#ab47bc", markersize=4.5, alpha=0.45, label="−Phase excluded"
        )
        (self.phase_fit_artist,) = self.phase_axes.plot(
            [], [], "-", color="#4a148c", linewidth=1.8, alpha=0.8, label="−Phase fit"
        )
        (self.phase_fit_points_included_artist,) = self.phase_axes.plot(
            [],
            [],
            "o",
            color="#8e24aa",
            markersize=2.5,
            alpha=0.55,
            label="−Phase fit at measured frequencies",
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
            label="−Phase measured-to-fit difference",
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
            fontsize=8,
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
        if self.phase_axes is not None:
            (self.drt_phase_fit_artist,) = self.phase_axes.plot(
                [], [], "-", color="#00897b", linewidth=1.6, alpha=0.9, label="−Phase DRT fit"
            )
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
            self.kk_axes.legend(loc="best", fontsize=8)
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
        self.paste_metadata_button = ttk.Button(
            explorer_header,
            text="+",
            width=3,
            command=self.paste_metadata_column_from_clipboard,
        )
        self.paste_metadata_button.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Checkbutton(
            explorer_header,
            text="Natural sort",
            variable=self.natural_sort_var,
            command=self._toggle_natural_sort,
        ).pack(side=tk.LEFT, padx=(8, 0))
        group.configure(labelwidget=explorer_header)
        group.pack(fill=tk.BOTH, expand=True)
        group.columnconfigure(0, weight=1)
        group.rowconfigure(0, weight=1)
        columns = (
            "fitted",
            "source",
            "cycle",
            "potential",
            "current",
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
            "source": "Source file",
            "cycle": "Cycle",
            "potential": "Voltage (V)",
            "current": "Current (mA)",
            "points": "Points",
            "f_min": "Min frequency (Hz)",
            "f_max": "Max frequency (Hz)",
        }
        self._explorer_attributes = {
            "fitted": None,
            "drt": None,
            "source": None,
            "cycle": "cycle",
            "potential": "potential_v",
            "current": "current_ma",
            "points": "point_count",
            "f_min": "minimum_frequency_hz",
            "f_max": "maximum_frequency_hz",
        }
        self._explorer_sort_reverse: dict[str, bool] = {}
        self._explorer_sort_columns: list[tuple[str, bool]] = []
        self._explorer_selected_column = "cycle"
        widths = {
            "fitted": 62,
            "drt": 48,
            "source": 190,
            "cycle": 65,
            "potential": 105,
            "current": 110,
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
                column, width=widths[column], minwidth=55, anchor=anchor
            )
        scrollbar = ttk.Scrollbar(
            group, orient=tk.VERTICAL, command=self.explorer.yview
        )
        self.explorer.configure(yscrollcommand=scrollbar.set)
        self.explorer.grid(row=0, column=0, sticky="nsew")
        self.explorer.tag_configure(
            "focus_row",
            background="#dff3df",
            font=("Segoe UI", 9, "bold"),
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.explorer.bind("<Button-1>", self._on_explorer_click, add="+")
        self.explorer.bind("<Double-Button-1>", self._on_explorer_double_click, add="+")
        self.explorer.bind("<Button-1>", self._on_explorer_heading_click, add="+")
        self.explorer.bind("<<TreeviewSelect>>", self._select_explorer_spectrum)
        self.explorer.bind("<Delete>", lambda _event: self.delete_selected_spectrum())
        self.explorer.bind("<Up>", lambda event: self._on_explorer_arrow(event, -1))
        self.explorer.bind("<Down>", lambda event: self._on_explorer_arrow(event, 1))
        self.explorer.bind("<Control-a>", self.select_all_spectra)

        explorer_actions = ttk.Frame(group)
        explorer_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
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
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

    def _explorer_base_columns(self) -> tuple[str, ...]:
        return (
            "fitted",
            "drt",
            "source",
            "cycle",
            "potential",
            "current",
            "points",
            "f_min",
            "f_max",
        )

    def _explorer_columns(self) -> list[str]:
        return [*self._explorer_base_columns(), *self._custom_metadata_columns]

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
                    if is_ewe_data and name in {
                        WORKING_POTENTIAL_COLUMN,
                        COUNTER_POTENTIAL_COLUMN,
                    }:
                        continue
                    if name not in columns:
                        columns.append(name)
        self._custom_metadata_columns = columns

    def _refresh_explorer_schema(self) -> None:
        if not hasattr(self, "explorer"):
            return
        columns = self._explorer_columns()
        self.explorer.configure(columns=columns)
        headings = {
            "fitted": "Fitted",
            "drt": "DRT",
            "source": "Source file",
            "cycle": "Cycle",
            "potential": "Voltage (V)",
            "current": "Current (mA)",
            "points": "Points",
            "f_min": "Min frequency (Hz)",
            "f_max": "Max frequency (Hz)",
        }
        widths = {
            "fitted": 62,
            "drt": 48,
            "source": 190,
            "cycle": 65,
            "potential": 105,
            "current": 110,
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
            self.explorer.column(column, width=widths[column], minwidth=55, anchor=anchor)

    @staticmethod
    def _format_explorer_value(value, column: str | None = None) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return ""
        numeric_column = column in {"potential", "current", "f_min", "f_max"}
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
        if column == "source":
            return loaded.state.source_path.name
        if column == "cycle":
            return spectrum.cycle
        if column == "potential":
            return spectrum.potential_v
        if column == "current":
            return spectrum.current_ma
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
        columns = list(self._explorer_columns())
        if not column_index.startswith("#"):
            return "break"
        index = int(column_index[1:]) - 1
        if index < 0 or index >= len(columns):
            return "break"
        column = columns[index]
        if not self._explorer_rows:
            return "break"
        shift_pressed = bool(event.state & 0x0001)
        if not shift_pressed:
            reverse = self._explorer_sort_reverse.get(column, False)
            self._explorer_sort_columns = [(column, reverse)]
            self._explorer_selected_column = column
            self._apply_explorer_sort()
            self._explorer_sort_reverse[column] = not reverse
            return "break"
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
        return "break"

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
        elif column == "source":
            value = loaded.state.source_path.name
        elif column == "cycle":
            value = spectrum.cycle
        elif column == "potential":
            value = spectrum.potential_v
        elif column == "current":
            value = spectrum.current_ma
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

    def _on_alt_a(self, event):
        if event.state & 0x0001 or event.keysym == "A":
            self.copy_neighbor_fit_settings(-1)
        else:
            self.copy_neighbor_fit(-1)
        return "break"

    def _on_alt_d(self, event):
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
            new_selection = visible_items[min(start, end) : max(start, end) + 1]
        elif control_pressed:
            new_selection = selected if item in selected else [*selected, item]
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
        columns = self._explorer_columns()
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
        primary = self._explorer_primary_item
        if primary is not None and self.explorer.exists(primary):
            self.explorer.item(primary, tags=("focus_row",))

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
            values=("EEC", "DRT"),
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
            values=MODEL_PRESETS,
            state="normal",
        )
        self.model_box.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.model_box.bind("<Return>", lambda _event: self.apply_model())
        self.model_button = ttk.Button(
            model_group, text="Set model", command=self.apply_model
        )
        self.model_button.grid(row=0, column=1)
        self.model_selected_button = ttk.Button(
            model_group,
            text="Set model for selected",
            command=self.apply_model_to_selected,
        )
        self.model_selected_button.grid(row=0, column=2, padx=(5, 0))
        self.sort_tau_button = ttk.Button(
            model_group,
            text="Sort by tau: current",
            command=self.sort_parameters_by_tau,
        )
        self.sort_tau_button.grid(row=1, column=0, padx=(0, 3), pady=(5, 0), sticky="ew")
        self.sort_tau_selected_button = ttk.Button(
            model_group,
            text="Sort by tau: selected",
            command=self.sort_selected_parameters_by_tau,
        )
        self.sort_tau_selected_button.grid(row=1, column=1, columnspan=2, padx=(3, 0), pady=(5, 0), sticky="ew")
        self.switch_blocks_button = ttk.Button(
            model_group,
            text="Switch blocks: current",
            command=self.switch_parameter_blocks,
        )
        self.switch_blocks_button.grid(
            row=2, column=0, padx=(0, 3), pady=(5, 0), sticky="ew"
        )
        self.switch_blocks_selected_button = ttk.Button(
            model_group,
            text="Switch blocks: selected",
            command=self.switch_selected_parameter_blocks,
        )
        self.switch_blocks_selected_button.grid(
            row=2, column=1, columnspan=2, padx=(3, 0), pady=(5, 0), sticky="ew"
        )
        self.open_eec_analysis_button = ttk.Button(
            model_group,
            text="Open EEC analysis window",
            command=self.open_eec_analysis_window,
        )
        self.open_eec_analysis_button.grid(
            row=3, column=0, columnspan=3, pady=(6, 0), sticky="ew"
        )

        parameters_group = ttk.LabelFrame(parent, text="Circuit parameters", padding=8)
        self.parameters_group = parameters_group
        parameters_group.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        parent.rowconfigure(2, weight=1)
        self.parameter_table = ParameterTable(parameters_group)
        self.parameter_table.pack(fill=tk.BOTH, expand=True)
        parameter_actions = ttk.Frame(parameters_group)
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

        options_group = ttk.LabelFrame(parent, text="Selection", padding=8)
        self.options_group = options_group
        options_group.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        options_group.columnconfigure(1, weight=1)
        ttk.Label(options_group, text="Min frequency (Hz)").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(options_group, textvariable=self.minimum_frequency_var).grid(
            row=0, column=1, padx=(8, 0), pady=2, sticky="ew"
        )
        ttk.Label(options_group, text="Max frequency (Hz)").grid(
            row=1, column=0, sticky="w"
        )
        ttk.Entry(options_group, textvariable=self.maximum_frequency_var).grid(
            row=1, column=1, padx=(8, 0), pady=2, sticky="ew"
        )
        self.frequency_button = ttk.Button(
            options_group,
            text="Apply to current cycle",
            command=self.apply_frequency_window,
        )
        self.frequency_button.grid(
            row=2, column=0, padx=(0, 4), pady=(6, 0), sticky="ew"
        )
        self.frequency_selected_button = ttk.Button(
            options_group,
            text="Apply to selected spectra",
            command=self.apply_frequency_window_to_selected,
        )
        self.frequency_selected_button.grid(
            row=2, column=1, padx=(4, 0), pady=(6, 0), sticky="ew"
        )
        ttk.Label(options_group, text="Outlier threshold").grid(
            row=3, column=0, pady=(8, 2), sticky="w"
        )
        ttk.Entry(options_group, textvariable=self.threshold_var).grid(
            row=3, column=1, padx=(8, 0), pady=(8, 2), sticky="ew"
        )
        self.outlier_button = ttk.Button(
            options_group, text="Outliers: current", command=self.find_outliers
        )
        self.outlier_button.grid(
            row=4, column=0, padx=(0, 4), pady=(6, 0), sticky="ew"
        )
        self.outlier_selected_button = ttk.Button(
            options_group,
            text="Outliers: selected spectra",
            command=self.find_outliers_for_selected,
        )
        self.outlier_selected_button.grid(
            row=4, column=1, padx=(4, 0), pady=(6, 0), sticky="ew"
        )
        self.reset_button = ttk.Button(
            options_group, text="Reset points", command=self.reset_points
        )
        self.reset_button.grid(
            row=5, column=0, columnspan=2, pady=(6, 0), sticky="ew"
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
        self.ridge_drt_button = ttk.Button(
            actions, text="Ridge DRT", command=self.calculate_ridge_drt
        )
        self.ridge_drt_button.grid(row=2, column=0, padx=(0, 4), pady=3, sticky="ew")
        self.hybrid_drt_button = ttk.Button(
            actions, text="Hybrid DRT", command=self.calculate_hybrid_drt
        )
        self.hybrid_drt_button.grid(row=2, column=1, padx=(4, 0), pady=3, sticky="ew")
        self.ridge_drt_selected_button = ttk.Button(
            actions,
            text="Ridge DRT: selected",
            command=self.calculate_selected_ridge_drts,
        )
        self.ridge_drt_selected_button.grid(
            row=3, column=0, padx=(0, 4), pady=3, sticky="ew"
        )
        self.hybrid_drt_selected_button = ttk.Button(
            actions,
            text="Hybrid DRT: selected",
            command=self.calculate_selected_hybrid_drts,
        )
        self.hybrid_drt_selected_button.grid(
            row=3, column=1, padx=(4, 0), pady=3, sticky="ew"
        )
        self.batch_fit_button = ttk.Button(
            actions,
            text="Batch fit from current",
            command=self.batch_fit,
        )
        self.batch_fit_button.grid(row=4, column=0, columnspan=2, pady=3, sticky="ew")
        self.python_export_button = ttk.Button(
            actions,
            text="Export to Python",
            command=self.export_python_workspace,
        )
        self.python_export_button.grid(
            row=5, column=0, columnspan=2, pady=3, sticky="ew"
        )
        self.stop_fit_button = ttk.Button(
            actions, text="Stop fit", command=self._cancel_fit, state="disabled"
        )
        self.stop_fit_button.grid(row=6, column=0, columnspan=2, pady=3, sticky="ew")
        self.drt_tools_group = ttk.LabelFrame(parent, text="DRT analysis", padding=8)
        self.drt_tools_group.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.drt_tools_group.columnconfigure(1, weight=1)
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
            command=self.add_gaussian_peak,
        )
        self.add_gaussian_peak_button.grid(
            row=2, column=0, padx=(0, 3), pady=(6, 0), sticky="ew"
        )
        self.fit_peaks_button = ttk.Button(
            self.drt_tools_group,
            text="Fit peaks",
            command=self.fit_drt_peaks,
        )
        self.fit_peaks_button.grid(
            row=2, column=1, padx=(3, 0), pady=(6, 0), sticky="ew"
        )
        self.send_drt_initials_button = ttk.Button(
            self.drt_tools_group,
            text="Send initials",
            command=self.send_drt_initials,
        )
        self.send_drt_initials_button.grid(
            row=4, column=0, columnspan=2, pady=(6, 0), sticky="ew"
        )
        self.open_drt_analysis_button = ttk.Button(
            self.drt_tools_group,
            text="Open DRT analysis window",
            command=self.open_drt_analysis_window,
        )
        self.open_drt_analysis_button.grid(
            row=5, column=0, columnspan=2, pady=(6, 0), sticky="ew"
        )
        self.drt_peak_table = ParameterTable(self.drt_tools_group)
        self.drt_peak_table.grid(
            row=3, column=0, columnspan=2, pady=(6, 0), sticky="ew"
        )
        self.drt_tools_group.grid_remove()
        self.action_buttons = (
            self.fit_button,
            self.fit_selected_button,
            self.drt_fit_button,
            self.send_drt_initials_button,
            self.initial_values_button,
            self.ridge_drt_button,
            self.hybrid_drt_button,
            self.ridge_drt_selected_button,
            self.hybrid_drt_selected_button,
            self.outlier_button,
            self.outlier_selected_button,
            self.reset_button,
            self.batch_fit_button,
            self.python_export_button,
            self.toggle_points_button,
            self.auto_fit_points_button,
            self.reset_view_button,
            self.toggle_plot_mode_button,
            self.delete_spectrum_button,
            self.plot_selected_button,
            self.plot_three_electrode_button,
            self.plot_fit_parameters_button,
            self.plot_drt_parameters_button,
            self.paste_metadata_button,
            self.frequency_button,
            self.frequency_selected_button,
            self.model_button,
            self.model_selected_button,
            self.sort_tau_button,
            self.sort_tau_selected_button,
            self.switch_blocks_button,
            self.switch_blocks_selected_button,
            self.open_eec_analysis_button,
            self.open_drt_analysis_button,
            self.parameters_selected_button,
            self.apply_fix_selected_button,
            self.apply_initial_selected_button,
            self.apply_lower_selected_button,
            self.apply_upper_selected_button,
        )

    def _on_analysis_mode_selected(self, _event=None) -> None:
        drt_mode = self.analysis_mode_var.get() == "DRT"
        if drt_mode:
            self.model_group.grid_remove()
            self.parameters_group.grid_remove()
            self.drt_tools_group.grid()
            self._update_status("DRT analysis mode")
        else:
            self.model_group.grid()
            self.parameters_group.grid()
            self.drt_tools_group.grid_remove()
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
            values=MODEL_PRESETS,
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
        table = ParameterTable(table_group)
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
            command=lambda: run_action(self.add_gaussian_peak),
        ).grid(row=1, column=0, padx=(0, 3), sticky="ew")
        ttk.Button(
            buttons,
            text="Fit peaks",
            command=lambda: run_action(self.fit_drt_peaks),
        ).grid(row=1, column=1, padx=3, sticky="ew")
        ttk.Button(
            buttons,
            text="Send initials",
            command=lambda: run_action(self.send_drt_initials),
        ).grid(row=1, column=2, padx=(3, 0), sticky="ew")

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
            lambda: load_projects(
                [self.path],
                self.control,
                self.circuit,
                self.requested_cycle,
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

    def _submit(
        self,
        work: Callable[[], object],
        success: Callable[[object], None],
        error_title: str,
    ) -> None:
        if self.busy:
            return
        self._fit_cancel_requested = False
        self._fit_parameter_snapshot = copy.deepcopy(self.state)
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
            self._set_controls_enabled(self.state is not None)
            self.status_var.set(f"Error: {error}")
            messagebox.showerror(
                error_title, f"{type(error).__name__}: {error}", parent=self.root
            )
            return
        if self._fit_cancel_requested:
            if self._fit_parameter_snapshot is not None and self.state is not None:
                self.state = copy.deepcopy(self._fit_parameter_snapshot)
                refresh = getattr(self, "_refresh_parameter_table", None)
                if refresh is not None:
                    refresh()
            self._update_status("fit cancelled")
        else:
            success(result)
        self._fit_parameter_snapshot = None
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
            for label in self._project_menu_actions[1:]:
                self.file_menu.entryconfigure(label, state=menu_state)
        if hasattr(self, "fit_menu"):
            for label in self._fit_menu_actions:
                self.fit_menu.entryconfigure(label, state=menu_state)
        if hasattr(self, "export_menu"):
            for label in self._export_menu_actions:
                self.export_menu.entryconfigure(label, state=menu_state)

    def _restore_controls(self) -> None:
        if self.state is None:
            return
        cycle = self.state.active
        self.parameter_table.set_parameters(self.state.parameters_for(cycle.cycle))
        self.model_var.set(cycle.model(self.state.circuit))
        if cycle.frequency_window is not None:
            self.minimum_frequency_var.set(f"{cycle.frequency_window[0]:g}")
            self.maximum_frequency_var.set(f"{cycle.frequency_window[1]:g}")

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

    def _refresh_plot(self, rescale: bool = False) -> None:
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
            self._drt_aux_parameter_limits = {}
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
        self._refresh_explorer_values()
        self.canvas.draw_idle()

    def _autoscale_to_included(self, cycle) -> None:
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
    def _gaussian_peak_values(log_tau, peak):
        return peak["height"] * np.exp(
            -0.5
            * ((log_tau - peak["center_log10"]) / max(peak["sigma_log10"], 1e-6))
            ** 2
        )

    @staticmethod
    def _peak_summary(peak):
        tau = 10.0 ** peak["center_log10"]
        sigma = max(peak["sigma_log10"], 1e-6)
        area = peak["height"] * sigma * np.sqrt(2.0 * np.pi) * np.log(10.0)
        half_width = np.sqrt(2.0 * np.log(2.0)) * sigma
        fwhm = tau * (10.0**half_width - 10.0 ** (-half_width))
        return tau, area, fwhm

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
                cycle.saved_ridge_inductance if mode == "Ridge DRT" else None
            )
            for name, value, unit in (
                ("R0", resistance, "Ohm"),
                ("L0", inductance, "H"),
            ):
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
            for suffix, value, error_key in (
                ("area", area, "area_error_percent"),
                ("tau", tau, "tau_error_percent"),
                ("fwhm", fwhm, "fwhm_error_percent"),
            ):
                lower_key = f"{suffix}_lower"
                upper_key = f"{suffix}_upper"
                magnitude = max(abs(value), np.finfo(float).eps)
                default_limits = {
                    "tau": (1e-5, 10.0),
                    "area": (0.0, 1e3),
                    "fwhm": (0.0, 1.0),
                }
                default_lower, default_upper = default_limits[suffix]
                lower = peak.get(lower_key, default_lower)
                upper = peak.get(upper_key, default_upper)
                parameters.append(
                    ParameterValue(
                        f"Peak{index}_{suffix}",
                        "s" if suffix == "tau" else "Ohm",
                        value,
                        lower,
                        upper,
                        peak.get(error_key),
                        bool(peak.get(f"{suffix}_fixed", False)),
                    )
                )
        self.drt_peak_table.set_parameters(parameters)

    def _clamp_drt_peak_parameters_to_limits(self) -> None:
        for peak in self.drt_peak_parameters:
            tau, area, fwhm = self._peak_summary(peak)
            values = {"tau": tau, "area": area, "fwhm": fwhm}
            for suffix, value in values.items():
                defaults = {"tau": (1e-5, 10.0), "area": (0.0, 1e3), "fwhm": (0.0, 1.0)}
                lower = peak.get(f"{suffix}_lower", defaults[suffix][0])
                upper = peak.get(f"{suffix}_upper", defaults[suffix][1])
                values[suffix] = self._clamp_parameter_value(value, lower, upper)
            tau = max(values["tau"], 1e-300)
            fwhm = max(values["fwhm"], 1e-300)
            ratio = max(fwhm / tau, 1e-12)
            peak["center_log10"] = float(np.log10(tau))
            peak["sigma_log10"] = float(
                np.arcsinh(ratio / 2.0)
                / (np.log(10.0) * np.sqrt(2.0 * np.log(2.0)))
            )
            peak["height"] = float(
                values["area"]
                / (peak["sigma_log10"] * np.log(10.0) * np.sqrt(2.0 * np.pi))
            )
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
                fwhm = by_name[f"Peak{index}_fwhm"]
                if tau.initial <= 0 or fwhm.initial <= 0:
                    raise ValueError(f"Peak {index}: tau and FWHM must be positive")
                peak["center_log10"] = float(np.log10(tau.initial))
                ratio = max(fwhm.initial / tau.initial, 1e-12)
                peak["sigma_log10"] = float(
                    np.arcsinh(ratio / 2.0)
                    / (np.log(10.0) * np.sqrt(2.0 * np.log(2.0)))
                )
                peak["height"] = float(
                    area.initial
                    / (peak["sigma_log10"] * np.log(10.0) * np.sqrt(2.0 * np.pi))
                )
                for source, target in (
                    (tau, "tau"),
                    (area, "area"),
                    (fwhm, "fwhm"),
                ):
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
            values = self._gaussian_peak_values(log_tau_grid, peak)
            total += values
            line, = self.drt_axes.plot(
                tau_grid,
                values,
                "--",
                linewidth=1.2,
                alpha=0.85,
                label=f"Gaussian {index + 1}",
            )
            center_tau = 10.0 ** peak["center_log10"]
            half_width = np.sqrt(2.0 * np.log(2.0)) * peak["sigma_log10"]
            left_tau = 10.0 ** (peak["center_log10"] - half_width)
            right_tau = 10.0 ** (peak["center_log10"] + half_width)
            top, = self.drt_axes.plot(
                [center_tau], [peak["height"]], "o", color=line.get_color(), ms=6
            )
            left, = self.drt_axes.plot(
                [left_tau], [peak["height"] / 2.0], "s", color=line.get_color(), ms=5
            )
            right, = self.drt_axes.plot(
                [right_tau], [peak["height"] / 2.0], "s", color=line.get_color(), ms=5
            )
            self._drt_peak_artists.extend((line, top, left, right))
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
        self.drt_axes.legend(loc="best", fontsize=8)

    def add_gaussian_peak(self) -> None:
        tau, gamma = self._current_drt_arrays()
        if tau is None:
            self._update_status("calculate a DRT before adding Gaussian peaks")
            return
        peak_index = int(np.nanargmax(gamma))
        center = float(np.log10(tau[peak_index]))
        if self.drt_peak_parameters:
            center = float(np.mean(np.log10(tau)))
        height = max(float(gamma[peak_index]), 0.0)
        if height == 0.0:
            height = float(np.nanmax(np.abs(gamma)))
        self.drt_peak_parameters.append(
            {"center_log10": center, "height": height, "sigma_log10": 0.12}
        )
        self._store_current_drt_peaks()
        self._refresh_drt_peak_artists()
        self.canvas.draw_idle()

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
        initial = []
        lower = []
        upper = []
        fixed = []
        for index, peak in enumerate(self.drt_peak_parameters, 1):
            peak_tau, peak_area, peak_fwhm = self._peak_summary(peak)
            initial.extend((peak_tau, peak_area, peak_fwhm))
            lower.extend(
                (
                    peak.get("tau_lower", peak_tau / 10.0),
                    peak.get("area_lower", min(0.0, peak_area)),
                    peak.get("fwhm_lower", peak_fwhm / 10.0),
                )
            )
            upper.extend(
                (
                    peak.get("tau_upper", peak_tau * 10.0),
                    peak.get("area_upper", max(abs(peak_area) * 10.0, 1e-12)),
                    peak.get("fwhm_upper", peak_fwhm * 10.0),
                )
            )
            fixed.extend(
                (
                    bool(peak.get("tau_fixed", False)),
                    bool(peak.get("area_fixed", False)),
                    bool(peak.get("fwhm_fixed", False)),
                )
            )

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
            for index in range(count):
                peak_tau, area, fwhm = all_parameters[index * 3 : index * 3 + 3]
                ratio = max(fwhm / max(peak_tau, 1e-300), 1e-12)
                sigma = np.arcsinh(ratio / 2.0) / (
                    np.log(10.0) * np.sqrt(2.0 * np.log(2.0))
                )
                height = area / (sigma * np.log(10.0) * np.sqrt(2.0 * np.pi))
                result += height * np.exp(
                    -0.5 * ((values - np.log10(peak_tau)) / sigma) ** 2
                )
            return result
        fit_initial = initial[free_indices]
        fit_lower = lower[free_indices].copy()
        fit_upper = upper[free_indices].copy()
        for index in range(fit_initial.size):
            parameter_index = free_indices[index]
            if fit_lower[index] <= 0 and parameter_index % 3 in {0, 2}:
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
        for index, peak in enumerate(self.drt_peak_parameters):
            peak_tau, peak_area, peak_fwhm = fitted[index * 3 : index * 3 + 3]
            peak["center_log10"] = float(np.log10(peak_tau))
            ratio = max(peak_fwhm / max(peak_tau, 1e-300), 1e-12)
            peak["sigma_log10"] = float(
                np.arcsinh(ratio / 2.0)
                / (np.log(10.0) * np.sqrt(2.0 * np.log(2.0)))
            )
            peak["height"] = float(
                peak_area
                / (peak["sigma_log10"] * np.log(10.0) * np.sqrt(2.0 * np.pi))
            )
            for offset, key, value in (
                (0, "tau", peak_tau),
                (1, "area", peak_area),
                (2, "fwhm", peak_fwhm),
            ):
                peak[f"{key}_error_percent"] = float(
                    100.0 * errors[index * 3 + offset] / max(abs(value), 1e-300)
                )
        self._store_current_drt_peaks()
        self._refresh_drt_peak_artists()
        self.canvas.draw_idle()
        self._update_status(f"fitted {count} Gaussian DRT peaks")

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
        if self.state is None:
            return
        self._refresh_plot(rescale=True)
        self._update_status("zoom reset to active points")

    def toggle_plot_mode(self) -> None:
        self.plot_mode = "bode" if self.plot_mode == "nyquist" else "nyquist"
        self._configure_plot_layout()
        if self.state is None:
            self.axes.set_title("No spectrum loaded")
            if self.drt_axes is not None:
                self.drt_axes.set_title("Ridge DRT")
            self.canvas.draw_idle()
            return
        self._refresh_plot(rescale=True)
        self._update_status(f"{self.plot_mode.title()} view")

    def toggle_drt_view(self) -> None:
        self._configure_plot_layout()
        if self.state is None:
            self.axes.set_title("No spectrum loaded")
            if self.drt_axes is not None:
                self.drt_axes.set_title("Ridge DRT")
        else:
            self._refresh_plot(rescale=True)
        self.canvas.draw_idle()

    def toggle_spectrum_view(self) -> None:
        self._configure_plot_layout()
        if self.state is None:
            self.axes.set_title("No spectrum loaded")
            if self.drt_axes is not None:
                self.drt_axes.set_title("Ridge DRT")
        else:
            self._refresh_plot(rescale=True)
        self.canvas.draw_idle()

    def _calculate_drt_fit_impedance(self, cycle):
        if not self.drt_peak_parameters:
            return None, None
        frequency = np.asarray(cycle.frequency_hz, dtype=float)
        finite = np.isfinite(frequency) & (frequency > 0)
        if not np.any(finite):
            return None, None
        frequency = np.geomspace(
            float(np.min(frequency[finite])),
            float(np.max(frequency[finite])),
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
        if "L0" in table_values:
            inductance = table_values["L0"].initial
        impedance = np.full(frequency.size, float(resistance or 0.0), dtype=complex)
        omega = 2.0 * np.pi * frequency
        if inductance is not None:
            impedance += 1j * omega * float(inductance)
        for peak in self.drt_peak_parameters:
            tau, area, _fwhm = self._peak_summary(peak)
            impedance += area / (1.0 + 1j * omega * tau)
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
        if self.show_drt_fit_var.get() and not self._sync_drt_peak_parameters_from_table():
            self.show_drt_fit_var.set(False)
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
        self.root.title("EIS Fitting")
        self._set_controls_enabled(False)

    def _start_drt_peak_drag(self, event) -> bool:
        if self.busy or self.state is None or not self.drt_peak_parameters:
            return False
        if event.x is None or event.y is None:
            return False
        best = None
        for index, peak in enumerate(self.drt_peak_parameters):
            center_tau = 10.0 ** peak["center_log10"]
            half_width = np.sqrt(2.0 * np.log(2.0)) * peak["sigma_log10"]
            points = (
                ("top", center_tau, peak["height"]),
                (
                    "left",
                    10.0 ** (peak["center_log10"] - half_width),
                    peak["height"] / 2.0,
                ),
                (
                    "right",
                    10.0 ** (peak["center_log10"] + half_width),
                    peak["height"] / 2.0,
                ),
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
            or event.button != 1
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
            self._drt_peak_drag = None
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
                    distance / np.sqrt(2.0 * np.log(2.0)), 1e-4
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

    def _fit_after_point_edit(self) -> None:
        if self.point_auto_fit and not self.busy:
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
        updated = 0
        for _dataset_id, loaded, spectrum in selected_rows:
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            cycle.frequency_window = window
            cycle.invalidate_drt_cache()
            cycle.clear_fit()
            updated += 1
        self._refresh_plot(rescale=True)
        self._update_status(
            f"frequency range applied to {updated} selected spectra"
        )

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
            parameters = circuit_parameters(circuit)
        except Exception as error:
            messagebox.showerror(
                "Invalid model",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            self.model_var.set(self.state.active.model(self.state.circuit))
            return
        self.state.active.circuit = circuit
        self.state.active.parameters = parameters
        self.state.active.clear_fit()
        self.state.active.invalidate_drt_cache()
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
            parameters = circuit_parameters(circuit)
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
            cycle.clear_fit()
            cycle.invalidate_drt_cache()
            updated += 1
        self._restore_controls()
        self._refresh_explorer_values()
        self._refresh_plot(rescale=True)
        self._update_status(f"model applied to {updated} selected spectra")

    @staticmethod
    def _switch_parameter_blocks(state: ProjectState, cycle) -> bool:
        circuit = re.sub(r"\s+", "", cycle.model(state.circuit))
        if circuit != "R0-L0-p(R1,CPE1)-p(R2,CPE2)":
            return False
        parameters = {parameter.name: parameter for parameter in cycle.parameters}
        names = (
            "R1",
            "R2",
            "CPE1_0",
            "CPE2_0",
            "CPE1_1",
            "CPE2_1",
        )
        if any(name not in parameters for name in names):
            return False
        for first_name, second_name in (
            ("R1", "R2"),
            ("CPE1_0", "CPE2_0"),
            ("CPE1_1", "CPE2_1"),
        ):
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
            for first_name, second_name in (
                ("R1", "R2"),
                ("CPE1_0", "CPE2_0"),
                ("CPE1_1", "CPE2_1"),
            ):
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

    def switch_parameter_blocks(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        changed = self._switch_parameter_blocks(self.state, self.state.active)
        self._restore_controls()
        self._refresh_explorer_values()
        self._refresh_plot(rescale=True)
        self._update_status(
            "parameter blocks switched"
            if changed
            else "switch blocks requires R0-L0-p(R1,CPE1)-p(R2,CPE2)"
        )

    def switch_selected_parameter_blocks(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        selected_rows = self._selected_spectrum_rows()
        if not selected_rows:
            self._update_status("select one or more spectra in the explorer first")
            return
        changed = 0
        for _dataset_id, loaded, spectrum in selected_rows:
            cycle = self._loaded_cycle_for_popup(loaded, spectrum.cycle)
            if self._switch_parameter_blocks(loaded.state, cycle):
                changed += 1
        self._restore_controls()
        self._refresh_explorer_values()
        self._refresh_plot(rescale=True)
        self._update_status(
            f"parameter blocks switched for {changed} selected spectra"
            if changed
            else "switch blocks requires R0-L0-p(R1,CPE1)-p(R2,CPE2)"
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

    def sort_parameters_by_tau(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        changed = self._sort_cycle_parameters_by_tau(self.state, self.state.active)
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._update_status(
            "parameters sorted by tau" if changed else "parameters are already sorted by tau"
        )

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
            ),
            self._finish_all_outliers,
            "File-wide outlier search failed",
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
                )
                for dataset_id, (loaded, project) in selected_projects.items()
            },
            self._finish_all_outliers,
            "Selected-spectra outlier search failed",
        )

    def _finish_all_outliers(self, results) -> None:
        if self.state is None:
            return
        peak_count = 0
        spectra_count = 0
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
                spectra_count += 1
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._update_status(
            f"outliers and ridge initial values calculated for {spectra_count} spectra "
            f"({peak_count} peaks)"
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
        if self.state.active_cycle == cycle_number:
            self._refresh_plot(rescale=True)
            self._update_status(
                f"ridge DRT calculated: "
                f"R∞={analysis.ohmic_resistance:.3g} Ω, "
                f"{analysis.peak_count} peaks"
            )

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
        )
        if self.state.active_cycle == cycle_number:
            self._refresh_plot(rescale=True)
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
        self._submit(
            lambda: {
                dataset_id: {
                    cycle_number: (
                        project.cycles[cycle_number],
                        self._cached_ridge_analysis(
                            project.cycles[cycle_number],
                            threshold,
                            project.parameters_for(cycle_number),
                        )
                        or analyze_outliers(
                            project.cycles[cycle_number],
                            threshold,
                            project.parameters_for(cycle_number),
                        )
                    )
                    for cycle_number in project.available_cycles
                }
                for dataset_id, (loaded, project) in selected_projects.items()
            },
            self._finish_selected_outliers,
            "Selected-spectra outlier search failed",
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
        self._submit(
            lambda: {
                dataset_id: {
                    cycle_number: analyze_outliers(
                        project.cycles[cycle_number],
                        threshold,
                        project.parameters_for(cycle_number),
                    )
                    for cycle_number in project.available_cycles
                }
                for dataset_id, (_loaded, project) in batches.items()
            },
            self._finish_selected_ridge_drts,
            "Selected ridge DRT calculation failed",
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
        self._submit(
            lambda: {
                dataset_id: {
                    cycle_number: calculate_hybrid_drt(project.cycles[cycle_number])
                    for cycle_number in project.available_cycles
                }
                for dataset_id, (_loaded, project) in batches.items()
            },
            self._finish_selected_hybrid_drts,
            "Selected hybrid DRT calculation failed",
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
        self._update_status(f"hybrid DRT recalculated for {spectra_count} selected spectra")

    def fit(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        cycle_number = self.state.active_cycle
        cycle = self.state.active
        parameters = self.state.parameters_for(cycle_number)
        self.status_var.set(f"Cycle {cycle_number} · fitting…")
        self._submit(
            lambda: fit_cycle(cycle, cycle.model(self.state.circuit), parameters),
            lambda result: self._finish_fit(cycle_number, parameters, result),
            "Fit failed",
        )

    def _finish_fit(self, cycle_number, parameters, result) -> None:
        if self.state is None:
            return
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
        self.status_var.set(f"Fitting {len(targets)} selected spectra…")
        self._submit(
            lambda: batch_fit_spectra(
                targets,
                parameters,
                use_target_initial_parameters=True,
            ),
            self._finish_explorer_batch_fit,
            "Selected fit failed",
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
            ),
            self._finish_batch_fit,
            "Batch fit failed",
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
            self.state.cycles[cycle.cycle] = cycle
        self._restore_controls()
        self._refresh_plot(rescale=True)
        if report.failed_cycle is None:
            self._update_status(f"batch fit completed for {len(report.fits)} cycles")
            return
        self._update_status(
            f"batch fit stopped at cycle {report.failed_cycle}; "
            f"{len(report.fits)} cycles completed"
        )
        messagebox.showwarning(
            "Batch fit stopped",
            f"Cycle {report.failed_cycle}: {report.error}\n\n"
            f"The {len(report.fits)} successful fits were retained.",
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
            lambda: batch_fit_spectra(targets, parameters),
            self._finish_explorer_batch_fit,
            "Explorer batch fit failed",
        )

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
            lambda: batch_fit_spectra(targets, parameters),
            self._finish_explorer_batch_fit,
            "Selected batch fit failed",
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
            result.loaded.state.cycles[cycle.cycle] = cycle
        self._restore_controls()
        self._refresh_plot(rescale=True)
        if report.failed_label is None:
            self._update_status(
                f"explorer batch fit completed for {len(report.fits)} spectra"
            )
            return
        self._update_status(
            f"explorer batch stopped at {report.failed_label}; "
            f"{len(report.fits)} spectra completed"
        )
        messagebox.showwarning(
            "Explorer batch fit stopped",
            f"{report.failed_label}: {report.error}\n\n"
            f"The {len(report.fits)} successful fits were retained.",
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
        self._batch_fit_both_pending = False
        self.batch_fit_selected_down(-1)

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
        if source_model != current_model:
            self._update_status("neighboring spectrum uses a different fitting model")
            return
        current_parameters = self.state.parameters_for(self.state.active_cycle)
        fitted = np.asarray(source.fit_parameters).reshape(-1)
        if fitted.size != len(current_parameters):
            self._update_status("neighboring fit uses incompatible parameters")
            return
        for parameter, value in zip(current_parameters, fitted):
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
        if source_model != current_model:
            self._update_status("neighboring spectrum uses a different fitting model")
            return
        current_parameters = self.state.parameters_for(self.state.active_cycle)
        source_parameters = source_loaded.state.parameters_for(source_spectrum.cycle)
        fitted = np.asarray(source.fit_parameters).reshape(-1)
        if (
            fitted.size != len(current_parameters)
            or len(source_parameters) != len(current_parameters)
            or [parameter.name for parameter in source_parameters]
            != [parameter.name for parameter in current_parameters]
        ):
            self._update_status("neighboring fit uses incompatible parameters")
            return
        copied_parameters = []
        for target, source_parameter, value in zip(
            current_parameters, source_parameters, fitted
        ):
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
                        continue
                    if np.isfinite(numeric_value):
                        record[name] = numeric_value
                records.append(record)
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
                        continue
                    if np.isfinite(numeric_value):
                        record[name] = numeric_value
                records.append(record)
        return records

    def open_drt_parameters_explorer(self) -> None:
        if self.busy or self.state is None:
            return
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
        parameter_fields = [
            field
            for field in fields
            if field == "R0" or field == "L0" or field.startswith("peak")
        ]
        x_default = "current_mA" if "current_mA" in fields else fields[0]
        y_default = parameter_fields[0] if parameter_fields else fields[0]

        popup = tk.Toplevel(self.root)
        self.drt_parameters_popup = popup
        popup.title(f"DRT Parameters Explorer — {mode.title()}")
        popup.geometry("1180x760")
        popup.minsize(900, 600)
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(1, weight=1)

        def close_popup() -> None:
            self.drt_parameters_popup = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", close_popup)
        controls = ttk.Frame(popup, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        for column in range(5):
            controls.columnconfigure(column, weight=1)
        x_var = tk.StringVar(value=x_default)
        y_var = tk.StringVar(value=y_default)
        split_var = tk.StringVar(value="None")
        x_equation = tk.StringVar(value="x")
        y_equation = tk.StringVar(value="y")
        x_log = tk.BooleanVar(value=False)
        y_log = tk.BooleanVar(value=False)
        ttk.Label(controls, text="X axis").grid(row=0, column=0, sticky="w")
        ttk.Label(controls, text="Y axis").grid(row=0, column=1, sticky="w")
        ttk.Label(controls, text="Split by").grid(row=0, column=2, sticky="w")
        x_box = ttk.Combobox(controls, textvariable=x_var, state="readonly")
        y_box = ttk.Combobox(controls, textvariable=y_var, state="readonly")
        split_box = ttk.Combobox(controls, textvariable=split_var, state="readonly")
        x_box.grid(row=1, column=0, padx=(0, 6), sticky="ew")
        y_box.grid(row=1, column=1, padx=3, sticky="ew")
        split_box.grid(row=1, column=2, padx=(6, 12), sticky="ew")
        ttk.Checkbutton(controls, text="Log X", variable=x_log).grid(row=1, column=3, sticky="w")
        ttk.Checkbutton(controls, text="Log Y", variable=y_log).grid(row=1, column=4, sticky="w")
        ttk.Label(controls, text="X equation").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(controls, text="Y equation").grid(row=2, column=1, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=x_equation).grid(row=3, column=0, padx=(0, 6), sticky="ew")
        ttk.Entry(controls, textvariable=y_equation).grid(row=3, column=1, padx=3, sticky="ew")
        ttk.Label(
            controls,
            text="Use x, y, column names, and np functions",
        ).grid(row=3, column=2, columnspan=3, padx=(6, 0), sticky="w")

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
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(canvas, chart_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        range_state: dict[str, tuple[tk.DoubleVar, tk.DoubleVar, float, float]] = {}
        range_labels: dict[str, tk.StringVar] = {}
        range_widgets: dict[str, list[tk.Widget]] = {}

        def bounds(field: str) -> tuple[float, float]:
            values = []
            for record in records:
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
                [x_var.get(), y_var.get()]
                + ([] if split_var.get() == "None" else [split_var.get()])
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
            x_field, y_field, split_field = x_var.get(), y_var.get(), split_var.get()
            filtered = []
            selected_fields = [x_field, y_field] + ([] if split_field == "None" else [split_field])
            for record in records:
                try:
                    if any(
                        field not in record
                        or not np.isfinite(float(record[field]))
                        or not value_at(field, range_state[field][0].get()) <= float(record[field]) <= value_at(field, range_state[field][1].get())
                        for field in selected_fields
                    ):
                        continue
                    x_value = evaluate(record, x_equation.get(), "x", x_field, y_field)
                    y_value = evaluate(record, y_equation.get(), "y", x_field, y_field)
                    if np.isfinite(x_value) and np.isfinite(y_value):
                        filtered.append((record, x_value, y_value))
                except (KeyError, TypeError, ValueError, SyntaxError, ZeroDivisionError):
                    continue
            groups: dict[object, list[tuple[dict[str, object], float, float]]] = {}
            if split_field == "None":
                groups["DRT"] = filtered
            else:
                for row in filtered:
                    groups.setdefault(row[0][split_field], []).append(row)
            ordered_groups = sorted(groups.items(), key=lambda item: str(item[0]))
            color_scale = colormaps["rainbow"]
            for index, (group, values) in enumerate(ordered_groups):
                if not values:
                    continue
                color = color_scale(index / max(len(ordered_groups) - 1, 1))
                axes.plot(
                    [value[1] for value in values],
                    [value[2] for value in values],
                    "o-",
                    color=color,
                    linewidth=1.1,
                    markersize=4,
                    label=str(group),
                )
            axes.set_xlabel(x_equation.get() or x_field)
            axes.set_ylabel(y_equation.get() or y_field)
            axes.set_xscale("log" if x_log.get() else "linear")
            axes.set_yscale("log" if y_log.get() else "linear")
            if ordered_groups:
                axes.legend(loc="best", fontsize=8)
            canvas.draw_idle()

        def refresh_data() -> None:
            nonlocal fields
            records[:] = self._collect_drt_parameter_records(mode)
            fields = numeric_fields()
            x_values = [field for field in fields]
            x_box.configure(values=x_values)
            y_box.configure(values=x_values)
            split_box.configure(values=["None", *x_values])
            if x_var.get() not in x_values:
                x_var.set(x_values[0])
            if y_var.get() not in x_values:
                y_var.set(x_values[0])
            refresh_ranges()

        x_box.bind("<<ComboboxSelected>>", lambda _event: refresh_ranges())
        y_box.bind("<<ComboboxSelected>>", lambda _event: refresh_ranges())
        split_box.bind("<<ComboboxSelected>>", lambda _event: refresh_ranges())
        x_equation.trace_add("write", lambda *_args: refresh_plot())
        y_equation.trace_add("write", lambda *_args: refresh_plot())
        x_log.trace_add("write", lambda *_args: refresh_plot())
        y_log.trace_add("write", lambda *_args: refresh_plot())
        x_box.configure(values=fields)
        y_box.configure(values=fields)
        split_box.configure(values=["None", *fields])
        ttk.Button(controls, text="Refresh", command=refresh_data).grid(row=4, column=0, pady=(6, 0), sticky="w")
        refresh_ranges()

    def open_fit_parameters_explorer(self) -> None:
        if self.busy or self.state is None:
            return
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
                        continue
                    if np.isfinite(numeric_value):
                        record[name] = numeric_value
                records.append(record)
        if not records:
            self._update_status("no fitted spectra are available")
            return

        numeric_fields = []
        for field in dict.fromkeys(
            key for record in records for key in record if key != "circuit" and key != "source_file"
        ):
            values = [record.get(field) for record in records]
            if any(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
                numeric_fields.append(field)
        parameter_fields = [
            field
            for field in numeric_fields
            if any(field.startswith(prefix) for prefix in ("R", "C", "L", "W"))
        ]
        x_default = "current_mA" if "current_mA" in numeric_fields else numeric_fields[0]
        y_default = parameter_fields[0] if parameter_fields else numeric_fields[0]

        popup = tk.Toplevel(self.root)
        self.fit_parameters_popup = popup
        popup.title("Fit Parameters Explorer")
        popup.geometry("1180x760")
        popup.minsize(900, 600)
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(1, weight=1)

        def close_popup() -> None:
            self.fit_parameters_popup = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", close_popup)

        controls = ttk.Frame(popup, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        for column in range(5):
            controls.columnconfigure(column, weight=1)

        x_var = tk.StringVar(value=x_default)
        y_var = tk.StringVar(value=y_default)
        split_var = tk.StringVar(value="None")
        x_equation = tk.StringVar(value="x")
        y_equation = tk.StringVar(value="y")
        x_log = tk.BooleanVar(value=False)
        y_log = tk.BooleanVar(value=False)
        ttk.Label(controls, text="X axis").grid(row=0, column=0, sticky="w")
        ttk.Label(controls, text="Y axis").grid(row=0, column=1, sticky="w")
        ttk.Label(controls, text="Split by").grid(row=0, column=2, sticky="w")
        x_box = ttk.Combobox(controls, textvariable=x_var, values=numeric_fields, state="readonly")
        y_box = ttk.Combobox(controls, textvariable=y_var, values=numeric_fields, state="readonly")
        split_box = ttk.Combobox(
            controls, textvariable=split_var, values=["None", *numeric_fields], state="readonly"
        )
        x_box.grid(row=1, column=0, padx=(0, 6), sticky="ew")
        y_box.grid(row=1, column=1, padx=3, sticky="ew")
        split_box.grid(row=1, column=2, padx=(6, 12), sticky="ew")
        ttk.Checkbutton(controls, text="Log X", variable=x_log).grid(row=1, column=3, sticky="w")
        ttk.Checkbutton(controls, text="Log Y", variable=y_log).grid(row=1, column=4, sticky="w")
        ttk.Label(controls, text="X equation").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(controls, text="Y equation").grid(row=2, column=1, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=x_equation).grid(row=3, column=0, padx=(0, 6), sticky="ew")
        ttk.Entry(controls, textvariable=y_equation).grid(row=3, column=1, padx=3, sticky="ew")
        ttk.Label(
            controls,
            text="Use column names and np functions, e.g. 1/R1 or np.log10(current_mA)",
        ).grid(row=3, column=2, columnspan=3, padx=(6, 0), sticky="w")
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
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(canvas, chart_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        range_state: dict[str, tuple[tk.DoubleVar, tk.DoubleVar, float, float]] = {}
        range_labels: dict[str, tk.StringVar] = {}
        range_widgets: dict[str, list[tk.Widget]] = {}

        def field_bounds(field: str) -> tuple[float, float]:
            values = np.asarray(
                [float(record[field]) for record in records if field in record and np.isfinite(float(record[field]))],
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
            fields = [x_var.get(), y_var.get()]
            if split_var.get() != "None":
                fields.append(split_var.get())
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
            y_field = y_var.get()
            split_field = split_var.get()
            filtered = []
            fields = [x_field, y_field]
            if split_field != "None":
                fields.append(split_field)
            for record in records:
                try:
                    if any(
                        field not in record
                        or not np.isfinite(float(record[field]))
                        or not (
                            range_value(field, range_state[field][0].get())
                            <= float(record[field])
                            <= range_value(field, range_state[field][1].get())
                        )
                        for field in dict.fromkeys(fields)
                    ):
                        continue
                    x_value = evaluate(
                        record, x_equation.get(), x_field, x_field, y_field
                    )
                    y_value = evaluate(
                        record, y_equation.get(), y_field, x_field, y_field
                    )
                    if not np.isfinite(x_value) or not np.isfinite(y_value):
                        continue
                    if x_log.get() and x_value <= 0 or y_log.get() and y_value <= 0:
                        continue
                    filtered.append((record, x_value, y_value))
                except Exception:
                    continue
            groups: dict[object, list[tuple[float, float]]] = {}
            for record, x_value, y_value in filtered:
                group = record.get(split_field) if split_field != "None" else "All spectra"
                groups.setdefault(group, []).append((x_value, y_value))
            color_scale = colormaps["rainbow"]
            ordered_groups = sorted(groups.items(), key=lambda item: item[0])
            for index, (group, points) in enumerate(ordered_groups):
                points.sort(key=lambda point: point[0])
                x_values, y_values = zip(*points)
                axes.plot(
                    x_values,
                    y_values,
                    "o-",
                    color=color_scale(index / max(len(groups) - 1, 1)),
                    label=str(group),
                )
            axes.relim()
            axes.autoscale(enable=True, axis="both", tight=False)
            axes.autoscale_view()
            plotted_values = [
                point
                for _group, points in ordered_groups
                for point in points
            ]
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
            axes.set_xlabel(x_equation.get().strip() or x_field)
            axes.set_ylabel(y_equation.get().strip() or y_field)
            axes.set_xscale("log" if x_log.get() else "linear")
            axes.set_yscale("log" if y_log.get() else "linear")
            if split_field != "None" and groups:
                axes.legend(loc="best", fontsize=8)
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

        for variable in (x_var, y_var, split_var, x_equation, y_equation, x_log, y_log):
            variable.trace_add("write", lambda *_args: refresh_ranges())
        x_box.bind("<<ComboboxSelected>>", lambda _event: refresh_ranges())
        y_box.bind("<<ComboboxSelected>>", lambda _event: refresh_ranges())
        split_box.bind("<<ComboboxSelected>>", lambda _event: refresh_ranges())
        refresh_ranges()

    def refresh_fit_parameters_explorer(self, popup: tk.Toplevel) -> None:
        if popup.winfo_exists():
            popup.destroy()
        self.open_fit_parameters_explorer()

    def paste_metadata_column_from_clipboard(self) -> None:
        if self.busy or self.state is None:
            return
        visible_items = list(self.explorer.get_children(""))
        if not visible_items:
            self._update_status("no spectra are available")
            return

        dialog = MetadataColumnDialog(self.root, len(visible_items))
        self.root.wait_window(dialog)
        if dialog.result is None:
            return
        column_name, values = dialog.result

        reserved_names = {
            "source_file",
            "source_path",
            "circuit",
            "potential_V",
            "current_mA",
            "included_points",
            "total_points",
            "outlier_points",
            "minimum_frequency_Hz",
            "maximum_frequency_Hz",
            "active_minimum_frequency_Hz",
            "active_maximum_frequency_Hz",
        }
        known_names = [
            *self._explorer_base_columns(),
            *self._custom_metadata_columns,
            *self._explorer_headings.values(),
            *reserved_names,
        ]
        for loaded in self.loaded_projects.values():
            known_names.extend(str(name) for name in loaded.dataframe.columns)
            for parameter in loaded.state.default_parameters:
                known_names.extend(
                    (parameter.name, f"{parameter.name}_error_percent")
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

        for item, value in zip(visible_items, values):
            _dataset_id, loaded, spectrum = self._explorer_rows[item]
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
                rows = loaded.dataframe["cycle_number"] == spectrum.cycle
                loaded.dataframe.loc[rows, column_name] = value
            else:
                loaded.dataframe[column_name] = value

        self._populate_explorer()
        self._update_status(f"metadata column '{column_name}' added")

    def reset_points(self) -> None:
        if self.state is None:
            return
        self.state.active.reset_selection()
        self._refresh_plot(rescale=True)
        self._update_status("selection reset")

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

        for _item, _dataset_id, loaded, spectrum in rows:
            loaded.state.cycles.pop(spectrum.cycle, None)
            if spectrum.cycle in loaded.state.available_cycles:
                loaded.state.available_cycles.remove(spectrum.cycle)
            loaded.spectra = [
                entry for entry in loaded.spectra if entry.cycle != spectrum.cycle
            ]

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
            self._submit(
                lambda: [
                    (
                        loaded,
                        cycle.cycle,
                        analyze_outliers(
                            cycle,
                            threshold,
                            loaded.state.parameters_for(cycle.cycle),
                        ),
                    )
                    for loaded, cycle in missing
                ],
                lambda results: self._finish_saved_ridge_batch(results, on_ready),
                "Ridge DRT calculation failed",
            )
            return
        self.status_var.set(f"Calculating hybrid DRT for {len(missing)} spectra...")
        self._submit(
            lambda: [
                (loaded, cycle.cycle, calculate_hybrid_drt(cycle))
                for loaded, cycle in missing
            ],
            lambda results: self._finish_saved_hybrid_batch(results, on_ready),
            "Hybrid DRT calculation failed",
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
        popup_axes: dict[str, object | None] = {"main": None, "phase": None}

        def _render_popup() -> None:
            figure.clear()
            color_scale = colormaps["rainbow"]
            if mode_state["value"] == "bode":
                axes = figure.add_subplot(111)
                phase_axes = axes.twinx()
                axes.set_xscale("log")
                axes.set_xlabel("Frequency / Hz")
                axes.set_ylabel("|Z| / Ω")
                phase_axes.set_ylabel("−Phase / °")
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
                    fontsize=8,
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
                axes.legend(loc="best", fontsize=8)
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
            axes.legend(loc="best", fontsize=8)
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
            initialdir=str(self._current_directory()),
            filetypes=[("BioLogic MPT", "*.mpt"), ("All files", "*.*")],
        )
        if not selected:
            return
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
            lambda: load_projects(new_paths, control, circuit),
            self._finish_imports,
            "Data import failed",
        )

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
        if self._fit_parameter_snapshot is not None and self.state is not None:
            self.state = copy.deepcopy(self._fit_parameter_snapshot)
            refresh = getattr(self, "_refresh_parameter_table", None)
            if refresh is not None:
                refresh()
        self._update_status("fit cancellation requested")

    def save_project(self, path: Path | None = None) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        if path is None:
            selected = filedialog.asksaveasfilename(
                parent=self.root,
                title="Save EIS fitting project",
                initialdir=str(self._current_directory()),
                initialfile=f"{self._current_stem()}.eisfit.json",
                defaultextension=".eisfit.json",
                filetypes=[("EIS fitting project", "*.eisfit.json"), ("JSON", "*.json")],
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
            save_project_file(self.state, project_path, datasets=datasets)
        except Exception as error:
            messagebox.showerror(
                "Project save failed",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            return
        self.project_path = project_path.resolve()
        self._saved_project_signature = self._project_signature()
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

    def load_project(self) -> None:
        if self.busy:
            return
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Load EIS fitting project",
            initialdir=str(self._current_directory()),
            filetypes=[("EIS fitting project", "*.eisfit.json"), ("JSON", "*.json")],
        )
        if not selected:
            return
        project_path = Path(selected)
        self.status_var.set(f"Loading project {project_path.name}…")
        self._submit(
            lambda: self._load_saved_project(project_path),
            lambda result: self._finish_project_load(result, project_path),
            "Project load failed",
        )

    @staticmethod
    def _load_saved_project(
        path: Path,
    ) -> list[tuple[str, LoadedProject, ProjectState]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
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
                        cycle_number: cycle.custom_metadata
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
                cycle_number: cycle.custom_metadata
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
        self._saved_project_signature = self._project_signature()
        self.control = restored.control
        self.circuit = restored.circuit
        self.model_var.set(restored.circuit)
        self.cycle_var.set(str(restored.active_cycle))
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
