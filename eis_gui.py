from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Callable

import numpy as np

from eis_model import ParameterValue, ProjectState
from eis_project import (
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
        headers = ("Parameter", "Error (%)", "Fix", "Initial", "Lower", "Upper")
        for column, text in enumerate(headers):
            ttk.Label(self, text=text, style="Heading.TLabel").grid(
                row=0, column=column, padx=3, pady=(0, 4), sticky="ew"
            )
        for column in range(6):
            self.columnconfigure(column, weight=1 if column >= 3 else 0)

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
            error_text, error_color = self._format_error(parameter.error_percent)
            tk.Label(
                self,
                text=error_text,
                background=error_color,
                relief=tk.SOLID,
                borderwidth=1,
                width=10,
            ).grid(row=row, column=1, padx=3, pady=2, sticky="ew")
            ttk.Checkbutton(self, variable=fixed).grid(
                row=row, column=2, padx=3, pady=2
            )
            for column, variable in enumerate((initial, lower, upper), start=3):
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
        self.current_dataset_id: str | None = None
        self.requested_cycle = cycle
        self.control = control
        self.circuit = circuit
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
        self._plot_imports = None
        self.plot_mode = "nyquist"

        self.threshold_var = tk.StringVar(value=f"{threshold:g}")
        self.model_var = tk.StringVar(value=circuit)
        self.show_drt_var = tk.BooleanVar(value=False)
        self.show_kk_var = tk.BooleanVar(value=False)
        self.hide_legends_var = tk.BooleanVar(value=False)
        self.minimum_frequency_var = tk.StringVar()
        self.maximum_frequency_var = tk.StringVar()
        self.cycle_var = tk.StringVar(value=str(cycle))
        self.status_var = tk.StringVar(value="Opening application…")

        self._configure_window()
        self._build_menu()
        self._build_interface()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Up>", lambda _event: self.change_cycle(-1))
        self.root.bind("<Down>", lambda _event: self.change_cycle(1))
        self.root.bind("<Shift-Up>", lambda _event: self.change_cycle(-1, True))
        self.root.bind("<Shift-Down>", lambda _event: self.change_cycle(1, True))
        self.root.bind("<Control-Up>", lambda _event: self.change_cycle(-1, focus_only=True))
        self.root.bind("<Control-Down>", lambda _event: self.change_cycle(1, focus_only=True))
        self.root.bind("<Control-a>", self.select_all_spectra)
        self.root.bind("<Control-e>", self.open_export_menu)
        self.root.bind("<Control-s>", lambda _event: self.save_project())
        self.root.bind("<Control-Shift-O>", lambda _event: self.load_project())
        self.root.bind("<Control-o>", lambda _event: self.import_data())
        self.root.bind("<Alt-a>", lambda _event: self.copy_neighbor_fit(-1))
        self.root.bind("<Alt-d>", lambda _event: self.copy_neighbor_fit(1))
        self.root.bind("<Alt-e>", self.toggle_point_edit_mode)
        self.root.bind("<Alt-s>", lambda _event: self.fit())
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
            label="Batch fit selected",
            command=self.batch_fit_selected,
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
            "Batch fit selected",
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
        self._pan_state = None
        self.plot_controls = ttk.Frame(self.plot_frame)
        self.plot_controls.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        self.toggle_points_button = ttk.Button(
            self.plot_controls,
            text="Edit points: Off",
            command=self.toggle_point_edit_mode,
        )
        self.toggle_points_button.pack(side=tk.LEFT)
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
        self._update_plot_mode_button()
        if self.show_drt_var.get() and self.show_kk_var.get():
            grid = self.figure.add_gridspec(
                2,
                2,
                width_ratios=[1.55, 1.0],
                height_ratios=[1.0, 0.42],
            )
            self.axes = self.figure.add_subplot(grid[0, 0])
            self.kk_axes = self.figure.add_subplot(grid[1, 0])
            self.drt_axes = self.figure.add_subplot(grid[:, 1])
        elif self.show_drt_var.get():
            grid = self.figure.add_gridspec(1, 2, width_ratios=[1.55, 1.0])
            self.axes = self.figure.add_subplot(grid[0, 0])
            self.drt_axes = self.figure.add_subplot(grid[0, 1])
        elif self.show_kk_var.get():
            grid = self.figure.add_gridspec(2, 1, height_ratios=[1.0, 0.42])
            self.axes = self.figure.add_subplot(grid[0, 0])
            self.kk_axes = self.figure.add_subplot(grid[1, 0])
            self.drt_axes = None
        else:
            self.axes = self.figure.add_subplot(111)
            self.drt_axes = None
        if self.plot_mode == "bode":
            self._configure_bode_plot()
        else:
            self._configure_nyquist_plot()
        if self.kk_axes is not None:
            self.kk_axes.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.5)
            self.kk_axes.set_xscale("log")
            self.kk_axes.grid(True, alpha=0.2)
            self.kk_axes.set_xlabel("Frequency / Hz")
            self.kk_axes.set_ylabel("KK residual")
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
        self.canvas.draw_idle()

    def _build_explorer(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, padding=6)
        explorer_header = ttk.Frame(group)
        ttk.Label(explorer_header, text="Spectra explorer").pack(side=tk.LEFT)
        self.paste_metadata_button = ttk.Button(
            explorer_header,
            text="+",
            width=3,
            command=self.paste_metadata_column_from_clipboard,
        )
        self.paste_metadata_button.pack(side=tk.LEFT, padx=(6, 0))
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
            return (1, text.casefold())
        if np.isfinite(number):
            return (0, number)
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

    def _on_drt_mode_selected(self, _event=None):
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
        model_group = ttk.LabelFrame(parent, text="Fitting model", padding=8)
        model_group.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        model_group.columnconfigure(0, weight=1)
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

        parameters_group = ttk.LabelFrame(parent, text="Circuit parameters", padding=8)
        parameters_group.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        parent.rowconfigure(1, weight=1)
        self.parameter_table = ParameterTable(parameters_group)
        self.parameter_table.pack(fill=tk.BOTH, expand=True)
        self.parameters_selected_button = ttk.Button(
            parameters_group,
            text="Apply parameters to selected spectra",
            command=self.apply_parameters_to_selected,
        )
        self.parameters_selected_button.pack(fill=tk.X, pady=(8, 0))

        options_group = ttk.LabelFrame(parent, text="Selection", padding=8)
        options_group.grid(row=2, column=0, sticky="ew", pady=(0, 8))
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
        ttk.Checkbutton(
            options_group,
            text="Show DRT next to Nyquist",
            variable=self.show_drt_var,
            command=self.toggle_drt_view,
        ).grid(row=6, column=0, columnspan=2, pady=(8, 0), sticky="w")
        ttk.Checkbutton(
            options_group,
            text="Show KK residuals",
            variable=self.show_kk_var,
            command=self.toggle_kk_view,
        ).grid(row=7, column=0, columnspan=2, pady=(6, 0), sticky="w")
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
        actions.grid(row=3, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.fit_button = ttk.Button(actions, text="Fit spectrum", command=self.fit)
        self.fit_button.grid(row=0, column=0, padx=(0, 4), pady=3, sticky="ew")
        self.initial_values_button = ttk.Button(
            actions, text="Initial values", command=self.initialize_from_ridge
        )
        self.initial_values_button.grid(
            row=0, column=1, padx=(4, 0), pady=3, sticky="ew"
        )
        self.ridge_drt_button = ttk.Button(
            actions, text="Ridge DRT", command=self.calculate_ridge_drt
        )
        self.ridge_drt_button.grid(row=1, column=0, padx=(0, 4), pady=3, sticky="ew")
        self.hybrid_drt_button = ttk.Button(
            actions, text="Hybrid DRT", command=self.calculate_hybrid_drt
        )
        self.hybrid_drt_button.grid(row=1, column=1, padx=(4, 0), pady=3, sticky="ew")
        self.ridge_drt_selected_button = ttk.Button(
            actions,
            text="Ridge DRT: selected",
            command=self.calculate_selected_ridge_drts,
        )
        self.ridge_drt_selected_button.grid(
            row=2, column=0, padx=(0, 4), pady=3, sticky="ew"
        )
        self.hybrid_drt_selected_button = ttk.Button(
            actions,
            text="Hybrid DRT: selected",
            command=self.calculate_selected_hybrid_drts,
        )
        self.hybrid_drt_selected_button.grid(
            row=2, column=1, padx=(4, 0), pady=3, sticky="ew"
        )
        self.batch_fit_button = ttk.Button(
            actions,
            text="Batch fit from current",
            command=self.batch_fit,
        )
        self.batch_fit_button.grid(row=3, column=0, columnspan=2, pady=3, sticky="ew")
        self.python_export_button = ttk.Button(
            actions,
            text="Export to Python",
            command=self.export_python_workspace,
        )
        self.python_export_button.grid(
            row=4, column=0, columnspan=2, pady=3, sticky="ew"
        )
        self.action_buttons = (
            self.fit_button,
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
            self.reset_view_button,
            self.toggle_plot_mode_button,
            self.delete_spectrum_button,
            self.plot_selected_button,
            self.plot_three_electrode_button,
            self.paste_metadata_button,
            self.frequency_button,
            self.frequency_selected_button,
            self.model_button,
            self.parameters_selected_button,
        )
        self._set_controls_enabled(False)

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
            state.cycles[cycle_number] = cycle
        state.active_cycle = cycle_number
        self.path = state.source_path.resolve()
        self.current_dataset_id = dataset_id
        self.loaded = loaded
        self.state = state
        self.control = state.control
        self.circuit = state.circuit
        self.model_var.set(self.state.circuit)
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
        self.busy = True
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
        success(result)
        self._set_controls_enabled(self.state is not None)

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled and not self.busy else tk.DISABLED
        for button in getattr(self, "action_buttons", ()):
            button.configure(state=state)
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
        if cycle.frequency_window is not None:
            self.minimum_frequency_var.set(f"{cycle.frequency_window[0]:g}")
            self.maximum_frequency_var.set(f"{cycle.frequency_window[1]:g}")

    def _capture_controls(self) -> bool:
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
                if not parameter.lower <= parameter.initial <= parameter.upper:
                    raise ValueError(
                        f"{parameter.name}: initial value is outside its bounds"
                    )
        except ValueError as error:
            messagebox.showerror("Invalid value", str(error), parent=self.root)
            return False
        self.state.remember_parameters(parameters)
        previous_window = self.state.active.frequency_window
        self.state.active.frequency_window = (minimum, maximum)
        if previous_window != self.state.active.frequency_window:
            self.state.active.invalidate_drt_cache()
        return True

    def apply_parameters_to_selected(self) -> None:
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
                copied.append(
                    ParameterValue(
                        target.name,
                        target.unit,
                        source.initial,
                        source.lower,
                        source.upper,
                        target.error_percent,
                        source.fixed,
                    )
                )
            cycle.parameters = copied
            cycle.clear_fit()
            updated += 1
        self._refresh_explorer_values()
        self._restore_controls()
        self._refresh_plot(rescale=True)
        suffix = f", {skipped} skipped" if skipped else ""
        self._update_status(f"parameter settings applied to {updated} spectra{suffix}")

    def _refresh_plot(self, rescale: bool = False) -> None:
        if self.state is None:
            return
        cycle = self.state.active
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
        if self.kk_axes is not None and self.kk_real_artist is not None and self.kk_imag_artist is not None:
            if (
                cycle.kk_cache_matches()
                and cycle.kk_residual_real is not None
                and cycle.kk_residual_imag is not None
            ):
                x_values = cycle.frequency_hz[cycle.included]
                self.kk_real_artist.set_data(x_values, cycle.kk_residual_real)
                self.kk_imag_artist.set_data(x_values, cycle.kk_residual_imag)
            else:
                self.kk_real_artist.set_data([], [])
                self.kk_imag_artist.set_data([], [])
        self.axes.set_title(
            (
                f"{self.loaded.dataset_label if self.loaded is not None else self._current_name()}\n"
                f"Cycle {cycle.cycle} · {self.state.circuit}"
            )
        )
        if self.drt_artist is not None and self.drt_axes is not None:
            drt_tau_s, drt_gamma_ohm, drt_label = self._apply_saved_drt_mode(cycle)
            if drt_tau_s is None or drt_gamma_ohm is None:
                self.drt_artist.set_data([], [])
            else:
                self.drt_artist.set_data(drt_tau_s, drt_gamma_ohm)
            self.drt_axes.set_title(drt_label)
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

    def _autoscale_kk(self, cycle) -> None:
        if (
            self.kk_axes is None
            or cycle.kk_residual_real is None
            or cycle.kk_residual_imag is None
            or not cycle.kk_cache_matches()
        ):
            return
        y_values = np.concatenate((cycle.kk_residual_real, cycle.kk_residual_imag))
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

    def _on_plot_button_press(self, event) -> None:
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
        if event.button != 2 or self._pan_state is None:
            return
        self._pan_state = None
        if hasattr(self, "zoom_selector"):
            self.zoom_selector.set_active(not self.point_toggle_mode)

    def _on_plot_motion(self, event) -> None:
        if self._pan_state is None:
            return
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

    def toggle_point_edit_mode(self, _event=None) -> str | None:
        self.point_toggle_mode = not self.point_toggle_mode
        self.toggle_points_button.configure(
            text=f"Edit points: {'On' if self.point_toggle_mode else 'Off'}"
        )
        if hasattr(self, "zoom_selector"):
            self.zoom_selector.set_active(not self.point_toggle_mode)
        self._update_status()
        return "break" if _event is not None else None

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
            self.model_var.set(self.state.circuit)
            return
        self.state.replace_circuit(circuit, parameters)
        self.circuit = circuit
        self.parameter_table.set_parameters(
            self.state.parameters_for(self.state.active_cycle)
        )
        self._refresh_plot(rescale=True)
        self._update_status("fitting model changed")

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
        cycle.parameters = analysis.parameters
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
            self.parameter_table.set_parameters(analysis.parameters)
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
            lambda: fit_cycle(cycle, self.state.circuit, parameters),
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

    def batch_fit_selected(self) -> None:
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
        batch_items = [
            item
            for item in visible_items[start_index:]
            if item in self.explorer.selection()
        ]
        if len(batch_items) < 2:
            self._update_status(
                "select spectra from the displayed one downward in the explorer"
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
            f"Batch fitting {len(targets)} selected spectra from the displayed one..."
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

    def copy_neighbor_fit(self, direction: int) -> None:
        if self.state is None or self.busy or not self._capture_controls():
            return
        cycles = self.state.available_cycles
        current_index = cycles.index(self.state.active_cycle)
        source_index = current_index + direction
        if not 0 <= source_index < len(cycles):
            self._update_status("no neighboring cycle in that direction")
            return
        source_cycle_number = cycles[source_index]
        source = self.state.cycles.get(source_cycle_number)
        if source is None or source.fit_parameters is None:
            self._update_status(f"cycle {source_cycle_number} has no fit to copy")
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
            f"initial parameters copied from cycle {source_cycle_number}"
        )

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
            color_scale = colormaps["tab20"]
            if mode_state["value"] == "bode":
                axes = figure.add_subplot(111)
                phase_axes = axes.twinx()
                axes.set_xscale("log")
                axes.set_xlabel("Frequency / Hz")
                axes.set_ylabel("|Z| / Ω")
                phase_axes.set_ylabel("−Phase / °")
                axes.grid(True, alpha=0.25)
                for index, (loaded, cycle) in enumerate(plotted_cycles):
                    color = color_scale(index % color_scale.N)
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
                for index, (loaded, cycle) in enumerate(plotted_cycles):
                    color = color_scale(index % color_scale.N)
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
            color_scale = colormaps["tab20"]
            for index, (loaded, cycle) in enumerate(plotted_cycles):
                color = color_scale(index % color_scale.N)
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
        circuit = self.state.circuit if self.state is not None else self.circuit
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
        self._populate_explorer()
        if report.loaded:
            dataset_id, loaded = report.loaded[0]
            self._switch_dataset(
                dataset_id,
                loaded,
                loaded.state.active_cycle,
                capture_current=False,
            )
        if report.errors:
            details = "\n".join(
                f"{path.name}: {error}" for path, error in report.errors
            )
            messagebox.showwarning(
                "Some files were not imported",
                details,
                parent=self.root,
            )
        self._update_status(
            f"added {len(report.loaded)} files; "
            f"{len(self._explorer_rows)} spectra loaded"
        )

    def save_project(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
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
            save_project_file(self.state, Path(selected), datasets=datasets)
        except Exception as error:
            messagebox.showerror(
                "Project save failed",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            return
        self._update_status(f"project saved as {Path(selected).name}")

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
        self.loaded_projects.clear()
        self._dataset_order.clear()
        for dataset_id, loaded, _restored in result:
            self._register_dataset(dataset_id, loaded)
        dataset_id, loaded, restored = result[0]
        self.current_dataset_id = dataset_id
        self.loaded = loaded
        self.state = restored
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
