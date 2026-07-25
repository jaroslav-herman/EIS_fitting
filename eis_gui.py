from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
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
    export_fit_parameters,
    export_python_workspace as write_python_workspace,
    load_project_file,
    save_project_file,
)
from eis_services import (
    BatchFitReport,
    LoadedProject,
    ProjectImportReport,
    RidgeInitialization,
    SpectrumBatchReport,
    SpectrumFitTarget,
    SpectrumMetadata,
    batch_fit_from_cycle,
    batch_fit_spectra,
    catalog_spectra,
    circuit_parameters,
    analyze_outliers,
    find_outliers_for_all_cycles,
    fit_cycle,
    load_cycle,
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
                tk.StringVar,
                tk.StringVar,
                tk.StringVar,
            ]
        ] = []
        headers = ("Parameter", "Error (%)", "Initial", "Lower", "Upper")
        for column, text in enumerate(headers):
            ttk.Label(self, text=text, style="Heading.TLabel").grid(
                row=0, column=column, padx=3, pady=(0, 4), sticky="ew"
            )
        for column in range(5):
            self.columnconfigure(column, weight=1 if column >= 2 else 0)

    def set_parameters(self, parameters: list[ParameterValue]) -> None:
        for child in self.grid_slaves():
            if int(child.grid_info()["row"]) > 0:
                child.destroy()
        self._rows.clear()
        for row, parameter in enumerate(parameters, start=1):
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
            for column, variable in enumerate((initial, lower, upper), start=2):
                ttk.Entry(self, textvariable=variable, width=10).grid(
                    row=row, column=column, padx=3, pady=2, sticky="ew"
                )
            self._rows.append(
                (
                    parameter.name,
                    parameter.unit,
                    parameter.error_percent,
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
        for name, unit, error_percent, initial, lower, upper in self._rows:
            parameters.append(
                ParameterValue(
                    name,
                    unit,
                    float(initial.get()),
                    float(lower.get()),
                    float(upper.get()),
                    error_percent,
                )
            )
        return parameters


class EISApplication:
    def __init__(
        self,
        root: tk.Tk,
        path: Path,
        cycle: int,
        control: str,
        threshold: float,
        circuit: str,
    ) -> None:
        self.root = root
        self.path = path.resolve()
        self.requested_cycle = cycle
        self.control = control
        self.circuit = circuit
        self.loaded: LoadedProject | None = None
        self.state: ProjectState | None = None
        self.loaded_projects: dict[Path, LoadedProject] = {}
        self._dataset_order: list[Path] = []
        self._explorer_rows: dict[str, tuple[Path, LoadedProject, SpectrumMetadata]] = (
            {}
        )
        self._explorer_lookup: dict[tuple[Path, int], str] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="eis-worker"
        )
        self.busy = False
        self._plot_imports = None

        self.threshold_var = tk.StringVar(value=f"{threshold:g}")
        self.model_var = tk.StringVar(value=circuit)
        self.minimum_frequency_var = tk.StringVar()
        self.maximum_frequency_var = tk.StringVar()
        self.cycle_var = tk.StringVar(value=str(cycle))
        self.status_var = tk.StringVar(value="Opening application…")

        self._configure_window()
        self._build_menu()
        self._build_interface()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Left>", lambda _event: self.change_cycle(-1))
        self.root.bind("<Right>", lambda _event: self.change_cycle(1))
        self.root.bind("<Control-s>", lambda _event: self.save_project())
        self.root.bind("<Control-Shift-O>", lambda _event: self.load_project())
        self.root.bind("<Control-o>", lambda _event: self.import_data())
        self.root.bind("<Alt-a>", lambda _event: self.copy_neighbor_fit(-1))
        self.root.bind("<Alt-d>", lambda _event: self.copy_neighbor_fit(1))
        self.root.bind("<Alt-s>", lambda _event: self.fit())
        self.root.after(30, self._begin_loading)

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
        self.file_menu.add_command(
            label="Export fit parameters…",
            command=self.export_fits,
        )
        self.file_menu.add_command(
            label="Export Python workspace…",
            command=self.export_python_workspace,
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.close)
        menu_bar.add_cascade(label="File", menu=self.file_menu)
        self._project_menu_actions = (
            "Load project…",
            "Save project…",
            "Save current mask…",
            "Export fit parameters…",
            "Export Python workspace…",
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
        self.root.configure(menu=menu_bar)
        self._fit_menu_actions = (
            "Fit selected spectrum",
            "Batch down",
            "Batch up",
            "Batch down to metadata value…",
            "Batch up to metadata value…",
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
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
        from matplotlib.collections import LineCollection
        from matplotlib.figure import Figure

        figure = Figure(figsize=(7.5, 6.5), dpi=100, constrained_layout=True)
        self.axes = figure.add_subplot(111)
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
        self.residual_artist = LineCollection(
            [],
            colors="#777777",
            linewidths=0.9,
            linestyles="dashed",
            alpha=0.3,
            zorder=1,
            label="Measured-to-fit difference",
        )
        self.excluded_residual_artist = LineCollection(
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
        self.axes.legend(loc="best")
        self.canvas = FigureCanvasTkAgg(figure, master=self.plot_frame)
        self.canvas.draw()
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

    def _build_explorer(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Spectra explorer", padding=6)
        group.pack(fill=tk.BOTH, expand=True)
        group.columnconfigure(0, weight=1)
        group.rowconfigure(0, weight=1)
        columns = (
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
            selectmode="browse",
            height=7,
        )
        self._explorer_headings = {
            "source": "Source file",
            "cycle": "Cycle",
            "potential": "Voltage (V)",
            "current": "Current (mA)",
            "points": "Points",
            "f_min": "Min frequency (Hz)",
            "f_max": "Max frequency (Hz)",
        }
        self._explorer_attributes = {
            "source": None,
            "cycle": "cycle",
            "potential": "potential_v",
            "current": "current_ma",
            "points": "point_count",
            "f_min": "minimum_frequency_hz",
            "f_max": "maximum_frequency_hz",
        }
        self._explorer_sort_reverse: dict[str, bool] = {}
        self._explorer_selected_column = "cycle"
        widths = {
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
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.explorer.bind("<<TreeviewSelect>>", self._select_explorer_spectrum)

    def _populate_explorer(self) -> None:
        self.explorer.delete(*self.explorer.get_children())
        self._explorer_rows.clear()
        self._explorer_lookup.clear()
        for dataset_index, path in enumerate(self._dataset_order):
            loaded = self.loaded_projects[path]
            for spectrum in loaded.spectra:
                item = f"dataset_{dataset_index}_cycle_{spectrum.cycle}"
                self._explorer_rows[item] = (path, loaded, spectrum)
                self._explorer_lookup[(path, spectrum.cycle)] = item
                self.explorer.insert(
                    "",
                    tk.END,
                    iid=item,
                    values=(
                        path.name,
                        spectrum.cycle,
                        f"{spectrum.potential_v:.8g}",
                        f"{spectrum.current_ma:.8g}",
                        spectrum.point_count,
                        f"{spectrum.minimum_frequency_hz:.8g}",
                        f"{spectrum.maximum_frequency_hz:.8g}",
                    ),
                )

    def _sort_explorer(self, column: str) -> None:
        if not self._explorer_rows:
            return
        reverse = self._explorer_sort_reverse.get(column, False)
        self._explorer_selected_column = column
        attribute = self._explorer_attributes[column]
        ordered = sorted(
            self._explorer_rows.items(),
            key=(
                (lambda item: item[1][0].name.casefold())
                if attribute is None
                else (lambda item: getattr(item[1][2], attribute))
            ),
            reverse=reverse,
        )
        for index, (item, _row) in enumerate(ordered):
            self.explorer.move(item, "", index)
        for name, label in self._explorer_headings.items():
            marker = ""
            if name == column:
                marker = " ▼" if reverse else " ▲"
            self.explorer.heading(name, text=f"{label}{marker}")
        self._explorer_sort_reverse[column] = not reverse

    def _select_explorer_spectrum(self, _event=None) -> None:
        if self.busy or self.state is None:
            return
        selected = self.explorer.selection()
        if selected:
            path, loaded, spectrum = self._explorer_rows[selected[0]]
            if loaded is self.loaded:
                self._activate_cycle(spectrum.cycle)
            else:
                self._switch_dataset(path, loaded, spectrum.cycle)

    def _highlight_explorer_cycle(self, cycle_number: int) -> None:
        item = self._explorer_lookup.get((self.path.resolve(), cycle_number))
        if item is not None and self.explorer.exists(item):
            self.explorer.selection_set(item)
            self.explorer.focus(item)
            self.explorer.see(item)

    def _build_controls(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        cycle_group = ttk.LabelFrame(parent, text="Cycle", padding=8)
        cycle_group.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        cycle_group.columnconfigure(1, weight=1)
        self.previous_button = ttk.Button(
            cycle_group, text="◀", width=4, command=lambda: self.change_cycle(-1)
        )
        self.previous_button.grid(row=0, column=0, padx=(0, 5))
        self.cycle_box = ttk.Combobox(
            cycle_group, textvariable=self.cycle_var, state="readonly", width=12
        )
        self.cycle_box.grid(row=0, column=1, sticky="ew")
        self.cycle_box.bind("<<ComboboxSelected>>", self._select_cycle)
        self.next_button = ttk.Button(
            cycle_group, text="▶", width=4, command=lambda: self.change_cycle(1)
        )
        self.next_button.grid(row=0, column=2, padx=(5, 0))

        model_group = ttk.LabelFrame(parent, text="Fitting model", padding=8)
        model_group.grid(row=1, column=0, sticky="ew", pady=(0, 8))
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
        parameters_group.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        parent.rowconfigure(2, weight=1)
        self.parameter_table = ParameterTable(parameters_group)
        self.parameter_table.pack(fill=tk.BOTH, expand=True)

        options_group = ttk.LabelFrame(parent, text="Selection", padding=8)
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
        self.frequency_all_button = ttk.Button(
            options_group,
            text="Apply to all cycles",
            command=self.apply_frequency_window_to_all,
        )
        self.frequency_all_button.grid(
            row=2, column=1, padx=(4, 0), pady=(6, 0), sticky="ew"
        )
        ttk.Label(options_group, text="Outlier threshold").grid(
            row=3, column=0, pady=(8, 2), sticky="w"
        )
        ttk.Entry(options_group, textvariable=self.threshold_var).grid(
            row=3, column=1, padx=(8, 0), pady=(8, 2), sticky="ew"
        )

        actions = ttk.LabelFrame(parent, text="Actions", padding=8)
        actions.grid(row=4, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.fit_button = ttk.Button(actions, text="Fit spectrum", command=self.fit)
        self.fit_button.grid(row=0, column=0, padx=(0, 4), pady=3, sticky="ew")
        self.outlier_button = ttk.Button(
            actions, text="Outliers: current", command=self.find_outliers
        )
        self.outlier_button.grid(row=0, column=1, padx=(4, 0), pady=3, sticky="ew")
        self.outlier_all_button = ttk.Button(
            actions,
            text="Outliers: all cycles",
            command=self.find_outliers_for_all,
        )
        self.outlier_all_button.grid(row=1, column=0, padx=(0, 4), pady=3, sticky="ew")
        self.reset_button = ttk.Button(
            actions, text="Reset points", command=self.reset_points
        )
        self.reset_button.grid(row=1, column=1, padx=(4, 0), pady=3, sticky="ew")
        self.batch_fit_button = ttk.Button(
            actions,
            text="Batch fit from current",
            command=self.batch_fit,
        )
        self.batch_fit_button.grid(row=2, column=0, columnspan=2, pady=3, sticky="ew")
        self.python_export_button = ttk.Button(
            actions,
            text="Export to Python",
            command=self.export_python_workspace,
        )
        self.python_export_button.grid(
            row=3, column=0, columnspan=2, pady=3, sticky="ew"
        )
        self.action_buttons = (
            self.fit_button,
            self.outlier_button,
            self.outlier_all_button,
            self.reset_button,
            self.batch_fit_button,
            self.python_export_button,
            self.frequency_button,
            self.frequency_all_button,
            self.model_button,
            self.previous_button,
            self.next_button,
        )
        self._set_controls_enabled(False)

    def _begin_loading(self) -> None:
        self.status_var.set(f"Loading {self.path.name}…")
        self._submit(
            lambda: load_project(
                self.path,
                self.requested_cycle,
                self.control,
                self.circuit,
            ),
            self._finish_loading,
            "Could not open the spectrum",
        )

    def _finish_loading(self, loaded: LoadedProject) -> None:
        self._register_dataset(self.path, loaded)
        self._populate_explorer()
        self._switch_dataset(
            self.path,
            loaded,
            loaded.state.active_cycle,
            capture_current=False,
        )
        self._set_controls_enabled(True)
        self._update_status()

    def _register_dataset(self, path: Path, loaded: LoadedProject) -> None:
        path = path.resolve()
        loaded.state.source_path = path
        if path not in self.loaded_projects:
            self._dataset_order.append(path)
        self.loaded_projects[path] = loaded

    def _switch_dataset(
        self,
        path: Path,
        loaded: LoadedProject,
        cycle_number: int,
        *,
        capture_current: bool = True,
    ) -> None:
        if capture_current and self.state is not None and not self._capture_controls():
            self._highlight_explorer_cycle(self.state.active_cycle)
            return
        path = path.resolve()
        state = loaded.state
        if cycle_number not in state.cycles:
            cycle = load_cycle(loaded.dataframe, cycle_number, state.control)
            if state.all_frequency_window is not None:
                cycle.frequency_window = state.all_frequency_window
            cycle.parameters = state.parameters_for(cycle_number)
            state.cycles[cycle_number] = cycle
        state.active_cycle = cycle_number
        self.path = path
        self.loaded = loaded
        self.state = state
        self.control = state.control
        self.circuit = state.circuit
        self.model_var.set(self.state.circuit)
        self.root.title(f"EIS Fitting — {self.path.name}")
        self.cycle_box.configure(
            values=[str(cycle) for cycle in self.state.available_cycles]
        )
        self.cycle_var.set(str(self.state.active_cycle))
        self._highlight_explorer_cycle(self.state.active_cycle)
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._update_status(f"source: {self.path.name}")

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
        if hasattr(self, "cycle_box"):
            self.cycle_box.configure(
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
            for label in self._project_menu_actions:
                self.file_menu.entryconfigure(label, state=menu_state)
        if hasattr(self, "fit_menu"):
            for label in self._fit_menu_actions:
                self.fit_menu.entryconfigure(label, state=menu_state)

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
        self.state.active.frequency_window = (minimum, maximum)
        return True

    def _refresh_plot(self, rescale: bool = False) -> None:
        if self.state is None:
            return
        cycle = self.state.active
        included = cycle.included
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
        self.axes.set_title(
            f"{self.path.name}\nCycle {cycle.cycle} · {self.state.circuit}"
        )
        if rescale:
            self._autoscale_to_included(cycle)
        self.canvas.draw_idle()

    def _autoscale_to_included(self, cycle) -> None:
        included = cycle.included
        if not np.any(included):
            included = np.ones(cycle.frequency_hz.size, dtype=bool)
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

    def _on_plot_click(self, event) -> None:
        if self.busy or self.state is None or event.inaxes is not self.axes:
            return
        if event.x is None or event.y is None:
            return
        cycle = self.state.active
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

    def _select_cycle(self, _event=None) -> None:
        if self.state is None:
            return
        self._activate_cycle(int(self.cycle_var.get()))

    def change_cycle(self, direction: int) -> None:
        if self.busy or self.state is None:
            return
        cycles = self.state.available_cycles
        index = cycles.index(self.state.active_cycle)
        next_index = max(0, min(len(cycles) - 1, index + direction))
        self._activate_cycle(cycles[next_index])

    def _activate_cycle(self, cycle_number: int) -> None:
        if self.state is None or cycle_number == self.state.active_cycle:
            return
        if not self._capture_controls():
            self.cycle_var.set(str(self.state.active_cycle))
            self._highlight_explorer_cycle(self.state.active_cycle)
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
        self._highlight_explorer_cycle(cycle_number)
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._update_status()

    def apply_frequency_window(self) -> None:
        if self._capture_controls():
            self.state.active.clear_fit()
            self._refresh_plot(rescale=True)
            self._update_status("frequency range applied")

    def apply_frequency_window_to_all(self) -> None:
        if self.state is None or not self._capture_controls():
            return
        window = self.state.active.frequency_window
        assert window is not None
        self.state.apply_frequency_window_to_all(window)
        self._refresh_plot(rescale=True)
        self._update_status("frequency range applied to all cycles")

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
            lambda: analyze_outliers(cycle, threshold, parameters),
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

    def find_outliers_for_all(self) -> None:
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
            attribute = self._explorer_attributes[column]
            if attribute is None:
                messagebox.showerror(
                    "Select numeric metadata",
                    "Choose a numeric explorer column such as voltage or current first.",
                    parent=self.root,
                )
                return
            label = self._explorer_headings[column]
            target_value = simpledialog.askfloat(
                "Batch fit limit",
                f"Stop near which {label} value?",
                parent=self.root,
            )
            if target_value is None:
                return
            nearest_index = min(
                range(len(batch_items)),
                key=lambda index: abs(
                    float(
                        getattr(
                            self._explorer_rows[batch_items[index]][2],
                            attribute,
                        )
                    )
                    - target_value
                ),
            )
            batch_items = batch_items[: nearest_index + 1]
            target_description = f" toward {label}={target_value:g}"

        targets = []
        for item in batch_items:
            path, loaded, spectrum = self._explorer_rows[item]
            targets.append(
                SpectrumFitTarget(
                    loaded=loaded,
                    cycle=spectrum.cycle,
                    label=f"{path.name}, cycle {spectrum.cycle}",
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

    def reset_points(self) -> None:
        if self.state is None:
            return
        self.state.active.reset_selection()
        self._refresh_plot(rescale=True)
        self._update_status("selection reset")

    def save_mask(self) -> None:
        if self.busy or self.state is None:
            return
        default_name = (
            f"{self.path.stem}_cycle{self.state.active_cycle}_mask_included.npy"
        )
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save included-point mask",
            initialdir=str(self.path.parent),
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
            initialdir=str(self.path.parent),
            filetypes=[("BioLogic MPT", "*.mpt"), ("All files", "*.*")],
        )
        if not selected:
            return
        selected_paths = list(
            dict.fromkeys(Path(value).resolve() for value in selected)
        )
        new_paths = [
            path for path in selected_paths if path not in self.loaded_projects
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
        for path, loaded in report.loaded:
            self._register_dataset(path, loaded)
        self._populate_explorer()
        if report.loaded:
            path, loaded = report.loaded[0]
            self._switch_dataset(
                path,
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
            initialdir=str(self.path.parent),
            initialfile=f"{self.path.stem}.eisfit.json",
            defaultextension=".eisfit.json",
            filetypes=[("EIS fitting project", "*.eisfit.json"), ("JSON", "*.json")],
        )
        if not selected:
            return
        try:
            save_project_file(self.state, Path(selected))
        except Exception as error:
            messagebox.showerror(
                "Project save failed",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            return
        self._update_status(f"project saved as {Path(selected).name}")

    def load_project(self) -> None:
        if self.busy or self.state is None or self.loaded is None:
            return
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Load EIS fitting project",
            initialdir=str(self.path.parent),
            filetypes=[("EIS fitting project", "*.eisfit.json"), ("JSON", "*.json")],
        )
        if not selected:
            return
        current = self.state
        dataframe = self.loaded.dataframe
        project_path = Path(selected)
        self.status_var.set(f"Loading project {project_path.name}…")
        self._submit(
            lambda: load_project_file(current, dataframe, project_path),
            lambda restored: self._finish_project_load(restored, project_path),
            "Project load failed",
        )

    def _finish_project_load(self, restored: ProjectState, path: Path) -> None:
        self.state = restored
        assert self.loaded is not None
        self.loaded.state = restored
        self.loaded.spectra = catalog_spectra(
            self.loaded.dataframe,
            restored.available_cycles,
            restored.control,
        )
        self.loaded_projects[self.path.resolve()] = self.loaded
        self.control = restored.control
        self.circuit = restored.circuit
        self.model_var.set(restored.circuit)
        self.cycle_var.set(str(restored.active_cycle))
        self.cycle_box.configure(
            values=[str(cycle) for cycle in restored.available_cycles]
        )
        self._populate_explorer()
        self._highlight_explorer_cycle(restored.active_cycle)
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._update_status(f"project loaded from {path.name}")

    def export_fits(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export fitted parameters",
            initialdir=str(self.path.parent),
            initialfile=f"{self.path.stem}_fit_parameters.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not selected:
            return
        try:
            count = export_fit_parameters(self.state, Path(selected))
        except Exception as error:
            messagebox.showerror(
                "Fit export failed",
                f"{type(error).__name__}: {error}",
                parent=self.root,
            )
            return
        self._update_status(f"exported fit parameters for {count} cycles")

    def export_python_workspace(self) -> None:
        if self.busy or self.state is None or not self._capture_controls():
            return
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export parameters and metadata for Python",
            initialdir=str(self.path.parent),
            initialfile=f"{self.path.stem}_analysis.csv",
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
        states = [
            self.loaded_projects[path].state for path in self._dataset_order
        ]
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
    mpt_path: Path,
    cycle: int = 1,
    control: str = "Ewe",
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
