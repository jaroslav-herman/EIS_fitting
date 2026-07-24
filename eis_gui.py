from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import numpy as np

from eis_model import ParameterValue, ProjectState
from eis_project import export_fit_parameters, load_project_file, save_project_file
from eis_services import (
    BatchFitReport,
    LoadedProject,
    batch_fit_from_cycle,
    circuit_parameters,
    find_outlier_indices,
    find_outliers_for_all_cycles,
    fit_cycle,
    load_cycle,
    load_project,
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
        self._rows: list[tuple[str, str, tk.StringVar, tk.StringVar, tk.StringVar]] = []
        headers = ("Parameter", "Unit", "Initial", "Lower", "Upper")
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
            ttk.Label(self, text=parameter.unit).grid(row=row, column=1, padx=3, pady=2)
            for column, variable in enumerate((initial, lower, upper), start=2):
                ttk.Entry(self, textvariable=variable, width=10).grid(
                    row=row, column=column, padx=3, pady=2, sticky="ew"
                )
            self._rows.append((parameter.name, parameter.unit, initial, lower, upper))

    def values(self) -> list[ParameterValue]:
        parameters = []
        for name, unit, initial, lower, upper in self._rows:
            parameters.append(
                ParameterValue(
                    name,
                    unit,
                    float(initial.get()),
                    float(lower.get()),
                    float(upper.get()),
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
        self.path = path
        self.requested_cycle = cycle
        self.control = control
        self.circuit = circuit
        self.loaded: LoadedProject | None = None
        self.state: ProjectState | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eis-worker")
        self.busy = False
        self._plot_imports = None

        self.threshold_var = tk.StringVar(value=f"{threshold:g}")
        self.model_var = tk.StringVar(value=circuit)
        self.minimum_frequency_var = tk.StringVar()
        self.maximum_frequency_var = tk.StringVar()
        self.cycle_var = tk.StringVar(value=str(cycle))
        self.status_var = tk.StringVar(value="Opening application…")

        self._configure_window()
        self._build_interface()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Left>", lambda _event: self.change_cycle(-1))
        self.root.bind("<Right>", lambda _event: self.change_cycle(1))
        self.root.bind("<Control-s>", lambda _event: self.save_mask())
        self.root.bind("<Alt-a>", lambda _event: self.copy_neighbor_fit(-1))
        self.root.bind("<Alt-d>", lambda _event: self.copy_neighbor_fit(1))
        self.root.after(30, self._begin_loading)

    def _configure_window(self) -> None:
        self.root.title("EIS Fitting")
        self.root.geometry("1220x760")
        self.root.minsize(940, 620)
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Heading.TLabel", font=("Segoe UI", 9, "bold"))

    def _build_interface(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.grid(row=0, column=0, sticky="nsew")

        self.plot_frame = ttk.Frame(body, padding=8)
        controls = ttk.Frame(body, padding=(8, 10, 12, 8), width=390)
        body.add(self.plot_frame, weight=4)
        body.add(controls, weight=0)
        self._build_plot()
        self._build_controls(controls)

        ttk.Separator(self.root).grid(row=1, column=0, sticky="ew")
        ttk.Label(self.root, textvariable=self.status_var, padding=(8, 5)).grid(
            row=2, column=0, sticky="ew"
        )

    def _build_plot(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.collections import LineCollection
        from matplotlib.figure import Figure

        figure = Figure(figsize=(7.5, 6.5), dpi=100, constrained_layout=True)
        self.axes = figure.add_subplot(111)
        self.axes.set_xlabel("Re(Z) / Ω")
        self.axes.set_ylabel("−Im(Z) / Ω")
        self.axes.set_aspect("equal", adjustable="datalim")
        self.axes.grid(True, alpha=0.25)
        self.included_artist, = self.axes.plot(
            [], [], "o", color="#1769aa", markersize=5, label="Included"
        )
        self.excluded_artist, = self.axes.plot(
            [], [], "x", color="#c62828", markersize=6, label="Excluded"
        )
        self.fit_artist, = self.axes.plot(
            [], [], "-", color="#202020", linewidth=2, alpha=0.8, label="Fit"
        )
        self.fit_points_artist, = self.axes.plot(
            [],
            [],
            "o",
            color="#f57c00",
            markersize=4,
            label="Fit at measured frequencies",
        )
        self.residual_artist = LineCollection(
            [],
            colors="#777777",
            linewidths=0.9,
            linestyles="dashed",
            alpha=0.35,
            zorder=1,
            label="Measured-to-fit difference",
        )
        self.axes.add_collection(self.residual_artist)
        self.axes.legend(loc="best")
        self.canvas = FigureCanvasTkAgg(figure, master=self.plot_frame)
        self.canvas.draw()
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

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
        self.model_button = ttk.Button(model_group, text="Set model", command=self.apply_model)
        self.model_button.grid(row=0, column=1)

        parameters_group = ttk.LabelFrame(parent, text="Circuit parameters", padding=8)
        parameters_group.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        parent.rowconfigure(2, weight=1)
        self.parameter_table = ParameterTable(parameters_group)
        self.parameter_table.pack(fill=tk.BOTH, expand=True)

        options_group = ttk.LabelFrame(parent, text="Selection", padding=8)
        options_group.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        options_group.columnconfigure(1, weight=1)
        ttk.Label(options_group, text="Min frequency (Hz)").grid(row=0, column=0, sticky="w")
        ttk.Entry(options_group, textvariable=self.minimum_frequency_var).grid(
            row=0, column=1, padx=(8, 0), pady=2, sticky="ew"
        )
        ttk.Label(options_group, text="Max frequency (Hz)").grid(row=1, column=0, sticky="w")
        ttk.Entry(options_group, textvariable=self.maximum_frequency_var).grid(
            row=1, column=1, padx=(8, 0), pady=2, sticky="ew"
        )
        self.frequency_button = ttk.Button(
            options_group,
            text="Apply to current cycle",
            command=self.apply_frequency_window,
        )
        self.frequency_button.grid(row=2, column=0, padx=(0, 4), pady=(6, 0), sticky="ew")
        self.frequency_all_button = ttk.Button(
            options_group,
            text="Apply to all cycles",
            command=self.apply_frequency_window_to_all,
        )
        self.frequency_all_button.grid(row=2, column=1, padx=(4, 0), pady=(6, 0), sticky="ew")
        ttk.Label(options_group, text="Outlier threshold").grid(row=3, column=0, pady=(8, 2), sticky="w")
        ttk.Entry(options_group, textvariable=self.threshold_var).grid(
            row=3, column=1, padx=(8, 0), pady=(8, 2), sticky="ew"
        )

        actions = ttk.LabelFrame(parent, text="Actions", padding=8)
        actions.grid(row=4, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.fit_button = ttk.Button(actions, text="Fit spectrum", command=self.fit)
        self.fit_button.grid(row=0, column=0, padx=(0, 4), pady=3, sticky="ew")
        self.outlier_button = ttk.Button(actions, text="Outliers: current", command=self.find_outliers)
        self.outlier_button.grid(row=0, column=1, padx=(4, 0), pady=3, sticky="ew")
        self.outlier_all_button = ttk.Button(
            actions,
            text="Outliers: all cycles",
            command=self.find_outliers_for_all,
        )
        self.outlier_all_button.grid(row=1, column=0, padx=(0, 4), pady=3, sticky="ew")
        self.reset_button = ttk.Button(actions, text="Reset points", command=self.reset_points)
        self.reset_button.grid(row=1, column=1, padx=(4, 0), pady=3, sticky="ew")
        self.batch_fit_button = ttk.Button(
            actions,
            text="Batch fit from current",
            command=self.batch_fit,
        )
        self.batch_fit_button.grid(row=2, column=0, columnspan=2, pady=3, sticky="ew")
        self.save_button = ttk.Button(actions, text="Save mask", command=self.save_mask)
        self.save_button.grid(row=3, column=0, columnspan=2, pady=3, sticky="ew")

        files = ttk.LabelFrame(parent, text="Project and export", padding=8)
        files.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        files.columnconfigure(0, weight=1)
        files.columnconfigure(1, weight=1)
        self.save_project_button = ttk.Button(
            files, text="Save project", command=self.save_project
        )
        self.save_project_button.grid(row=0, column=0, padx=(0, 4), pady=3, sticky="ew")
        self.load_project_button = ttk.Button(
            files, text="Load project", command=self.load_project
        )
        self.load_project_button.grid(row=0, column=1, padx=(4, 0), pady=3, sticky="ew")
        self.export_fits_button = ttk.Button(
            files, text="Export fit parameters", command=self.export_fits
        )
        self.export_fits_button.grid(
            row=1, column=0, columnspan=2, pady=3, sticky="ew"
        )
        self.action_buttons = (
            self.fit_button,
            self.outlier_button,
            self.outlier_all_button,
            self.reset_button,
            self.batch_fit_button,
            self.save_button,
            self.frequency_button,
            self.frequency_all_button,
            self.model_button,
            self.save_project_button,
            self.load_project_button,
            self.export_fits_button,
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
        self.loaded = loaded
        self.state = loaded.state
        self.model_var.set(self.state.circuit)
        self.cycle_box.configure(values=[str(cycle) for cycle in self.state.available_cycles])
        self.cycle_var.set(str(self.state.active_cycle))
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._set_controls_enabled(True)
        self._update_status()

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
            messagebox.showerror(error_title, f"{type(error).__name__}: {error}", parent=self.root)
            return
        success(result)
        self._set_controls_enabled(self.state is not None)

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled and not self.busy else tk.DISABLED
        for button in getattr(self, "action_buttons", ()):
            button.configure(state=state)
        if hasattr(self, "cycle_box"):
            self.cycle_box.configure(state="readonly" if enabled and not self.busy else "disabled")
        if hasattr(self, "model_box"):
            self.model_box.configure(state="normal" if enabled and not self.busy else "disabled")

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
                    raise ValueError(f"{parameter.name}: lower bound exceeds upper bound")
                if not parameter.lower <= parameter.initial <= parameter.upper:
                    raise ValueError(f"{parameter.name}: initial value is outside its bounds")
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
            self.fit_artist.set_data(cycle.fit_impedance.real, -cycle.fit_impedance.imag)
        if cycle.fit_at_data_impedance is None:
            self.fit_points_artist.set_data([], [])
            self.residual_artist.set_segments([])
        else:
            fitted = cycle.fit_at_data_impedance
            self.fit_points_artist.set_data(fitted.real, -fitted.imag)
            measured_points = np.column_stack((real, negative_imaginary))
            fitted_points = np.column_stack((fitted.real, -fitted.imag))
            self.residual_artist.set_segments(
                np.stack((measured_points, fitted_points), axis=1)
            )
        self.axes.set_title(
            f"{self.path.name}\nCycle {cycle.cycle} · {self.state.circuit}"
        )
        if rescale:
            self.axes.relim()
            self.axes.autoscale_view()
        self.canvas.draw_idle()

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
        distances = np.hypot(display_points[:, 0] - event.x, display_points[:, 1] - event.y)
        if distances.size == 0:
            return
        index = int(np.argmin(distances))
        if distances[index] > 10:
            return
        cycle.toggle_point(index)
        self._refresh_plot()
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
            return
        if cycle_number not in self.state.cycles:
            assert self.loaded is not None
            new_cycle = load_cycle(self.loaded.dataframe, cycle_number, self.state.control)
            if self.state.all_frequency_window is not None:
                new_cycle.frequency_window = self.state.all_frequency_window
            new_cycle.parameters = self.state.parameters_for(cycle_number)
            self.state.cycles[cycle_number] = new_cycle
        self.state.active_cycle = cycle_number
        self.cycle_var.set(str(cycle_number))
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
            messagebox.showerror("Invalid model", "Enter a circuit model", parent=self.root)
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
        self.parameter_table.set_parameters(self.state.parameters_for(self.state.active_cycle))
        self._refresh_plot(rescale=True)
        self._update_status("fitting model changed")

    def find_outliers(self) -> None:
        if self.state is None or not self._capture_controls():
            return
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("Invalid threshold", "Enter a numeric threshold", parent=self.root)
            return
        cycle_number = self.state.active_cycle
        cycle = self.state.active
        self.status_var.set(f"Cycle {cycle_number} · finding outliers…")
        self._submit(
            lambda: find_outlier_indices(cycle, threshold),
            lambda indices: self._finish_outliers(cycle_number, indices),
            "Outlier search failed",
        )

    def _finish_outliers(self, cycle_number: int, indices: np.ndarray) -> None:
        if self.state is None:
            return
        self.state.cycles[cycle_number].apply_outliers(indices)
        if self.state.active_cycle == cycle_number:
            self._refresh_plot()
            self._update_status("outlier search complete")

    def find_outliers_for_all(self) -> None:
        if self.state is None or self.loaded is None or not self._capture_controls():
            return
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("Invalid threshold", "Enter a numeric threshold", parent=self.root)
            return
        cycle_count = len(self.state.available_cycles)
        self.status_var.set(f"Finding outliers in all {cycle_count} cycles…")
        self._submit(
            lambda: find_outliers_for_all_cycles(
                self.loaded.dataframe,
                self.state.available_cycles,
                self.state.control,
                threshold,
            ),
            self._finish_all_outliers,
            "File-wide outlier search failed",
        )

    def _finish_all_outliers(self, results) -> None:
        if self.state is None:
            return
        for cycle_number, (loaded_cycle, indices) in results.items():
            if cycle_number in self.state.cycles:
                cycle = self.state.cycles[cycle_number]
            else:
                cycle = loaded_cycle
                cycle.parameters = self.state.parameters_for(cycle_number)
                self.state.cycles[cycle_number] = cycle
            if self.state.all_frequency_window is not None:
                cycle.frequency_window = self.state.all_frequency_window
            cycle.apply_outliers(indices)
        self._restore_controls()
        self._refresh_plot()
        self._update_status(f"outliers calculated for {len(results)} cycles")

    def fit(self) -> None:
        if self.state is None or not self._capture_controls():
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
        fitted_parameters, fit_frequency, fit_impedance, fit_at_data = result
        cycle = self.state.cycles[cycle_number]
        cycle.fit_parameters = fitted_parameters
        cycle.fit_frequency_hz = fit_frequency
        cycle.fit_impedance = fit_impedance
        cycle.fit_at_data_impedance = fit_at_data
        for parameter, fitted in zip(parameters, fitted_parameters):
            parameter.initial = float(fitted)
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
                self.state.available_cycles.index(start_cycle):
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
        self._refresh_plot()
        self._update_status("selection reset")

    def save_mask(self) -> None:
        if self.state is None:
            return
        default_name = f"{self.path.stem}_cycle{self.state.active_cycle}_mask_included.npy"
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

    def save_project(self) -> None:
        if self.state is None or not self._capture_controls():
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
        if self.state is None or self.loaded is None:
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
        self.control = restored.control
        self.circuit = restored.circuit
        self.model_var.set(restored.circuit)
        self.cycle_var.set(str(restored.active_cycle))
        self.cycle_box.configure(values=[str(cycle) for cycle in restored.available_cycles])
        self._restore_controls()
        self._refresh_plot(rescale=True)
        self._update_status(f"project loaded from {path.name}")

    def export_fits(self) -> None:
        if self.state is None or not self._capture_controls():
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
