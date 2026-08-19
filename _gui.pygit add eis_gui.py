warning: in the working copy of 'eis_gui.py', LF will be replaced by CRLF the next time Git touches it
[1mdiff --git a/eis_gui.py b/eis_gui.py[m
[1mindex 718c6c13..8014220b 100644[m
[1m--- a/eis_gui.py[m
[1m+++ b/eis_gui.py[m
[36m@@ -8,6 +8,8 @@[m [mfrom pathlib import Path[m
 import re[m
 import shutil[m
 import subprocess[m
[32m+[m[32mimport tempfile[m
[32m+[m[32mimport threading[m
 import tkinter as tk[m
 from tkinter import filedialog, messagebox, simpledialog, ttk[m
 from typing import Callable[m
[36m@@ -18,7 +20,7 @@[m [mfrom scipy.optimize import curve_fit[m
 from scipy.special import voigt_profile[m
 from wepy.eis import tau as cpe_tau[m
 [m
[31m-from eis_model import ParameterValue, ProjectState[m
[32m+[m[32mfrom eis_model import CycleState, ParameterValue, ProjectState[m
 from eis_project import ([m
     _dataframe_to_payload,[m
     _state_to_payload,[m
[36m@@ -59,6 +61,7 @@[m [mfrom eis_services import ([m
     analyze_outliers,[m
     find_outliers_for_all_cycles,[m
     fit_cycle,[m
[32m+[m[32m    inspect_eis_file_spectrum_kinds,[m
     refine_fit_cycle,[m
     load_cycle,[m
     load_project_from_dataframe,[m
[36m@@ -66,6 +69,18 @@[m [mfrom eis_services import ([m
     load_projects,[m
     select_eec_model_from_hybrid_drt,[m
 )[m
[32m+[m[32mfrom ml.gui_results import MLResult, load_ml_results, suggested_eec[m
[32m+[m[32mfrom ml.results_schema import spectrum_identifier[m
[32m+[m[32mfrom ml.point_validity import detect_outliers_in_active_points[m
[32m+[m[32mfrom spectrum_simulator import logarithmic_frequencies, simulate_spectrum[m
[32m+[m[32mfrom extract_relaxis import export_to_eisfit_json[m
[32m+[m[32mfrom explorer_filter import ([m
[32m+[m[32m    FilterCondition,[m
[32m+[m[32m    FilterDefinition,[m
[32m+[m[32m    apply_filters,[m
[32m+[m[32m    field_is_numeric,[m
[32m+[m[32m    field_operators,[m
[32m+[m[32m)[m
 [m
 MODEL_PRESETS = ([m
     "R0-L0-p(R1,CPE1)",[m
[36m@@ -462,6 +477,72 @@[m [mclass MetadataEditDialog(tk.Toplevel):[m
         self.destroy()[m
 [m
 [m
[32m+[m[32mclass ElectrodeSelectionDialog(tk.Toplevel):[m
[32m+[m[32m    _labels = {[m
[32m+[m[32m        "working": "WE–RE",[m
[32m+[m[32m        "cell": "WE–CE",[m
[32m+[m[32m        "counter": "CE–RE",[m
[32m+[m[32m    }[m
[32m+[m
[32m+[m[32m    def __init__(self, parent: tk.Tk, path: Path, available: list[str]) -> None:[m
[32m+[m[32m        super().__init__(parent)[m
[32m+[m[32m        self.result: tuple[list[str], bool] | None = None[m
[32m+[m[32m        self.title(f"Select spectra — {path.name}")[m
[32m+[m[32m        self.transient(parent)[m
[32m+[m[32m        self.resizable(False, False)[m
[32m+[m[32m        self.protocol("WM_DELETE_WINDOW", self.destroy)[m
[32m+[m[32m        body = ttk.Frame(self, padding=12)[m
[32m+[m[32m        body.pack(fill=tk.BOTH, expand=True)[m
[32m+[m[32m        ttk.Label([m
[32m+[m[32m            body,[m
[32m+[m[32m            text="This file contains multiple electrode-pair spectra.\nSelect the spectra to import:",[m
[32m+[m[32m            justify=tk.LEFT,[m
[32m+[m[32m        ).pack(anchor="w", pady=(0, 8))[m
[32m+[m[32m        self._variables = {[m
[32m+[m[32m            kind: tk.BooleanVar(value=True) for kind in available[m
[32m+[m[32m        }[m
[32m+[m[32m        for kind in ("working", "cell", "counter"):[m
[32m+[m[32m            if kind in self._variables:[m
[32m+[m[32m                ttk.Checkbutton([m
[32m+[m[32m                    body,[m
[32m+[m[32m                    text=self._labels[kind],[m
[32m+[m[32m                    variable=self._variables[kind],[m
[32m+[m[32m                ).pack(anchor="w")[m
[32m+[m[32m        self.apply_to_all_var = tk.BooleanVar(value=False)[m
[32m+[m[32m        ttk.Checkbutton([m
[32m+[m[32m            body,[m
[32m+[m[32m            text="Apply to all subsequent applicable files in this import",[m
[32m+[m[32m            variable=self.apply_to_all_var,[m
[32m+[m[32m        ).pack(anchor="w", pady=(8, 0))[m
[32m+[m[32m        buttons = ttk.Frame(body)[m
[32m+[m[32m        buttons.pack(fill=tk.X, pady=(12, 0))[m
[32m+[m[32m        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)[m
[32m+[m[32m        ttk.Button(buttons, text="Import selected", command=self._accept).pack([m
[32m+[m[32m            side=tk.RIGHT, padx=(0, 6)[m
[32m+[m[32m        )[m
[32m+[m[32m        self.bind("<Return>", lambda _event: self._accept())[m
[32m+[m[32m        self.grab_set()[m
[32m+[m
[32m+[m[32m    def _accept(self) -> None:[m
[32m+[m[32m        selected = [kind for kind, variable in self._variables.items() if variable.get()][m
[32m+[m[32m        if not selected:[m
[32m+[m[32m            messagebox.showerror([m
[32m+[m[32m                "No spectra selected",[m
[32m+[m[32m                "Select at least one electrode-pair spectrum.",[m
[32m+[m[32m                parent=self,[m
[32m+[m[32m            )[m
[32m+[m[32m            return[m
[32m+[m[32m        self.result = (selected, self.apply_to_all_var.get())[m
[32m+[m[32m        self.destroy()[m
[32m+[m
[32m+[m
[32m+[m[32mdef _compatible_spectrum_selection([m
[32m+[m[32m    selection: list[str], available: list[str][m
[32m+[m[32m) -> list[str] | None:[m
[32m+[m[32m    compatible = [kind for kind in selection if kind in available][m
[32m+[m[32m    return compatible or None[m
[32m+[m
[32m+[m
 class EISApplication:[m
     def __init__([m
         self,[m
[36m@@ -491,6 +572,9 @@[m [mclass EISApplication:[m
         self._explorer_rows: dict[str, tuple[str, LoadedProject, SpectrumMetadata]] = ([m
             {}[m
         )[m
[32m+[m[32m        self._explorer_current_column_order: list[str] | None = None[m
[32m+[m[32m        self._fit_explorer_filter = FilterDefinition()[m
[32m+[m[32m        self._drt_explorer_filter = FilterDefinition()[m
         self._explorer_lookup: dict[tuple[str, int], str] = {}[m
         self._explorer_anchor_item: str | None = None[m
         self._explorer_primary_item: str | None = None[m
[36m@@ -512,8 +596,13 @@[m [mclass EISApplication:[m
         self._plot_imports = None[m
         self.plot_mode = "nyquist"[m
         self.procedure_blocks: dict[str, list[dict[str, str]]] = {}[m
[32m+[m[32m        self.simulator_spectrum = None[m
[32m+[m[32m        self.simulator_parameters: list[ParameterValue] = [][m
[32m+[m[32m        self.simulator_drt_result = None[m
[32m+[m[32m        self.simulator_drt_mode_var = tk.StringVar(value="Ridge DRT")[m
 [m
         self.threshold_var = tk.StringVar(value=f"{threshold:g}")[m
[32m+[m[32m        self.deterministic_threshold_var = tk.StringVar(value="4")[m
         self.refine_z_threshold_var = tk.StringVar(value="3.5")[m
         self.refine_max_iterations_var = tk.StringVar(value="5")[m
         self.model_var = tk.StringVar(value=circuit)[m
[36m@@ -524,6 +613,13 @@[m [mclass EISApplication:[m
         self.show_drt_fit_var = tk.BooleanVar(value=False)[m
         self.show_drt_recovered_var = tk.BooleanVar(value=False)[m
         self.hide_legends_var = tk.BooleanVar(value=False)[m
[32m+[m[32m        self.show_ml_frequency_ranges_var = tk.BooleanVar(value=False)[m
[32m+[m[32m        self.show_ml_active_points_var = tk.BooleanVar(value=False)[m
[32m+[m[32m        self.show_ml_model_var = tk.BooleanVar(value=False)[m
[32m+[m[32m        self.show_ml_residuals_var = tk.BooleanVar(value=False)[m
[32m+[m[32m        self.ml_results: dict[str, MLResult] = {}[m
[32m+[m[32m        self.ml_results_directory: Path | None = None[m
[32m+[m[32m        self.ml_results_status_var = tk.StringVar(value="No ML results loaded")[m
         self.minimum_frequency_var = tk.StringVar()[m
         self.maximum_frequency_var = tk.StringVar()[m
         self.auto_max_frequency_var = tk.BooleanVar(value=False)[m
[36m@@ -63