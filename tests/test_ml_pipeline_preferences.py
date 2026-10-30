import json
import tempfile
import unittest
from pathlib import Path

import eis_gui
from eis_gui import (
    DEFAULT_ML_PIPELINES,
    ML_PIPELINE_ACTIONS,
    ML_PIPELINE_ACTION_LABELS,
    ML_TRAINED_MODELS,
    normalize_ml_pipeline,
    normalize_ml_pipelines,
)


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _application_with_preferences_path(path: Path):
    """Create an EISApplication shell without running __init__ (no Tk needed)."""
    app = object.__new__(eis_gui.EISApplication)
    app._preferences_path = path
    return app


class MlPipelineConstantsTests(unittest.TestCase):
    def test_every_trained_model_has_a_default_pipeline(self):
        self.assertEqual(set(DEFAULT_ML_PIPELINES), set(ML_TRAINED_MODELS))
        for name, pipeline in DEFAULT_ML_PIPELINES.items():
            self.assertEqual(pipeline["actions"], list(ML_PIPELINE_ACTIONS), name)
            self.assertTrue(all(action in ML_PIPELINE_ACTION_LABELS for action in pipeline["actions"]))

    def test_action_labels_cover_all_actions(self):
        self.assertEqual(set(ML_PIPELINE_ACTIONS), set(ML_PIPELINE_ACTION_LABELS))
        self.assertEqual(len(ML_PIPELINE_ACTIONS), len(set(ML_PIPELINE_ACTIONS)))


class NormalizeMlPipelineTests(unittest.TestCase):
    def test_valid_pipeline_is_normalized(self):
        pipeline = normalize_ml_pipeline(["model", "fit"], "6", "2.5", "3")
        self.assertEqual(pipeline["actions"], ["model", "fit"])
        self.assertEqual(pipeline["outlier_threshold"], "6")
        self.assertEqual(pipeline["refine_z_threshold"], "2.5")
        self.assertEqual(pipeline["refine_max_iterations"], "3")

    def test_unknown_actions_are_dropped(self):
        pipeline = normalize_ml_pipeline(["model", "legacy_step", "fit"], 4, 3.5, 5)
        self.assertEqual(pipeline["actions"], ["model", "fit"])

    def test_empty_action_list_rejected(self):
        with self.assertRaises(ValueError):
            normalize_ml_pipeline([], "4", "3.5", "5")

    def test_non_positive_threshold_rejected(self):
        for bad in ("0", "-1", "nan"):
            with self.assertRaises(ValueError):
                normalize_ml_pipeline(["fit"], bad, "3.5", "5")

    def test_invalid_refine_settings_rejected(self):
        with self.assertRaises(ValueError):
            normalize_ml_pipeline(["fit"], "4", "0", "5")
        with self.assertRaises(ValueError):
            normalize_ml_pipeline(["fit"], "4", "3.5", "0")

    def test_non_list_actions_rejected(self):
        with self.assertRaises(ValueError):
            normalize_ml_pipeline("fit", "4", "3.5", "5")


class NormalizeMlPipelinesTests(unittest.TestCase):
    def test_missing_models_fall_back_to_defaults(self):
        pipelines = normalize_ml_pipelines({})
        self.assertEqual(set(pipelines), set(ML_TRAINED_MODELS))
        for name in ML_TRAINED_MODELS:
            self.assertEqual(pipelines[name], DEFAULT_ML_PIPELINES[name])
            self.assertIsNot(pipelines[name], DEFAULT_ML_PIPELINES[name])

    def test_saved_model_entry_is_used(self):
        payload = {
            "AEM-WE": {
                "actions": ["model", "fit"],
                "outlier_threshold": "6",
                "refine_z_threshold": "2",
                "refine_max_iterations": "3",
            }
        }
        pipelines = normalize_ml_pipelines(payload)
        self.assertEqual(pipelines["AEM-WE"]["actions"], ["model", "fit"])
        self.assertEqual(pipelines["AEM-WE"]["outlier_threshold"], "6")
        self.assertEqual(pipelines["Sputtered cathode"], DEFAULT_ML_PIPELINES["Sputtered cathode"])

    def test_broken_entries_fall_back_to_defaults(self):
        payload = {"Flex": {"actions": [], "outlier_threshold": "4", "refine_z_threshold": "3.5", "refine_max_iterations": "5"}}
        pipelines = normalize_ml_pipelines(payload)
        self.assertEqual(pipelines["Flex"], DEFAULT_ML_PIPELINES["Flex"])

    def test_non_dict_payload_returns_defaults(self):
        pipelines = normalize_ml_pipelines(["not", "a", "dict"])
        self.assertEqual(set(pipelines), set(ML_TRAINED_MODELS))


class MlPipelinePreferencesPersistenceTests(unittest.TestCase):
    def _write_preferences(self, payload: dict) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "preferences.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_load_defaults_when_key_missing(self):
        path = self._write_preferences({"fit_timeout_seconds": 10.0})
        app = _application_with_preferences_path(path)
        app._load_preferences()
        self.assertEqual(app._ml_model_pipelines, DEFAULT_ML_PIPELINES)

    def test_load_saved_pipelines_and_fallbacks(self):
        saved = {
            "ml_model_pipelines": {
                "Flex": {
                    "actions": ["outliers", "model", "fit"],
                    "outlier_threshold": "7",
                    "refine_z_threshold": "4.5",
                    "refine_max_iterations": "2",
                },
                "AEM-WE": {"actions": "broken"},
            }
        }
        path = self._write_preferences(saved)
        app = _application_with_preferences_path(path)
        app._load_preferences()
        self.assertEqual(app._ml_model_pipelines["Flex"]["actions"], ["outliers", "model", "fit"])
        self.assertEqual(app._ml_model_pipelines["Flex"]["outlier_threshold"], "7")
        self.assertEqual(app._ml_model_pipelines["AEM-WE"], DEFAULT_ML_PIPELINES["AEM-WE"])
        self.assertEqual(app._ml_model_pipelines["Sputtered cathode"], DEFAULT_ML_PIPELINES["Sputtered cathode"])

    def test_save_writes_pipelines_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preferences.json"
            app = _application_with_preferences_path(path)
            app._load_preferences()
            app._ml_model_pipelines = normalize_ml_pipelines(
                {"AEM-WE": {"actions": ["model", "initial_parameters", "fit", "refine"], "outlier_threshold": "5", "refine_z_threshold": "3", "refine_max_iterations": "4"}}
            )
            app.threshold_var = _FakeVar("1")
            app.deterministic_threshold_var = _FakeVar("4")
            app.refine_z_threshold_var = _FakeVar("3.5")
            app.refine_max_iterations_var = _FakeVar("5")
            app.fit_pipeline_var = _FakeVar("local only")
            app.fit_seed_var = _FakeVar("")
            app.fit_population_var = _FakeVar("30")
            app.fit_iterations_var = _FakeVar("200")
            app.fit_weight_modulus_var = _FakeVar(False)
            app.fit_jacobian_mode_var = _FakeVar("Numerical only")
            app._save_preferences(
                ("R0",),
                "I_mA",
                "I_mA",
                "R0",
                "R0",
                [],
                [],
                "end",
                {},
                {},
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("ml_model_pipelines", payload)
            reloaded = _application_with_preferences_path(path)
            reloaded._load_preferences()
            self.assertEqual(reloaded._ml_model_pipelines, app._ml_model_pipelines)
            self.assertEqual(
                reloaded._ml_model_pipelines["AEM-WE"]["actions"],
                ["model", "initial_parameters", "fit", "refine"],
            )

    def test_source_of_ml_processing_dialog_offers_saved_pipeline_checkbox(self):
        source = Path(eis_gui.__file__).read_text(encoding="utf-8")
        self.assertIn("Apply saved pipeline for the selected model", source)
        self.assertIn("ml_model_pipelines", source)
        self.assertIn('text="ML pipelines"', source)


if __name__ == "__main__":
    unittest.main()
