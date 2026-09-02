from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ml.dataset import SpectrumRecord
from ml.gui_results import MLResult, load_ml_results, suggested_eec
from ml.preprocessing import SpectrumPreprocessor
from ml.topology_classifier import run_topology_experiment
from ml.frequency_range import run_frequency_range_experiment


def records():
    frequency = np.logspace(5, -1, 12)
    result = []
    for sample in ("sample_a", "sample_b", "sample_c"):
        for index in range(4):
            label = "R0-p(R1,CPE1)" if index % 2 == 0 else "R0-L0-p(R1,CPE1)"
            scale = 1.0 + index / 10
            result.append(SpectrumRecord(
                f"{sample}-{index}", "synthetic", sample, index,
                1.0, 2.0, float(index), frequency,
                np.full(frequency.size, scale),
                -np.full(frequency.size, scale), label,
            ))
    return result


class MlTests(unittest.TestCase):
    def test_ml_eec_topology_mapping(self):
        cases = (
            (1, False, "R0-p(R1,CPE1)"),
            (1, True, "R0-L0-p(R1,CPE1)"),
            (2, False, "R0-p(R1,CPE1)-p(R2,CPE2)"),
            (2, True, "R0-L0-p(R1,CPE1)-p(R2,CPE2)"),
            (3, False, "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)"),
            (3, True, "R0-L0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)"),
        )
        for process_count, l0_required, expected in cases:
            result = MLResult(
                "test",
                predicted_process_count=process_count,
                predicted_l0_required=l0_required,
            )
            self.assertEqual(suggested_eec(result), expected)
        self.assertIsNone(suggested_eec(MLResult("missing")))
        self.assertIsNone(suggested_eec(MLResult("invalid", suggested_eec="invalid")))
        explicit_three_block = "R0-L0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)"
        self.assertEqual(
            suggested_eec(MLResult("explicit", suggested_eec=explicit_three_block)),
            explicit_three_block,
        )

    def test_ml_json_keeps_source_cycle_identity_and_metadata(self):
        spectra = []
        for source_name, value in (("source_A", 1.0), ("source_B", 2.0)):
            spectra.append({
                "spectrum_id": f"{source_name}::working::1",
                "source_name": source_name,
                "cycle": 1,
                "metadata": {
                    "Time": 1234.5 + value,
                    "Cycle mod 15": 7,
                    "User label": source_name,
                },
                "frequency": [10.0, 1.0],
                "z_real": [value, value + 1.0],
                "z_imag": [-value, -value - 1.0],
                "ml_envelope_mask": [True, False],
                "hgb_topology": "ONE_PROCESS",
                "hgb_confidence": value / 2.0,
            })
        payload = {"ml_results": {"source_file": "input.eisfit.json", "spectra": spectra}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.eisfit.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            results = load_ml_results(path)
        self.assertEqual(set(results), {"source_A::working::1", "source_B::working::1"})
        self.assertEqual(results["source_A::working::1"].cycle, 1)
        self.assertEqual(results["source_B::working::1"].cycle, 1)
        self.assertEqual(results["source_A::working::1"].metadata["Cycle mod 15"], 7)
        self.assertEqual(results["source_B::working::1"].metadata["User label"], "source_B")
        np.testing.assert_array_equal(results["source_A::working::1"].active_mask, [True, False])
        self.assertEqual(results["source_B::working::1"].confidence, 1.0)

    def test_ml_initialization_project_is_adapted_for_gui_inspection(self):
        payload = {
            "format": "eis-fitting-project",
            "source_path": "C:/data/sample.mpt",
            "control": "working",
            "circuit": "R0-L0-p(R1,CPE1)",
            "cycles": {
                "7": {
                    "circuit": "R0-L0-p(R1,CPE1)",
                    "frequency_window": [0.5, 10000.0],
                    "manually_included": [True, True, False],
                    "outliers": [False, False, True],
                    "custom_metadata": {"Spectrum": "WE"},
                    "parameters": [
                        {"name": "R0", "initial": 0.1},
                        {"name": "L0", "initial": 1e-8},
                        {"name": "R1", "initial": 2.0},
                        {"name": "CPE1_0", "initial": 0.01},
                        {"name": "CPE1_1", "initial": 0.9},
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "178_ML_initial_parameters.eisfit.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            results = load_ml_results(path)
        result = results["C:/data/sample.mpt::working::7"]
        self.assertEqual(suggested_eec(result), "R0-L0-p(R1,CPE1)")
        self.assertEqual(result.model_parameters["CPE1_0"], 0.01)
        self.assertEqual(result.initial_sources["R0"], "DRT")
        self.assertEqual(result.initial_sources["R1"], "ML")
        self.assertEqual(result.frequency_ranges, [(0.5, 10000.0)])
        np.testing.assert_array_equal(result.active_mask, [True, True, False])
    def test_preprocessor_handles_variable_frequency_grids_without_extrapolation(self):
        source = records()
        source[1] = SpectrumRecord(**{**source[1].__dict__, "frequency": source[1].frequency[2:-2]})
        processor = SpectrumPreprocessor(grid_size=16).fit(source[:4])
        values = processor.transform([source[1]])
        self.assertEqual(values.shape, (1, 48))
        self.assertTrue(np.isfinite(values).all())

    def test_sample_based_validation_and_probabilities(self):
        result = run_topology_experiment(records(), model_names=("logistic_regression",))
        self.assertEqual(set(result.predictions["validation_fold"]), {"sample_a", "sample_b", "sample_c"})
        probability_columns = [column for column in result.predictions if column.startswith("probability_")]
        self.assertTrue(probability_columns)
        self.assertTrue(np.isfinite(result.predictions[probability_columns].to_numpy()).all())
        self.assertNotIn("sample_id", result.predictions.drop(columns=["sample_id"]).columns)

    def test_metadata_is_optional_and_separate(self):
        result = run_topology_experiment(records(), model_names=("random_forest",), use_metadata=True)
        self.assertTrue((result.predictions["feature_mode"] == "spectrum_plus_metadata").all())

    def test_frequency_range_targets_are_logarithmic_and_ordered(self):
        source = records()
        source = [
            SpectrumRecord(**{**record.__dict__, "manual_f_min": 1.0 + record.cycle, "manual_f_max": 1000.0})
            for record in source
        ]
        result = run_frequency_range_experiment(source, feature_mode="voltage_only", model_names=("ridge",))
        self.assertTrue((result.predictions["predicted_log_f_min"] < result.predictions["predicted_log_f_max"]).all())
        self.assertTrue((result.predictions["predicted_f_min"] < result.predictions["predicted_f_max"]).all())
        self.assertTrue((result.predictions["range_iou"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
