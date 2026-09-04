from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.dataset import SpectrumRecord, load_eisfit_projects
from ml.frequency_range import active_frequency_bounds, _targets
from ml.number_aware_pipeline import deterministic_masks, _candidate_topologies, _is_positive_parameter, _physical_initial_cap, _parameter_features, _usable_fit_parameter
from ml.preprocessing import SpectrumPreprocessor


def _project(path: Path, circuit: str, *, with_window: bool = True) -> None:
    frequency = np.logspace(4, 0, 12)
    frame = pd.DataFrame({
        "freq_hz": frequency,
        "re_zwe_ce_ohm": np.full(frequency.size, 2.0),
        "minus_im_zwe_ce_ohm": np.linspace(-1.0, -0.1, frequency.size),
        "ewe_ece_v": np.full(frequency.size, 1.6),
        "i_ma": np.full(frequency.size, 4.0),
        "time_s": np.arange(frequency.size, dtype=float),
        "cycle_number": np.ones(frequency.size, dtype=int),
    })
    saved = {
        "circuit": circuit,
        "potential_v": 1.6,
        "current_ma": 4.0,
        "time_s": 1.0,
        "frequency_window": [1.0, 10000.0] if with_window else None,
        "manually_included": [True] * frequency.size,
        "outliers": [False] * frequency.size,
        "fit_parameters": [0.1, 1e-7, 1.0, 0.01, 0.9, 0.5, 0.02, 0.85],
    }
    state = {"circuit": circuit, "control": "cell", "cycles": {"1": saved}}
    payload = {"format": "eis-fitting-project", "version": 4, "circuit": circuit, "control": "cell", "datasets": [{"dataset_id": "sample.mpr::cell", "state": state, "dataframe": frame.to_json(orient="split")}]}
    path.write_text(json.dumps(payload), encoding="utf-8")


class NumberAwarePipelineTests(unittest.TestCase):
    def test_loader_accepts_numbered_r3_cpe3_and_preserves_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.eisfit.json"
            _project(path, "R0-L0-p(R1,CPE1)-p(R3,CPE3)")
            report = load_eisfit_projects([path], {str(path): "sample"})
        self.assertEqual(len(report.records), 1)
        self.assertEqual(report.records[0].original_eec_topology, "R0-L0-p(R1,CPE1)-p(R3,CPE3)")
        self.assertEqual(report.records[0].control, "cell")

    def test_raw_validation_can_omit_frequency_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.eisfit.json"
            _project(path, "R0-L0-p(R1,CPE1)", with_window=False)
            report = load_eisfit_projects([path], {str(path): "validation"}, require_fit=False, require_frequency_window=False)
        self.assertEqual(len(report.records), 1)
        self.assertIsNone(report.records[0].manual_f_min)

    def test_deterministic_mask_keeps_raw_length_and_rejects_outside_window(self):
        frequency = np.logspace(4, 0, 12)
        from ml.dataset import SpectrumRecord
        record = SpectrumRecord("id", "source", "sample", 1, 1.0, 2.0, 3.0, frequency, np.ones(12), -np.ones(12), "R0-L0-p(R1,CPE1)")
        masks, diagnostics = deterministic_masks([record], {"id": (10.0, 1000.0)}, threshold=4.0)
        self.assertEqual(masks["id"].size, frequency.size)
        self.assertTrue((frequency[masks["id"]] >= 10.0).all())
        self.assertEqual(diagnostics["id"]["rejected_points"], int((~masks["id"]).sum()))

    def test_parameter_transform_classifies_alpha_as_non_positive(self):
        self.assertTrue(_is_positive_parameter("R3"))
        self.assertTrue(_is_positive_parameter("CPE3_0"))
        self.assertFalse(_is_positive_parameter("CPE3_1"))

    def test_physical_cap_blocks_ohmic_scale_excursion(self):
        record = SpectrumRecord("id", "source", "sample", 1, 1.0, 2.0, 3.0,
                                np.asarray([1.0, 10.0, 100.0, 1000.0]),
                                np.asarray([0.02, 0.021, 0.025, 0.03]),
                                np.asarray([0.0, 0.0, 0.001, 0.002]),
                                "R0-L0-p(R1,CPE1)")
        self.assertLess(_physical_initial_cap(record, "R0", 10.0), 0.03)

    def test_optimizer_collapse_labels_are_excluded_for_scale_parameters(self):
        frequency = np.logspace(4, 0, 12)
        record = SpectrumRecord(
            "id", "source", "sample", 1, 1.6, 100.0, 1.0, frequency,
            np.full(12, 0.1), -np.full(12, 0.01), "R0-L0-p(R1,CPE1)",
        )
        self.assertFalse(_usable_fit_parameter(record, "R0", 1e-12))
        self.assertTrue(_usable_fit_parameter(record, "R0", 0.02))
        self.assertFalse(_usable_fit_parameter(record, "L0", 1e-14))
        self.assertTrue(_usable_fit_parameter(record, "L0", 1e-7))

    def test_parameter_features_include_current_and_interactions(self):
        frequency = np.logspace(4, 0, 12)
        records = [
            SpectrumRecord("a", "source", "sample", 1, 1.6, 10.0, 1.0, frequency, np.ones(12), -np.ones(12), "R0-p(R1,CPE1)"),
            SpectrumRecord("b", "source", "sample", 2, 1.7, 100.0, 2.0, frequency, np.ones(12), -np.ones(12), "R0-p(R1,CPE1)"),
        ]
        processor = SpectrumPreprocessor(grid_size=8, include_impedance_scale=True).fit(records)
        features = _parameter_features(processor, records)
        self.assertEqual(features.shape, (2, 8 * 3 + 1 + 11))
        self.assertNotEqual(features[0, -8], features[1, -8])  # log(current)
        self.assertNotEqual(features[0, -7], features[1, -7])  # 1/current

    def test_boundary_features_mark_voltage_extrapolation(self):
        frequency = np.logspace(4, 0, 12)
        records = [SpectrumRecord("a", "source", "sample", 1, 1.6, 10.0, 1.0, frequency, np.ones(12), -np.ones(12), "R0-p(R1,CPE1)")]
        processor = SpectrumPreprocessor(grid_size=8, include_impedance_scale=True).fit(records)
        processor.parameter_voltage_mean_ = 1.6
        processor.parameter_voltage_scale_ = 0.1
        processor.parameter_voltage_min_ = 1.5
        processor.parameter_voltage_max_ = 1.7
        outside = SpectrumRecord("b", "source", "sample", 2, 1.9, 10.0, 1.0, frequency, np.ones(12), -np.ones(12), "R0-p(R1,CPE1)")
        features = _parameter_features(processor, [outside])
        self.assertEqual(features[0, -1], 1.0)
        self.assertGreater(features[0, -3], 0.0)

    def test_candidate_topologies_include_competitive_alternative(self):
        class Bundle:
            circuit_classes = ("R0-L0-p(R1,CPE1)-p(R2,CPE2)", "R0-L0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)")
            parameter_preprocessor = type("Pre", (), {"parameter_voltage_min_": 1.4, "parameter_voltage_max_": 1.8})()
        frequency = np.logspace(4, 0, 12)
        record = SpectrumRecord("id", "source", "sample", 1, 1.9, 10.0, 1.0, frequency, np.ones(12), -np.ones(12), "R0-L0-p(R1,CPE1)-p(R2,CPE2)")
        candidates = _candidate_topologies(Bundle(), "R0-L0-p(R1,CPE1)-p(R2,CPE2)", {Bundle.circuit_classes[0]: 0.55, Bundle.circuit_classes[1]: 0.45}, record)
        self.assertEqual(candidates[0], Bundle.circuit_classes[0])
        self.assertIn(Bundle.circuit_classes[1], candidates)

    def test_frequency_target_uses_active_points_not_saved_boundaries(self):
        record = SpectrumRecord("id", "source", "sample", 1, 1.0, 2.0, 3.0,
                                np.asarray([1.0, 10.0, 100.0, 1000.0]),
                                np.ones(4), np.ones(4), "R0-L0-p(R1,CPE1)",
                                manual_f_min=1.0, manual_f_max=1000.0,
                                cleaned_frequency=np.asarray([10.0, 100.0]),
                                cleaned_z_real=np.ones(2), cleaned_z_imag=np.ones(2))
        self.assertEqual(active_frequency_bounds(record), (10.0, 100.0))
        np.testing.assert_allclose(_targets([record])[0], [1.5, np.log(1.0)])


if __name__ == "__main__":
    unittest.main()
