from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.dataset import SpectrumRecord, load_eisfit_projects
from ml.frequency_range import active_frequency_bounds, _targets
from ml.number_aware_pipeline import deterministic_masks, _is_positive_parameter, _physical_initial_cap


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
