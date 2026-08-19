from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ml.gui_results import load_ml_results
from ml.migrate_embedded_results import split_project


class MlResultsArchitectureTests(unittest.TestCase):
    def test_embedded_results_split_without_raw_arrays_in_sidecar(self):
        payload = {
            "format": "eis-fitting-project",
            "source_path": "sample.mpt",
            "datasets": [],
            "ml_results": {
                "schema_version": "test",
                "spectra": [{
                    "spectrum_id": "legacy::sample::working::1",
                    "source_name": "sample.mpt::working",
                    "cycle": 1,
                    "frequency": [10.0, 1.0],
                    "z_real": [1.0, 2.0],
                    "z_imag": [-1.0, -2.0],
                    "predicted_eec_model": "R0-p(R1,CPE1)",
                    "ml_eec_parameters": {"R0": 1.0},
                }],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input.eisfit.json"
            project = Path(temporary) / "sample.eisfit.json"
            results = Path(temporary) / "sample_ml_results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            split_project(source, project, results)
            normal = json.loads(project.read_text(encoding="utf-8"))
            sidecar = json.loads(results.read_text(encoding="utf-8"))
            self.assertNotIn("ml_results", normal)
            self.assertEqual(normal["format"], "eis-fitting-project")
            self.assertEqual(sidecar["format"], "eis-fitting-ml-results")
            self.assertNotIn("frequency", sidecar["spectra"][0])
            loaded = load_ml_results(results)
            self.assertEqual(len(loaded), 1)
            self.assertTrue(next(iter(loaded.values())).spectrum_key)


if __name__ == "__main__":
    unittest.main()
