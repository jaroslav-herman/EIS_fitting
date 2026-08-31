import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eis_model import CycleState, ParameterValue, ProjectState
from eis_project import load_json_payload, load_project_file, save_project_file


class ProjectCompressionTests(unittest.TestCase):
    def _state_and_dataframe(self):
        dataframe = pd.DataFrame(
            {
                "freq_hz": [100.0, 10.0, 1.0],
                "re_zwe_ce_ohm": [2.0, 3.0, 4.0],
                "minus_im_zwe_ce_ohm": [0.1, 0.2, 0.3],
                "ewe_ece_v": [1.0, 1.0, 1.0],
                "i_ma": [2.0, 2.0, 2.0],
                "cycle_number": [1, 1, 1],
            }
        )
        parameter = ParameterValue("R0", "Ohm", 2.0, 0.0, 10.0, 1.5)
        cycle = CycleState(
            1,
            np.array([100.0, 10.0, 1.0]),
            np.array([2.0 - 0.1j, 3.0 - 0.2j, 4.0 - 0.3j]),
            manually_included=np.array([True, False, True]),
            outliers=np.array([False, True, False]),
            parameters=[parameter],
            fit_parameters=np.array([2.1]),
            fit_frequency_hz=np.array([100.0, 1.0]),
            fit_impedance=np.array([2.1 - 0.1j, 4.1 - 0.3j]),
            fit_at_data_impedance=np.array([2.1 - 0.1j, 3.1 - 0.2j, 4.1 - 0.3j]),
        )
        cycle.store_hybrid_drt(
            np.array([0.01, 0.1]), np.array([1.0, 2.0]), 2.0, 0.01
        )
        state = ProjectState(
            Path("source.mpt"),
            "R0",
            "cell",
            [1],
            1,
            [parameter],
            {1: cycle},
        )
        return state, dataframe

    def test_compressed_round_trip_preserves_scientific_state(self):
        state, dataframe = self._state_and_dataframe()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project.eisfit.json.gz"
            save_project_file(
                state,
                path,
                datasets=[("source::cell", state, dataframe)],
            )

            self.assertEqual(path.read_bytes()[:2], b"\x1f\x8b")
            payload = load_json_payload(path)
            restored = load_project_file(state, dataframe, path, payload=payload)
            cycle = restored.cycles[1]
            np.testing.assert_array_equal(cycle.frequency_hz, state.cycles[1].frequency_hz)
            np.testing.assert_array_equal(cycle.impedance, state.cycles[1].impedance)
            np.testing.assert_array_equal(
                cycle.manually_included, state.cycles[1].manually_included
            )
            np.testing.assert_array_equal(cycle.fit_impedance, state.cycles[1].fit_impedance)
            np.testing.assert_array_equal(
                cycle.saved_hybrid_tau_s, state.cycles[1].saved_hybrid_tau_s
            )

    def test_uncompressed_projects_remain_readable(self):
        state, dataframe = self._state_and_dataframe()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project.eisfit.json"
            save_project_file(state, path)
            self.assertEqual(path.read_bytes()[:1], b"{")
            self.assertEqual(load_json_payload(path)["format"], "eis-fitting-project")

    def test_compressed_payload_is_compact_json(self):
        state, dataframe = self._state_and_dataframe()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project.eisfit.json.gz"
            save_project_file(state, path)
            text = gzip.decompress(path.read_bytes()).decode("utf-8")
            self.assertNotIn("\n", text)
            self.assertEqual(json.loads(text)["version"], 4)


if __name__ == "__main__":
    unittest.main()
