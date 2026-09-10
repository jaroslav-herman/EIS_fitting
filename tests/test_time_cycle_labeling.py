from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import pandas as pd

from eis_services import load_project_from_dataframe
from eis_gui import EISApplication
from load_and_label_eis import label_project_catalog


class TimeCycleLabelingTests(unittest.TestCase):
    @staticmethod
    def _project(source: str):
        rows = []
        voltages = [1.0, 1.1, 1.0, 1.1]
        for cycle, voltage in enumerate(voltages, start=1):
            for frequency in (100.0, 10.0):
                rows.append(
                    {
                        "freq_hz": frequency,
                        "cycle_number": cycle,
                        "ewe_ece_v": voltage,
                        "i_ma": 1.0,
                        "re_zwe_ce_ohm": 2.0,
                        "minus_im_zwe_ce_ohm": 0.1,
                    }
                )
        return load_project_from_dataframe(
            pd.DataFrame(rows),
            Path(source),
            1,
            "cell",
            "R0-L0-p(R1,CPE1)",
        )

    def test_labels_dataframe_state_and_catalog(self):
        project = self._project("new.mpr")

        loops, pattern_length = label_project_catalog(project)

        self.assertEqual((loops, pattern_length), (2, 2))
        self.assertEqual(project.dataframe["Time"].drop_duplicates().tolist(), [1, 2])
        self.assertEqual(project.dataframe["Cycle mod"].drop_duplicates().tolist(), [1, 2])
        self.assertEqual(project.state.cycles[3].custom_metadata["Time"], 2)
        self.assertEqual(project.spectra[3].custom_metadata["Cycle mod"], 2)

    def test_new_project_can_continue_after_existing_loops(self):
        project = self._project("new.mpr")

        loops, _pattern_length = label_project_catalog(project, time_offset=7)

        self.assertEqual(loops, 2)
        self.assertEqual(project.state.cycles[1].custom_metadata["Time"], 8)
        self.assertEqual(project.state.cycles[4].custom_metadata["Time"], 9)

    def test_raw_measurement_time_is_not_used_as_label_offset(self):
        application = object.__new__(EISApplication)
        application.loaded_projects = {
            "raw": SimpleNamespace(
                dataframe=pd.DataFrame({"Time": [21.0, 22.0]}),
                state=SimpleNamespace(cycles={}),
            )
        }

        self.assertEqual(application._existing_time_label_max(), 0)


if __name__ == "__main__":
    unittest.main()
