from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from eis_services import _read_eis_dataframe, load_projects_for_file
from eis_gui import EISApplication, _compatible_spectrum_selection


GALVANI_FIXTURE = (
    Path(__file__).parents[2]
    / "galvani"
    / "tests"
    / "testdata"
    / "v1150"
    / "v1150_PEIS.mpr"
)


class MprImportTests(unittest.TestCase):
    class _FakeRoot:
        def wait_window(self, _dialog):
            return None

    class _FakeDialog:
        def __init__(self, _root, _path, _available, result):
            self.result = result

    def test_apply_to_all_reuses_selection_for_compatible_files(self):
        application = object.__new__(EISApplication)
        application.root = self._FakeRoot()
        responses = iter(
            [self._FakeDialog(None, None, None, (["cell", "counter"], True))]
        )
        with patch(
            "eis_gui.ElectrodeSelectionDialog",
            side_effect=lambda *args: next(responses),
        ) as dialog:
            selected = application._select_import_spectrum_kinds(
                [
                    (Path("a.mpt"), ["cell", "counter", "working"]),
                    (Path("b.mpr"), ["cell", "counter"]),
                ]
            )
        self.assertEqual(dialog.call_count, 1)
        self.assertEqual(selected[Path("b.mpr").resolve()], ["cell", "counter"])

    def test_apply_to_all_falls_back_when_no_selection_is_compatible(self):
        application = object.__new__(EISApplication)
        application.root = self._FakeRoot()
        responses = iter(
            [
                self._FakeDialog(None, None, None, (["working"], True)),
                self._FakeDialog(None, None, None, (["cell"], False)),
            ]
        )
        with patch(
            "eis_gui.ElectrodeSelectionDialog",
            side_effect=lambda *args: next(responses),
        ) as dialog:
            selected = application._select_import_spectrum_kinds(
                [
                    (Path("a.mpt"), ["cell", "working"]),
                    (Path("b.mpr"), ["cell", "counter"]),
                ]
            )
        self.assertEqual(dialog.call_count, 2)
        self.assertEqual(selected[Path("b.mpr").resolve()], ["cell"])

    def test_selection_without_apply_to_all_shows_each_dialog(self):
        application = object.__new__(EISApplication)
        application.root = self._FakeRoot()
        responses = iter(
            [
                self._FakeDialog(None, None, None, (["cell"], False)),
                self._FakeDialog(None, None, None, (["counter"], False)),
            ]
        )
        with patch(
            "eis_gui.ElectrodeSelectionDialog",
            side_effect=lambda *args: next(responses),
        ) as dialog:
            application._select_import_spectrum_kinds(
                [
                    (Path("a.mpt"), ["cell", "counter"]),
                    (Path("b.mpr"), ["cell", "counter"]),
                ]
            )
        self.assertEqual(dialog.call_count, 2)

    def test_single_role_skips_dialog_and_compatibility_is_safe(self):
        application = object.__new__(EISApplication)
        application.root = self._FakeRoot()
        with patch("eis_gui.ElectrodeSelectionDialog") as dialog:
            selected = application._select_import_spectrum_kinds(
                [(Path("single.mpr"), ["cell"])]
            )
        dialog.assert_not_called()
        self.assertEqual(selected, {})
        self.assertEqual(_compatible_spectrum_selection(["working"], ["cell"]), None)

    @staticmethod
    def _three_electrode_dataframe() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "freq_hz": [10.0, 1.0, 0.0, 10.0, 1.0, 0.0],
                "cycle_number": [1, 1, 1, 2, 2, 2],
                "re_z_ohm": [1.0, 2.0, 0.0, 3.0, 4.0, 0.0],
                "minus_im_z_ohm": [0.1, 0.2, 0.0, 0.3, 0.4, 0.0],
                "re_zwe_ce_ohm": [5.0, 6.0, 0.0, 7.0, 8.0, 0.0],
                "minus_im_zwe_ce_ohm": [0.5, 0.6, 0.0, 0.7, 0.8, 0.0],
                "re_zce_ohm": [9.0, 10.0, 0.0, 11.0, 12.0, 0.0],
                "minus_im_zce_ohm": [0.9, 1.0, 0.0, 1.1, 1.2, 0.0],
                "ewe_v": [1.0] * 6,
                "ece_v": [0.1] * 6,
                "ewe_ece_v": [0.9] * 6,
                "i_ma": [1.0] * 6,
            }
        )

    def test_two_electrode_has_one_automatic_role(self):
        dataframe = self._three_electrode_dataframe().drop(
            columns=[
                "re_z_ohm",
                "minus_im_z_ohm",
                "re_zce_ohm",
                "minus_im_zce_ohm",
                "ece_v",
            ]
        )
        with patch(
            "eis_services._read_eis_dataframe",
            return_value=(dataframe, {"Potential control": "Ewe-Ece"}, "PEIS"),
        ):
            projects = load_projects_for_file(
                Path("two-electrode.mpr"), 1, "cell", "R0-L0-p(R1,CPE1)"
            )
        self.assertEqual([project.state.control for project in projects], ["cell"])

    def test_three_electrode_selection_limits_imported_roles(self):
        with patch(
            "eis_services._read_eis_dataframe",
            return_value=(
                self._three_electrode_dataframe(),
                {"Potential control": "Ewe-Ece"},
                "PEIS",
            ),
        ):
            all_projects = load_projects_for_file(
                Path("three-electrode.mpr"), 1, "cell", "R0-L0-p(R1,CPE1)"
            )
            selected_projects = load_projects_for_file(
                Path("three-electrode.mpr"),
                1,
                "cell",
                "R0-L0-p(R1,CPE1)",
                ["cell", "counter"],
            )
        self.assertEqual(
            {project.state.control for project in all_projects},
            {"working", "cell", "counter"},
        )
        self.assertEqual(
            {project.state.control for project in selected_projects},
            {"cell", "counter"},
        )

    @unittest.skipUnless(GALVANI_FIXTURE.exists(), "Galvani fixture is not available")
    def test_biologic_mpr_fixture_uses_binary_parser(self):
        dataframe, _metadata, technique = _read_eis_dataframe(GALVANI_FIXTURE)
        self.assertEqual(technique, "PEIS")
        self.assertGreater(len(dataframe), 0)
        self.assertIn("freq_hz", dataframe)
        self.assertIn("cycle_number", dataframe)

    def test_invalid_mpr_reports_binary_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.mpr"
            path.write_bytes(b"not a BioLogic file\x00\x00")
            with self.assertRaisesRegex(ValueError, "not a recognized BioLogic"):
                _read_eis_dataframe(path)

    def test_non_mpr_files_remain_on_wepy_path(self):
        expected = pd.DataFrame({"freq_hz": [1.0], "re_z_ohm": [2.0]})
        with patch("wepy.read_mpt_dataframe", return_value=(expected, {}, "PEIS")) as reader:
            dataframe, metadata, technique = _read_eis_dataframe(Path("sample.mpt"))
        reader.assert_called_once_with(Path("sample.mpt"))
        self.assertEqual(dataframe.iloc[0, 0], 1.0)
        self.assertEqual(metadata, {})
        self.assertEqual(technique, "PEIS")


if __name__ == "__main__":
    unittest.main()
