import unittest

from eis_gui import (
    extract_metadata_value_from_filename,
    suggest_metadata_filename_regex,
)


class MetadataFilenameTests(unittest.TestCase):
    def test_matches_stem_and_types_integer(self):
        self.assertEqual(
            extract_metadata_value_from_filename(
                "GEISs_at_N2_flow_80_sccm_automated_C02.mpt",
                r"flow_(?P<flow>\d+)_sccm",
            ),
            80,
        )

    def test_falls_back_to_filename_with_extension(self):
        self.assertEqual(
            extract_metadata_value_from_filename(
                "sample.dat", r"sample\.(?P<kind>dat)"
            ),
            "dat",
        )

    def test_decimal_and_text_values(self):
        self.assertEqual(
            extract_metadata_value_from_filename("gas_flow_2.5_sccm", r"flow_(?P<x>\S+)_sccm"),
            2.5,
        )
        self.assertEqual(
            extract_metadata_value_from_filename("gas_N2_run", r"gas_(?P<gas>[A-Z0-9]+)_run"),
            "N2",
        )

    def test_named_group_can_be_selected(self):
        self.assertEqual(
            extract_metadata_value_from_filename(
                "N2_flow_2", r"(?P<gas>N2)_flow_(?P<flow>\d+)", "flow"
            ),
            2,
        )

    def test_suggests_shared_numeric_filename_pattern(self):
        expression = suggest_metadata_filename_regex(
            [
                "GEISs_at_N2_flow_2_sccm_automated_C02.mpt",
                "GEISs_at_N2_flow_80_sccm_automated_C02.mpt",
            ]
        )
        self.assertIsNotNone(expression)
        self.assertEqual(
            extract_metadata_value_from_filename(
                "GEISs_at_N2_flow_2_sccm_automated_C02.mpt", expression, "value"
            ),
            2,
        )
        self.assertEqual(
            extract_metadata_value_from_filename(
                "GEISs_at_N2_flow_80_sccm_automated_C02.mpt", expression, "value"
            ),
            80,
        )

    def test_suggests_text_pattern_and_returns_none_without_variation(self):
        expression = suggest_metadata_filename_regex(
            ["sample_N2_run.mpt", "sample_O2_run.mpt"]
        )
        self.assertIsNotNone(expression)
        self.assertEqual(
            extract_metadata_value_from_filename("sample_N2_run.mpt", expression),
            "N2",
        )
        self.assertIsNone(suggest_metadata_filename_regex(["same.mpt", "same.mpt"]))

    def test_invalid_rules_and_unmatched_files(self):
        cases = [
            ("sample", r"[", None),
            ("sample", r"sample", None),
            ("sample", r"sample_(?P<one>\d+)", "missing"),
            ("sample", r"other_(?P<value>\d+)", None),
            ("sample_nan", r"sample_(?P<value>nan)", None),
        ]
        for filename, expression, group in cases:
            with self.subTest(filename=filename, expression=expression):
                with self.assertRaises(ValueError):
                    extract_metadata_value_from_filename(filename, expression, group)


if __name__ == "__main__":
    unittest.main()
