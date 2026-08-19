import unittest

import numpy as np

from explorer_filter import FilterCondition, FilterDefinition, apply_filters


class ExplorerFilterTests(unittest.TestCase):
    records = [
        {"Day": 5, "I_mA": 20, "Spectrum": "Cell", "label": "baseline"},
        {"Day": 5, "I_mA": 120, "Spectrum": "WE", "label": "heated"},
        {"Day": 6, "I_mA": 80, "Spectrum": "CE", "label": "baseline-2"},
        {"Day": None, "I_mA": 200, "Spectrum": "Cell", "label": None},
    ]

    def test_numeric_operators(self):
        definition = FilterDefinition(
            [FilterCondition("Day", "=", "5"), FilterCondition("I_mA", "<", "100")]
        )
        self.assertEqual([self.records[0]], apply_filters(self.records, definition))

    def test_text_operators_are_case_insensitive(self):
        definition = FilterDefinition([FilterCondition("Spectrum", "=", "cell")])
        self.assertEqual([self.records[0], self.records[3]], apply_filters(self.records, definition))
        definition.conditions[0] = FilterCondition("label", "contains", "BASE")
        self.assertEqual([self.records[0], self.records[2]], apply_filters(self.records, definition))

    def test_any_combines_conditions(self):
        definition = FilterDefinition(
            [FilterCondition("Day", "=", "6"), FilterCondition("Spectrum", "=", "WE")],
            match="any",
        )
        self.assertEqual([self.records[1], self.records[2]], apply_filters(self.records, definition))

    def test_missing_values_do_not_match(self):
        definition = FilterDefinition([FilterCondition("Day", ">", "0")])
        self.assertEqual(self.records[:3], apply_filters(self.records, definition))
        records = [{"value": 1}, {"value": np.nan}]
        self.assertEqual([records[0]], apply_filters(records, definition=FilterDefinition([
            FilterCondition("value", ">", "0")
        ])))
        records_with_text = [{"value": "not numeric"}]
        self.assertEqual([], apply_filters(records_with_text, FilterDefinition([
            FilterCondition("value", ">", "0")
        ])))

    def test_empty_result_is_supported(self):
        definition = FilterDefinition([FilterCondition("Day", "=", "999")])
        self.assertEqual([], apply_filters(self.records, definition))


if __name__ == "__main__":
    unittest.main()
