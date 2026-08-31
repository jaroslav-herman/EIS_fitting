import unittest

from eis_gui import EISApplication


class ExplorerColumnTests(unittest.TestCase):
    def test_hidden_columns_are_removed_from_display_without_changing_order(self):
        application = object.__new__(EISApplication)
        application._custom_metadata_columns = ["Voltage setpoint", "_ml_outliers"]
        application._explorer_column_order_preference = ["points", "cycle", "Voltage setpoint"]
        application._explorer_hidden_columns_preference = ["points", "_ml_outliers"]
        application._explorer_current_column_order = None
        application._explorer_new_columns_position = "end"

        self.assertEqual(
            [
                "cycle",
                "Voltage setpoint",
                "fitted",
                "drt",
                "model",
                "source",
                "potential",
                "current",
                "time",
                "f_min",
                "f_max",
            ],
            application._explorer_display_columns(),
        )


if __name__ == "__main__":
    unittest.main()
