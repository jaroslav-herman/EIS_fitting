import unittest

import matplotlib

import eis_gui


class PlainPlotTextTests(unittest.TestCase):
    def test_matplotlib_math_rendering_is_disabled_at_import(self):
        self.assertFalse(matplotlib.rcParams["text.usetex"])
        self.assertFalse(matplotlib.rcParams["text.parse_math"])


if __name__ == "__main__":
    unittest.main()
