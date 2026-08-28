import unittest

import matplotlib

import eis_gui


class PlainPlotTextTests(unittest.TestCase):
    def test_external_tex_is_disabled_and_default_math_text_is_enabled(self):
        self.assertFalse(matplotlib.rcParams["text.usetex"])
        self.assertEqual(
            matplotlib.rcParams["text.parse_math"],
            matplotlib.rcParamsDefault["text.parse_math"],
        )

    def test_gui_uses_matplotlib_defaults_for_plot_text(self):
        for key in (
            "font.family",
            "font.size",
            "axes.labelsize",
            "xtick.labelsize",
            "ytick.labelsize",
            "legend.fontsize",
            "axes.titlesize",
            "figure.titlesize",
        ):
            self.assertEqual(matplotlib.rcParams[key], matplotlib.rcParamsDefault[key])


if __name__ == "__main__":
    unittest.main()
