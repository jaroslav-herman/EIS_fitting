import unittest
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from eis_gui import EISApplication


class PlotNavigationTests(unittest.TestCase):
    def setUp(self):
        self.axes = Figure(figsize=(4, 3)).add_subplot(111)
        self.axes.set_xlim(0.0, 10.0)
        self.axes.set_ylim(0.0, 10.0)
        self.axes.figure.canvas.draw()

    def test_control_modifier_is_detected_from_matplotlib_key(self):
        self.assertTrue(EISApplication._event_has_control(SimpleNamespace(key="control")))
        self.assertFalse(EISApplication._event_has_control(SimpleNamespace(key=None)))

    def test_wheel_zoom_keeps_cursor_position_anchor(self):
        event = SimpleNamespace(xdata=2.0, ydata=3.0)
        EISApplication._zoom_axes_at_event(self.axes, event, 0.5)
        self.assertEqual(self.axes.get_xlim(), (1.0, 6.0))
        self.assertEqual(self.axes.get_ylim(), (1.5, 6.5))

    def test_log_zoom_preserves_positive_limits(self):
        self.axes.set_xscale("log")
        self.axes.set_yscale("log")
        self.axes.set_xlim(1.0, 100.0)
        self.axes.set_ylim(0.1, 10.0)
        event = SimpleNamespace(xdata=10.0, ydata=1.0)
        EISApplication._zoom_axes_at_event(self.axes, event, 0.5)
        self.assertGreater(self.axes.get_xlim()[0], 0.0)
        self.assertGreater(self.axes.get_ylim()[0], 0.0)

    def test_middle_drag_pan_preserves_view_span(self):
        state = {
            "axes": self.axes,
            "x": 100.0,
            "y": 100.0,
            "xlim": self.axes.get_xlim(),
            "ylim": self.axes.get_ylim(),
        }
        before_x = self.axes.get_xlim()
        before_y = self.axes.get_ylim()
        EISApplication._pan_axes_from_state(
            state,
            SimpleNamespace(inaxes=self.axes, x=120.0, y=80.0),
        )
        after_x = self.axes.get_xlim()
        after_y = self.axes.get_ylim()
        self.assertAlmostEqual(after_x[1] - after_x[0], before_x[1] - before_x[0])
        self.assertAlmostEqual(after_y[1] - after_y[0], before_y[1] - before_y[0])
        self.assertNotEqual(after_x, before_x)
        self.assertNotEqual(after_y, before_y)
        EISApplication._pan_axes_from_state(
            state,
            SimpleNamespace(inaxes=self.axes, x=120.0, y=80.0),
        )
        self.assertEqual(self.axes.get_xlim(), after_x)
        self.assertEqual(self.axes.get_ylim(), after_y)

    def test_zoom_rectangle_tracks_dragged_area(self):
        rectangle = EISApplication._create_zoom_rectangle(self.axes, 2.0, 3.0)
        try:
            EISApplication._update_zoom_rectangle(rectangle, 7.0, 8.0)
            self.assertEqual(rectangle.get_xy(), (2.0, 3.0))
            self.assertEqual(rectangle.get_width(), 5.0)
            self.assertEqual(rectangle.get_height(), 5.0)
            EISApplication._update_zoom_rectangle(rectangle, 1.0, 1.0)
            self.assertEqual(rectangle.get_xy(), (1.0, 1.0))
            self.assertEqual(rectangle.get_width(), 1.0)
            self.assertEqual(rectangle.get_height(), 2.0)
        finally:
            EISApplication._remove_zoom_rectangle(rectangle)


if __name__ == "__main__":
    unittest.main()
