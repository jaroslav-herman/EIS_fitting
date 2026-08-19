from __future__ import annotations

import unittest

import numpy as np

from ml.low_frequency_selector import select_low_frequency_boundary


def spectrum(points=121, scale=1.0):
    frequency = np.logspace(5, -1, points)
    x = np.log10(frequency)
    impedance = scale * (2.0 + 0.4 / (1.0 + np.exp(x - 2.0))) - 1j * scale * (0.2 + 0.05 * np.sin(x))
    return frequency, impedance


class LowFrequencySelectorTests(unittest.TestCase):
    def test_no_degradation_returns_measured_minimum(self):
        f, z = spectrum()
        result = select_low_frequency_boundary(f, z, method="combined", threshold=4.0)
        self.assertEqual(result.predicted_f_min, np.min(f))

    def test_persistent_low_frequency_noise_is_trimmed(self):
        f, z = spectrum()
        low = np.arange(f.size - 20, f.size)
        rng = np.random.default_rng(4)
        z[low] += rng.normal(0, 8, low.size) + 1j * rng.normal(0, 8, low.size)
        result = select_low_frequency_boundary(f, z, method="local_residual", threshold=1.0, persistence_window=5, min_fraction=0.5)
        self.assertGreater(result.predicted_f_min, np.min(f))

    def test_single_spike_does_not_define_boundary(self):
        f, z = spectrum()
        z[10] += 10 + 10j
        result = select_low_frequency_boundary(f, z, method="local_residual", threshold=3.0, persistence_window=7)
        self.assertEqual(result.predicted_f_min, np.min(f))

    def test_scale_and_order_invariance(self):
        f, z = spectrum()
        order = np.random.default_rng(1).permutation(f.size)
        a = select_low_frequency_boundary(f, z, method="rolling_stability")
        b = select_low_frequency_boundary(f[order], z[order] * 1000, method="rolling_stability")
        self.assertEqual(a.predicted_f_min, b.predicted_f_min)

    def test_invalid_input_is_rejected(self):
        f, z = spectrum(20)
        f[0] = np.nan
        result = select_low_frequency_boundary(f, z)
        self.assertTrue(np.isfinite(result.predicted_f_min))
        self.assertTrue(np.isnan(result.score[0]))


if __name__ == "__main__":
    unittest.main()
