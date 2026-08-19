from __future__ import annotations

import unittest

import numpy as np

from ml.point_validity import detect_outliers_in_active_points, detect_valid_points


def smooth_spectrum(size=81, scale=1.0):
    frequency = np.logspace(5, -1, size)
    x = np.log10(frequency)
    impedance = scale * (2.0 + 0.5 / (1.0 + np.exp(x - 2.0)) - 1j * (0.2 + 0.1 * np.sin(x)))
    return frequency, impedance


class PointValidityTests(unittest.TestCase):
    def test_active_detector_only_removes_active_points(self):
        frequency, impedance = smooth_spectrum()
        impedance[40] += 10.0 + 10.0j
        impedance[41] += 10.0 + 10.0j
        active = np.ones(frequency.size, dtype=bool)
        active[41] = False
        removed, _ = detect_outliers_in_active_points(
            frequency, impedance, active, threshold=4.0
        )
        self.assertIn(40, removed)
        self.assertNotIn(41, removed)
        self.assertTrue(np.all(active[~active] == False))

    def test_active_detector_respects_existing_mask_and_is_repeatable(self):
        frequency, impedance = smooth_spectrum()
        impedance[40] += 10.0 + 10.0j
        active = np.ones(frequency.size, dtype=bool)
        active[10] = False
        removed, _ = detect_outliers_in_active_points(
            frequency, impedance, active, threshold=4.0
        )
        active[removed] = False
        repeated, _ = detect_outliers_in_active_points(
            frequency, impedance, active, threshold=4.0
        )
        self.assertNotIn(10, removed)
        self.assertEqual(repeated.size, 0)
        self.assertTrue(np.all(active[~active] == False))

    def test_active_detector_threshold_and_validation(self):
        frequency, impedance = smooth_spectrum()
        impedance[40] += 10.0 + 10.0j
        active = np.ones(frequency.size, dtype=bool)
        low_threshold, _ = detect_outliers_in_active_points(
            frequency, impedance, active, threshold=4.0
        )
        high_threshold, _ = detect_outliers_in_active_points(
            frequency, impedance, active, threshold=100.0
        )
        self.assertGreaterEqual(low_threshold.size, high_threshold.size)
        with self.assertRaises(ValueError):
            detect_outliers_in_active_points(
                frequency, impedance, active, threshold=0.0
            )

    def test_smooth_spectrum_has_no_rejections(self):
        frequency, impedance = smooth_spectrum()
        mask, score, diagnostics = detect_valid_points(frequency, impedance, threshold=4.0)
        self.assertTrue(mask.all())
        self.assertEqual(diagnostics["rejection_reason"].tolist().count("local_anomaly"), 0)
        self.assertTrue(np.isfinite(score).sum() >= frequency.size - 2)

    def test_isolated_spike_is_detected(self):
        frequency, impedance = smooth_spectrum()
        impedance[40] += 10.0 + 10.0j
        mask, score, diagnostics = detect_valid_points(frequency, impedance, threshold=4.0)
        self.assertFalse(mask[40])
        self.assertEqual(diagnostics["rejection_reason"][40], "local_anomaly")
        self.assertGreater(score[40], 4.0)

    def test_adjacent_spikes_are_reported_as_a_limitation(self):
        frequency, impedance = smooth_spectrum()
        impedance[40:42] += 10.0 + 10.0j
        mask, _, diagnostics = detect_valid_points(frequency, impedance, threshold=4.0)
        self.assertFalse(mask[40:42].all())
        self.assertTrue(np.any(diagnostics["rejection_reason"] == "local_anomaly"))

    def test_realistic_noise_does_not_reject_excessively(self):
        rng = np.random.default_rng(42)
        frequency, impedance = smooth_spectrum(121)
        impedance = impedance + rng.normal(0, 0.002, impedance.size) + 1j * rng.normal(0, 0.002, impedance.size)
        mask, _, _ = detect_valid_points(frequency, impedance, threshold=5.0)
        self.assertGreater(mask.mean(), 0.90)

    def test_scale_normalization(self):
        frequency, impedance = smooth_spectrum()
        noisy = impedance.copy(); noisy[40] += 0.5 + 0.5j
        small_mask, _, _ = detect_valid_points(frequency, noisy, threshold=4.0)
        large_mask, _, _ = detect_valid_points(frequency, noisy * 1000.0, threshold=4.0)
        self.assertEqual(small_mask.tolist(), large_mask.tolist())

    def test_nonuniform_frequency_and_unsorted_duplicate_input(self):
        frequency, impedance = smooth_spectrum(60)
        order = np.random.default_rng(1).permutation(frequency.size)
        frequency, impedance = frequency[order], impedance[order]
        frequency = np.insert(frequency, 10, frequency[10])
        impedance = np.insert(impedance, 10, impedance[10])
        mask, _, diagnostics = detect_valid_points(frequency, impedance)
        self.assertEqual(mask.size, frequency.size)
        self.assertEqual(diagnostics["frequency"].size, frequency.size)

    def test_invalid_and_edge_points_are_diagnosed(self):
        frequency, impedance = smooth_spectrum(12)
        frequency[0] = np.nan
        impedance[-1] = np.inf + 1j
        mask, _, diagnostics = detect_valid_points(frequency, impedance)
        self.assertFalse(mask[0]); self.assertFalse(mask[-1])
        self.assertEqual(diagnostics["rejection_reason"][0], "input_invalid")
        self.assertEqual(diagnostics["rejection_reason"][-1], "input_invalid")

    def test_small_and_nearly_constant_spectra_are_stable(self):
        frequency = np.logspace(4, -2, 6)
        impedance = np.full(6, 3.0 + 1e-12j)
        mask, score, diagnostics = detect_valid_points(frequency, impedance, min_points=3)
        self.assertTrue(np.isfinite(score[mask]).all())
        self.assertTrue(mask.all())
        self.assertTrue(np.isfinite(diagnostics["local_scale"][mask]).all())


if __name__ == "__main__":
    unittest.main()
