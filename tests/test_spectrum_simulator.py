from __future__ import annotations

import unittest
import warnings

import numpy as np

from spectrum_simulator import logarithmic_frequencies, simulate_spectrum


class SpectrumSimulatorTests(unittest.TestCase):
    def test_frequency_grid_is_logarithmic_and_descending(self):
        frequency = logarithmic_frequencies(1.0, 1000.0, 10)
        self.assertEqual(len(frequency), 31)
        self.assertTrue(np.all(np.diff(frequency) < 0))
        self.assertAlmostEqual(frequency[0], 1000.0)
        self.assertAlmostEqual(frequency[-1], 1.0)

    def test_noiseless_simulation_is_the_ideal_curve(self):
        frequency = logarithmic_frequencies(1.0, 1e4, 5)
        result = simulate_spectrum("R0-p(R1,CPE1)", frequency, [1, 10, 1e-3, 0.9])
        np.testing.assert_array_equal(result.impedance, result.ideal_impedance)

    def test_noise_is_reproducible_for_a_seed(self):
        frequency = logarithmic_frequencies(1.0, 1e4, 5)
        arguments = dict(
            circuit="R0-p(R1,CPE1)",
            frequency_hz=frequency,
            parameters=[1, 10, 1e-3, 0.9],
            noise_enabled=True,
            noise_level_percent=2,
            seed=42,
        )
        first = simulate_spectrum(**arguments)
        second = simulate_spectrum(**arguments)
        np.testing.assert_array_equal(first.impedance, second.impedance)
        self.assertFalse(np.array_equal(first.impedance, first.ideal_impedance))

    def test_explicit_simulation_parameters_do_not_warn_as_initial_guess(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            simulate_spectrum(
                "R0-p(R1,CPE1)", [10.0, 1.0], [1.0, 2.0, 1e-3, 0.9]
            )
        self.assertFalse(
            any(
                "Simulating circuit based on initial parameters" in str(item.message)
                for item in caught
            )
        )

    def test_invalid_frequency_range_is_rejected(self):
        with self.assertRaises(ValueError):
            logarithmic_frequencies(0, 10, 10)
        with self.assertRaises(ValueError):
            logarithmic_frequencies(10, 1, 10)


if __name__ == "__main__":
    unittest.main()
