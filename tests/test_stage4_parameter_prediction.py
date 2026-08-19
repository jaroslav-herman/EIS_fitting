import unittest

import numpy as np

from ml.dataset import SpectrumRecord
from ml.run_stage4b_parameters import (
    ABSOLUTE_FEATURE_NAMES,
    _fold_append,
    calculate_absolute_impedance_features,
)

from ml.parameter_prediction import (
    bounds_from_residuals,
    enforce_bounds,
    inverse_target,
    parameter_mapping,
    topology_for_circuit,
    transform_target,
)
from ml.run_stage4a_parameters import loso_splits, multiplicative_error


class Stage4ParameterTests(unittest.TestCase):
    def test_parameter_to_topology_mapping_excludes_l0(self):
        names = ["R0", "L0", "R1", "CPE1_0", "CPE1_1", "R2", "CPE2_0", "CPE2_1"]
        values = [1.0, 2.0, 3.0, 4e-3, .8, 5.0, 6e-3, .7]
        result = parameter_mapping("R0-L0-p(R1,CPE1)-p(R2,CPE2)", values, names)
        self.assertEqual(set(result), {"R0", "R1", "Q1", "alpha1", "R2", "Q2", "alpha2"})
        self.assertNotIn("L0", result)
        self.assertEqual(topology_for_circuit("R0-p(R1,CPE1)"), "ONE_PROCESS")

    def test_positive_log_transform_round_trip(self):
        values = np.array([1e-6, .1, 10.0])
        np.testing.assert_allclose(inverse_target(transform_target(values, "R1"), "R1"), values)

    def test_alpha_transform_round_trip_and_physical_range(self):
        values = np.array([.1, .5, .9])
        result = inverse_target(transform_target(values, "alpha1"), "alpha1")
        np.testing.assert_allclose(result, values)
        self.assertTrue(np.all((result > 0) & (result < 1)))

    def test_residual_bounds_and_physical_enforcement(self):
        predicted = np.array([1.0, 2.0, 3.0])
        bounds = bounds_from_residuals(predicted, np.array([-.1, 0.0, .1]), "R1")
        low, high, _ = bounds["95"]
        self.assertTrue(np.all(low > 0)); self.assertTrue(np.all(high > low))
        low, high, clipped = enforce_bounds(np.array([.5]), np.array([-.1]), np.array([1.2]), "alpha1")
        self.assertGreaterEqual(low[0], 0); self.assertLessEqual(high[0], 1); self.assertGreaterEqual(clipped, 1)

    def test_missing_parameter_is_not_encoded_as_zero(self):
        one = parameter_mapping("R0-p(R1,CPE1)", [1.0, 2.0, .01, .8], ["R0", "R1", "CPE1_0", "CPE1_1"])
        self.assertNotIn("R2", one); self.assertNotIn("Q2", one); self.assertNotIn("alpha2", one)

    def test_stage4a_loso_split_excludes_held_out_sample(self):
        folds = list(loso_splits(("129", "140", "150")))
        self.assertEqual(folds[0], ("129", ("140", "150")))
        self.assertTrue(all(held_out not in train for held_out, train in folds))

    def test_stage4a_multiplicative_error(self):
        self.assertEqual(multiplicative_error(2.0, 1.0), 2.0)
        self.assertEqual(multiplicative_error(2.0, 4.0), 2.0)

    def test_stage4b_absolute_features_use_log_scale_and_robust_endpoints(self):
        frequency = np.array([1.0, 10.0, 100.0, 1000.0])
        z_real = np.array([10.0, 5.0, 2.0, 1.0])
        z_imag = np.array([-4.0, -2.0, -1.0, -.5])
        record = SpectrumRecord(
            spectrum_id="synthetic", source_project="synthetic", sample_id="1", cycle=1,
            voltage=1.0, current=2.0, time=3.0, frequency=frequency,
            z_real=z_real, z_imag=z_imag, topology_label="R0-p(R1,CPE1)",
        )
        features = calculate_absolute_impedance_features(record)
        self.assertEqual(tuple(features), ABSOLUTE_FEATURE_NAMES)
        self.assertAlmostEqual(features["log10_median_abs_Z"], np.log10(np.median(np.hypot(z_real, z_imag))))
        self.assertAlmostEqual(features["Re_Z_high"], np.median(z_real[-4:]))
        self.assertAlmostEqual(features["Im_Z_low"], np.median(z_imag[:4]))

    def test_stage4b_metadata_fill_uses_training_rows_only(self):
        train, test, _ = _fold_append(
            np.empty((2, 0)), np.empty((1, 0)),
            np.array([[1.0], [3.0]]), np.array([[100.0]]), ("voltage",),
        )
        self.assertAlmostEqual(test[0, 0], 100.0)
        self.assertEqual(train.shape, (2, 1))
        self.assertEqual(test.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
