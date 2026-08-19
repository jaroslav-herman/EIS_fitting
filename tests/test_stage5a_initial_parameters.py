import unittest

from ml.run_stage5a_initial_parameters import (
    build_eec_model,
    inverse_stage4b_prediction,
    l0_status_from_ridge,
)
from ml.l0_decision import decide_l0, high_frequency_inductive_diagnostic
from ml.parameter_limits import build_limit_strategy, make_parameter_limit


class Stage5AInitialParameterTests(unittest.TestCase):
    def test_model_strings(self):
        self.assertEqual(build_eec_model(1, False), "R0-p(R1,CPE1)")
        self.assertEqual(build_eec_model(1, True), "R0-L0-p(R1,CPE1)")
        self.assertEqual(build_eec_model(2, False), "R0-p(R1,CPE1)-p(R2,CPE2)")
        self.assertEqual(build_eec_model(2, True), "R0-L0-p(R1,CPE1)-p(R2,CPE2)")

    def test_stage4b_inverse_transforms(self):
        self.assertAlmostEqual(inverse_stage4b_prediction(2.0, "R1"), 100.0)
        alpha = inverse_stage4b_prediction(0.0, "alpha1")
        self.assertGreater(alpha, 0.0)
        self.assertLess(alpha, 1.0)

    def test_l0_status_does_not_invent_missing_value(self):
        self.assertEqual(l0_status_from_ridge(None), ("unavailable", None, False))
        self.assertEqual(l0_status_from_ridge(-1.0), ("not_required", None, False))
        status, value, required = l0_status_from_ridge(1e-7)
        self.assertEqual(status, "required")
        self.assertEqual(value, 1e-7)
        self.assertTrue(required)

    def test_l0_diagnostic_uses_gui_nyquist_convention_and_persistence(self):
        frequency = [1000.0, 800.0, 600.0, 400.0, 100.0]
        # Raw Im(Z) positive means GUI Nyquist ordinate -Im(Z) is below zero.
        impedance = [1 + 0.1j, 1 + 0.2j, 1 + 0.3j, 1 - 0.1j, 1 - 0.1j]
        diagnostic = high_frequency_inductive_diagnostic(frequency, impedance)
        self.assertEqual(diagnostic["negative_imaginary_consecutive_points"], 3)
        self.assertGreater(diagnostic["high_frequency_inductive_strength"], 0.0)
        decision = decide_l0(diagnostic, strength_threshold=0.01)
        self.assertTrue(decision["l0_required"])

    def test_parameter_limits_are_broad_and_alpha_is_physical(self):
        class Metadata:
            def get(self, name, default=None):
                return {"R1": [1.0, 10.0, 100.0], "R2": [1.0, 1000.0, 100000.0], "alpha1": [0.7, 0.8]}.get(name, default)
        strategy = build_limit_strategy(Metadata())
        positive = make_parameter_limit("R1", 10.0, 0.0, 1e6, strategy["R1"])
        self.assertLess(positive["lower_limit"], 10.0)
        self.assertGreater(positive["upper_limit"], 10.0)
        alpha = make_parameter_limit("alpha1", 0.2, 0.5, 1.0, strategy["alpha1"])
        self.assertEqual(alpha["lower_limit"], 1e-4)
        self.assertEqual(alpha["upper_limit"], 0.9999)


if __name__ == "__main__":
    unittest.main()
