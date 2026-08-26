import unittest

import numpy as np

from eis_model import CycleState, ParameterValue
from eis_services import FitOptions, FitResult, fit_cycle
from spectrum_simulator import simulate_spectrum


class FitOptimizerTests(unittest.TestCase):
    def _cycle(self):
        frequency = np.logspace(4, -1, 20)
        impedance = simulate_spectrum(
            "R0-p(R1,CPE1)", frequency, [1.0, 2.0, 0.01, 0.9]
        ).impedance
        parameters = [
            ParameterValue("R0", "ohm", 1.0, 0.0, 10.0),
            ParameterValue("R1", "ohm", 2.0, 0.0, 10.0),
            ParameterValue("CPE1_0", "F", 0.01, 1e-6, 1.0),
            ParameterValue("CPE1_1", "", 0.9, 0.5, 1.0),
        ]
        return CycleState(1, frequency, impedance, parameters=parameters), parameters

    def test_default_fit_returns_structured_result_and_legacy_unpacking(self):
        cycle, parameters = self._cycle()
        result = fit_cycle(cycle, "R0-p(R1,CPE1)", parameters)
        self.assertIsInstance(result, FitResult)
        fitted, errors, *_ = result
        np.testing.assert_allclose(fitted, [1.0, 2.0, 0.01, 0.9])
        self.assertEqual(errors.shape, fitted.shape)
        self.assertTrue(result.converged)
        self.assertEqual(result.options.stages(), ("least_squares",))

    def test_fit_options_pipeline_and_validation(self):
        options = FitOptions(pipeline=("pso", "least_squares"), seed=7)
        self.assertEqual(options.stages(), ("pso", "least_squares"))
        with self.assertRaises(ValueError):
            FitOptions(pipeline=("not-an-optimizer",)).validated()

    def test_fixed_parameter_is_preserved(self):
        cycle, parameters = self._cycle()
        parameters[0].fixed = True
        parameters[0].initial = 1.25
        result = fit_cycle(cycle, "R0-p(R1,CPE1)", parameters)
        self.assertEqual(result.fitted_parameters[0], 1.25)


if __name__ == "__main__":
    unittest.main()
