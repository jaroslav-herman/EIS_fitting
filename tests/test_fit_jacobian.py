import unittest

import numpy as np

from eis_services import FitOptions, JacobianUnsupported, circuit_jacobian, fit_cycle
from eis_model import CycleState, ParameterValue
from spectrum_simulator import simulate_spectrum
from impedance.models.circuits import CustomCircuit


class FitJacobianTests(unittest.TestCase):
    def _derivative_error(self, circuit, values):
        frequency = np.logspace(4, -1, 19)
        model = CustomCircuit(circuit, initial_guess=values)
        names, _ = model.get_param_names()
        values = np.asarray(values, dtype=float)
        analytical = circuit_jacobian(frequency, circuit, values, names)
        numerical = np.empty_like(analytical)
        for index in range(values.size):
            step = 1e-6 * max(abs(values[index]), 1.0)
            plus, minus = values.copy(), values.copy()
            plus[index] += step
            minus[index] -= step
            numerical[:, index] = (model.__class__(circuit, initial_guess=plus).predict(frequency)
                                   - model.__class__(circuit, initial_guess=minus).predict(frequency)) / (2 * step)
        np.testing.assert_allclose(analytical, numerical, rtol=2e-4, atol=2e-6)

    def test_common_circuits_match_finite_difference(self):
        self._derivative_error("R0-p(R1,CPE1)", [1.0, 2.0, 0.01, 0.9])
        self._derivative_error("R0-L0-p(R1,CPE1)", [1.0, 1e-5, 2.0, 0.01, 0.9])
        self._derivative_error("R0-p(R1,CPE1)-p(R2,C2)", [1.0, 2.0, .01, .9, 3.0, .02])
        self._derivative_error("R0-p(R1-W1,CPE1)", [1.0, 2.0, .01, .9, 1.0])

    def test_unsupported_element_is_explicit(self):
        with self.assertRaises(JacobianUnsupported):
            circuit_jacobian([1.0, 10.0], "R0-TLMQ1", [1.0, 1.0, 1.0, 1.0, 1.0],
                             ["R0", "TLMQ1_0", "TLMQ1_1", "TLMQ1_2", "TLMQ1_3"])

    def test_analytical_fit_reports_mode(self):
        frequency = np.logspace(4, -1, 25)
        impedance = simulate_spectrum("R0-p(R1,CPE1)", frequency, [1.0, 2.0, .01, .9]).impedance
        parameters = [
            ParameterValue("R0", "ohm", 1.0, 0.0, 10.0),
            ParameterValue("R1", "ohm", 2.0, 0.0, 10.0),
            ParameterValue("CPE1_0", "F", .01, 1e-6, 1.0),
            ParameterValue("CPE1_1", "", .9, .5, 1.0),
        ]
        cycle = CycleState(1, frequency, impedance, parameters=parameters)
        result = fit_cycle(cycle, "R0-p(R1,CPE1)", parameters,
                           options=FitOptions(jacobian_mode="analytical"))
        self.assertEqual(result.jacobian_mode, "analytical")
        self.assertIsNone(result.jacobian_fallback_reason)


if __name__ == "__main__":
    unittest.main()
