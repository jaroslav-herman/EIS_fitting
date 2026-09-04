import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from eis_model import CycleState, ParameterValue, ProjectState
from eis_services import FitOptions, FitResult, LoadedProject, SpectrumMetadata, SpectrumFitTarget, batch_fit_candidate_spectra, fit_cycle, refine_fit_cycle
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

    def test_refine_can_run_more_than_one_iteration(self):
        frequency = np.arange(1.0, 9.0)
        impedance = np.array([0, 0, 0, 0, 0, 0, 10, 100], dtype=complex)
        parameters = [ParameterValue("R0", "ohm", 1.0, 0.0, 10.0)]
        cycle = CycleState(
            1,
            frequency,
            impedance,
            parameters=parameters,
            fit_parameters=np.array([1.0]),
            fit_frequency_hz=frequency.copy(),
            fit_impedance=np.zeros(frequency.size, dtype=complex),
            fit_at_data_impedance=np.zeros(frequency.size, dtype=complex),
        )
        updated_fit = FitResult(
            fitted_parameters=np.array([1.0]),
            errors_percent=np.array([0.0]),
            fit_frequency_hz=frequency.copy(),
            fit_impedance=np.zeros(frequency.size, dtype=complex),
            fit_at_data_impedance=np.zeros(frequency.size, dtype=complex),
            objective=0.0,
            rmse=0.0,
            converged=True,
        )

        with patch("eis_services.fit_cycle", return_value=updated_fit) as fit:
            result, removed_indices, iterations = refine_fit_cycle(
                cycle, "R0", parameters, z_threshold=0.5, max_iterations=2
            )

        self.assertIsInstance(result, FitResult)
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(iterations, 2)
        self.assertGreater(removed_indices.size, 0)

    def test_candidate_fit_selects_lower_normalized_impedance_error(self):
        cycle, parameters = self._cycle()
        cycle.circuit = "R0-p(R1,CPE1)"
        project = ProjectState(
            source_path=Path("sample.eisfit.json"),
            circuit="R0-p(R1,CPE1)", control="cell", available_cycles=[1],
            active_cycle=1, default_parameters=parameters, cycles={1: cycle},
        )
        loaded = LoadedProject(None, project, "PEIS", [SpectrumMetadata(1, 1.6, 10.0, 1.0, 20, 0.1, 10000.0)], "sample", "sample")
        report = batch_fit_candidate_spectra(
            [SpectrumFitTarget(loaded, 1, "sample")],
            {"sample": ["R0", "R0-p(R1,CPE1)"]},
        )
        self.assertEqual(len(report.fits), 1)
        provenance = report.fits[0].fit.fit_provenance
        self.assertEqual(provenance["selected_circuit"], "R0-p(R1,CPE1)")
        self.assertEqual(len(provenance["candidate_scores"]), 2)


if __name__ == "__main__":
    unittest.main()
