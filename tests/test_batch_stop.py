import threading
import unittest
from unittest.mock import patch

import numpy as np

from eis_model import ParameterValue
from eis_services import (
    BatchCycleFit,
    LoadedProject,
    SpectrumBatchFit,
    SpectrumBatchReport,
    SpectrumFitTarget,
    batch_fit_spectra,
)


class FakeCycle:
    def __init__(self, cycle, circuit="R0", parameters=None):
        self.cycle = cycle
        self.circuit = circuit
        self.parameters = parameters or [ParameterValue("R0", "Ohm", 1.0, 0.0, 10.0, None)]

    def model(self, circuit):
        return self.circuit or circuit


class FakeState:
    circuit = "R0"
    control = "Ewe"
    all_frequency_window = None

    def __init__(self, cycles):
        self.cycles = {cycle.cycle: cycle for cycle in cycles}

    def parameters_for(self, cycle_number):
        return self.cycles[cycle_number].parameters


class FakeLoaded:
    dataframe = None
    dataset_label = "sample"

    def __init__(self, cycles):
        self.state = FakeState(cycles)


def fake_fit(cycle, _circuit, parameters):
    values = np.asarray([parameter.initial for parameter in parameters], dtype=float)
    return values, np.zeros_like(values), np.array([1.0]), np.array([1.0 + 0j]), np.array([1.0 + 0j])


class BatchStopTests(unittest.TestCase):
    def setUp(self):
        self.loaded = FakeLoaded([FakeCycle(1), FakeCycle(2), FakeCycle(3)])
        self.parameters = self.loaded.state.parameters_for(1)
        self.targets = [
            SpectrumFitTarget(self.loaded, cycle, f"cycle {cycle}")
            for cycle in (1, 2, 3)
        ]

    @patch("eis_services.fit_cycle", side_effect=fake_fit)
    def test_without_stop_processes_all(self, _fit):
        report = batch_fit_spectra(self.targets, self.parameters)
        self.assertEqual(3, len(report.fits))
        self.assertFalse(report.stopped)

    @patch("eis_services.fit_cycle", side_effect=fake_fit)
    def test_stop_before_first_skips_all(self, _fit):
        event = threading.Event()
        event.set()
        report = batch_fit_spectra(self.targets, self.parameters, stop_event=event)
        self.assertEqual([], report.fits)
        self.assertEqual([target.label for target in self.targets], report.skipped_labels)
        self.assertTrue(report.stopped)

    @patch("eis_services.fit_cycle", side_effect=fake_fit)
    def test_stop_after_current_retains_completed_results(self, fit):
        event = threading.Event()

        def fit_and_stop(*args):
            result = fake_fit(*args)
            if fit.call_count == 2:
                event.set()
            return result

        fit.side_effect = fit_and_stop
        report = batch_fit_spectra(self.targets, self.parameters, stop_event=event)
        self.assertEqual(2, len(report.fits))
        self.assertEqual(["cycle 3"], report.skipped_labels)
        self.assertTrue(report.stopped)

    @patch("eis_services.fit_cycle", side_effect=fake_fit)
    def test_batch_accepts_equivalent_element_numbering(self, _fit):
        source_parameters = [
            ParameterValue("R0", "Ohm", 1.0, 0.0, 10.0, None),
            ParameterValue("L0", "H", 1.0, 0.0, 10.0, None),
            ParameterValue("R1", "Ohm", 1.0, 0.0, 10.0, None),
            ParameterValue("CPE1_0", "F", 1.0, 0.0, 10.0, None),
            ParameterValue("CPE1_1", "", 0.8, 0.0, 1.0, None),
        ]
        target_parameters = [
            ParameterValue("R0", "Ohm", 1.0, 0.0, 10.0, None),
            ParameterValue("R2", "Ohm", 1.0, 0.0, 10.0, None),
            ParameterValue("CPE3_0", "F", 1.0, 0.0, 10.0, None),
            ParameterValue("CPE3_1", "", 0.8, 0.0, 1.0, None),
            ParameterValue("L5", "H", 1.0, 0.0, 10.0, None),
        ]
        loaded = FakeLoaded([])
        first = FakeCycle(1, "R0-L0-p(R1,CPE1)", source_parameters)
        second = FakeCycle(2, "R0-p(CPE3,R2)-L5", target_parameters)
        loaded.state = FakeState([first, second])
        targets = [
            SpectrumFitTarget(loaded, 1, "cycle 1"),
            SpectrumFitTarget(loaded, 2, "cycle 2"),
        ]
        report = batch_fit_spectra(
            targets,
            source_parameters,
            initial_circuit="R0-L0-p(R1,CPE1)",
        )
        self.assertEqual(2, len(report.fits))
        self.assertIsNone(report.failed_label)

    @patch("eis_services.fit_cycle", side_effect=fake_fit)
    def test_selected_fit_clamps_previous_initials_to_current_bounds(self, fit):
        parameters = [ParameterValue("R0", "Ohm", 20.0, 0.0, 10.0, None)]
        loaded = FakeLoaded([FakeCycle(1, parameters=parameters), FakeCycle(2, parameters=parameters)])
        targets = [
            SpectrumFitTarget(loaded, cycle, f"cycle {cycle}")
            for cycle in (1, 2)
        ]

        report = batch_fit_spectra(
            targets,
            parameters,
            use_target_initial_parameters=True,
        )

        self.assertEqual(2, len(report.fits))
        self.assertEqual([10.0, 10.0], [call.args[2][0].initial for call in fit.call_args_list])
        self.assertEqual(20.0, parameters[0].initial)


if __name__ == "__main__":
    unittest.main()
