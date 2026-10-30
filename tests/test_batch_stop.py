import threading
import unittest
from unittest.mock import patch

import numpy as np

from eis_model import ParameterValue
from eis_services import (
    BatchCycleFit,
    FitResult,
    LoadedProject,
    SpectrumBatchFit,
    SpectrumBatchReport,
    SpectrumFitTarget,
    batch_fit_from_cycle,
    batch_fit_spectra,
    find_outliers_for_all_cycles,
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

    @patch("eis_services.fit_cycle")
    def test_fit_failure_retains_completed_results_and_stops_batch(self, fit):
        fit.side_effect = [
            fake_fit(self.loaded.state.cycles[1], "R0", self.parameters),
            RuntimeError("synthetic fit failure"),
        ]
        report = batch_fit_spectra(self.targets, self.parameters)
        self.assertEqual(1, len(report.fits))
        self.assertEqual("cycle 2", report.failed_label)
        self.assertIn("synthetic fit failure", report.error)
        self.assertEqual(["cycle 3"], report.skipped_labels)
        self.assertFalse(report.stopped)

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


class BatchProgressTests(unittest.TestCase):
    def setUp(self):
        self.loaded = FakeLoaded([FakeCycle(1), FakeCycle(2), FakeCycle(3)])
        self.parameters = self.loaded.state.parameters_for(1)
        self.targets = [
            SpectrumFitTarget(self.loaded, cycle, f"cycle {cycle}")
            for cycle in (1, 2, 3)
        ]

    @patch("eis_services.fit_cycle", side_effect=fake_fit)
    def test_progress_reports_start_and_completion_per_spectrum(self, _fit):
        events = []
        report = batch_fit_spectra(
            self.targets,
            self.parameters,
            progress_callback=lambda done, total, label: events.append((done, total, label)),
        )
        self.assertEqual(3, len(report.fits))
        self.assertEqual(
            [
                (0, 3, "cycle 1"),
                (1, 3, "cycle 1"),
                (1, 3, "cycle 2"),
                (2, 3, "cycle 2"),
                (2, 3, "cycle 3"),
                (3, 3, "cycle 3"),
            ],
            events,
        )

    @patch("eis_services.fit_cycle", side_effect=fake_fit)
    def test_progress_counts_failed_spectra_as_done(self, _fit):
        _fit.side_effect = [
            fake_fit(self.loaded.state.cycles[1], "R0", self.parameters),
            RuntimeError("synthetic fit failure"),
            fake_fit(self.loaded.state.cycles[3], "R0", self.parameters),
        ]
        events = []
        report = batch_fit_spectra(
            self.targets,
            self.parameters,
            progress_callback=lambda done, total, label: events.append((done, total, label)),
        )
        self.assertEqual(3, len(report.fits))
        self.assertEqual((3, 3, "cycle 3"), events[-1])

    @patch("eis_services.fit_cycle", side_effect=fake_fit)
    def test_progress_stops_when_batch_is_stopped(self, _fit):
        event = threading.Event()

        def fit_and_stop(*args):
            result = fake_fit(*args)
            if _fit.call_count == 1:
                event.set()
            return result

        _fit.side_effect = fit_and_stop
        events = []
        report = batch_fit_spectra(
            self.targets,
            self.parameters,
            stop_event=event,
            progress_callback=lambda done, total, label: events.append((done, total, label)),
        )
        self.assertTrue(report.stopped)
        self.assertEqual((1, 3, "cycle 1"), events[-1])

    @patch("eis_services.analyze_outliers")
    def test_find_outliers_reports_progress_per_cycle(self, analyze):
        analyze.return_value = object()
        loaded = FakeLoaded([FakeCycle(1), FakeCycle(2)])
        loaded.state.available_cycles = [1, 2]
        events = []
        find_outliers_for_all_cycles(
            None,
            loaded.state,
            3.0,
            progress_callback=lambda done, total, cycle: events.append((done, total, cycle)),
        )
        self.assertEqual([(0, 2, 1), (1, 2, 1), (1, 2, 2), (2, 2, 2)], events)

    @patch("eis_services.fit_cycle", side_effect=fake_fit)
    def test_batch_fit_from_cycle_reports_progress(self, _fit):
        loaded = FakeLoaded([FakeCycle(1), FakeCycle(2)])
        loaded.state.available_cycles = [1, 2]
        events = []
        report = batch_fit_from_cycle(
            None,
            loaded.state,
            1,
            self.parameters,
            progress_callback=lambda done, total, cycle: events.append((done, total, cycle)),
        )
        self.assertEqual(2, len(report.fits))
        self.assertEqual([(0, 2, 1), (1, 2, 1), (1, 2, 2), (2, 2, 2)], events)


def fake_refine(state, _circuit, parameters, z_threshold, max_iterations, *args, **kwargs):
    values = np.asarray([parameter.initial for parameter in parameters], dtype=float) * 2.0
    result = FitResult(
        fitted_parameters=values,
        errors_percent=np.full_like(values, 0.5),
        fit_frequency_hz=np.array([1.0]),
        fit_impedance=np.array([2.0 + 0j]),
        fit_at_data_impedance=np.array([2.0 + 0j]),
        objective=0.0,
        rmse=0.0,
        converged=True,
    )
    return result, np.asarray([0], dtype=int), 1


class BatchFitWithRefineTests(unittest.TestCase):
    def setUp(self):
        self.loaded = FakeLoaded([FakeCycle(1), FakeCycle(2), FakeCycle(3)])
        self.parameters = self.loaded.state.parameters_for(1)
        self.targets = [
            SpectrumFitTarget(self.loaded, cycle, f"cycle {cycle}")
            for cycle in (1, 2, 3)
        ]

    @patch("eis_services.refine_fit_cycle", side_effect=fake_refine)
    @patch("eis_services.fit_cycle", side_effect=fake_fit)
    def test_refined_parameters_become_initials_for_next_spectrum(self, fit, _refine):
        report = batch_fit_spectra(
            self.targets,
            self.parameters,
            refine_z_threshold=3.5,
            refine_max_iterations=5,
        )
        self.assertEqual(3, len(report.fits))
        self.assertEqual(
            [1.0, 2.0, 4.0],
            [call.args[2][0].initial for call in fit.call_args_list],
        )
        self.assertEqual([2.0, 4.0, 8.0], [f.fit.fitted_parameters[0] for f in report.fits])
        for result in report.fits:
            self.assertEqual([0], list(result.fit.refined_outlier_indices))
            self.assertEqual(0.5, result.fit.parameters[0].error_percent)

    @patch("eis_services.refine_fit_cycle", side_effect=fake_refine)
    @patch("eis_services.fit_cycle", side_effect=fake_fit)
    def test_stop_between_fit_and_refine_keeps_fit_without_refine(self, fit, _refine):
        event = threading.Event()

        def fit_and_stop(*args):
            result = fake_fit(*args)
            if fit.call_count == 2:
                event.set()
            return result

        fit.side_effect = fit_and_stop
        report = batch_fit_spectra(
            self.targets,
            self.parameters,
            stop_event=event,
            refine_z_threshold=3.5,
            refine_max_iterations=5,
        )
        self.assertTrue(report.stopped)
        self.assertEqual(2, len(report.fits))
        self.assertEqual(["cycle 3"], report.skipped_labels)
        self.assertEqual([0], list(report.fits[0].fit.refined_outlier_indices))
        self.assertEqual([], list(report.fits[1].fit.refined_outlier_indices))

    @patch("eis_services.fit_cycle", side_effect=fake_fit)
    def test_refine_failure_continues_with_next_spectrum(self, _fit):
        with patch(
            "eis_services.refine_fit_cycle",
            side_effect=[fake_refine(*[None, None, self.parameters, 3.5, 5]), RuntimeError("refine failed"), fake_refine(*[None, None, self.parameters, 3.5, 5])],
        ):
            report = batch_fit_spectra(
                self.targets,
                self.parameters,
                refine_z_threshold=3.5,
                refine_max_iterations=5,
            )
        self.assertEqual(3, len(report.fits))
        self.assertIsNone(report.fits[0].error)
        self.assertIn("refine failed", report.fits[1].error)
        self.assertIsNone(report.fits[2].error)
        self.assertFalse(report.stopped)

    @patch("eis_services.refine_fit_cycle", side_effect=fake_refine)
    @patch("eis_services.fit_cycle", side_effect=fake_fit)
    def test_batch_fit_from_cycle_chains_refined_parameters(self, fit, _refine):
        loaded = FakeLoaded([FakeCycle(1), FakeCycle(2)])
        loaded.state.available_cycles = [1, 2]
        report = batch_fit_from_cycle(
            None,
            loaded.state,
            1,
            self.parameters,
            refine_z_threshold=3.5,
            refine_max_iterations=5,
        )
        self.assertEqual(2, len(report.fits))
        self.assertEqual(
            [1.0, 2.0],
            [call.args[2][0].initial for call in fit.call_args_list],
        )
        self.assertEqual([2.0, 4.0], [f.fitted_parameters[0] for f in report.fits])
        for result in report.fits:
            self.assertEqual([0], list(result.refined_outlier_indices))

    @patch("eis_services.refine_fit_cycle", side_effect=fake_refine)
    @patch("eis_services.fit_cycle", side_effect=fake_fit)
    def test_batch_fit_from_cycle_stop_skips_remaining_cycles(self, fit, _refine):
        event = threading.Event()
        event.set()
        loaded = FakeLoaded([FakeCycle(1), FakeCycle(2)])
        loaded.state.available_cycles = [1, 2]
        report = batch_fit_from_cycle(
            None,
            loaded.state,
            1,
            self.parameters,
            stop_event=event,
            refine_z_threshold=3.5,
            refine_max_iterations=5,
        )
        self.assertEqual([], report.fits)
        self.assertTrue(report.stopped)
        self.assertEqual([1, 2], report.skipped_cycles)


if __name__ == "__main__":
    unittest.main()
