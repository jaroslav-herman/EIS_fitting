import unittest
from types import SimpleNamespace

import numpy as np

from circuit_structure import circuits_equivalent, parameter_name_mapping
from eis_gui import EISApplication
from eis_model import CycleState, ParameterValue


class CircuitStructureTests(unittest.TestCase):
    def test_element_numbers_do_not_change_topology(self):
        self.assertTrue(
            circuits_equivalent("R0-L0-p(R1,CPE1)", "R0-L5-p(R2,CPE3)")
        )

    def test_series_and_parallel_order_is_normalized(self):
        self.assertTrue(
            circuits_equivalent(
                "R0-L0-p(R1,CPE1)", "R0-p(CPE3,R2)-L5"
            )
        )

    def test_different_hierarchy_is_not_equivalent(self):
        self.assertFalse(
            circuits_equivalent(
                "R0-p(R1,CPE1)-p(R2,CPE2)",
                "R0-p(R1,p(CPE1,R2))-CPE2",
            )
        )

    def test_parameter_mapping_follows_physical_elements(self):
        mapping = parameter_name_mapping(
            "R0-L0-p(R1,CPE1)", "R0-p(CPE3,R2)-L5"
        )
        self.assertEqual(
            {"R0": "R0", "L0": "L5", "R1": "R2", "CPE1": "CPE3"},
            mapping,
        )

    def test_rcpe_block_switch_supports_more_than_two_blocks(self):
        circuit = "R0-L0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)"
        self.assertEqual(("1", "2", "3"), EISApplication._rcpe_block_ids(circuit))
        parameters = [
            ParameterValue("R0", "Ohm", 0.0, 0.0, 10.0),
            ParameterValue("L0", "H", 0.0, 0.0, 10.0),
            ParameterValue("R1", "Ohm", 1.0, 0.0, 10.0),
            ParameterValue("CPE1_0", "F", 11.0, 0.0, 20.0),
            ParameterValue("CPE1_1", "", 0.81, 0.0, 1.0),
            ParameterValue("R2", "Ohm", 2.0, 0.0, 10.0),
            ParameterValue("CPE2_0", "F", 22.0, 0.0, 20.0),
            ParameterValue("CPE2_1", "", 0.82, 0.0, 1.0),
            ParameterValue("R3", "Ohm", 3.0, 0.0, 10.0),
            ParameterValue("CPE3_0", "F", 33.0, 0.0, 20.0),
            ParameterValue("CPE3_1", "", 0.83, 0.0, 1.0),
        ]
        cycle = CycleState(
            1,
            np.array([1.0, 2.0, 3.0]),
            np.ones(3, dtype=complex),
            parameters=parameters,
            fit_parameters=np.arange(len(parameters), dtype=float),
            circuit=circuit,
        )
        state = SimpleNamespace(circuit=circuit)

        self.assertTrue(EISApplication._switch_parameter_blocks(state, cycle, "1", "3"))
        self.assertEqual(3.0, cycle.parameters[2].initial)
        self.assertEqual(33.0, cycle.parameters[3].initial)
        self.assertEqual(0.83, cycle.parameters[4].initial)
        self.assertEqual(1.0, cycle.parameters[8].initial)
        self.assertEqual(11.0, cycle.parameters[9].initial)
        self.assertEqual(0.81, cycle.parameters[10].initial)
        self.assertEqual([8.0, 9.0, 10.0], cycle.fit_parameters[2:5].tolist())
        self.assertEqual([2.0, 3.0, 4.0], cycle.fit_parameters[8:11].tolist())

    def test_block_pair_from_selection_accepts_any_two_blocks(self):
        group_ids = ("1", "2", "3")
        self.assertEqual(("1", "3"), EISApplication._block_pair_from_selection(group_ids, (0, 2)))
        self.assertEqual(("2", "3"), EISApplication._block_pair_from_selection(group_ids, (1, 2)))
        self.assertIsNone(EISApplication._block_pair_from_selection(group_ids, (0,)))
        self.assertIsNone(EISApplication._block_pair_from_selection(group_ids, (0, 1, 2)))
        self.assertIsNone(EISApplication._block_pair_from_selection(group_ids, (1, 1)))
        self.assertIsNone(EISApplication._block_pair_from_selection(group_ids, (0, 5)))

    def test_switch_parameter_blocks_swaps_values_limits_and_fit(self):
        circuit = "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)"
        parameters = [
            ParameterValue("R0", "Ohm", 0.0, 0.0, 10.0),
            ParameterValue("R1", "Ohm", 1.0, 0.1, 11.0, 5.0, True),
            ParameterValue("CPE1_0", "F", 11.0, 1.0, 21.0),
            ParameterValue("CPE1_1", "", 0.81, 0.5, 1.0),
            ParameterValue("R2", "Ohm", 2.0, 0.2, 22.0),
            ParameterValue("CPE2_0", "F", 22.0, 2.0, 32.0),
            ParameterValue("CPE2_1", "", 0.82, 0.5, 1.0),
            ParameterValue("R3", "Ohm", 3.0, 0.3, 33.0, 7.0, False),
            ParameterValue("CPE3_0", "F", 33.0, 3.0, 43.0),
            ParameterValue("CPE3_1", "", 0.83, 0.6, 1.0),
        ]
        cycle = CycleState(
            1,
            np.array([1.0, 2.0, 3.0]),
            np.ones(3, dtype=complex),
            parameters=parameters,
            fit_parameters=np.arange(len(parameters), dtype=float),
            circuit=circuit,
        )
        state = SimpleNamespace(circuit=circuit)

        self.assertTrue(EISApplication._switch_parameter_blocks(state, cycle, "1", "3"))
        by_name = {parameter.name: parameter for parameter in cycle.parameters}
        self.assertEqual("R1", by_name["R1"].name)
        self.assertEqual(3.0, by_name["R1"].initial)
        self.assertEqual((0.3, 33.0), (by_name["R1"].lower, by_name["R1"].upper))
        self.assertEqual(7.0, by_name["R1"].error_percent)
        self.assertFalse(by_name["R1"].fixed)
        self.assertEqual(33.0, by_name["CPE1_0"].initial)
        self.assertEqual((3.0, 43.0), (by_name["CPE1_0"].lower, by_name["CPE1_0"].upper))
        self.assertEqual(0.83, by_name["CPE1_1"].initial)
        self.assertEqual((0.6, 1.0), (by_name["CPE1_1"].lower, by_name["CPE1_1"].upper))
        self.assertEqual(1.0, by_name["R3"].initial)
        self.assertEqual((0.1, 11.0), (by_name["R3"].lower, by_name["R3"].upper))
        self.assertEqual(5.0, by_name["R3"].error_percent)
        self.assertTrue(by_name["R3"].fixed)
        self.assertEqual(11.0, by_name["CPE3_0"].initial)
        self.assertEqual((1.0, 21.0), (by_name["CPE3_0"].lower, by_name["CPE3_0"].upper))
        self.assertEqual(0.81, by_name["CPE3_1"].initial)
        self.assertEqual(2.0, by_name["R2"].initial)
        self.assertEqual((0.2, 22.0), (by_name["R2"].lower, by_name["R2"].upper))
        self.assertEqual([7.0, 8.0, 9.0], cycle.fit_parameters[1:4].tolist())
        self.assertEqual([4.0, 5.0, 6.0], cycle.fit_parameters[4:7].tolist())
        self.assertEqual([1.0, 2.0, 3.0], cycle.fit_parameters[7:10].tolist())

    def test_sort_by_tau_orders_blocks_values_and_limits(self):
        circuit = "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)"
        parameters = [
            ParameterValue("R0", "Ohm", 0.02, 0.0, 10.0),
            ParameterValue("R1", "Ohm", 0.3, 1.0, 10.0),
            ParameterValue("CPE1_0", "F", 1.0, 0.0, 20.0),
            ParameterValue("CPE1_1", "", 1.0, 0.5, 1.0),
            ParameterValue("R2", "Ohm", 0.1, 2.0, 20.0),
            ParameterValue("CPE2_0", "F", 1.0, 0.0, 20.0),
            ParameterValue("CPE2_1", "", 1.0, 0.5, 1.0),
            ParameterValue("R3", "Ohm", 0.2, 3.0, 30.0),
            ParameterValue("CPE3_0", "F", 1.0, 0.0, 20.0),
            ParameterValue("CPE3_1", "", 1.0, 0.5, 1.0),
        ]
        cycle = CycleState(
            1,
            np.array([1.0, 2.0, 3.0]),
            np.ones(3, dtype=complex),
            parameters=parameters,
            fit_parameters=np.arange(len(parameters), dtype=float),
            circuit=circuit,
        )
        state = SimpleNamespace(circuit=circuit)
        app = object.__new__(EISApplication)

        self.assertTrue(EISApplication._sort_cycle_parameters_by_tau(app, state, cycle))
        by_name = {parameter.name: parameter for parameter in cycle.parameters}
        self.assertEqual(0.1, by_name["R1"].initial)
        self.assertEqual(0.2, by_name["R2"].initial)
        self.assertEqual(0.3, by_name["R3"].initial)
        self.assertEqual((2.0, 20.0), (by_name["R1"].lower, by_name["R1"].upper))
        self.assertEqual((3.0, 30.0), (by_name["R2"].lower, by_name["R2"].upper))
        self.assertEqual((1.0, 10.0), (by_name["R3"].lower, by_name["R3"].upper))
        self.assertEqual([4.0, 5.0, 6.0], cycle.fit_parameters[1:4].tolist())
        self.assertEqual([7.0, 8.0, 9.0], cycle.fit_parameters[4:7].tolist())
        self.assertEqual([1.0, 2.0, 3.0], cycle.fit_parameters[7:10].tolist())

        self.assertFalse(EISApplication._sort_cycle_parameters_by_tau(app, state, cycle))


if __name__ == "__main__":
    unittest.main()
