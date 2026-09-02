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


if __name__ == "__main__":
    unittest.main()
