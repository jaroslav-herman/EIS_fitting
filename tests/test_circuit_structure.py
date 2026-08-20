import unittest

from circuit_structure import circuits_equivalent, parameter_name_mapping


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


if __name__ == "__main__":
    unittest.main()
