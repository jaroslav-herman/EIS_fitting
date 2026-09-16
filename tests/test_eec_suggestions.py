import unittest

from eis_services import FittedEECRecord, suggest_eec_parameters


def record(
    identity,
    cycle,
    value,
    *,
    circuit="R0",
    names=("R0",),
    potential=0.0,
    current=0.0,
    metadata=None,
):
    return FittedEECRecord(
        identity=identity,
        cycle=cycle,
        potential_v=potential,
        current_ma=current,
        circuit=circuit,
        parameter_names=tuple(names),
        fitted_parameters=tuple(value),
        custom_metadata=dict(metadata or {}),
    )


class EECSuggestionTests(unittest.TestCase):
    def test_weighted_positive_values_use_log_average_and_exclude_target(self):
        target = record("target", 3, (), current=3.0)
        result = suggest_eec_parameters(
            target,
            [
                record("left", 2, (1.0,), current=2.0),
                record("right", 4, (9.0,), current=4.0),
                record("target", 3, (100.0,), current=3.0),
            ],
        )
        self.assertAlmostEqual(result.values["R0"], 3.0)
        self.assertEqual([item.identity for item in result.contributors], ["left", "right"])

    def test_coordinates_are_normalized_and_cycle_mod_is_used_but_time_is_not(self):
        target = record(
            "target", 10, (), potential=1.0, current=100.0,
            metadata={"Cycle mod 15": 0, "time/s": 100000.0},
        )
        result = suggest_eec_parameters(
            target,
            [
                record("near", 11, (2.0,), potential=1.01, current=101.0,
                       metadata={"Cycle mod 15": 1, "time/s": 0.0}),
                record("far", 100, (20.0,), potential=1.5, current=500.0,
                       metadata={"Cycle mod 15": 0, "time/s": 100001.0}),
            ],
            max_neighbors=1,
        )
        self.assertEqual(result.contributors[0].identity, "near")
        self.assertIn("Cycle mod 15", result.coordinate_names)
        self.assertNotIn("time/s", result.coordinate_names)

    def test_equivalent_circuit_parameter_names_are_mapped(self):
        target = record(
            "target", 3, (), circuit="R0-p(CPE3,R2)-L5",
            names=("R0", "L5", "R2", "CPE3_0", "CPE3_1"),
        )
        source = record(
            "source", 2, (2.0, 3.0, 4.0, 5.0, 0.8),
            circuit="R0-L0-p(R1,CPE1)",
            names=("R0", "L0", "R1", "CPE1_0", "CPE1_1"),
            current=1.0,
        )
        result = suggest_eec_parameters(target, [source])
        self.assertAlmostEqual(result.values["R0"], 2.0)
        self.assertAlmostEqual(result.values["L5"], 3.0)
        self.assertAlmostEqual(result.values["R2"], 4.0)
        self.assertAlmostEqual(result.values["CPE3_0"], 5.0)
        self.assertAlmostEqual(result.values["CPE3_1"], 0.8)

    def test_invalid_candidates_and_missing_coordinates_are_reported(self):
        target = record("target", 3, ())
        result = suggest_eec_parameters(
            target,
            [
                record("different", 2, (1.0,), circuit="R0-L0"),
                record("missing", 1, (2.0,), current=float("nan")),
                record("complete", 4, (3.0,), current=1.0),
            ],
        )
        self.assertTrue(result.values)
        self.assertTrue(any("incompatible circuit" in warning for warning in result.warnings))
        self.assertTrue(any("missing suggestion coordinate" in warning for warning in result.warnings))

    def test_constant_coordinates_have_no_suggestion(self):
        target = record("target", 3, (), current=1.0)
        result = suggest_eec_parameters(target, [record("source", 3, (2.0,), current=1.0)])
        self.assertFalse(result.values)
        self.assertIn("constant", " ".join(result.warnings))


if __name__ == "__main__":
    unittest.main()
