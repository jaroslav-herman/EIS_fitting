from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from eis_gui import EISApplication
from extract_relaxis import export_to_eisfit_json


class RelaxisMaskLimitModelTests(unittest.TestCase):
    @staticmethod
    def _create_database(path: Path) -> None:
        con = sqlite3.connect(path)
        try:
            con.executescript(
                """
                CREATE TABLE Files (
                    ID INTEGER PRIMARY KEY, groupname TEXT, datasource TEXT,
                    fitted INTEGER, lasttransferfunction TEXT,
                    lowfreqlimit REAL, highfreqlimit REAL
                );
                CREATE TABLE Datapoints (
                    ID INTEGER PRIMARY KEY, file_id INTEGER, frequency REAL,
                    zreal REAL, zimag REAL, active INTEGER
                );
                CREATE TABLE Fitparameters (
                    file_id INTEGER, pindex INTEGER, name TEXT, fixed INTEGER,
                    value REAL, error REAL, lowerlimit REAL, upperlimit REAL
                );
                CREATE TABLE FileInformation (file_id INTEGER, name TEXT, value TEXT);
                """
            )
            files = [
                (1, "R-(R)(P)", "spectrum-a", 1, "Impedance", 2.0, 8.0),
                (2, "R-I-(R)(P)-(R)(P)", "spectrum-b", 1, "Impedance", 2.0, 8.0),
                (3, "R-(R)(P)", "spectrum-c", 1, "Impedance", 2.0, 8.0),
            ]
            con.executemany("INSERT INTO Files VALUES (?, ?, ?, ?, ?, ?, ?)", files)
            frequencies = (1.0, 2.0, 4.0, 8.0)
            for file_id in (1, 2, 3):
                con.executemany(
                    "INSERT INTO Datapoints VALUES (?, ?, ?, ?, ?, ?)",
                    [(file_id * 10 + i, file_id, f, 1.0 + f, -0.1 * f, int(i < 2)) for i, f in enumerate(frequencies)],
                )
            con.commit()
        finally:
            con.close()

    def test_mask_is_associated_after_descending_sort_and_limits_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "masks.eis3"
            self._create_database(source)
            converted = export_to_eisfit_json(source, root / "converted")
            loaded = EISApplication._load_saved_project(converted)
            state = loaded[0][2]
            cycle = state.active

            by_frequency = dict(zip(cycle.frequency_hz.tolist(), cycle.manually_included.tolist()))
            self.assertEqual(by_frequency, {8.0: False, 4.0: False, 2.0: True, 1.0: True})
            self.assertEqual(cycle.frequency_window, (2.0, 8.0))
            self.assertEqual(dict(zip(cycle.frequency_hz.tolist(), cycle.included.tolist())),
                             {8.0: False, 4.0: False, 2.0: True, 1.0: False})

    def test_each_spectrum_keeps_its_relaxis_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "models.eis3"
            self._create_database(source)
            converted = export_to_eisfit_json(source, root / "converted")
            payload = json.loads(converted.read_text(encoding="utf-8"))
            circuits = [dataset["state"]["circuit"] for dataset in payload["datasets"]]
            self.assertEqual(circuits, [
                "R0-p(R1,CPE1)",
                "R0-L0-p(R1,CPE1)-p(R2,CPE2)",
                "R0-p(R1,CPE1)",
            ])
            loaded = EISApplication._load_saved_project(converted)
            self.assertEqual([state.circuit for _, _, state in loaded], circuits)

    def test_unknown_model_requires_explicit_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "unknown.eis3"
            self._create_database(source)
            con = sqlite3.connect(source)
            try:
                con.execute("UPDATE Files SET groupname = 'R-X-(Q)', lasttransferfunction = 'Impedance' WHERE ID = 1")
                con.commit()
            finally:
                con.close()
            with self.assertRaisesRegex(ValueError, "cannot be mapped automatically"):
                export_to_eisfit_json(source, root / "failed")
            converted = export_to_eisfit_json(
                source,
                root / "fallback",
                unmapped_model_handler=lambda _model, _instance: {"circuit": "R0-p(R1,CPE1)"},
            )
            self.assertTrue(converted.is_file())

    def test_conflict_handler_override_can_apply_to_remaining_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "conflicts.eis3"
            self._create_database(source)
            con = sqlite3.connect(source)
            try:
                con.execute("UPDATE Files SET datasource = 'Copy of spectrum-a', groupname = 'R-I-(R)(P)' WHERE ID = 2")
                con.execute("UPDATE Files SET datasource = 'spectrum-b', groupname = 'R-(R)(P)' WHERE ID = 3")
                con.execute("INSERT INTO Files VALUES (4, 'R-I-(R)(P)', 'Copy of spectrum-b', 1, 'Impedance', 2.0, 8.0)")
                con.executemany(
                    "INSERT INTO Datapoints VALUES (?, 4, ?, ?, ?, ?)",
                    [(40 + i, f, 1.0 + f, -0.1 * f, int(i < 2)) for i, f in enumerate((1.0, 2.0, 4.0, 8.0))],
                )
                con.commit()
            finally:
                con.close()
            calls = []
            override = {"model": "R-I-(R)(P)", "circuit": "R0-L0-p(R1,CPE1)"}

            def handler(_datasource, instances):
                calls.append(instances)
                if len(calls) == 1:
                    return {"index": 1, **override}
                return override

            converted = export_to_eisfit_json(source, root / "converted", model_conflict_handler=handler)
            payload = json.loads(converted.read_text(encoding="utf-8"))
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(dataset["state"]["circuit"] == override["circuit"] for dataset in payload["datasets"]))


if __name__ == "__main__":
    unittest.main()
