from __future__ import annotations

import sqlite3
from pathlib import Path
import tempfile
import unittest

from eis_gui import EISApplication
from extract_relaxis import export_to_eisfit_json


class RelaxisProjectLoadTests(unittest.TestCase):
    @staticmethod
    def _create_database(path: Path) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.executescript(
                """
                CREATE TABLE Files (
                    ID INTEGER PRIMARY KEY, groupname TEXT, datasource TEXT,
                    fitted INTEGER, lasttransferfunction TEXT
                );
                CREATE TABLE Datapoints (
                    ID INTEGER PRIMARY KEY, file_id INTEGER, frequency REAL,
                    zreal REAL, zimag REAL, active INTEGER
                );
                CREATE TABLE Fitparameters (
                    file_id INTEGER, pindex INTEGER, name TEXT, fixed INTEGER,
                    value REAL, error REAL, lowerlimit REAL, upperlimit REAL
                );
                CREATE TABLE FileInformation (
                    file_id INTEGER, name TEXT, value TEXT
                );
                """
            )
            connection.execute(
                "INSERT INTO Files VALUES (1, 'R-(R)(P)', 'sample', 1, 'R-(R)(P)')"
            )
            connection.executemany(
                "INSERT INTO Datapoints VALUES (?, 1, ?, ?, ?, ?)",
                [(1, 10.0, 1.0, -0.1, 1), (2, 1.0, 2.0, -0.2, 0)],
            )
            connection.executemany(
                "INSERT INTO Fitparameters VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (0, "Resistance 1", 0, 1.5, 0.1, 0.1, 10.0),
                    (1, "CPE Q 1", 1, 0.01, 0.001, 1e-6, 1.0),
                    (2, "CPE Alpha 1", 0, 0.9, 0.01, 0.5, 1.0),
                ],
            )
            connection.executemany(
                "INSERT INTO FileInformation VALUES (1, ?, ?)",
                [("DCVoltage", "0.25"), ("Current", "3.5"), ("IsEpsOnlyData", "0")],
            )
            connection.commit()
        finally:
            connection.close()

    def test_relaxis_conversion_loads_as_native_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sample.eis3"
            self._create_database(source)
            converted = export_to_eisfit_json(source, root / "converted")
            loaded = EISApplication._load_saved_project(converted)

            self.assertEqual(len(loaded), 1)
            _dataset_id, project, state = loaded[0]
            cycle = state.cycles[1]
            self.assertEqual(cycle.model(state.circuit), "R0-p(R1,CPE1)")
            self.assertEqual(cycle.frequency_window, (1.0, 10.0))
            self.assertEqual(cycle.manually_included.tolist(), [True, False])
            self.assertEqual((-cycle.impedance.imag).tolist(), [0.1, 0.2])
            self.assertEqual(cycle.custom_metadata["Ecell_V"], "0.25")
            self.assertEqual(cycle.custom_metadata["I_mA"], "3.5")
            parameters = {parameter.name: parameter for parameter in cycle.parameters}
            self.assertEqual(parameters["R0"].initial, 1.5)
            self.assertEqual(parameters["R0"].lower, 0.1)
            self.assertTrue(parameters["CPE1_0"].fixed)


if __name__ == "__main__":
    unittest.main()
