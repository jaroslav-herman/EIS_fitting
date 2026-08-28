from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from eis_model import CycleState, ParameterValue
from ml.runtime_inference import infer_runtime, make_runtime_spectrum, save_runtime_results


class RuntimeInferenceTests(unittest.TestCase):
    def _spectrum(self, number: int, circuit: str):
        frequency = np.logspace(4, -1, 12)
        parameters = [
            ParameterValue("R0", "ohm", 1.0, 0.0, 100.0),
            ParameterValue("R1", "ohm", 2.0, 0.0, 100.0),
            ParameterValue("CPE1_0", "", 0.01, 1e-8, 1.0),
            ParameterValue("CPE1_1", "", 0.8, 1e-4, 0.9999),
        ]
        cycle = CycleState(
            number,
            frequency,
            np.ones(frequency.size, dtype=complex) * (number + 1 - 0.1j),
            frequency_window=(1.0, 1000.0),
            parameters=parameters,
            fit_parameters=np.array([1.0, 2.0, 0.01, 0.8]),
            circuit=circuit,
        )
        return make_runtime_spectrum(f"source{number}::working::{number}", cycle, circuit)

    def test_runtime_prediction_and_sidecar(self):
        circuit = "R0-p(R1,CPE1)"
        training = [self._spectrum(index, circuit) for index in range(4)]
        target = self._spectrum(9, circuit)
        prediction = infer_runtime(
            training,
            [target],
            operations={"frequency", "model", "initial_parameters", "active_points"},
        )
        self.assertEqual(len(prediction), 1)
        self.assertEqual(prediction[0]["predicted_eec_model"], circuit)
        self.assertEqual(len(prediction[0]["final_ml_active_mask"]), 12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime_ml_results.json"
            save_runtime_results(path, prediction, training_count=4, operations={"model"})
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["format"], "eis-fitting-ml-results")
        self.assertEqual(payload["pipeline"]["training_spectra"], 4)


if __name__ == "__main__":
    unittest.main()
