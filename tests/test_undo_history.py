import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eis_model import CycleState, ParameterValue, ProjectState, copy_project_state
from eis_project import (
    _dataframe_to_payload,
    _state_to_payload,
    describe_history_change,
    history_to_payload,
    load_json_payload,
    load_project_file,
    load_project_history,
    save_project_file,
)
from eis_services import (
    HistoryEntry,
    UndoHistory,
    copy_loaded_project,
    load_project_from_dataframe,
)


def _dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "freq_hz": [100.0, 10.0, 1.0],
            "re_zwe_ce_ohm": [2.0, 3.0, 4.0],
            "minus_im_zwe_ce_ohm": [0.1, 0.2, 0.3],
            "ewe_ece_v": [1.0, 1.0, 1.0],
            "i_ma": [2.0, 2.0, 2.0],
            "cycle_number": [1, 1, 1],
        }
    )


def _state() -> ProjectState:
    parameter = ParameterValue("R0", "Ohm", 2.0, 0.0, 10.0, 1.5)
    cycle = CycleState(
        1,
        np.array([100.0, 10.0, 1.0]),
        np.array([2.0 - 0.1j, 3.0 - 0.2j, 4.0 - 0.3j]),
        manually_included=np.array([True, False, True]),
        outliers=np.array([False, True, False]),
        parameters=[parameter],
        fit_parameters=np.array([2.1]),
        fit_frequency_hz=np.array([100.0, 1.0]),
        fit_impedance=np.array([2.1 - 0.1j, 4.1 - 0.3j]),
        fit_at_data_impedance=np.array([2.1 - 0.1j, 3.1 - 0.2j, 4.1 - 0.3j]),
    )
    return ProjectState(
        Path("source.mpt"),
        "R0",
        "cell",
        [1],
        1,
        [parameter],
        {1: cycle},
    )


def _loaded_project() -> tuple[object, ProjectState, pd.DataFrame]:
    dataframe = _dataframe()
    loaded = load_project_from_dataframe(
        dataframe, Path("source.mpt"), 1, "cell", "R0"
    )
    return loaded, loaded.state, dataframe


def _payload_for(state: ProjectState, dataframe) -> list:
    return [
        {
            "dataset_id": "ds",
            "state": _state_to_payload(state),
            "dataframe": _dataframe_to_payload(dataframe),
        }
    ]


class UndoHistoryTests(unittest.TestCase):
    def _entry(self, label: str) -> HistoryEntry:
        return HistoryEntry(label=label, current_dataset_id=None, datasets=[])

    def test_push_pop_and_redo_semantics(self):
        history = UndoHistory()
        self.assertFalse(history.can_undo)
        self.assertFalse(history.can_redo)
        history.push(self._entry("one"))
        history.push(self._entry("two"))
        self.assertTrue(history.can_undo)
        self.assertEqual(history.undo_label, "two")
        entry = history.pop_undo()
        self.assertEqual(entry.label, "two")
        history.push_redo(entry)
        self.assertTrue(history.can_redo)
        self.assertEqual(history.redo_label, "two")
        redo_entry = history.pop_redo()
        history.push_undo(redo_entry)
        self.assertEqual(history.undo_label, "two")
        self.assertFalse(history.can_redo)

    def test_new_change_clears_redo_stack(self):
        history = UndoHistory()
        history.push(self._entry("one"))
        history.pop_undo()
        history.push_redo(self._entry("one"))
        history.push(self._entry("two"))
        self.assertFalse(history.can_redo)
        self.assertEqual([entry.label for entry in history.undo_stack], ["two"])

    def test_limit_drops_oldest_entries(self):
        history = UndoHistory(limit=3)
        for index in range(5):
            history.push(self._entry(f"change-{index}"))
        self.assertEqual(len(history.undo_stack), 3)
        self.assertEqual(
            [entry.label for entry in history.undo_stack],
            ["change-2", "change-3", "change-4"],
        )

    def test_clear(self):
        history = UndoHistory()
        history.push(self._entry("one"))
        history.clear()
        self.assertFalse(history.can_undo)
        self.assertFalse(history.can_redo)


class CopyStateTests(unittest.TestCase):
    def test_copy_project_state_is_deep(self):
        state = _state()
        copied = copy_project_state(state)
        state.active.manually_included[0] = False
        state.active.parameters[0].initial = 99.0
        state.active.fit_impedance[0] = 0j
        state.active.custom_metadata["key"] = "changed"
        state.circuit = "R0-p(R1,CPE1)"
        self.assertTrue(copied.active.manually_included[0])
        self.assertEqual(copied.active.parameters[0].initial, 2.0)
        self.assertEqual(copied.active.fit_impedance[0], 2.1 - 0.1j)
        self.assertNotIn("key", copied.active.custom_metadata)
        self.assertEqual(copied.circuit, "R0")

    def test_copy_loaded_project_is_deep(self):
        loaded, _state, _dataframe = _loaded_project()
        copied = copy_loaded_project(loaded)
        loaded.state.active.manually_included[0] = False
        loaded.dataframe.loc[0, "re_zwe_ce_ohm"] = 999.0
        self.assertTrue(copied.state.active.manually_included[0])
        self.assertEqual(copied.dataframe.loc[0, "re_zwe_ce_ohm"], 2.0)


class HistoryPersistenceTests(unittest.TestCase):
    def test_history_round_trip_through_project_file(self):
        loaded, state, dataframe = _loaded_project()
        before = copy_loaded_project(loaded)
        before.state.active.toggle_point(0)
        entry = HistoryEntry(
            label="cycle 1: points",
            current_dataset_id=loaded.dataset_id,
            datasets=[(loaded.dataset_id, before)],
            procedure_blocks={"block": [{"name": "Fit"}]},
            procedures={},
        )
        history = UndoHistory()
        history.push(entry)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project.eisfit.json"
            save_project_file(
                state,
                path,
                datasets=[(loaded.dataset_id, state, dataframe)],
                history=history,
            )
            payload = load_json_payload(path)
            self.assertEqual(payload["version"], 4)
            self.assertIn("history", payload)
            restored_main = load_project_file(state, dataframe, path, payload=payload)
            self.assertIn(1, restored_main.cycles)

            restored_history = load_project_history(payload["history"], path)
            self.assertTrue(restored_history.can_undo)
            self.assertFalse(restored_history.can_redo)
            restored_entry = restored_history.undo_stack[0]
            self.assertEqual(restored_entry.label, "cycle 1: points")
            self.assertEqual(restored_entry.current_dataset_id, loaded.dataset_id)
            self.assertEqual(
                restored_entry.procedure_blocks, {"block": [{"name": "Fit"}]}
            )
            self.assertEqual(
                [dataset_id for dataset_id, _ in restored_entry.datasets],
                [loaded.dataset_id],
            )
            restored_loaded = restored_entry.datasets[0][1]
            np.testing.assert_array_equal(
                restored_loaded.state.active.manually_included,
                before.state.active.manually_included,
            )
            self.assertEqual(
                restored_loaded.state.active.parameters[0].initial,
                before.state.active.parameters[0].initial,
            )
            pd.testing.assert_frame_equal(
                restored_loaded.dataframe, before.dataframe, check_dtype=False
            )

    def test_history_payload_round_trip_without_file(self):
        loaded, _state, _dataframe = _loaded_project()
        entry = HistoryEntry(
            label="fit",
            current_dataset_id=loaded.dataset_id,
            datasets=[(loaded.dataset_id, copy_loaded_project(loaded))],
        )
        history = UndoHistory()
        history.push(entry)
        payload = history_to_payload(history)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project.eisfit.json"
            path.write_text("{}", encoding="utf-8")
            restored = load_project_history(payload, path)
        self.assertEqual(restored.undo_label, "fit")
        self.assertEqual(restored.limit, history.limit)

    def test_missing_history_defaults_to_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project.eisfit.json"
            path.write_text("{}", encoding="utf-8")
            self.assertFalse(load_project_history(None, path).can_undo)
            self.assertFalse(load_project_history({}, path).can_undo)

    def test_older_payload_without_history_still_loads(self):
        state, dataframe = _state(), _dataframe()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project.eisfit.json"
            save_project_file(state, path)
            payload = load_json_payload(path)
            self.assertNotIn("history", payload)
            restored = load_project_file(state, dataframe, path, payload=payload)
            np.testing.assert_array_equal(
                restored.cycles[1].manually_included,
                state.cycles[1].manually_included,
            )
            self.assertFalse(load_project_history(payload.get("history"), path).can_undo)


class DescribeHistoryChangeTests(unittest.TestCase):
    def test_point_toggle_is_labeled(self):
        state = _state()
        old = _payload_for(state, _dataframe())
        changed = copy_project_state(state)
        changed.cycles[1].manually_included[0] = False
        new = _payload_for(changed, _dataframe())
        label = describe_history_change(old, new)
        self.assertIn("cycle 1", label)
        self.assertIn("points", label)

    def test_circuit_change_is_labeled(self):
        state = _state()
        old = _payload_for(state, _dataframe())
        changed = copy_project_state(state)
        changed.circuit = "R0-p(R1,CPE1)"
        new = _payload_for(changed, _dataframe())
        self.assertIn("circuit", describe_history_change(old, new))

    def test_added_dataset_is_labeled(self):
        state = _state()
        old = _payload_for(state, _dataframe())
        new = _payload_for(state, _dataframe()) + [
            {
                "dataset_id": "other",
                "state": _state_to_payload(state),
                "dataframe": _dataframe_to_payload(_dataframe()),
            }
        ]
        self.assertIn("added dataset other", describe_history_change(old, new))

    def test_unchanged_payloads_give_generic_label(self):
        state = _state()
        payload = _payload_for(state, _dataframe())
        self.assertEqual(describe_history_change(payload, payload), "project change")
        self.assertEqual(describe_history_change(None, payload), "project change")


if __name__ == "__main__":
    unittest.main()
