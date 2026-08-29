import tempfile
import unittest
from pathlib import Path

from defectdock.config import RunConfig
from defectdock.db import RunStore
from defectdock.domain import RunStatus


class RunStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.store = RunStore(root / "metadata" / "defectdock.db")
        self.config = RunConfig.model_validate(
            {
                "project": "board-test",
                "dataset": {"path": "coco8.yaml", "version": "smoke-v1"},
                "train": {"epochs": 1},
            }
        )
        self.output_dir = root / "outputs" / "run-1"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_and_complete_run(self):
        created = self.store.create_run(self.config, self.output_dir, run_id="run-1")
        self.assertEqual(created.status, RunStatus.CREATED)
        running = self.store.update_status("run-1", RunStatus.RUNNING)
        self.assertIsNotNone(running.started_at)
        completed = self.store.update_status(
            "run-1", RunStatus.SUCCEEDED, metrics={"standard": {"map50": 0.9}}
        )
        self.assertEqual(completed.metrics["standard"]["map50"], 0.9)
        self.assertIsNotNone(completed.finished_at)

    def test_terminal_run_cannot_restart(self):
        self.store.create_run(self.config, self.output_dir, run_id="run-2")
        self.store.update_status("run-2", RunStatus.CANCELLED)
        with self.assertRaises(ValueError):
            self.store.update_status("run-2", RunStatus.RUNNING)

    def test_list_can_filter_by_project(self):
        self.store.create_run(self.config, self.output_dir, run_id="run-3")
        records = self.store.list_runs(project="board-test")
        self.assertEqual([record.run_id for record in records], ["run-3"])
        self.assertEqual(self.store.list_runs(project="missing"), [])


if __name__ == "__main__":
    unittest.main()
