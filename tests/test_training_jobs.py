import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from defectdock.config import RunConfig
from defectdock.db import RunStore
from defectdock.domain import RunStatus
from defectdock.engines import EngineResult, TrainingCancelled
from defectdock.services import TrainingJobManager


class TrainingJobManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = RunStore(self.root / ".defectdock" / "defectdock.db")
        self.config = RunConfig.model_validate(
            {
                "project": "queue-test",
                "dataset": {"path": "datasets/example/data.yaml", "version": "v1"},
                "train": {"epochs": 1, "pretrained": False},
            }
        )
        self.managers: list[TrainingJobManager] = []

    def tearDown(self):
        for manager in self.managers:
            manager.shutdown(wait=True)
        self.temp_dir.cleanup()

    def _manager(self, runner) -> TrainingJobManager:
        manager = TrainingJobManager(self.store, self.root, runner=runner)
        self.managers.append(manager)
        return manager

    def _wait_for(self, run_id: str, status: RunStatus, timeout: float = 5) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.store.get_run(run_id).status == status:
                return
            time.sleep(0.01)
        self.fail(f"run {run_id} did not reach {status.value}")

    def test_successful_job_persists_status_metrics_and_workspace(self):
        observed = {}

        def runner(config, run_dir, on_event, should_cancel, *, project_root):
            observed["project_root"] = Path(project_root)
            observed["cancelled"] = should_cancel()
            return EngineResult(str(run_dir / "trainer_output"), None, None, {"score": 0.9})

        manager = self._manager(runner)
        record = manager.submit(self.config)
        self._wait_for(record.run_id, RunStatus.SUCCEEDED)
        completed = self.store.get_run(record.run_id)
        self.assertEqual(completed.metrics, {"score": 0.9})
        self.assertEqual(observed["project_root"], self.root.resolve())
        self.assertFalse(observed["cancelled"])
        manifest = json.loads(
            (Path(completed.output_dir) / "run.manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["config_hash"], self.config.config_hash)
        events = [
            json.loads(line)
            for line in (Path(completed.output_dir) / "events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        self.assertEqual(events[0]["event"], "run_queued")

    def test_running_job_can_be_cancelled_cooperatively(self):
        started = threading.Event()

        def runner(config, run_dir, on_event, should_cancel, *, project_root):
            started.set()
            while not should_cancel():
                time.sleep(0.01)
            raise TrainingCancelled("cancelled in test runner")

        manager = self._manager(runner)
        record = manager.submit(self.config)
        self.assertTrue(started.wait(timeout=3))
        manager.cancel(record.run_id)
        self._wait_for(record.run_id, RunStatus.CANCELLED)
        self.assertIn("cancelled", self.store.get_run(record.run_id).error)

    def test_restart_marks_incomplete_runs_failed(self):
        run_dir = self.root / "outputs" / "incomplete"
        self.store.create_run(self.config, run_dir, run_id="incomplete")
        self.store.update_status("incomplete", RunStatus.QUEUED)

        manager = self._manager(lambda *args, **kwargs: None)

        recovered = self.store.get_run("incomplete")
        self.assertEqual(recovered.status, RunStatus.FAILED)
        self.assertIn("service restarted", recovered.error)
        self.assertEqual([record.run_id for record in manager.recovered_runs], ["incomplete"])


if __name__ == "__main__":
    unittest.main()
