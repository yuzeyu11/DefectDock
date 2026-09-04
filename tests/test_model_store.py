import tempfile
import unittest
from pathlib import Path

from defectdock.config import RunConfig
from defectdock.db import ModelStore, RunStore
from defectdock.domain import RunStatus


class ModelStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "metadata" / "defectdock.db"
        self.runs = RunStore(self.db_path)
        self.models = ModelStore(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _completed_run(self, run_id: str, content: bytes):
        config = RunConfig.model_validate(
            {
                "project": "registry-test",
                "dataset": {"path": "dataset.yaml", "version": f"snapshot-{run_id}"},
                "train": {"epochs": 1, "pretrained": False},
            }
        )
        run_dir = self.root / "outputs" / run_id
        artifact = run_dir / "trainer_output" / "weights" / "best.ckpt"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(content)
        (run_dir / "run.manifest.json").write_text('{"run":"ok"}', encoding="utf-8")
        self.runs.create_run(config, run_dir, run_id=run_id)
        self.runs.update_status(run_id, RunStatus.RUNNING)
        run = self.runs.update_status(run_id, RunStatus.SUCCEEDED, metrics={"recall": 0.9})
        return run, artifact

    def test_register_is_idempotent_and_integrity_is_enforced(self):
        run, artifact = self._completed_run("run-one", b"checkpoint-one")
        registered = self.models.register_run(run, artifact, actor="tester")
        repeated = self.models.register_run(run, artifact, actor="tester")
        self.assertEqual(repeated.model_version_id, registered.model_version_id)
        self.assertEqual(registered.artifact_size, len(b"checkpoint-one"))
        self.assertEqual(registered.created_by, "tester")
        self.models.verify_artifact(registered)

        artifact.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.models.verify_artifact(registered)
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.models.register_run(run, artifact, actor="tester")

    def test_activation_history_and_rollback_target(self):
        run_one, artifact_one = self._completed_run("run-one", b"one")
        run_two, artifact_two = self._completed_run("run-two", b"two")
        first = self.models.register_run(run_one, artifact_one, actor="tester")
        second = self.models.register_run(run_two, artifact_two, actor="tester")
        first = self.models.approve(first.model_version_id, actor="reviewer")
        second = self.models.approve(second.model_version_id, actor="reviewer")
        self.assertEqual(first.approved_by, "reviewer")

        first_event = self.models.activate(
            first.model_version_id,
            actor="tester",
            expected_previous_id=None,
        )
        second_event = self.models.activate(
            second.model_version_id,
            actor="tester",
            expected_previous_id=first.model_version_id,
        )
        self.assertIsNone(first_event.previous_model_version_id)
        self.assertEqual(second_event.previous_model_version_id, first.model_version_id)
        self.assertEqual(self.models.get_active_model().model_version_id, second.model_version_id)
        self.assertEqual(self.models.rollback_target().model_version_id, first.model_version_id)
        self.assertEqual(
            [event.model_version_id for event in self.models.activation_history()],
            [second.model_version_id, first.model_version_id],
        )

        rollback = self.models.activate(
            first.model_version_id,
            actor="tester",
            action="rollback",
            expected_previous_id=second.model_version_id,
        )
        self.assertEqual(rollback.action, "rollback")
        self.assertTrue(self.models.get_model(first.model_version_id).is_active)
        self.assertFalse(self.models.get_model(second.model_version_id).is_active)

    def test_activation_compare_and_swap_rejects_stale_state(self):
        run, artifact = self._completed_run("run-one", b"one")
        model = self.models.register_run(run, artifact, actor="tester")
        with self.assertRaisesRegex(ValueError, "approved"):
            self.models.activate(
                model.model_version_id,
                actor="tester",
                expected_previous_id=None,
            )
        model = self.models.approve(model.model_version_id, actor="reviewer")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.models.activate(
                model.model_version_id,
                actor="tester",
                expected_previous_id="stale-model",
            )
        self.assertIsNone(self.models.get_active_model(required=False))


if __name__ == "__main__":
    unittest.main()
