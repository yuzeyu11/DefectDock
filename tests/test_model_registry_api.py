import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from defectdock.api import create_app
from defectdock.config import RunConfig
from defectdock.domain import RunStatus


class ModelRegistryApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.app = create_app(workspace=self.root, training_enabled=False)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.app.state.jobs.shutdown(wait=True)
        self.temp_dir.cleanup()

    def _completed_run(self, run_id: str, content: bytes) -> Path:
        run_dir = self.root / "outputs" / run_id
        artifact = run_dir / "trainer_output" / "weights" / "best.ckpt"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(content)
        config = RunConfig.model_validate(
            {
                "project": "api-registry",
                "dataset": {"path": "dataset.yaml", "version": f"snapshot-{run_id}"},
                "train": {"epochs": 1, "pretrained": False, "score_threshold": 0.4},
            }
        )
        self.app.state.store.create_run(config, run_dir, run_id=run_id)
        self.app.state.store.update_status(run_id, RunStatus.RUNNING)
        self.app.state.store.update_status(run_id, RunStatus.SUCCEEDED)
        return artifact

    def test_compatibility_activation_registers_models_and_can_rollback(self):
        self._completed_run("run-one", b"one")
        self._completed_run("run-two", b"two")

        first_response = self.client.post("/api/runs/run-one/activate")
        self.assertEqual(first_response.status_code, 200, first_response.text)
        first = first_response.json()["model_version"]
        second_response = self.client.post("/api/runs/run-two/activate")
        self.assertEqual(second_response.status_code, 200, second_response.text)
        second = second_response.json()["model_version"]

        listing = self.client.get("/api/models")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(len(listing.json()), 2)
        self.assertEqual(self.client.get("/api/models/active").json()["model_version_id"], second["model_version_id"])

        rollback = self.client.post("/api/models/rollback")
        self.assertEqual(rollback.status_code, 200, rollback.text)
        self.assertEqual(rollback.json()["model_version"]["model_version_id"], first["model_version_id"])
        history = self.client.get("/api/models/activation-history").json()
        self.assertEqual([event["action"] for event in history], ["rollback", "activate", "activate"])
        active_config = json.loads(self.app.state.active_model_config.read_text(encoding="utf-8"))
        self.assertEqual(active_config["model_version_id"], first["model_version_id"])
        self.assertEqual(active_config["artifact_sha256"], first["artifact_sha256"])

    def test_tampered_artifact_cannot_replace_active_model(self):
        self._completed_run("run-one", b"one")
        artifact_two = self._completed_run("run-two", b"two")
        active = self.client.post("/api/runs/run-one/activate").json()["model_version"]
        candidate = self.client.post("/api/runs/run-two/register").json()["model_version"]
        previous_config = self.app.state.active_model_config.read_bytes()
        artifact_two.write_bytes(b"tampered")

        rejected = self.client.post(f"/api/models/{candidate['model_version_id']}/activate")
        self.assertEqual(rejected.status_code, 409)
        self.assertIn("integrity", rejected.json()["detail"])
        self.assertEqual(self.app.state.active_model_config.read_bytes(), previous_config)
        self.assertEqual(self.app.state.models.get_active_model().model_version_id, active["model_version_id"])

    def test_registered_model_requires_explicit_approval_before_activation(self):
        self._completed_run("run-one", b"one")
        model = self.client.post("/api/runs/run-one/register").json()["model_version"]
        self.assertEqual(model["approval_status"], "candidate")
        rejected = self.client.post(f"/api/models/{model['model_version_id']}/activate")
        self.assertEqual(rejected.status_code, 409, rejected.text)
        self.assertIn("approved", rejected.json()["detail"])

        approved = self.client.post(f"/api/models/{model['model_version_id']}/approve")
        self.assertEqual(approved.status_code, 200, approved.text)
        self.assertEqual(approved.json()["approval_status"], "approved")
        self.assertEqual(approved.json()["approved_by"], "local")
        activated = self.client.post(f"/api/models/{model['model_version_id']}/activate")
        self.assertEqual(activated.status_code, 200, activated.text)

    def test_database_failure_restores_previous_pointer(self):
        self._completed_run("run-one", b"one")
        self._completed_run("run-two", b"two")
        active = self.client.post("/api/runs/run-one/activate").json()["model_version"]
        candidate = self.client.post("/api/runs/run-two/register").json()["model_version"]
        previous_config = self.app.state.active_model_config.read_bytes()

        with patch.object(self.app.state.models, "activate", side_effect=ValueError("database failed")):
            rejected = self.client.post(f"/api/models/{candidate['model_version_id']}/activate")
        self.assertEqual(rejected.status_code, 409)
        self.assertEqual(self.app.state.active_model_config.read_bytes(), previous_config)
        self.assertEqual(self.app.state.models.get_active_model().model_version_id, active["model_version_id"])

    def test_restart_uses_registry_as_authority_and_rejects_tampering(self):
        artifact = self._completed_run("run-one", b"one")
        activated = self.client.post("/api/runs/run-one/activate")
        self.assertEqual(activated.status_code, 200)

        restarted = create_app(workspace=self.root, training_enabled=False)
        restarted_client = TestClient(restarted)
        try:
            health = restarted_client.get("/api/health").json()
            self.assertEqual(health["active_model_integrity"], "verified")
            self.assertTrue(health["inference_ready"])
        finally:
            restarted_client.close()
            restarted.state.jobs.shutdown(wait=True)

        artifact.write_bytes(b"tampered-after-activation")
        rejected = create_app(workspace=self.root, training_enabled=False)
        rejected_client = TestClient(rejected)
        try:
            health = rejected_client.get("/api/health").json()
            self.assertEqual(health["active_model_integrity"], "failed")
            self.assertFalse(health["inference_ready"])
        finally:
            rejected_client.close()
            rejected.state.jobs.shutdown(wait=True)

    def test_registered_model_can_publish_and_query_verified_onnx_export(self):
        self._completed_run("run-one", b"one")
        model = self.client.post("/api/runs/run-one/register").json()["model_version"]

        def fake_runner(checkpoint, model_path, opset, warmup_runs, benchmark_runs):
            model_path.write_bytes(b"onnx")
            return {
                "classes": ["object"],
                "input_size": 640,
                "validation": {"passed": True},
                "benchmark": {
                    "provider": "fake",
                    "warmup_runs": warmup_runs,
                    "measured_runs": benchmark_runs,
                },
                "runtime": {"opset": opset},
            }

        with patch("defectdock.exports.onnx._run_torchvision_export", side_effect=fake_runner):
            created = self.client.post(
                f"/api/models/{model['model_version_id']}/exports/onnx",
                json={"opset": 18, "warmup_runs": 0, "benchmark_runs": 1},
            )
        self.assertEqual(created.status_code, 201, created.text)
        self.assertTrue(created.json()["created"])
        queried = self.client.get(f"/api/models/{model['model_version_id']}/exports/onnx")
        self.assertEqual(queried.status_code, 200, queried.text)
        self.assertEqual(queried.json()["source"]["checkpoint_sha256"], model["artifact_sha256"])
        repeated = self.client.post(f"/api/models/{model['model_version_id']}/exports/onnx")
        self.assertEqual(repeated.status_code, 201)
        self.assertFalse(repeated.json()["created"])


if __name__ == "__main__":
    unittest.main()
