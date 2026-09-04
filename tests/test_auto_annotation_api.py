import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from defectdock.api import create_app
from defectdock.config import RunConfig
from defectdock.domain import RunStatus


class AutoAnnotationApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.app = create_app(workspace=self.root, training_enabled=False)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.app.state.jobs.shutdown(wait=True)
        self.temp_dir.cleanup()

    @staticmethod
    def _png_bytes() -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (100, 50), color=(80, 90, 100)).save(buffer, format="PNG")
        return buffer.getvalue()

    def _registered_model(self) -> dict:
        config = RunConfig.model_validate(
            {
                "project": "auto-label",
                "dataset": {"path": "source.yaml", "version": "source-v1"},
                "train": {"epochs": 1, "pretrained": False},
            }
        )
        run_dir = self.root / "outputs" / "auto-run"
        artifact = run_dir / "trainer_output" / "weights" / "best.ckpt"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"checkpoint")
        self.app.state.store.create_run(config, run_dir, run_id="auto-run")
        self.app.state.store.update_status("auto-run", RunStatus.RUNNING)
        self.app.state.store.update_status("auto-run", RunStatus.SUCCEEDED)
        response = self.client.post("/api/runs/auto-run/register")
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["model_version"]

    def test_model_predictions_require_review_before_dataset_can_freeze(self):
        upload = self.client.post(
            "/api/datasets",
            data={"name": "自动标注数据", "scene": "board", "labels": "pit"},
            files=[("files", ("board.png", self._png_bytes(), "image/png"))],
        )
        dataset_id = upload.json()["dataset"]["dataset_id"]
        model = self._registered_model()
        unapproved = self.client.post(
            f"/api/datasets/{dataset_id}/auto-annotations",
            json={"model_version_id": model["model_version_id"]},
        )
        self.assertEqual(unapproved.status_code, 409, unapproved.text)
        self.assertIn("approved", unapproved.json()["detail"])
        approved_model = self.client.post(f"/api/models/{model['model_version_id']}/approve")
        self.assertEqual(approved_model.status_code, 200, approved_model.text)

        class FakePredictor:
            class_names = ["pit"]

            def __init__(self, *args, **kwargs):
                pass

            def predict(self, image_bytes, filename):
                return {
                    "width": 100,
                    "height": 50,
                    "detections": [
                        {"class_id": 0, "confidence": 0.95, "x1": 10, "y1": 5, "x2": 50, "y2": 25}
                    ],
                }

        with patch("defectdock.api.app.DetectionInferenceService", FakePredictor):
            generated = self.client.post(
                f"/api/datasets/{dataset_id}/auto-annotations",
                json={"model_version_id": model["model_version_id"], "confidence": 0.5},
            )
        self.assertEqual(generated.status_code, 201, generated.text)
        version = generated.json()["annotation_version"]
        self.assertEqual(version["review_status"], "candidate")
        self.assertEqual(generated.json()["detection_count"], 1)

        detail = self.client.get(f"/api/datasets/{dataset_id}").json()
        self.assertEqual(detail["images"][0]["boxes"][0]["cx"], 0.3)
        blocked = self.client.post(f"/api/datasets/{dataset_id}/freeze")
        self.assertEqual(blocked.status_code, 409)
        self.assertIn("approved", blocked.json()["detail"])

        approved = self.client.post(
            f"/api/datasets/{dataset_id}/annotation-versions/{version['annotation_version_id']}/approve"
        )
        self.assertEqual(approved.status_code, 200, approved.text)
        self.assertEqual(approved.json()["review_status"], "approved")
        self.assertEqual(approved.json()["reviewed_by"], "local")
        frozen = self.client.post(f"/api/datasets/{dataset_id}/freeze")
        self.assertEqual(frozen.status_code, 200, frozen.text)


if __name__ == "__main__":
    unittest.main()
