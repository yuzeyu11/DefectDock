import io
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from defectdock.api import create_app
from defectdock.config import RunConfig, load_run_config
from defectdock.engines import EngineResult


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = create_app(
            Path(self.temp_dir.name) / "defectdock.db",
            datasets_root=Path(self.temp_dir.name) / "datasets",
            workspace=Path(self.temp_dir.name),
            training_enabled=True,
        )
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.app.state.jobs.shutdown(wait=True)
        self.temp_dir.cleanup()

    def test_health_and_service_root(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(health.json()["service"], "defectdock")
        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        self.assertEqual(root.json()["api_docs"], "/docs")

    def test_example_configs_and_validation(self):
        response = self.client.get("/api/configs/examples")
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(len(response.json()), 1)
        config = load_run_config("configs/examples/gc10-torchvision.yaml")
        validation = self.client.post(
            "/api/configs/validate", json=config.model_dump(mode="json")
        )
        self.assertEqual(validation.status_code, 200)
        self.assertTrue(validation.json()["valid"])
        self.assertTrue(
            validation.json()["plan"]["dataset"].replace("\\", "/").endswith(
                "datasets/gc10-v1/data.yaml"
            )
        )

    def test_runs_are_read_from_store(self):
        config = load_run_config("configs/examples/gc10-torchvision.yaml")
        run_dir = Path(self.temp_dir.name) / "run"
        run_dir.mkdir()
        self.app.state.store.create_run(config, run_dir, run_id="api-run")
        (run_dir / "events.jsonl").write_text(
            '{"event":"training_started"}\n{"epoch":1,"event":"epoch_end"}\n',
            encoding="utf-8",
        )
        listing = self.client.get("/api/runs")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()[0]["run_id"], "api-run")
        detail = self.client.get("/api/runs/api-run")
        self.assertEqual(detail.status_code, 200)
        events = self.client.get("/api/runs/api-run/events")
        self.assertEqual(events.status_code, 200)
        self.assertEqual(events.json()[1]["epoch"], 1)
        missing = self.client.get("/api/runs/missing")
        self.assertEqual(missing.status_code, 404)

    def test_truncated_events_tail_does_not_break_endpoint(self):
        # M7 回归：训练中断留下的半行 JSON 不应让 /events 返回 500，
        # 坏行被跳过，前面合法行仍然返回。
        config = load_run_config("configs/examples/gc10-torchvision.yaml")
        run_dir = Path(self.temp_dir.name) / "run2"
        run_dir.mkdir()
        self.app.state.store.create_run(config, run_dir, run_id="api-run-2")
        (run_dir / "events.jsonl").write_text(
            '{"event":"training_started"}\n{"event":"epoch_end","epoch":3\n',
            encoding="utf-8",
        )
        events = self.client.get("/api/runs/api-run-2/events")
        self.assertEqual(events.status_code, 200)
        self.assertEqual(len(events.json()), 1)
        self.assertEqual(events.json()[0]["event"], "training_started")

    def test_real_images_can_be_uploaded_deduplicated_and_frozen(self):
        image_bytes = self._png_bytes()
        response = self.client.post(
            "/api/datasets",
            data={"name": "真实板材", "scene": "board", "labels": "pit,scratch"},
            files=[
                ("files", ("board-a.png", image_bytes, "image/png")),
                ("files", ("board-copy.png", image_bytes, "image/png")),
            ],
        )
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()
        self.assertEqual(payload["upload"]["accepted_count"], 1)
        self.assertEqual(payload["upload"]["duplicate_count"], 1)
        dataset_id = payload["dataset"]["dataset_id"]
        detail = self.client.get(f"/api/datasets/{dataset_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["image_count"], 1)
        preview = self.client.get(detail.json()["images"][0]["preview_url"])
        self.assertEqual(preview.status_code, 200)
        frozen = self.client.post(f"/api/datasets/{dataset_id}/freeze")
        self.assertEqual(frozen.status_code, 200)
        self.assertEqual(frozen.json()["status"], "frozen")

    def test_direct_annotations_snapshot_training_and_activation_flow(self):
        upload = self.client.post(
            "/api/datasets",
            data={"name": "主链路数据", "scene": "board", "labels": "pit"},
            files=[
                ("files", ("board-a.png", self._png_bytes((120, 90, 60)), "image/png")),
                ("files", ("board-b.png", self._png_bytes((20, 140, 80)), "image/png")),
            ],
        )
        self.assertEqual(upload.status_code, 201, upload.text)
        dataset_id = upload.json()["dataset"]["dataset_id"]

        annotations = self.client.post(
            f"/api/datasets/{dataset_id}/annotations",
            files=[
                ("files", ("board-a.txt", b"0 0.5 0.5 0.2 0.2\n", "text/plain")),
                ("files", ("board-b.txt", b"0 0.5 0.5 0.3 0.3\n", "text/plain")),
            ],
        )
        self.assertEqual(annotations.status_code, 201, annotations.text)
        self.assertEqual(annotations.json()["annotation_version"]["labeled_count"], 2)
        version_id = annotations.json()["annotation_version"]["annotation_version_id"]
        versions = self.client.get(f"/api/datasets/{dataset_id}/annotation-versions")
        self.assertEqual(versions.status_code, 200, versions.text)
        self.assertEqual(versions.json()[0]["annotation_version_id"], version_id)
        self.assertTrue(versions.json()[0]["is_current"])
        detail = self.client.get(f"/api/datasets/{dataset_id}")
        self.assertEqual(
            detail.json()["current_annotation_version"]["annotation_version_id"],
            version_id,
        )

        frozen = self.client.post(f"/api/datasets/{dataset_id}/freeze")
        self.assertEqual(frozen.status_code, 200, frozen.text)
        snapshot_response = self.client.post(
            f"/api/datasets/{dataset_id}/training-snapshot",
            json={"seed": 11, "val_ratio": 0.5},
        )
        self.assertEqual(snapshot_response.status_code, 201, snapshot_response.text)
        snapshot = snapshot_response.json()["snapshot"]
        self.assertEqual(snapshot["image_count"], 2)
        self.assertEqual(snapshot["annotation_version"], version_id)
        self.assertEqual(
            snapshot["annotation_manifest_sha256"],
            annotations.json()["annotation_version"]["manifest_sha256"],
        )
        self.assertTrue(snapshot_response.json()["quality"]["ok"])

        def fake_runner(config, run_dir, on_event, should_cancel, *, project_root):
            weights = Path(run_dir) / "trainer_output" / "weights"
            weights.mkdir(parents=True)
            best = weights / "best.ckpt"
            last = weights / "last.ckpt"
            best.write_bytes(b"fake-checkpoint")
            last.write_bytes(b"fake-checkpoint")
            return EngineResult(
                trainer_output=str(weights.parent),
                best_model=str(best),
                last_model=str(last),
                metrics={"standard": {"recall": 1.0}},
            )

        self.app.state.jobs.runner = fake_runner
        config = RunConfig.model_validate(
            {
                "project": "api-flow",
                "dataset": {"path": snapshot["data_yaml"], "version": snapshot["snapshot_id"]},
                "train": {"epochs": 1, "pretrained": False},
            }
        )
        submitted = self.client.post("/api/runs", json=config.model_dump(mode="json"))
        self.assertEqual(submitted.status_code, 202, submitted.text)
        run_id = submitted.json()["run_id"]
        deadline = time.monotonic() + 5
        run = None
        while time.monotonic() < deadline:
            run = self.client.get(f"/api/runs/{run_id}").json()
            if run["status"] == "succeeded":
                break
            time.sleep(0.01)
        self.assertEqual(run["status"], "succeeded")
        artifacts = self.client.get(f"/api/runs/{run_id}/artifacts")
        self.assertTrue(artifacts.json()["best_model"].endswith("best.ckpt"))
        activated = self.client.post(f"/api/runs/{run_id}/activate")
        self.assertEqual(activated.status_code, 200, activated.text)
        self.assertTrue(activated.json()["inference"]["available"])

    def test_invalid_image_rolls_back_dataset(self):
        response = self.client.post(
            "/api/datasets",
            data={"name": "损坏数据", "labels": "pit"},
            files=[("files", ("broken.jpg", b"not-an-image", "image/jpeg"))],
        )
        self.assertEqual(response.status_code, 400)
        listing = self.client.get("/api/datasets")
        self.assertEqual(listing.json(), [])

    def test_real_image_can_be_sent_to_inference_service(self):
        class FakeInference:
            available = True

            def status(self):
                return {"configured": True, "available": True, "loaded": False}

            def predict(self, image_bytes, filename):
                self.filename = filename
                self.image_bytes = image_bytes
                return {
                    "filename": filename,
                    "width": 64,
                    "height": 48,
                    "model": "fake.pt",
                    "thresholds": {"confidence": 0.45, "iou": 0.35},
                    "timing_ms": {"inference": 12.0, "total": 14.0},
                    "detections": [],
                }

        fake = FakeInference()
        self.app.state.inference = fake
        response = self.client.post(
            "/api/inference/detect",
            files=[("file", ("board.png", self._png_bytes(), "image/png"))],
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["model"], "fake.pt")
        self.assertEqual(fake.filename, "board.png")
        self.assertGreater(len(fake.image_bytes), 0)

    def test_blocking_inference_does_not_stall_the_event_loop(self):
        # H1 回归：推理 predict 是阻塞调用；若它直接跑在事件循环线程里，
        # health 请求会等到推理结束才能响应。这里验证推理进行中 health 立即可达。
        class SlowInference:
            available = True

            def status(self):
                return {"configured": True, "available": True, "loaded": False}

            def predict(self, image_bytes, filename):
                time.sleep(1.5)
                return {"filename": filename, "detections": []}

        self.app.state.inference = SlowInference()
        detect_thread = threading.Thread(
            target=lambda: self.client.post(
                "/api/inference/detect",
                files=[("file", ("board.png", self._png_bytes(), "image/png"))],
            )
        )
        detect_thread.start()
        try:
            time.sleep(0.2)  # 等 detect 进入阻塞的 predict
            start = time.perf_counter()
            health = self.client.get("/api/health")
            elapsed = time.perf_counter() - start
            self.assertEqual(health.status_code, 200)
            self.assertLess(elapsed, 1.0, "event loop was blocked by inference")
        finally:
            detect_thread.join(timeout=10)
        self.assertFalse(detect_thread.is_alive())

    def test_dataset_can_create_cvat_task_through_adapter(self):
        class FakeCvat:
            def status(self):
                return {"url": "http://cvat.test", "reachable": True, "configured": True}

            def task_url(self, task_id):
                return f"http://cvat.test/tasks/{task_id}"

            def create_task(self, dataset, image_paths):
                self.dataset = dataset
                self.image_paths = image_paths
                return {"task_id": 42, "task_url": self.task_url(42)}

        fake = FakeCvat()
        self.app.state.cvat = fake
        upload = self.client.post(
            "/api/datasets",
            data={"name": "待标注板材", "labels": "pit,scratch"},
            files=[("files", ("board.png", self._png_bytes(), "image/png"))],
        )
        dataset_id = upload.json()["dataset"]["dataset_id"]
        response = self.client.post(f"/api/datasets/{dataset_id}/cvat-task")
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["task_id"], 42)
        self.assertEqual(response.json()["dataset"]["status"], "annotating")
        self.assertEqual(response.json()["dataset"]["annotation_url"], "http://cvat.test/tasks/42")
        self.assertEqual(len(fake.image_paths), 1)

    def test_dataset_can_sync_cvat_annotations(self):
        class FakeCvat:
            def status(self):
                return {"url": "http://cvat.test", "reachable": True, "configured": True}

            def task_url(self, task_id):
                return f"http://cvat.test/tasks/{task_id}"

            def create_task(self, dataset, image_paths):
                return {"task_id": 42, "task_url": self.task_url(42)}

            def task_status(self, task_id):
                return {
                    "task_id": task_id,
                    "task_url": self.task_url(task_id),
                    "status": "completed",
                    "size": 1,
                    "completed": True,
                }

            def export_dataset(self, task_id, destination):
                stored_name = self.stored_name
                destination = Path(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(destination, "w") as archive:
                    archive.writestr(
                        f"obj_train_data/{Path(stored_name).stem}.txt",
                        "0 0.5 0.5 0.2 0.2\n",
                    )
                    archive.writestr("obj.names", "pit\n")
                return {"task_id": task_id, "format": "YOLO 1.1", "archive_path": str(destination)}

        fake = FakeCvat()
        self.app.state.cvat = fake
        upload = self.client.post(
            "/api/datasets",
            data={"name": "待回传板材", "labels": "pit"},
            files=[("files", ("board.png", self._png_bytes(), "image/png"))],
        )
        dataset_id = upload.json()["dataset"]["dataset_id"]
        fake.stored_name = upload.json()["upload"]["accepted"][0]["stored_name"]
        self.client.post(f"/api/datasets/{dataset_id}/cvat-task")
        response = self.client.post(f"/api/datasets/{dataset_id}/cvat-sync")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["dataset"]["status"], "frozen")
        self.assertEqual(response.json()["annotation_version"]["labeled_count"], 1)
        status = self.client.get(f"/api/datasets/{dataset_id}/cvat-status")
        self.assertEqual(status.status_code, 200)
        self.assertTrue(status.json()["completed"])

    def test_incomplete_cvat_task_cannot_be_frozen(self):
        class FakeCvat:
            def status(self):
                return {"url": "http://cvat.test", "reachable": True, "configured": True}

            def task_url(self, task_id):
                return f"http://cvat.test/tasks/{task_id}"

            def create_task(self, dataset, image_paths):
                return {"task_id": 42, "task_url": self.task_url(42)}

            def task_status(self, task_id):
                return {
                    "task_id": task_id,
                    "task_url": self.task_url(task_id),
                    "status": "annotation",
                    "size": 1,
                    "completed": False,
                }

            def export_dataset(self, task_id, destination):
                raise AssertionError("incomplete task must not be exported")

        self.app.state.cvat = FakeCvat()
        upload = self.client.post(
            "/api/datasets",
            data={"name": "incomplete-cvat", "labels": "pit"},
            files=[("files", ("board.png", self._png_bytes(), "image/png"))],
        )
        dataset_id = upload.json()["dataset"]["dataset_id"]
        self.client.post(f"/api/datasets/{dataset_id}/cvat-task")
        response = self.client.post(f"/api/datasets/{dataset_id}/cvat-sync")
        self.assertEqual(response.status_code, 409)
        self.assertIn("mark all jobs completed", response.json()["detail"])

    @staticmethod
    def _png_bytes(color=(120, 90, 60)) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (64, 48), color).save(buffer, format="PNG")
        return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
