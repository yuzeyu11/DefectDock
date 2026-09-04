import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from defectdock.data.auto_annotations import generate_auto_annotations
from defectdock.domain import DatasetImageRecord, DatasetRecord, DatasetStatus


class AutoAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        image_dir = self.root / "dataset" / "images"
        image_dir.mkdir(parents=True)
        self.image_path = image_dir / "stored.png"
        Image.new("RGB", (100, 50), color=(20, 30, 40)).save(self.image_path)
        digest = hashlib.sha256(self.image_path.read_bytes()).hexdigest()
        self.dataset = DatasetRecord(
            dataset_id="ds-auto",
            name="Auto",
            scene="board",
            labels=["pit"],
            status=DatasetStatus.DRAFT,
            root_dir=str(self.root / "dataset"),
            image_count=1,
            total_bytes=self.image_path.stat().st_size,
            cvat_task_id=None,
            created_at="now",
            updated_at="now",
        )
        self.image = DatasetImageRecord(
            image_id="img-one",
            dataset_id="ds-auto",
            original_name="board.png",
            stored_name="stored.png",
            sha256=digest,
            size_bytes=self.image_path.stat().st_size,
            width=100,
            height=50,
            created_at="now",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_predictions_are_clipped_filtered_and_converted_to_normalized_xywh(self):
        class Predictor:
            def predict(self, image_bytes, filename):
                return {
                    "width": 100,
                    "height": 50,
                    "detections": [
                        {"class_id": 0, "confidence": 0.9, "x1": 10, "y1": 5, "x2": 50, "y2": 25},
                        {"class_id": 0, "confidence": 0.2, "x1": 1, "y1": 1, "x2": 2, "y2": 2},
                        {"class_id": 0, "confidence": 0.8, "x1": -5, "y1": -5, "x2": 10, "y2": 10},
                    ],
                }

        output = self.root / "dataset" / "annotations" / "versions" / "auto-v1"
        result = generate_auto_annotations(
            self.dataset,
            [self.image],
            Predictor(),
            output,
            model_version_id="model-one",
            model_sha256="abc",
            confidence=0.5,
        )
        self.assertEqual(result["detection_count"], 2)
        lines = (output / "labels" / "stored.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "0 0.30000000 0.30000000 0.40000000 0.40000000")
        self.assertEqual(lines[1], "0 0.05000000 0.10000000 0.10000000 0.20000000")
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["review_status"], "candidate")
        self.assertEqual(manifest["model_version_id"], "model-one")

    def test_changed_source_image_rejects_and_cleans_staging(self):
        self.image_path.write_bytes(b"tampered")
        output = self.root / "dataset" / "annotations" / "versions" / "auto-v1"
        with self.assertRaisesRegex(ValueError, "changed"):
            generate_auto_annotations(
                self.dataset,
                [self.image],
                object(),
                output,
                model_version_id="model-one",
                model_sha256="abc",
                confidence=0.5,
            )
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
