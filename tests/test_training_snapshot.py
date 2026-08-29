import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from defectdock.data.cv import build_training_snapshot
from defectdock.domain import (
    AnnotationVersionRecord,
    DatasetImageRecord,
    DatasetRecord,
    DatasetStatus,
)


class TrainingSnapshotTests(unittest.TestCase):
    def _fixture(self, root: Path):
        dataset_root = root / "dataset"
        images_root = dataset_root / "images"
        version_root = dataset_root / "annotations" / "versions" / "direct-v1"
        labels_root = version_root / "labels"
        images_root.mkdir(parents=True)
        labels_root.mkdir(parents=True)
        records = []
        manifest_images = []
        for index, payload in enumerate((b"image-one", b"image-two"), 1):
            stored_name = f"image-{index}.png"
            image_path = images_root / stored_name
            image_path.write_bytes(payload)
            sha256 = hashlib.sha256(payload).hexdigest()
            record = DatasetImageRecord(
                image_id=f"img-{index}",
                dataset_id="ds-snapshot",
                original_name=stored_name,
                stored_name=stored_name,
                sha256=sha256,
                size_bytes=len(payload),
                width=64,
                height=48,
                created_at="now",
            )
            records.append(record)
            label_name = f"image-{index}.txt"
            (labels_root / label_name).write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            label_sha256 = hashlib.sha256((labels_root / label_name).read_bytes()).hexdigest()
            manifest_images.append(
                {
                    "image_id": record.image_id,
                    "stored_name": stored_name,
                    "sha256": sha256,
                    "label": f"labels/{label_name}",
                    "label_sha256": label_sha256,
                }
            )
        (version_root / "manifest.json").write_text(
            json.dumps({"classes": ["pit"], "images": manifest_images}), encoding="utf-8"
        )
        dataset = DatasetRecord(
            dataset_id="ds-snapshot",
            name="snapshot-test",
            scene="board",
            labels=["pit"],
            status=DatasetStatus.FROZEN,
            root_dir=str(dataset_root),
            image_count=2,
            total_bytes=18,
            cvat_task_id=None,
            created_at="now",
            updated_at="now",
        )
        annotation_version = AnnotationVersionRecord(
            annotation_version_id="direct-v1",
            dataset_id=dataset.dataset_id,
            source="direct_upload",
            format="normalized-detection-text-v1",
            root_dir=str(version_root),
            manifest_path=str(version_root / "manifest.json"),
            manifest_sha256=hashlib.sha256(
                (version_root / "manifest.json").read_bytes()
            ).hexdigest(),
            labeled_count=2,
            unlabeled_count=0,
            created_at="now",
            is_current=True,
        )
        return dataset, records, annotation_version

    def test_snapshot_is_deterministic_hashed_and_independent_from_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset, records, annotation_version = self._fixture(Path(temp_dir))
            first = build_training_snapshot(
                dataset, records, annotation_version, seed=7, val_ratio=0.5
            )
            second = build_training_snapshot(
                dataset, records, annotation_version, seed=7, val_ratio=0.5
            )
            self.assertEqual(first["snapshot_id"], second["snapshot_id"])
            self.assertEqual(first["snapshot_sha256"], second["snapshot_sha256"])
            self.assertEqual(first["train_count"], 1)
            self.assertEqual(first["val_count"], 1)
            self.assertEqual(first["materialization"], "copy")
            self.assertTrue(Path(first["data_yaml"]).is_file())
            self.assertTrue(all(item["label_sha256"] for item in first["images"]))

            snapshot_image = Path(first["data_yaml"]).parent / first["images"][0]["image"]
            original_snapshot_payload = snapshot_image.read_bytes()
            source_image = Path(dataset.root_dir) / "images" / records[0].stored_name
            source_image.write_bytes(b"changed-source")
            self.assertEqual(snapshot_image.read_bytes(), original_snapshot_payload)

    def test_snapshot_rejects_source_changed_after_ingestion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset, records, annotation_version = self._fixture(Path(temp_dir))
            source = Path(dataset.root_dir) / "images" / records[0].stored_name
            source.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "changed after ingestion"):
                build_training_snapshot(dataset, records, annotation_version)

    def test_snapshot_rejects_annotation_manifest_changed_after_registration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset, records, annotation_version = self._fixture(Path(temp_dir))
            Path(annotation_version.manifest_path).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                build_training_snapshot(dataset, records, annotation_version)

    def test_snapshot_rejects_annotation_label_changed_after_registration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset, records, annotation_version = self._fixture(Path(temp_dir))
            manifest = json.loads(Path(annotation_version.manifest_path).read_text(encoding="utf-8"))
            label_path = Path(annotation_version.root_dir) / manifest["images"][0]["label"]
            label_path.write_text("0 0.4 0.4 0.1 0.1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "label changed"):
                build_training_snapshot(dataset, records, annotation_version)


if __name__ == "__main__":
    unittest.main()
