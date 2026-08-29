import tempfile
import unittest
import zipfile
from pathlib import Path

from defectdock.data import import_cvat_yolo_export
from defectdock.domain import DatasetImageRecord, DatasetRecord, DatasetStatus


class CvatImportTests(unittest.TestCase):
    def _dataset(self, root: Path) -> DatasetRecord:
        return DatasetRecord(
            dataset_id="ds-test",
            name="test",
            scene="board",
            labels=["pit"],
            status=DatasetStatus.ANNOTATING,
            root_dir=str(root),
            image_count=1,
            total_bytes=10,
            cvat_task_id=42,
            created_at="now",
            updated_at="now",
        )

    def _image(self) -> DatasetImageRecord:
        return DatasetImageRecord(
            image_id="img-1",
            dataset_id="ds-test",
            original_name="board.jpg",
            stored_name="abc123.jpg",
            sha256="abc123",
            size_bytes=10,
            width=64,
            height=48,
            created_at="now",
        )

    def test_imports_matching_yolo_labels_as_a_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "export.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("obj_train_data/abc123.txt", "0 0.5 0.5 0.2 0.2\n")
                archive.writestr("obj.names", "pit\n")
            result = import_cvat_yolo_export(
                archive_path,
                self._dataset(root),
                [self._image()],
                root / "version",
            )
            self.assertEqual(result["labeled_count"], 1)
            self.assertEqual(result["unlabeled_count"], 0)
            self.assertTrue((root / "version" / "labels" / "abc123.txt").is_file())
            self.assertTrue(Path(result["manifest_path"]).is_file())

    def test_rejects_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "export.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.txt", "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                import_cvat_yolo_export(
                    archive_path,
                    self._dataset(root),
                    [self._image()],
                    root / "version",
                )

    def test_rejects_changed_class_order_without_leaving_partial_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "export.zip"
            version_dir = root / "version"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("obj.names", "scratch\n")
                archive.writestr("obj_train_data/abc123.txt", "0 0.5 0.5 0.2 0.2\n")
            with self.assertRaisesRegex(ValueError, "class order"):
                import_cvat_yolo_export(
                    archive_path,
                    self._dataset(root),
                    [self._image()],
                    version_dir,
                )
            self.assertFalse(version_dir.exists())

    def test_accepts_completed_background_only_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "export.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("obj.names", "pit\n")
                archive.writestr("train.txt", "data/obj_train_data/abc123.jpg\n")
            result = import_cvat_yolo_export(
                archive_path,
                self._dataset(root),
                [self._image()],
                root / "version",
            )
            self.assertEqual(result["labeled_count"], 0)
            self.assertEqual(result["unlabeled_count"], 1)


if __name__ == "__main__":
    unittest.main()
