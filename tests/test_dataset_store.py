import json
import tempfile
import unittest
from pathlib import Path

from defectdock.db import DatasetStore, DuplicateImageError


class DatasetStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = DatasetStore(Path(self.temp_dir.name) / "defectdock.db")
        self.root = Path(self.temp_dir.name) / "dataset"
        self.dataset = self.store.create_dataset(
            "Board defects", "board", ["pit", "scratch"], self.root, dataset_id="ds-test"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_duplicate_and_freeze(self):
        self.store.add_image(
            "ds-test",
            original_name="one.png",
            stored_name="abc.png",
            sha256="abc",
            size_bytes=100,
            width=64,
            height=48,
        )
        with self.assertRaises(DuplicateImageError):
            self.store.add_image(
                "ds-test",
                original_name="copy.png",
                stored_name="copy.png",
                sha256="abc",
                size_bytes=100,
                width=64,
                height=48,
            )
        with self.assertRaisesRegex(ValueError, "annotation version"):
            self.store.freeze_dataset("ds-test")
        self._annotation_version("approved-v1", "approved")
        dataset = self.store.freeze_dataset("ds-test")
        self.assertEqual(dataset.status.value, "frozen")
        with self.assertRaises(ValueError):
            self.store.add_image(
                "ds-test",
                original_name="two.png",
                stored_name="def.png",
                sha256="def",
                size_bytes=100,
                width=64,
                height=48,
            )

    def test_annotation_versions_are_explicit_and_frozen_head_is_immutable(self):
        self.store.add_image(
            "ds-test",
            original_name="one.png",
            stored_name="abc.png",
            sha256="abc",
            size_bytes=100,
            width=64,
            height=48,
        )
        first = self._annotation_version("z-older", "first")
        second = self._annotation_version("a-newer", "second")
        self.assertFalse(self.store.get_annotation_version("ds-test", "z-older").is_current)
        self.assertTrue(second.is_current)
        selected = self.store.set_current_annotation_version("ds-test", first.annotation_version_id)
        self.assertTrue(selected.is_current)
        self.assertEqual(
            self.store.get_current_annotation_version("ds-test").annotation_version_id,
            "z-older",
        )
        self.store.freeze_dataset("ds-test")
        with self.assertRaisesRegex(ValueError, "Frozen datasets"):
            self.store.set_current_annotation_version("ds-test", "a-newer")

    def test_duplicate_annotation_manifest_is_a_domain_error(self):
        self._annotation_version("version-one", "same")
        with self.assertRaisesRegex(ValueError, "manifest already exists"):
            self._annotation_version("version-two", "same")
        self.assertEqual(
            self.store.get_current_annotation_version("ds-test").annotation_version_id,
            "version-one",
        )

    def _annotation_version(self, version_id: str, marker: str):
        version_root = self.root / "annotations" / "versions" / version_id
        version_root.mkdir(parents=True)
        manifest = version_root / "manifest.json"
        manifest.write_text(
            json.dumps({"classes": self.dataset.labels, "marker": marker}),
            encoding="utf-8",
        )
        return self.store.register_annotation_version(
            "ds-test",
            version_id,
            source="direct_upload",
            format="normalized-detection-text-v1",
            root_dir=version_root,
            manifest_path=manifest,
            labeled_count=1,
            unlabeled_count=0,
        )


if __name__ == "__main__":
    unittest.main()
