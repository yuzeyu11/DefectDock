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


if __name__ == "__main__":
    unittest.main()
