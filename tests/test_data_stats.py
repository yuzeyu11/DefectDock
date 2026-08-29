import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from defectdock.data.cv import compute_stats


def _make_image(path: Path, size: tuple[int, int] = (640, 480)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


class DataStatsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_dataset(self):
        config = {"train": "images/train", "val": "images/val", "nc": 2, "names": ["cat", "dog"]}
        (self.root / "data.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        _make_image(self.root / "images" / "train" / "a.jpg", (640, 480))
        _make_image(self.root / "images" / "train" / "b.jpg", (800, 600))
        _make_image(self.root / "images" / "val" / "c.jpg", (640, 480))
        label_dir = self.root / "labels" / "train"
        label_dir.mkdir(parents=True, exist_ok=True)
        (label_dir / "a.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.2 0.2\n1 0.8 0.8 0.1 0.1\n", encoding="utf-8"
        )
        (label_dir / "b.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
        val_label_dir = self.root / "labels" / "val"
        val_label_dir.mkdir(parents=True, exist_ok=True)
        (val_label_dir / "c.txt").write_text("1 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    def test_compute_stats(self):
        self._write_dataset()
        stats = compute_stats(self.root / "data.yaml")
        self.assertEqual(stats.total_images, 3)
        self.assertEqual(stats.total_boxes, 5)
        self.assertEqual(stats.splits, {"train": 2, "val": 1})
        by_class = {item.class_id: item for item in stats.per_class}
        self.assertEqual(by_class[0].box_count, 3)
        self.assertEqual(by_class[1].box_count, 2)
        self.assertEqual(stats.image_size["max_width"], 800)
        self.assertEqual(stats.image_size["min_height"], 480)

    def test_compute_stats_missing_yaml_raises(self):
        with self.assertRaises(ValueError):
            compute_stats(self.root / "data.yaml")


if __name__ == "__main__":
    unittest.main()
