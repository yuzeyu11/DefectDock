import tempfile
import unittest
from pathlib import Path

import yaml

from defectdock.data.cv import check_dataset


def _write_dataset(root: Path, data_yaml: dict, images: dict, labels: dict):
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump(data_yaml), encoding="utf-8")
    for split, names in images.items():
        image_dir = root / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            (image_dir / name).write_bytes(b"fake-image-bytes")
    for split, entries in labels.items():
        label_dir = root / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        for name, content in entries.items():
            (label_dir / name).write_text(content, encoding="utf-8")


def _base_yaml() -> dict:
    return {
        "train": "images/train",
        "nc": 2,
        "names": ["cat", "dog"],
    }


class DataCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_dataset_passes(self):
        config = _base_yaml()
        config["val"] = "images/val"
        _write_dataset(
            self.root,
            config,
            {"train": ["a.jpg"], "val": ["b.jpg"]},
            {
                "train": {"a.txt": "0 0.5 0.5 0.1 0.1\n"},
                "val": {"b.txt": "1 0.5 0.5 0.2 0.2\n"},
            },
        )
        report = check_dataset(self.root / "data.yaml")
        self.assertTrue(report.ok)
        self.assertEqual(report.errors, [])
        self.assertEqual(report.classes, ["cat", "dog"])
        self.assertEqual(report.splits["train"], 1)
        self.assertEqual(report.splits["val"], 1)

    def test_missing_data_yaml(self):
        report = check_dataset(self.root / "data.yaml")
        self.assertFalse(report.ok)
        self.assertTrue(any(issue.kind == "missing_data_yaml" for issue in report.errors))

    def test_missing_names(self):
        (self.root / "data.yaml").write_text(
            yaml.safe_dump({"train": "images/train"}), encoding="utf-8"
        )
        report = check_dataset(self.root / "data.yaml")
        self.assertFalse(report.ok)
        self.assertTrue(any(issue.kind == "invalid_names" for issue in report.errors))

    def test_nc_names_mismatch(self):
        config = _base_yaml()
        config["nc"] = 3
        _write_dataset(self.root, config, {"train": []}, {})
        report = check_dataset(self.root / "data.yaml")
        self.assertFalse(report.ok)
        self.assertTrue(any(issue.kind == "invalid_names" for issue in report.errors))

    def test_class_out_of_range(self):
        _write_dataset(
            self.root,
            _base_yaml(),
            {"train": ["a.jpg"]},
            {"train": {"a.txt": "2 0.5 0.5 0.1 0.1\n"}},
        )
        report = check_dataset(self.root / "data.yaml")
        self.assertFalse(report.ok)
        self.assertTrue(any(issue.kind == "class_out_of_range" for issue in report.errors))

    def test_box_out_of_range(self):
        _write_dataset(
            self.root,
            _base_yaml(),
            {"train": ["a.jpg"]},
            {"train": {"a.txt": "0 0.5 0.5 1.5 0.5\n"}},
        )
        report = check_dataset(self.root / "data.yaml")
        self.assertFalse(report.ok)
        self.assertTrue(any(issue.kind == "box_out_of_range" for issue in report.errors))

    def test_zero_size_box_is_an_error(self):
        _write_dataset(
            self.root,
            _base_yaml(),
            {"train": ["a.jpg"]},
            {"train": {"a.txt": "0 0.5 0.5 0.0 0.1\n"}},
        )
        report = check_dataset(self.root / "data.yaml")
        self.assertFalse(report.ok)
        self.assertTrue(any(issue.kind == "box_non_positive" for issue in report.errors))

    def test_malformed_label_line(self):
        _write_dataset(
            self.root,
            _base_yaml(),
            {"train": ["a.jpg"]},
            {"train": {"a.txt": "0 0.5 0.5\n"}},
        )
        report = check_dataset(self.root / "data.yaml")
        self.assertFalse(report.ok)
        self.assertTrue(any(issue.kind == "malformed_label_line" for issue in report.errors))

    def test_empty_label_is_warning(self):
        _write_dataset(
            self.root,
            _base_yaml(),
            {"train": ["a.jpg"]},
            {"train": {"a.txt": "\n"}},
        )
        report = check_dataset(self.root / "data.yaml")
        self.assertTrue(report.ok)
        self.assertTrue(any(issue.kind == "empty_label" for issue in report.warnings))

    def test_missing_label_is_warning(self):
        _write_dataset(self.root, _base_yaml(), {"train": ["a.jpg"]}, {})
        report = check_dataset(self.root / "data.yaml")
        self.assertTrue(report.ok)
        self.assertTrue(any(issue.kind == "missing_label" for issue in report.warnings))

    def test_orphan_label_is_warning(self):
        _write_dataset(
            self.root,
            _base_yaml(),
            {"train": ["a.jpg"]},
            {"train": {"a.txt": "0 0.5 0.5 0.1 0.1\n", "orphan.txt": "0 0.5 0.5 0.1 0.1\n"}},
        )
        report = check_dataset(self.root / "data.yaml")
        self.assertTrue(report.ok)
        self.assertTrue(any(issue.kind == "orphan_label" for issue in report.warnings))

    def test_same_source_in_train_and_val_is_an_error(self):
        config = _base_yaml()
        config["val"] = "images/train"
        _write_dataset(
            self.root,
            config,
            {"train": ["a.jpg"]},
            {"train": {"a.txt": "0 0.5 0.5 0.1 0.1\n"}},
        )
        report = check_dataset(self.root / "data.yaml")
        self.assertFalse(report.ok)
        self.assertTrue(any(issue.kind == "split_overlap" for issue in report.errors))


if __name__ == "__main__":
    unittest.main()
