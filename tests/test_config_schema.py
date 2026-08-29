import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from defectdock.config import RunConfig, load_run_config


class ConfigSchemaTests(unittest.TestCase):
    def test_example_config_loads_and_hash_is_stable(self):
        config = load_run_config("configs/examples/gc10-torchvision.yaml")
        clone = RunConfig.model_validate(config.model_dump())
        self.assertEqual(config.config_hash, clone.config_hash)
        self.assertEqual(config.dataset.path, "datasets/gc10-v1/data.yaml")
        self.assertEqual(config.train.epochs, 1)

    def test_unknown_fields_are_rejected(self):
        with self.assertRaises(ValidationError):
            RunConfig.model_validate(
                {
                    "project": "bad-config",
                    "dataset": {"path": "coco8.yaml"},
                    "unexpected": True,
                }
            )

    def test_image_size_must_be_stride_aligned(self):
        with self.assertRaises(ValidationError):
            RunConfig.model_validate(
                {
                    "project": "bad-image-size",
                    "dataset": {"path": "coco8.yaml"},
                    "train": {"imgsz": 650},
                }
            )

    def test_relative_dataset_path_resolves_from_project_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "datasets" / "demo" / "data.yaml"
            dataset.parent.mkdir(parents=True)
            dataset.write_text(yaml.safe_dump({"train": "images"}), encoding="utf-8")
            config = RunConfig.model_validate(
                {"project": "path-test", "dataset": {"path": "datasets/demo/data.yaml"}}
            )
            self.assertEqual(config.resolve_dataset_path(root), str(dataset.resolve()))


if __name__ == "__main__":
    unittest.main()
