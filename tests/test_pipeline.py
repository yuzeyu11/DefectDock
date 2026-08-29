import tempfile
import unittest
from pathlib import Path

from defectdock.pipeline import prepare_project, recommend_model


class ModelSelectionTests(unittest.TestCase):
    def test_small_dataset_uses_bootstrap_preset(self):
        recommendation = recommend_model(50, num_classes=10)
        self.assertEqual(recommendation.model, "fasterrcnn-resnet50-fpn-v2")
        self.assertEqual(recommendation.preset, "bootstrap")

    def test_many_classes_increase_epochs(self):
        base = recommend_model(1000, num_classes=10)
        many = recommend_model(1000, num_classes=30)
        self.assertGreater(many.epochs, base.epochs)

    def test_invalid_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            recommend_model(-1)


class PrepareProjectTests(unittest.TestCase):
    def test_prepare_project_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            raw.mkdir()
            (raw / "a.jpg").write_bytes(b"x")
            (raw / "b.png").write_bytes(b"x")
            (raw / "c.txt").write_bytes(b"ignored")
            project = root / "proj"
            result = prepare_project(raw, project)
            self.assertEqual(result["unlabeled_images"], 2)
            self.assertTrue((project / "unlabeled" / "a.jpg").is_file())
            self.assertTrue((project / "seed" / "images").is_dir())


if __name__ == "__main__":
    unittest.main()
