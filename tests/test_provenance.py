import json
import tempfile
import unittest
from pathlib import Path

from defectdock.config import RunConfig
from defectdock.provenance import file_sha256, write_run_manifest


class ProvenanceTests(unittest.TestCase):
    def test_manifest_captures_config_dataset_and_environment_without_secrets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "datasets" / "sample" / "data.yaml"
            dataset.parent.mkdir(parents=True)
            dataset.write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")
            config = RunConfig.model_validate(
                {
                    "project": "manifest-test",
                    "dataset": {"path": "datasets/sample/data.yaml", "version": "v1"},
                    "train": {"pretrained": False},
                }
            )
            run_dir = root / "outputs" / "run"
            manifest = write_run_manifest(run_dir, config, root)

            self.assertEqual(manifest["config_hash"], config.config_hash)
            self.assertEqual(manifest["dataset"]["config_sha256"], file_sha256(dataset))
            self.assertIn("python", manifest["environment"])
            self.assertIn("torchvision", manifest["environment"]["dependencies"])
            serialized = json.dumps(manifest).casefold()
            self.assertNotIn("access_token", serialized)
            self.assertNotIn("password", serialized)
            self.assertTrue((run_dir / "run.manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
