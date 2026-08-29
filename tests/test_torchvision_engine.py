import tempfile
import unittest
from pathlib import Path

from defectdock.config import RunConfig
from defectdock.engines import build_plan


class TorchvisionEngineTests(unittest.TestCase):
    def test_plan_is_scoped_and_declares_license_boundary(self):
        config = RunConfig.model_validate(
            {
                "project": "engine-test",
                "dataset": {"path": "datasets/example/data.yaml", "version": "v1"},
                "train": {"epochs": 2, "batch": 2, "workers": 0},
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = build_plan(config, Path(temp_dir) / "run")
        self.assertEqual(plan["engine"], "torchvision")
        self.assertEqual(plan["train"]["epochs"], 2)
        self.assertFalse(plan["license_boundary"]["agpl_runtime_included"])


if __name__ == "__main__":
    unittest.main()
