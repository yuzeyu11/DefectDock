import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from defectdock.resources import load_example_configs
from defectdock.settings import RuntimeSettings


class RuntimeSettingsTests(unittest.TestCase):
    def test_workspace_precedence_and_relative_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            environment = {"DEFECTDOCK_WORKSPACE": str(root / "from-env")}
            from_env = RuntimeSettings.from_sources(environ=environment, cwd=root)
            self.assertEqual(from_env.workspace, (root / "from-env").resolve())

            explicit = RuntimeSettings.from_sources(
                "explicit",
                db_path="state/custom.db",
                datasets_root="incoming",
                environ=environment,
                cwd=root,
            )
            self.assertEqual(explicit.workspace, (root / "explicit").resolve())
            self.assertEqual(explicit.db_path, (root / "explicit/state/custom.db").resolve())
            self.assertEqual(explicit.datasets_root, (root / "explicit/incoming").resolve())

    def test_constructing_settings_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "new-workspace"
            settings = RuntimeSettings.from_sources(workspace)
            self.assertEqual(settings.workspace, workspace.resolve())
            self.assertFalse(workspace.exists())

    def test_importing_api_module_does_not_write_runtime_state(self):
        source_root = Path(__file__).resolve().parents[1] / "src"
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = os.environ.copy()
            environment.pop("DEFECTDOCK_WORKSPACE", None)
            environment["PYTHONPATH"] = str(source_root)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [sys.executable, "-c", "import defectdock.api.app"],
                cwd=temp_dir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((Path(temp_dir) / ".defectdock").exists())
            self.assertFalse((Path(temp_dir) / "datasets").exists())

    def test_example_config_is_loaded_from_package_resources(self):
        examples = dict(load_example_configs())
        self.assertIn("gc10-torchvision", examples)
        self.assertEqual(examples["gc10-torchvision"].engine, "torchvision")


if __name__ == "__main__":
    unittest.main()
