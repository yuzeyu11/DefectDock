import contextlib
import io
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from defectdock.integrations import CvatSettings


@contextmanager
def isolated_cvat_env():
    """Drop CVAT_* env vars so local-config precedence is deterministic."""
    keys = ("CVAT_URL", "CVAT_ACCESS_TOKEN", "CVAT_USERNAME", "CVAT_PASSWORD", "DEFECTDOCK_CVAT_CONFIG")
    saved = {key: os.environ.pop(key, None) for key in keys}
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class CvatSettingsTests(unittest.TestCase):
    def _load(self, config_path: Path) -> CvatSettings:
        return CvatSettings.from_env(config_path)

    def test_corrupt_config_prints_warning_and_falls_back(self):
        # M9 回归：损坏的本地 CVAT 配置不能静默吞掉，必须给出可定位的警告。
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "cvat.json"
            config_path.write_text("{not-valid-json", encoding="utf-8")
            stderr = io.StringIO()
            with isolated_cvat_env(), contextlib.redirect_stderr(stderr):
                settings = self._load(config_path)
            self.assertIn("warning", stderr.getvalue())
            self.assertIn(str(config_path), stderr.getvalue())
            # 仍然可构造，并回退到默认 URL。
            self.assertEqual(settings.url, "http://localhost:8080")

    def test_non_object_config_prints_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "cvat.json"
            config_path.write_text("[1, 2, 3]", encoding="utf-8")
            stderr = io.StringIO()
            with isolated_cvat_env(), contextlib.redirect_stderr(stderr):
                self._load(config_path)
            self.assertIn("warning", stderr.getvalue())

    def test_valid_config_is_used(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "cvat.json"
            config_path.write_text(
                '{"url": "http://cvat.test:8080/", "access_token": "tok"}',
                encoding="utf-8",
            )
            with isolated_cvat_env():
                settings = self._load(config_path)
            self.assertEqual(settings.url, "http://cvat.test:8080")
            self.assertEqual(settings.access_token, "tok")
            self.assertTrue(settings.configured)


if __name__ == "__main__":
    unittest.main()
