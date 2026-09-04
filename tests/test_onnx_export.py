import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from defectdock.exports.onnx import _compare_outputs, export_onnx_package, verify_onnx_package


class OnnxExportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.checkpoint = self.root / "best.ckpt"
        self.checkpoint.write_bytes(b"checkpoint")

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _runner(checkpoint, model_path, opset, warmup_runs, benchmark_runs):
        model_path.write_bytes(b"onnx-model")
        return {
            "classes": ["pit", "scratch"],
            "input_size": 640,
            "validation": {
                "passed": True,
                "labels_exact": True,
                "boxes_max_abs_error": 0.0,
                "scores_max_abs_error": 0.0,
            },
            "benchmark": {
                "provider": "FakeExecutionProvider",
                "warmup_runs": warmup_runs,
                "measured_runs": benchmark_runs,
                "median_ms": 1.0,
            },
            "runtime": {"exporter": "fake", "opset": opset},
        }

    def test_export_writes_verified_immutable_package(self):
        output = self.root / "deployment" / "onnx-opset18"
        manifest = export_onnx_package(self.checkpoint, output, runner=self._runner)
        self.assertTrue((output / "model.onnx").is_file())
        self.assertEqual(
            manifest["source"]["checkpoint_sha256"],
            hashlib.sha256(b"checkpoint").hexdigest(),
        )
        self.assertEqual(verify_onnx_package(output)["artifact"]["path"], "model.onnx")
        with self.assertRaises(FileExistsError):
            export_onnx_package(self.checkpoint, output, runner=self._runner)

        (output / "model.onnx").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "integrity"):
            verify_onnx_package(output)

    def test_failed_validation_does_not_publish_partial_package(self):
        output = self.root / "failed"

        def failing_runner(checkpoint, model_path, opset, warmup_runs, benchmark_runs):
            result = self._runner(checkpoint, model_path, opset, warmup_runs, benchmark_runs)
            result["validation"] = {"passed": False}
            return result

        with self.assertRaisesRegex(ValueError, "consistency"):
            export_onnx_package(self.checkpoint, output, runner=failing_runner)
        self.assertFalse(output.exists())

    def test_numerical_comparison_enforces_shapes_labels_and_tolerances(self):
        native = [
            np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
            np.array([1], dtype=np.int64),
            np.array([0.9], dtype=np.float32),
        ]
        close = [native[0] + 0.0001, native[1].copy(), native[2] + 0.00001]
        self.assertTrue(_compare_outputs(native, close, np)["passed"])
        wrong_labels = [native[0], np.array([2], dtype=np.int64), native[2]]
        self.assertFalse(_compare_outputs(native, wrong_labels, np)["passed"])
        wrong_shape = [np.empty((0, 4), dtype=np.float32), native[1], native[2]]
        self.assertEqual(_compare_outputs(native, wrong_shape, np)["reason"], "output shape mismatch")


if __name__ == "__main__":
    unittest.main()
