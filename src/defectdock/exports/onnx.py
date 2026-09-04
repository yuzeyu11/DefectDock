"""Export a DefectDock TorchVision checkpoint as a verified ONNX package."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ExportRunner = Callable[[Path, Path, int, int, int], dict[str, Any]]


def export_onnx_package(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    opset: int = 18,
    warmup_runs: int = 2,
    benchmark_runs: int = 10,
    runner: ExportRunner | None = None,
) -> dict[str, Any]:
    """Create an immutable ONNX package and machine-readable verification manifest."""
    checkpoint = Path(checkpoint_path).resolve()
    destination = Path(output_dir).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not 17 <= opset <= 20:
        raise ValueError("ONNX opset must be between 17 and 20")
    if not 0 <= warmup_runs <= 20:
        raise ValueError("warmup_runs must be between 0 and 20")
    if not 1 <= benchmark_runs <= 100:
        raise ValueError("benchmark_runs must be between 1 and 100")
    if destination.exists():
        raise FileExistsError(f"ONNX export already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    model_path = staging / "model.onnx"
    try:
        result = (runner or _run_torchvision_export)(
            checkpoint,
            model_path,
            opset,
            warmup_runs,
            benchmark_runs,
        )
        if not model_path.is_file() or model_path.stat().st_size == 0:
            raise RuntimeError("ONNX exporter did not create a model artifact")
        if not result.get("validation", {}).get("passed"):
            raise ValueError("ONNX numerical consistency validation failed")

        classes = [str(item) for item in result["classes"]]
        (staging / "classes.json").write_text(
            json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "format": "onnx",
            "opset": opset,
            "source": {
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _file_sha256(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
            },
            "artifact": {
                "path": "model.onnx",
                "sha256": _file_sha256(model_path),
                "size": model_path.stat().st_size,
            },
            "classes": classes,
            "input": {
                "name": "images",
                "shape": [1, 3, int(result["input_size"]), int(result["input_size"])],
                "dtype": "float32",
                "range": [0.0, 1.0],
            },
            "outputs": ["boxes", "labels", "scores"],
            "preprocessing": "RGB float32 tensor scaled to [0,1]",
            "postprocessing": "model-native boxes in pixel xyxy coordinates",
            "validation": result["validation"],
            "benchmark": result["benchmark"],
            "runtime": result["runtime"],
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    manifest["package_dir"] = str(destination)
    manifest["manifest_path"] = str(destination / "manifest.json")
    return manifest


def verify_onnx_package(output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("ONNX package manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = destination / manifest["artifact"]["path"]
    if not artifact.is_file():
        raise ValueError("ONNX package artifact is missing")
    if artifact.stat().st_size != manifest["artifact"]["size"] or _file_sha256(artifact) != manifest[
        "artifact"
    ]["sha256"]:
        raise ValueError("ONNX package artifact failed integrity verification")
    return manifest


def _run_torchvision_export(
    checkpoint: Path,
    model_path: Path,
    opset: int,
    warmup_runs: int,
    benchmark_runs: int,
) -> dict[str, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export dependencies are missing; install defectdock[train,export]"
        ) from exc

    from defectdock.engines.torchvision import load_checkpoint

    model, classes, checkpoint_data = load_checkpoint(checkpoint, device="cpu")
    model.eval()
    input_size = int(checkpoint_data.get("input_size", 640))

    class DetectionExportWrapper(torch.nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector

        def forward(self, images):
            output = self.detector([images[0]])[0]
            return output["boxes"], output["labels"], output["scores"]

    wrapper = DetectionExportWrapper(model).eval()
    sample = torch.linspace(0.0, 1.0, steps=3 * input_size * input_size, dtype=torch.float32).reshape(
        1, 3, input_size, input_size
    )
    with torch.inference_mode():
        native = wrapper(sample)
        torch.onnx.export(
            wrapper,
            (sample,),
            str(model_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["boxes", "labels", "scores"],
            dynamic_axes={
                "boxes": {0: "detections"},
                "labels": {0: "detections"},
                "scores": {0: "detections"},
            },
            dynamo=False,
        )

    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    sample_numpy = sample.numpy()
    onnx_outputs = session.run(["boxes", "labels", "scores"], {"images": sample_numpy})
    native_outputs = [item.detach().cpu().numpy() for item in native]
    validation = _compare_outputs(native_outputs, onnx_outputs, np)

    for _ in range(warmup_runs):
        session.run(None, {"images": sample_numpy})
    elapsed_ms = []
    for _ in range(benchmark_runs):
        started = time.perf_counter()
        session.run(None, {"images": sample_numpy})
        elapsed_ms.append((time.perf_counter() - started) * 1000)
    elapsed_ms.sort()
    benchmark = {
        "provider": "CPUExecutionProvider",
        "warmup_runs": warmup_runs,
        "measured_runs": benchmark_runs,
        "median_ms": round(elapsed_ms[len(elapsed_ms) // 2], 3),
        "min_ms": round(elapsed_ms[0], 3),
        "max_ms": round(elapsed_ms[-1], 3),
        "note": "Synthetic fixed-shape smoke benchmark; repeat on target hardware and representative images.",
    }
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "torchvision": _package_version("torchvision"),
    }
    return {
        "classes": classes,
        "input_size": input_size,
        "validation": validation,
        "benchmark": benchmark,
        "runtime": runtime,
    }


def _compare_outputs(native: list[Any], exported: list[Any], np_module) -> dict[str, Any]:
    if len(native) != 3 or len(exported) != 3:
        return {"passed": False, "reason": "unexpected output count"}
    if any(left.shape != right.shape for left, right in zip(native, exported)):
        return {
            "passed": False,
            "reason": "output shape mismatch",
            "native_shapes": [list(item.shape) for item in native],
            "onnx_shapes": [list(item.shape) for item in exported],
        }
    boxes_error = _max_abs_error(native[0], exported[0], np_module)
    scores_error = _max_abs_error(native[2], exported[2], np_module)
    labels_match = bool(np_module.array_equal(native[1], exported[1]))
    return {
        "passed": labels_match and boxes_error <= 1e-3 and scores_error <= 1e-4,
        "labels_exact": labels_match,
        "boxes_max_abs_error": boxes_error,
        "scores_max_abs_error": scores_error,
        "boxes_tolerance": 1e-3,
        "scores_tolerance": 1e-4,
    }


def _max_abs_error(left, right, np_module) -> float:
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(np_module.max(np_module.abs(left - right)))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
