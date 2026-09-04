"""Generate reviewable annotation candidates from a registered detector."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

from defectdock.domain import DatasetImageRecord, DatasetRecord


def generate_auto_annotations(
    dataset: DatasetRecord,
    images: list[DatasetImageRecord],
    predictor: Any,
    version_dir: str | Path,
    *,
    model_version_id: str,
    model_sha256: str,
    confidence: float,
) -> dict:
    """Run batch inference and publish a candidate normalized-text version atomically."""
    if not images:
        raise ValueError("Cannot auto-label an empty dataset")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    version_dir = Path(version_dir).resolve()
    if version_dir.exists():
        raise FileExistsError(f"Annotation version already exists: {version_dir}")
    version_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{version_dir.name}-", dir=version_dir.parent))
    labels_dir = staging / "labels"
    labels_dir.mkdir()
    image_root = Path(dataset.root_dir).resolve() / "images"
    manifest_images = []
    detection_count = 0
    try:
        for record in images:
            image_path = (image_root / record.stored_name).resolve()
            if not image_path.is_relative_to(image_root) or not image_path.is_file():
                raise ValueError(f"Dataset image is missing or outside its root: {record.original_name}")
            if _file_sha256(image_path) != record.sha256:
                raise ValueError(f"Dataset image changed after upload: {record.original_name}")
            result = predictor.predict(image_path.read_bytes(), record.original_name)
            width = int(result.get("width", record.width))
            height = int(result.get("height", record.height))
            if width != record.width or height != record.height:
                raise ValueError(f"Inference dimensions do not match image metadata: {record.original_name}")
            lines = []
            for detection in result.get("detections", []):
                score = float(detection.get("confidence", 0.0))
                if score < confidence:
                    continue
                class_id = int(detection["class_id"])
                if not 0 <= class_id < len(dataset.labels):
                    raise ValueError(f"Prediction class is outside dataset labels: {class_id}")
                x1, y1, x2, y2 = (
                    float(detection[key]) for key in ("x1", "y1", "x2", "y2")
                )
                if not all(math.isfinite(value) for value in (x1, y1, x2, y2, score)):
                    raise ValueError("Prediction contains a non-finite value")
                x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
                y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
                if x2 <= x1 or y2 <= y1:
                    continue
                cx = (x1 + x2) / (2 * width)
                cy = (y1 + y2) / (2 * height)
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                lines.append(f"{class_id} {cx:.8f} {cy:.8f} {box_width:.8f} {box_height:.8f}")
            label_name = f"{Path(record.stored_name).stem}.txt"
            label_path = labels_dir / label_name
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            detection_count += len(lines)
            manifest_images.append(
                {
                    "image_id": record.image_id,
                    "original_name": record.original_name,
                    "stored_name": record.stored_name,
                    "sha256": record.sha256,
                    "label": f"labels/{label_name}",
                    "label_sha256": _file_sha256(label_path),
                    "prediction_count": len(lines),
                }
            )
        manifest = {
            "dataset_id": dataset.dataset_id,
            "source": "model_prediction",
            "format": "normalized-detection-text-v1",
            "review_status": "candidate",
            "classes": dataset.labels,
            "model_version_id": model_version_id,
            "model_sha256": model_sha256,
            "confidence": confidence,
            "image_count": len(images),
            "labeled_count": len(images),
            "unlabeled_count": 0,
            "detection_count": detection_count,
            "images": manifest_images,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging.replace(version_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "version_dir": str(version_dir),
        "manifest_path": str(version_dir / "manifest.json"),
        "labeled_count": len(images),
        "unlabeled_count": 0,
        "detection_count": detection_count,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
