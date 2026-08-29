"""Direct annotation upload for small local projects that do not use CVAT."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import UploadFile

from defectdock.data.cvat import _validate_yolo_label
from defectdock.domain import DatasetImageRecord, DatasetRecord

MAX_LABEL_BYTES = 5 * 1024 * 1024


async def import_uploaded_annotations(
    dataset: DatasetRecord,
    images: list[DatasetImageRecord],
    files: list[UploadFile],
    version_dir: str | Path,
) -> dict:
    """Version uploaded normalized label files and map them to stored images."""
    if not files:
        raise ValueError("At least one annotation text file is required")
    version_dir = Path(version_dir).resolve()
    if version_dir.exists():
        raise FileExistsError(f"Annotation version already exists: {version_dir}")
    version_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{version_dir.name}-", dir=version_dir.parent))
    labels_dir = staging / "labels"
    labels_dir.mkdir()

    aliases: dict[str, list[DatasetImageRecord]] = {}
    for record in images:
        for stem in {Path(record.original_name).stem.casefold(), Path(record.stored_name).stem.casefold()}:
            aliases.setdefault(stem, []).append(record)

    matched: dict[str, dict[str, str]] = {}
    try:
        for upload in files:
            filename = Path(upload.filename or "").name
            if not filename or Path(filename).suffix.casefold() != ".txt":
                raise ValueError(f"Annotation must be a .txt file: {filename or '<unnamed>'}")
            candidates = aliases.get(Path(filename).stem.casefold(), [])
            if len(candidates) != 1:
                raise ValueError(f"Annotation filename does not identify exactly one image: {filename}")
            record = candidates[0]
            if record.image_id in matched:
                raise ValueError(f"Duplicate annotation for image: {record.original_name}")
            payload = await upload.read(MAX_LABEL_BYTES + 1)
            if len(payload) > MAX_LABEL_BYTES:
                raise ValueError(f"Annotation exceeds 5 MB: {filename}")
            try:
                text = payload.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Annotation is not UTF-8 text: {filename}") from exc
            destination_name = f"{Path(record.stored_name).stem}.txt"
            destination = labels_dir / destination_name
            destination.write_text(text, encoding="utf-8")
            _validate_yolo_label(destination, len(dataset.labels))
            matched[record.image_id] = {
                "path": f"labels/{destination_name}",
                "sha256": _file_sha256(destination),
            }

        manifest_images = [
            {
                "image_id": record.image_id,
                "original_name": record.original_name,
                "stored_name": record.stored_name,
                "sha256": record.sha256,
                "label": matched.get(record.image_id, {}).get("path"),
                "label_sha256": matched.get(record.image_id, {}).get("sha256"),
            }
            for record in images
        ]
        manifest = {
            "dataset_id": dataset.dataset_id,
            "source": "direct_upload",
            "format": "normalized-detection-text-v1",
            "classes": dataset.labels,
            "image_count": len(images),
            "labeled_count": len(matched),
            "unlabeled_count": len(images) - len(matched),
            "images": manifest_images,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        staging.replace(version_dir)
        return {
            "version_dir": str(version_dir),
            "labels_dir": str(version_dir / "labels"),
            "manifest_path": str(version_dir / "manifest.json"),
            "labeled_count": len(matched),
            "unlabeled_count": len(images) - len(matched),
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        for upload in files:
            await upload.close()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
