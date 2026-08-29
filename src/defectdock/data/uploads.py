"""Streaming image upload with validation, hashing, and dataset deduplication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from defectdock.db import DatasetStore, DuplicateImageError

SUPPORTED_FORMATS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "BMP": ".bmp",
    "TIFF": ".tif",
    "WEBP": ".webp",
}


@dataclass(frozen=True)
class UploadLimits:
    max_files: int = 500
    max_total_bytes: int = 5 * 1024 * 1024 * 1024
    max_file_bytes: int = 100 * 1024 * 1024
    chunk_bytes: int = 1024 * 1024


def parse_labels(raw: str) -> list[str]:
    value = raw.strip()
    if not value:
        raise ValueError("At least one label is required")
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("labels JSON must be a list")
        labels = [str(item).strip() for item in parsed]
    else:
        labels = [item.strip() for item in value.split(",")]
    labels = [item for item in labels if item]
    if not labels:
        raise ValueError("At least one label is required")
    if len(labels) != len(set(labels)):
        raise ValueError("Labels must be unique")
    if len(labels) > 100:
        raise ValueError("At most 100 labels are allowed")
    return labels


async def ingest_images(
    store: DatasetStore,
    dataset_id: str,
    image_dir: Path,
    files: list[UploadFile],
    *,
    limits: UploadLimits = UploadLimits(),
) -> dict:
    if not files:
        raise ValueError("Select at least one image")
    if len(files) > limits.max_files:
        raise ValueError(f"A browser upload is limited to {limits.max_files} images")
    image_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    accepted = []
    duplicates = []

    for upload in files:
        original_name = Path(upload.filename or "unnamed").name
        incoming = image_dir / f".incoming-{uuid4().hex}"
        digest = hashlib.sha256()
        file_bytes = 0
        try:
            with incoming.open("wb") as handle:
                while chunk := await upload.read(limits.chunk_bytes):
                    file_bytes += len(chunk)
                    total_bytes += len(chunk)
                    if file_bytes > limits.max_file_bytes:
                        raise ValueError(f"Image exceeds 100MB limit: {original_name}")
                    if total_bytes > limits.max_total_bytes:
                        raise ValueError("Upload exceeds the 5GB browser batch limit")
                    digest.update(chunk)
                    handle.write(chunk)
            image_format, width, height = _inspect_image(incoming, original_name)
            sha256 = digest.hexdigest()
            stored_name = f"{sha256}{SUPPORTED_FORMATS[image_format]}"
            destination = image_dir / stored_name
            if not destination.exists():
                incoming.replace(destination)
            else:
                incoming.unlink(missing_ok=True)
            try:
                record = store.add_image(
                    dataset_id,
                    original_name=original_name,
                    stored_name=stored_name,
                    sha256=sha256,
                    size_bytes=file_bytes,
                    width=width,
                    height=height,
                )
            except DuplicateImageError:
                duplicates.append(original_name)
            else:
                accepted.append(record.model_dump(mode="json"))
        finally:
            incoming.unlink(missing_ok=True)
            await upload.close()

    return {
        "accepted_count": len(accepted),
        "duplicate_count": len(duplicates),
        "accepted": accepted,
        "duplicates": duplicates,
    }


def _inspect_image(path: Path, original_name: str) -> tuple[str, int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image_format = image.format or ""
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid or corrupted image: {original_name}") from exc
    if image_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format {image_format or 'unknown'}: {original_name}")
    if width < 16 or height < 16:
        raise ValueError(f"Image is too small: {original_name} ({width}x{height})")
    return image_format, width, height
