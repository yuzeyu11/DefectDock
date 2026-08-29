"""Import a versioned CVAT YOLO export into an uploaded dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

from defectdock.domain import DatasetImageRecord, DatasetRecord


def import_cvat_yolo_export(
    archive_path: str | Path,
    dataset: DatasetRecord,
    images: Iterable[DatasetImageRecord],
    version_dir: str | Path,
) -> dict:
    """Safely unpack a CVAT YOLO 1.1 export and version its label files.

    CVAT receives the stored image filenames, so exported label stems normally
    match ``stored_name``. Original filename stems are accepted only when they
    identify one uploaded image unambiguously.
    """
    archive_path = Path(archive_path).resolve()
    version_dir = Path(version_dir).resolve()
    if not archive_path.is_file():
        raise ValueError(f"CVAT export archive not found: {archive_path}")
    if version_dir.exists():
        raise FileExistsError(f"Annotation version already exists: {version_dir}")
    version_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{version_dir.name}-", dir=version_dir.parent)
    )
    extracted_dir = staging_dir / "export"
    extracted_dir.mkdir()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extracted_dir)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise ValueError("CVAT returned an invalid ZIP archive") from exc
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    try:
        exported_classes = _read_export_classes(extracted_dir)
        if exported_classes != dataset.labels:
            raise ValueError(
                "CVAT class order does not match the dataset: "
                f"expected {dataset.labels}, got {exported_classes}"
            )

        image_records = list(images)
        aliases: dict[str, list[DatasetImageRecord]] = {}
        for record in image_records:
            for stem in {Path(record.stored_name).stem, Path(record.original_name).stem}:
                aliases.setdefault(stem, []).append(record)

        matched: dict[str, Path] = {}
        unmatched_labels: list[str] = []
        for label_path in sorted(extracted_dir.rglob("*.txt")):
            if label_path.name.casefold() in {"train.txt", "valid.txt", "test.txt"}:
                continue
            candidates = aliases.get(label_path.stem, [])
            if not candidates:
                if label_path.parent.name == "obj_train_data":
                    unmatched_labels.append(str(label_path.relative_to(extracted_dir)))
                continue
            if len(candidates) != 1:
                raise ValueError(
                    f"CVAT label name is ambiguous for uploaded images: {label_path.name}"
                )
            record = candidates[0]
            if record.image_id in matched:
                raise ValueError(f"CVAT export has duplicate labels for {record.original_name}")
            _validate_yolo_label(label_path, len(dataset.labels))
            matched[record.image_id] = label_path
        if unmatched_labels:
            raise ValueError(
                "CVAT export contains labels that do not match uploaded images: "
                + ", ".join(unmatched_labels[:10])
            )

        labels_dir = staging_dir / "labels"
        labels_dir.mkdir()
        manifest_images = []
        for record in image_records:
            source = matched.get(record.image_id)
            label_name = f"{Path(record.stored_name).stem}.txt"
            if source is not None:
                shutil.copy2(source, labels_dir / label_name)
            copied_label = labels_dir / label_name
            manifest_images.append(
                {
                    "image_id": record.image_id,
                    "original_name": record.original_name,
                    "stored_name": record.stored_name,
                    "sha256": record.sha256,
                    "label": f"labels/{label_name}" if source is not None else None,
                    "label_sha256": _sha256(copied_label) if source is not None else None,
                }
            )

        archive_sha256 = _sha256(archive_path)
        manifest = {
            "dataset_id": dataset.dataset_id,
            "cvat_task_id": dataset.cvat_task_id,
            "format": "YOLO 1.1",
            "classes": dataset.labels,
            "archive": str(archive_path),
            "archive_sha256": archive_sha256,
            "image_count": len(image_records),
            "labeled_count": len(matched),
            "unlabeled_count": len(image_records) - len(matched),
            "images": manifest_images,
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging_dir.replace(version_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    labels_dir = version_dir / "labels"
    manifest_path = version_dir / "manifest.json"
    return {
        "version_dir": str(version_dir),
        "labels_dir": str(labels_dir),
        "manifest_path": str(manifest_path),
        "archive_sha256": archive_sha256,
        "labeled_count": len(matched),
        "unlabeled_count": len(image_records) - len(matched),
    }


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    seen: set[str] = set()
    total_size = 0
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Unsafe path in CVAT export: {member.filename}")
        mode = (member.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise ValueError(f"Symlink is not allowed in CVAT export: {member.filename}")
        if not member.is_dir():
            normalized = str(target).casefold()
            if normalized in seen:
                raise ValueError(f"Duplicate path in CVAT export: {member.filename}")
            seen.add(normalized)
            total_size += member.file_size
            if total_size > 512 * 1024 * 1024:
                raise ValueError("CVAT annotation export exceeds the 512 MB safety limit")
    archive.extractall(destination)


def _read_export_classes(extracted_dir: Path) -> list[str]:
    names_files = list(extracted_dir.rglob("obj.names"))
    if len(names_files) != 1:
        raise ValueError(
            f"CVAT YOLO 1.1 export must contain one obj.names file; found {len(names_files)}"
        )
    return [
        line.strip()
        for line in names_files[0].read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _validate_yolo_label(path: Path, class_count: int) -> None:
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label {path}:{line_number}: expected 5 fields")
        try:
            class_value = float(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO label {path}:{line_number}: non-numeric value") from exc
        if not class_value.is_integer() or not 0 <= int(class_value) < class_count:
            raise ValueError(f"Invalid class id in {path}:{line_number}: {parts[0]}")
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"Out-of-range box in {path}:{line_number}")
        cx, cy, width, height = coordinates
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"Non-positive box size in {path}:{line_number}")
        if cx - width / 2 < 0.0 or cy - height / 2 < 0.0 or cx + width / 2 > 1.0 or cy + height / 2 > 1.0:
            raise ValueError(f"Box exceeds image bounds in {path}:{line_number}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
