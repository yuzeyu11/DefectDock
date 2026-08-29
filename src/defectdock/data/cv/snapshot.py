"""Materialize an immutable, checked training snapshot from a frozen dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import yaml

from defectdock.domain import DatasetImageRecord, DatasetRecord, DatasetStatus


def build_training_snapshot(
    dataset: DatasetRecord,
    images: Iterable[DatasetImageRecord],
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """Create deterministic train/val trees from the latest annotation version."""
    if dataset.status != DatasetStatus.FROZEN:
        raise ValueError("Dataset must be frozen before creating a training snapshot")
    if not 0.05 <= val_ratio <= 0.5:
        raise ValueError("val_ratio must be between 0.05 and 0.5")

    dataset_root = Path(dataset.root_dir).resolve()
    versions_root = dataset_root / "annotations" / "versions"
    versions = sorted(path for path in versions_root.iterdir() if path.is_dir()) if versions_root.is_dir() else []
    if not versions:
        raise ValueError("Dataset has no synchronized annotation version")
    annotation_version = versions[-1]
    annotation_manifest = json.loads(
        (annotation_version / "manifest.json").read_text(encoding="utf-8")
    )
    if annotation_manifest.get("classes") != dataset.labels:
        raise ValueError("Annotation classes no longer match the dataset")

    records = {record.image_id: record for record in images}
    labeled: list[tuple[DatasetImageRecord, Path]] = []
    for item in annotation_manifest.get("images", []):
        label_relative = item.get("label")
        record = records.get(item.get("image_id"))
        if not label_relative or record is None:
            continue
        label_path = annotation_version / label_relative
        image_path = dataset_root / "images" / record.stored_name
        if label_path.is_file() and image_path.is_file():
            labeled.append((record, label_path))
    if len(labeled) < 2:
        raise ValueError("At least two labeled images are required for separate train and val splits")

    labeled.sort(key=lambda item: _stable_order(seed, item[0].sha256))
    val_count = min(len(labeled) - 1, max(1, round(len(labeled) * val_ratio)))
    val_ids = {record.image_id for record, _ in labeled[:val_count]}
    split_signature = hashlib.sha256(
        json.dumps(
            {
                "annotation": annotation_version.name,
                "seed": seed,
                "val_ratio": val_ratio,
                "images": [record.sha256 for record, _ in labeled],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    snapshot_id = f"{annotation_version.name}-s{seed}-{split_signature}"
    snapshots_root = dataset_root / "training" / "snapshots"
    snapshot_dir = snapshots_root / snapshot_id
    manifest_path = snapshot_dir / "manifest.json"
    if manifest_path.is_file():
        return _load_snapshot_manifest(snapshot_dir)

    snapshots_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}-", dir=snapshots_root))
    manifest_images: list[dict] = []
    try:
        for split in ("train", "val"):
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
        for record, label_path in labeled:
            split = "val" if record.image_id in val_ids else "train"
            image_source = dataset_root / "images" / record.stored_name
            if _file_sha256(image_source) != record.sha256:
                raise ValueError(f"Uploaded image changed after ingestion: {record.stored_name}")
            image_destination = staging / "images" / split / record.stored_name
            label_destination = staging / "labels" / split / f"{Path(record.stored_name).stem}.txt"
            # A hardlink would let later in-place changes to the upload mutate
            # the snapshot as well. Copies keep the content-addressed snapshot
            # independent from its mutable source tree.
            shutil.copy2(image_source, image_destination)
            shutil.copy2(label_path, label_destination)
            manifest_images.append(
                {
                    "image_id": record.image_id,
                    "image_sha256": _file_sha256(image_destination),
                    "label_sha256": _file_sha256(label_destination),
                    "split": split,
                    "image": str(image_destination.relative_to(staging)).replace("\\", "/"),
                    "label": str(label_destination.relative_to(staging)).replace("\\", "/"),
                }
            )

        data_yaml = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "nc": len(dataset.labels),
            "names": dataset.labels,
        }
        (staging / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.name,
            "annotation_version": annotation_version.name,
            "classes": dataset.labels,
            "seed": seed,
            "val_ratio": val_ratio,
            "train_count": len(labeled) - val_count,
            "val_count": val_count,
            "image_count": len(labeled),
            "materialization": "copy",
            "data_yaml": "data.yaml",
            "images": manifest_images,
        }
        manifest["snapshot_sha256"] = hashlib.sha256(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        staging.replace(snapshot_dir)
        return _load_snapshot_manifest(snapshot_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _stable_order(seed: int, sha256: str) -> str:
    return hashlib.sha256(f"{seed}:{sha256}".encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_snapshot_manifest(snapshot_dir: Path) -> dict:
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    relative_data_yaml = manifest["data_yaml"]
    manifest["data_yaml_relative"] = relative_data_yaml
    manifest["data_yaml"] = str((snapshot_dir / relative_data_yaml).resolve())
    return manifest
