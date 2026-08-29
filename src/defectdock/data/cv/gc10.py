"""GC10-DET to YOLO dataset importer.

The public GC10-DET layout uses ten numbered image directories and a shared
``lable`` directory containing Pascal VOC XML files.  This module imports the
complete dataset, applies the canonical ten-class mapping, creates deterministic
train/validation/test splits, and records source anomalies in a manifest.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml
from PIL import Image

from .convert import parse_voc_xml

GC10_CLASS_NAMES = [
    "punching_hole",
    "welding_line",
    "crescent_gap",
    "water_spot",
    "oil_spot",
    "silk_spot",
    "inclusion",
    "rolled_pit",
    "crease",
    "waist_folding",
]

GC10_SOURCE_LABELS = {
    "1_chongkong": "punching_hole",
    "2_hanfeng": "welding_line",
    "3_yueyawan": "crescent_gap",
    "4_shuiban": "water_spot",
    "5_youban": "oil_spot",
    "6_siban": "silk_spot",
    "7_yiwu": "inclusion",
    "8_yahen": "rolled_pit",
    "9_zhehen": "crease",
    "10_yaozhe": "waist_folding",
    # This spelling occurs in the distributed annotations and is an alias,
    # not an eleventh class.
    "10_yaozhed": "waist_folding",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Gc10Box:
    class_name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class Gc10Sample:
    image: Path
    annotation: Path | None
    source_folder: str
    width: int
    height: int
    boxes: tuple[Gc10Box, ...]

    @property
    def classes(self) -> frozenset[str]:
        return frozenset(box.class_name for box in self.boxes)


def import_gc10_dataset(
    source: str | Path,
    output: str | Path,
    *,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    include_unannotated: bool = False,
    strict_unknown_labels: bool = False,
    materialize: Literal["copy", "hardlink"] = "copy",
) -> dict:
    """Import a complete GC10-DET tree into a versioned YOLO layout.

    Images without XML are skipped by default because GC10 image folders hold
    defect samples, so treating missing annotations as confirmed negatives can
    corrupt training.  Set ``include_unannotated`` only after manual review.
    Unknown object labels are counted and skipped unless
    ``strict_unknown_labels`` is enabled.
    """
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    _validate_import_options(source_path, output_path, val_ratio, test_ratio, materialize)

    samples, audit = discover_gc10_samples(
        source_path,
        include_unannotated=include_unannotated,
        strict_unknown_labels=strict_unknown_labels,
    )
    if not samples:
        raise ValueError("GC10 source contains no importable images")
    splits = split_gc10_samples(samples, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.parent / f".{output_path.name}.tmp-{uuid4().hex[:8]}"
    staging.mkdir()
    try:
        manifest_samples: list[dict] = []
        split_summary: dict[str, dict] = {}
        class_ids = {name: index for index, name in enumerate(GC10_CLASS_NAMES)}
        for split_name, split_samples in splits.items():
            images_dir = staging / "images" / split_name
            labels_dir = staging / "labels" / split_name
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            box_counts: Counter[str] = Counter()
            image_counts: Counter[str] = Counter()
            for sample in sorted(split_samples, key=lambda item: item.image.name):
                destination = images_dir / sample.image.name
                _materialize_image(sample.image, destination, materialize)
                label_path = labels_dir / f"{sample.image.stem}.txt"
                lines = _yolo_lines(sample, class_ids)
                label_path.write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
                box_counts.update(box.class_name for box in sample.boxes)
                image_counts.update(sample.classes)
                manifest_samples.append(
                    {
                        "source_image": str(sample.image),
                        "source_annotation": str(sample.annotation) if sample.annotation else None,
                        "output_image": f"images/{split_name}/{sample.image.name}",
                        "output_label": f"labels/{split_name}/{sample.image.stem}.txt",
                        "source_folder": sample.source_folder,
                        "split": split_name,
                        "classes": sorted(sample.classes),
                        "box_count": len(sample.boxes),
                    }
                )
            split_summary[split_name] = {
                "images": len(split_samples),
                "boxes": sum(box_counts.values()),
                "box_counts": dict(sorted(box_counts.items())),
                "image_counts": dict(sorted(image_counts.items())),
            }

        data_yaml = {
            "path": output_path.as_posix(),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {index: name for index, name in enumerate(GC10_CLASS_NAMES)},
        }
        (staging / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        manifest = {
            "dataset": "GC10-DET",
            "source": str(source_path),
            "output": str(output_path),
            "seed": seed,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "materialize": materialize,
            "classes": GC10_CLASS_NAMES,
            "source_label_map": GC10_SOURCE_LABELS,
            "audit": audit,
            "splits": split_summary,
            "samples": manifest_samples,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging.replace(output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    manifest["data_yaml"] = str(output_path / "data.yaml")
    manifest["manifest_path"] = str(output_path / "manifest.json")
    return manifest


def discover_gc10_samples(
    source: str | Path,
    *,
    include_unannotated: bool = False,
    strict_unknown_labels: bool = False,
) -> tuple[list[Gc10Sample], dict]:
    """Discover and validate GC10 images and VOC annotations."""
    source_path = Path(source).resolve()
    annotation_dir = _annotation_dir(source_path)
    image_dirs = {path.name: path for path in source_path.iterdir() if path.is_dir() and path.name.isdigit()}
    missing_folders = [str(number) for number in range(1, 11) if str(number) not in image_dirs]
    if missing_folders:
        raise ValueError(f"GC10 source is missing image folders: {', '.join(missing_folders)}")

    samples: list[Gc10Sample] = []
    missing_annotations: list[str] = []
    empty_annotations: list[str] = []
    unknown_only_annotations: list[str] = []
    unknown_labels: Counter[str] = Counter()
    images_by_name: defaultdict[str, list[Path]] = defaultdict(list)
    for folder_number in range(1, 11):
        image_dir = image_dirs[str(folder_number)]
        for image_path in sorted(image_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images_by_name[image_path.name.casefold()].append(image_path)

    duplicate_sources: dict[str, list[str]] = {}
    representatives: list[Path] = []
    for paths in images_by_name.values():
        paths.sort(key=lambda path: (int(path.parent.name), path.name))
        if len(paths) > 1:
            pixel_hashes = {_pixel_sha256(path) for path in paths}
            if len(pixel_hashes) != 1:
                joined = ", ".join(str(path) for path in paths)
                raise ValueError(f"GC10 filename collision has different pixels: {joined}")
            duplicate_sources[paths[0].name] = [
                str(path.relative_to(source_path)) for path in paths
            ]
        representatives.append(paths[0])

    for image_path in sorted(representatives, key=lambda path: path.name):
        folder_number = int(image_path.parent.name)

        annotation_path = annotation_dir / f"{image_path.stem}.xml"
        if not annotation_path.is_file():
            missing_annotations.append(str(image_path.relative_to(source_path)))
            if not include_unannotated:
                continue
            with Image.open(image_path) as opened:
                width, height = opened.size
            samples.append(
                Gc10Sample(image_path, None, str(folder_number), width, height, ())
            )
            continue

        annotation = parse_voc_xml(annotation_path)
        with Image.open(image_path) as opened:
            width, height = opened.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions for {image_path}")
        if (annotation.width, annotation.height) != (width, height):
            raise ValueError(
                f"Image/XML size mismatch for {image_path.name}: "
                f"image={width}x{height}, xml={annotation.width}x{annotation.height}"
            )

        boxes: list[Gc10Box] = []
        for obj in annotation.objects:
            canonical_name = GC10_SOURCE_LABELS.get(obj.name.strip())
            if canonical_name is None:
                unknown_labels[obj.name.strip() or "<empty>"] += 1
                if strict_unknown_labels:
                    raise ValueError(f"Unknown GC10 label {obj.name!r} in {annotation_path}")
                continue
            if not (0 <= obj.xmin < obj.xmax <= width and 0 <= obj.ymin < obj.ymax <= height):
                raise ValueError(
                    f"Invalid bounding box {(obj.xmin, obj.ymin, obj.xmax, obj.ymax)} "
                    f"in {annotation_path}"
                )
            boxes.append(
                Gc10Box(canonical_name, obj.xmin, obj.ymin, obj.xmax, obj.ymax)
            )
        if not annotation.objects:
            empty_annotations.append(str(annotation_path.relative_to(source_path)))
        elif not boxes:
            unknown_only_annotations.append(str(annotation_path.relative_to(source_path)))
            continue
        samples.append(
            Gc10Sample(
                image_path,
                annotation_path,
                str(folder_number),
                width,
                height,
                tuple(boxes),
            )
        )

    source_image_count = sum(len(paths) for paths in images_by_name.values())
    audit = {
        "source_image_count": source_image_count,
        "unique_image_count": len(representatives),
        "duplicate_image_count": source_image_count - len(representatives),
        "duplicate_image_groups": duplicate_sources,
        "source_annotation_count": len(list(annotation_dir.glob("*.xml"))),
        "imported_image_count": len(samples),
        "missing_annotation_count": len(missing_annotations),
        "missing_annotations": missing_annotations,
        "empty_annotation_count": len(empty_annotations),
        "empty_annotations": empty_annotations,
        "unknown_only_annotation_count": len(unknown_only_annotations),
        "unknown_only_annotations": unknown_only_annotations,
        "included_unannotated": include_unannotated,
        "unknown_label_count": sum(unknown_labels.values()),
        "unknown_labels": dict(sorted(unknown_labels.items())),
    }
    return samples, audit


def split_gc10_samples(
    samples: list[Gc10Sample],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[Gc10Sample]]:
    """Deterministically split each source class folder to avoid class drift."""
    grouped: defaultdict[str, list[Gc10Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.source_folder].append(sample)

    splits: dict[str, list[Gc10Sample]] = {"train": [], "val": [], "test": []}
    for folder, group in sorted(grouped.items(), key=lambda item: int(item[0])):
        ordered = sorted(group, key=lambda item: item.image.name)
        random.Random(f"gc10:{seed}:{folder}").shuffle(ordered)
        test_count, val_count = _holdout_counts(len(ordered), test_ratio, val_ratio)
        splits["test"].extend(ordered[:test_count])
        splits["val"].extend(ordered[test_count : test_count + val_count])
        splits["train"].extend(ordered[test_count + val_count :])

    for split_name, split in splits.items():
        if not split:
            raise ValueError(f"GC10 {split_name} split is empty")
    return splits


def _holdout_counts(size: int, test_ratio: float, val_ratio: float) -> tuple[int, int]:
    if size < 3:
        return (0, 0)
    test_count = max(1, round(size * test_ratio)) if test_ratio > 0 else 0
    val_count = max(1, round(size * val_ratio)) if val_ratio > 0 else 0
    while test_count + val_count >= size:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
    return test_count, val_count


def _yolo_lines(sample: Gc10Sample, class_ids: dict[str, int]) -> list[str]:
    lines = []
    for box in sample.boxes:
        x_center = ((box.xmin + box.xmax) / 2) / sample.width
        y_center = ((box.ymin + box.ymax) / 2) / sample.height
        width = (box.xmax - box.xmin) / sample.width
        height = (box.ymax - box.ymin) / sample.height
        lines.append(
            f"{class_ids[box.class_name]} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"
        )
    return lines


def _annotation_dir(source: Path) -> Path:
    if not source.is_dir():
        raise FileNotFoundError(f"GC10 source directory not found: {source}")
    for name in ("lable", "label", "labels", "Annotations", "annotations"):
        candidate = source / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"GC10 annotation directory not found under: {source}")


def _validate_import_options(
    source: Path,
    output: Path,
    val_ratio: float,
    test_ratio: float,
    materialize: str,
) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"GC10 source directory not found: {source}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Expected val_ratio >= 0, test_ratio >= 0, and their sum < 1")
    if materialize not in {"copy", "hardlink"}:
        raise ValueError("materialize must be 'copy' or 'hardlink'")


def _materialize_image(source: Path, destination: Path, mode: str) -> None:
    if mode == "hardlink":
        destination.hardlink_to(source)
    else:
        shutil.copy2(source, destination)


def _pixel_sha256(path: Path) -> str:
    """Hash decoded pixels so metadata-only JPEG differences are deduplicated."""
    digest = hashlib.sha256()
    with Image.open(path) as image:
        digest.update(image.mode.encode("ascii"))
        digest.update(str(image.size).encode("ascii"))
        digest.update(image.tobytes())
    return digest.hexdigest()
