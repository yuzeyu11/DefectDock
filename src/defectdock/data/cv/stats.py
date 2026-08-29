"""Dataset statistics for YOLO detection datasets.

Computes per-class box counts, box size/area distributions (normalized), and
image size distributions. Box-size statistics matter for industrial defect
detection where small targets dominate. Reuses the data-layout helpers from
:mod:`defectdock.data.cv.check` and only reads image headers via Pillow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .check import (
    _iter_image_paths,
    _label_path_for,
    _parse_names,
    _split_sources,
    load_data_yaml,
    resolve_dataset_root,
)


@dataclass
class ClassStat:
    class_id: int
    name: str
    box_count: int
    image_count: int


@dataclass
class DatasetStats:
    data_yaml: str
    root: str
    classes: list[str]
    total_images: int
    total_boxes: int
    splits: dict[str, int]
    per_class: list[ClassStat]
    box_size: dict[str, float]
    box_area: dict[str, float]
    image_size: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size  # (width, height)
    except (OSError, ValueError):
        return None


def _min_mean_max(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "count": 0}
    return {
        "min": round(min(values), 4),
        "mean": round(sum(values) / len(values), 4),
        "max": round(max(values), 4),
        "count": len(values),
    }


def compute_stats(data_yaml_path: str | Path) -> DatasetStats:
    """Compute dataset statistics from a YOLO ``data.yaml``.

    Raises :class:`ValueError` on a malformed ``data.yaml`` (missing ``names``,
    non-mapping root, etc.) so callers can distinguish "no dataset" from
    "dataset with issues". Missing split directories and unreadable images are
    collected as warnings rather than errors.
    """
    data_yaml_path = Path(data_yaml_path).resolve()
    if not data_yaml_path.is_file():
        raise ValueError(f"data.yaml not found: {data_yaml_path}")
    config = load_data_yaml(data_yaml_path)
    classes, _ = _parse_names(config)
    root = resolve_dataset_root(data_yaml_path, config)

    warnings: list[str] = []
    splits: dict[str, int] = {}
    total_images = 0
    total_boxes = 0
    per_class_counts: dict[int, int] = {index: 0 for index in range(len(classes))}
    per_class_images: dict[int, set] = {index: set() for index in range(len(classes))}
    box_widths: list[float] = []
    box_heights: list[float] = []
    box_areas: list[float] = []
    image_widths: list[int] = []
    image_heights: list[int] = []

    for split_name in ("train", "val", "test"):
        raw = config.get(split_name)
        if raw is None:
            continue
        image_count = 0
        for source in _split_sources(root, raw):
            if not source.exists():
                warnings.append(f"{split_name}: path not found: {source}")
                continue
            for image_path in _iter_image_paths(source):
                image_count += 1
                total_images += 1
                size = _image_size(image_path)
                if size is None:
                    warnings.append(f"unreadable image: {image_path}")
                else:
                    image_widths.append(size[0])
                    image_heights.append(size[1])
                _accumulate_labels(
                    image_path, per_class_counts, per_class_images, box_widths, box_heights, box_areas
                )
        splits[split_name] = image_count

    for counts in per_class_counts.values():
        total_boxes += counts

    return DatasetStats(
        data_yaml=str(data_yaml_path),
        root=str(root),
        classes=classes,
        total_images=total_images,
        total_boxes=total_boxes,
        splits=splits,
        per_class=[
            ClassStat(
                class_id=class_id,
                name=classes[class_id],
                box_count=per_class_counts[class_id],
                image_count=len(per_class_images[class_id]),
            )
            for class_id in range(len(classes))
        ],
        box_size=_min_mean_max(box_widths + box_heights),
        box_area=_min_mean_max(box_areas),
        image_size={
            "min_width": min(image_widths) if image_widths else 0,
            "mean_width": round(sum(image_widths) / len(image_widths)) if image_widths else 0,
            "max_width": max(image_widths) if image_widths else 0,
            "min_height": min(image_heights) if image_heights else 0,
            "mean_height": round(sum(image_heights) / len(image_heights)) if image_heights else 0,
            "max_height": max(image_heights) if image_heights else 0,
        },
        warnings=warnings,
    )


def _accumulate_labels(
    image_path: Path,
    per_class_counts: dict[int, int],
    per_class_images: dict[int, set],
    box_widths: list[float],
    box_heights: list[float],
    box_areas: list[float],
) -> None:
    label_path = _label_path_for(image_path)
    if label_path is None or not label_path.is_file():
        return
    image_stem = image_path.stem
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    seen_in_image: set[int] = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            _, _, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            continue
        if class_id < 0 or class_id >= len(per_class_counts):
            continue
        if not (0.0 <= width <= 1.0 and 0.0 <= height <= 1.0):
            continue
        per_class_counts[class_id] += 1
        if class_id not in seen_in_image:
            seen_in_image.add(class_id)
            per_class_images[class_id].add(image_stem)
        box_widths.append(width)
        box_heights.append(height)
        box_areas.append(width * height)
