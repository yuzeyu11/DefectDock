"""YOLO detection dataset validation.

Validates a YOLO-format dataset (``data.yaml`` + ``images/`` + ``labels/``) and
reports missing images, missing labels, orphan labels, out-of-range classes,
out-of-bounds boxes, empty annotations, malformed label lines, and class/name
inconsistencies.

The validator targets the *detection* layout::

    <root>/
      data.yaml            # path / train / val / (test) / nc / names
      images/<split>/*.jpg
      labels/<split>/*.txt  # one line per box: "class cx cy w h" (normalized)

It intentionally avoids importing a training framework so it can run cheaply in
tests and minimal API installations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
_EPS = 1e-6


@dataclass
class Issue:
    severity: str
    kind: str
    message: str
    split: str | None = None
    path: str | None = None


@dataclass
class CheckReport:
    ok: bool
    data_yaml: str
    root: str | None
    classes: list[str]
    nc: int | None
    splits: dict[str, int]
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_data_yaml(path: str | Path) -> dict:
    """Load and shallow-validate a YOLO ``data.yaml`` as a mapping."""
    data_yaml_path = Path(path)
    with data_yaml_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("data.yaml root must be a mapping")
    return payload


def resolve_dataset_root(data_yaml_path: str | Path, config: dict) -> Path:
    """Resolve the dataset root from ``data.yaml``'s optional ``path`` field."""
    data_yaml_path = Path(data_yaml_path)
    yaml_dir = data_yaml_path.resolve().parent
    raw = config.get("path")
    if not raw:
        return yaml_dir
    candidate = Path(str(raw))
    if candidate.is_absolute():
        return candidate.resolve()
    return (yaml_dir / candidate).resolve()


def _parse_names(config: dict) -> tuple[list[str], int | None]:
    """Return ``(classes, nc)`` from the ``names`` / ``nc`` fields."""
    names = config.get("names")
    if names is None:
        raise ValueError("data.yaml is missing the required 'names' field")
    if isinstance(names, dict):
        ordered = sorted(names.keys(), key=lambda key: int(key))
        classes = [str(names[key]) for key in ordered]
        nc = len(classes)
    elif isinstance(names, (list, tuple)):
        classes = [str(item) for item in names]
        nc = len(classes)
    else:
        raise ValueError("'names' must be a list or a mapping of {index: name}")
    if not classes:
        raise ValueError("'names' must not be empty")
    declared_nc = config.get("nc")
    if declared_nc is not None and int(declared_nc) != nc:
        raise ValueError(f"'nc' ({declared_nc}) does not match the number of names ({nc})")
    return classes, nc


def _split_sources(root: Path, raw: Any) -> list[Path]:
    """Normalize a ``train``/``val``/``test`` value into a list of paths."""
    if isinstance(raw, (list, tuple)):
        values = raw
    else:
        values = [raw]
    sources: list[Path] = []
    for value in values:
        if value is None:
            continue
        candidate = Path(str(value))
        sources.append(candidate if candidate.is_absolute() else root / candidate)
    return sources


def _label_dir_for(image_dir: Path) -> Path | None:
    """Infer the parallel label directory from an image directory.

    Follows the YOLO convention ``images/<split>`` <-> ``labels/<split>`` by
    swapping the ``images`` path segment. Returns ``None`` when the convention
    cannot be applied.
    """
    parts = list(image_dir.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts)
    return None


def _iter_image_paths(source: Path) -> Iterator[Path]:
    """Yield image paths referenced by a directory, list file, or single image."""
    if source.is_dir():
        for candidate in sorted(source.rglob("*")):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                yield candidate
        return
    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            yield source
            return
        if source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                candidate = Path(line)
                if not candidate.is_absolute():
                    candidate = source.parent / candidate
                if candidate.suffix.lower() in IMAGE_EXTENSIONS:
                    yield candidate


def _label_path_for(image_path: Path) -> Path | None:
    """Map an image path to its parallel label path, or ``None`` if unknown."""
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return None


def iter_labeled_images(root: Path, raw: Any) -> Iterator[tuple[Path, Path]]:
    """Yield ``(image_path, label_path)`` pairs for images that have labels.

    Used by the evaluation module and by any consumer that needs to walk a
    dataset split while skipping unlabeled (background-only) images.
    """
    for source in _split_sources(root, raw):
        if not source.exists():
            continue
        for image_path in _iter_image_paths(source):
            label_path = _label_path_for(image_path)
            if label_path is not None and label_path.is_file():
                yield image_path, label_path


def check_dataset(data_yaml_path: str | Path) -> CheckReport:
    """Validate a YOLO detection dataset and return a structured report."""
    data_yaml_path = Path(data_yaml_path).resolve()
    errors: list[Issue] = []
    warnings: list[Issue] = []

    def report(
        severity: str,
        kind: str,
        message: str,
        *,
        split: str | None = None,
        path: str | Path | None = None,
    ) -> None:
        bucket = errors if severity == SEVERITY_ERROR else warnings
        bucket.append(
            Issue(severity, kind, message, split=split, path=str(path) if path else None)
        )

    if not data_yaml_path.is_file():
        report(SEVERITY_ERROR, "missing_data_yaml", f"data.yaml not found: {data_yaml_path}")
        return CheckReport(
            ok=False,
            data_yaml=str(data_yaml_path),
            root=None,
            classes=[],
            nc=None,
            splits={},
            errors=errors,
            warnings=warnings,
        )

    try:
        config = load_data_yaml(data_yaml_path)
    except (yaml.YAMLError, ValueError, OSError) as exc:
        report(SEVERITY_ERROR, "invalid_data_yaml", f"cannot parse data.yaml: {exc}")
        return CheckReport(
            ok=False,
            data_yaml=str(data_yaml_path),
            root=None,
            classes=[],
            nc=None,
            splits={},
            errors=errors,
            warnings=warnings,
        )

    try:
        classes, nc = _parse_names(config)
    except (ValueError, KeyError) as exc:
        report(SEVERITY_ERROR, "invalid_names", str(exc))
        return CheckReport(
            ok=False,
            data_yaml=str(data_yaml_path),
            root=str(resolve_dataset_root(data_yaml_path, config)),
            classes=[],
            nc=None,
            splits={},
            errors=errors,
            warnings=warnings,
        )

    root = resolve_dataset_root(data_yaml_path, config)
    raw_splits = {key: config[key] for key in ("train", "val", "test") if key in config}
    if "train" not in raw_splits and "val" not in raw_splits:
        report(SEVERITY_ERROR, "no_split", "data.yaml must declare 'train' and/or 'val'")
        return CheckReport(
            ok=False,
            data_yaml=str(data_yaml_path),
            root=str(root),
            classes=classes,
            nc=nc,
            splits={},
            errors=errors,
            warnings=warnings,
        )

    splits: dict[str, int] = {}
    split_images: dict[str, set[Path]] = {}
    for split_name, raw in raw_splits.items():
        sources = _split_sources(root, raw)
        missing_sources = [source for source in sources if not source.exists()]
        for source in missing_sources:
            if split_name == "test":
                report(
                    SEVERITY_WARNING,
                    "missing_split_dir",
                    f"split directory not found: {source}",
                    split=split_name,
                    path=source,
                )
            else:
                report(
                    SEVERITY_ERROR,
                    "missing_split_dir",
                    f"split directory not found: {source}",
                    split=split_name,
                    path=source,
                )

        image_count = 0
        image_paths: set[Path] = set()
        for source in sources:
            if not source.exists():
                continue
            for image_path in _iter_image_paths(source):
                image_count += 1
                image_paths.add(image_path.resolve())
                _check_image(image_path, nc, report, split_name)
        splits[split_name] = image_count
        split_images[split_name] = image_paths

    split_names = list(split_images)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            overlap = split_images[left_name] & split_images[right_name]
            if overlap:
                sample = sorted(overlap)[0]
                report(
                    SEVERITY_ERROR,
                    "split_overlap",
                    f"{len(overlap)} image(s) appear in both '{left_name}' and '{right_name}'",
                    path=sample,
                )

    _check_orphan_labels(root, raw_splits, report)

    return CheckReport(
        ok=not errors,
        data_yaml=str(data_yaml_path),
        root=str(root),
        classes=classes,
        nc=nc,
        splits=splits,
        errors=errors,
        warnings=warnings,
    )


def _check_image(
    image_path: Path,
    nc: int | None,
    report,
    split_name: str,
) -> None:
    label_path = _label_path_for(image_path)
    if label_path is None:
        report(
            SEVERITY_WARNING,
            "unknown_label_dir",
            "cannot infer label directory from image path (expected an 'images' segment)",
            split=split_name,
            path=image_path,
        )
        return
    if not label_path.is_file():
        report(
            SEVERITY_WARNING,
            "missing_label",
            "image has no corresponding label file",
            split=split_name,
            path=image_path,
        )
        return

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        report(SEVERITY_ERROR, "unreadable_label", str(exc), split=split_name, path=label_path)
        return

    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        report(
            SEVERITY_WARNING,
            "empty_label",
            "label file is empty (no boxes)",
            split=split_name,
            path=label_path,
        )
        return

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            report(
                SEVERITY_ERROR,
                "malformed_label_line",
                f"expected 5 fields (class cx cy w h), got {len(parts)}",
                split=split_name,
                path=f"{label_path}:{line_number}",
            )
            continue
        try:
            class_value = float(parts[0])
            cx, cy, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            report(
                SEVERITY_ERROR,
                "malformed_label_line",
                f"non-numeric fields: {line}",
                split=split_name,
                path=f"{label_path}:{line_number}",
            )
            continue
        if not class_value.is_integer():
            report(
                SEVERITY_ERROR,
                "malformed_label_line",
                f"class id must be an integer, got {parts[0]}",
                split=split_name,
                path=f"{label_path}:{line_number}",
            )
            continue
        class_id = int(class_value)
        if nc is not None and (class_id < 0 or class_id >= nc):
            report(
                SEVERITY_ERROR,
                "class_out_of_range",
                f"class id {class_id} outside [0, {nc})",
                split=split_name,
                path=f"{label_path}:{line_number}",
            )
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= width <= 1.0 and 0.0 <= height <= 1.0):
            report(
                SEVERITY_ERROR,
                "box_out_of_range",
                f"normalized box values outside [0, 1]: {cx} {cy} {width} {height}",
                split=split_name,
                path=f"{label_path}:{line_number}",
            )
        elif width <= 0.0 or height <= 0.0:
            report(
                SEVERITY_ERROR,
                "box_non_positive",
                f"box width and height must be positive: {cx} {cy} {width} {height}",
                split=split_name,
                path=f"{label_path}:{line_number}",
            )
        elif (
            cx - width / 2 < -_EPS
            or cy - height / 2 < -_EPS
            or cx + width / 2 > 1.0 + _EPS
            or cy + height / 2 > 1.0 + _EPS
        ):
            report(
                SEVERITY_ERROR,
                "box_out_of_range",
                f"box exceeds image bounds: {cx} {cy} {width} {height}",
                split=split_name,
                path=f"{label_path}:{line_number}",
            )


def _check_orphan_labels(
    root: Path,
    raw_splits: dict[str, Any],
    report,
) -> None:
    """Warn about label files that have no matching image."""
    for split_name, raw in raw_splits.items():
        sources = _split_sources(root, raw)
        label_dirs: set[Path] = set()
        for source in sources:
            if source.is_dir():
                label_dir = _label_dir_for(source)
                if label_dir is not None and label_dir.is_dir():
                    label_dirs.add(label_dir)
            elif source.is_file() and source.suffix.lower() == ".txt":
                label_dir = _label_dir_for(source.parent)
                if label_dir is not None and label_dir.is_dir():
                    label_dirs.add(label_dir)
        for label_dir in label_dirs:
            for label_path in sorted(label_dir.rglob("*.txt")):
                image_path = _image_path_for(label_path)
                if image_path is not None and not image_path.is_file():
                    report(
                        SEVERITY_WARNING,
                        "orphan_label",
                        "label file has no corresponding image",
                        split=split_name,
                        path=label_path,
                    )


def _image_path_for(label_path: Path) -> Path | None:
    """Map a label path back to its image path, or ``None`` if unknown.

    Returns the inferred image path even when the file does not exist, so
    callers can distinguish "cannot infer" from "label has no image".
    """
    parts = list(label_path.parts)
    for index, part in enumerate(parts):
        if part == "labels":
            parts[index] = "images"
            image_path = Path(*parts)
            for extension in IMAGE_EXTENSIONS:
                candidate = image_path.with_suffix(extension)
                if candidate.is_file():
                    return candidate
            return image_path.with_suffix(".jpg")
    return None
