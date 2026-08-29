"""Pascal VOC XML → YOLO detection format conversion.

Converts ``<annotation>`` XML files (with ``<bndbox>`` in pixel coordinates) into
normalized YOLO labels (``class_id cx cy w h``). Useful for importing datasets
such as GC10-DET that ship VOC annotations, and for the CVAT export pipeline.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass
class VocObject:
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class VocAnnotation:
    filename: str
    width: int
    height: int
    objects: list[VocObject] = field(default_factory=list)


def parse_voc_xml(xml_path: str | Path) -> VocAnnotation:
    """Parse a single Pascal VOC annotation file."""
    tree = ET.parse(Path(xml_path))
    root = tree.getroot()
    width = int(float(root.findtext("size/width") or 0))
    height = int(float(root.findtext("size/height") or 0))
    filename = root.findtext("filename") or Path(xml_path).stem
    objects: list[VocObject] = []
    for obj in root.findall("object"):
        name = obj.findtext("name") or ""
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        objects.append(
            VocObject(
                name=name,
                xmin=int(float(bndbox.findtext("xmin") or 0)),
                ymin=int(float(bndbox.findtext("ymin") or 0)),
                xmax=int(float(bndbox.findtext("xmax") or 0)),
                ymax=int(float(bndbox.findtext("ymax") or 0)),
            )
        )
    return VocAnnotation(filename=filename, width=width, height=height, objects=objects)


def box_to_yolo(
    obj: VocObject,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Convert a pixel ``bndbox`` to normalized ``(cx, cy, w, h)``."""
    cx = (obj.xmin + obj.xmax) / 2 / width
    cy = (obj.ymin + obj.ymax) / 2 / height
    w = (obj.xmax - obj.xmin) / width
    h = (obj.ymax - obj.ymin) / height
    return round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)


def build_class_map(names: list[str]) -> dict[str, int]:
    """Assign class ids in first-seen order, preserving caller-defined order."""
    return {name: index for index, name in enumerate(names)}


def annotation_to_yolo_lines(
    annotation: VocAnnotation,
    class_map: dict[str, int],
) -> list[str]:
    """Render a VOC annotation as YOLO label lines, skipping unknown classes."""
    lines: list[str] = []
    for obj in annotation.objects:
        class_id = class_map.get(obj.name)
        if class_id is None:
            continue
        cx, cy, w, h = box_to_yolo(obj, annotation.width, annotation.height)
        lines.append(f"{class_id} {cx} {cy} {w} {h}")
    return lines


def convert_voc_dataset(
    image_dir: str | Path,
    xml_dir: str | Path,
    out_dir: str | Path,
    *,
    class_map: dict[str, int] | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Convert a directory of VOC-annotated images into a YOLO layout.

    Writes ``<out_dir>/images/*`` (streamed copies) and ``<out_dir>/labels/*.txt``.
    Both images and labels are always overwritten, so re-running the conversion
    on a changed source cannot leave stale image/label pairs behind.

    Returns ``{class_map, image_count, label_count, skipped_no_xml,
    skipped_unknown_class}``.

    ``class_map`` maps VOC ``<name>`` values to class ids. When omitted, the
    first-seen order across all XML files is used (order may vary per filesystem
    iteration). Pass an explicit ``class_names`` list for a stable mapping.
    """
    image_dir = Path(image_dir)
    xml_dir = Path(xml_dir)
    out_dir = Path(out_dir)
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    resolved_map = dict(class_map) if class_map is not None else None
    if resolved_map is None:
        discovered: list[str] = []
        if class_names:
            discovered = list(class_names)
        else:
            for xml_path in sorted(xml_dir.glob("*.xml")):
                for obj in parse_voc_xml(xml_path).objects:
                    if obj.name not in discovered:
                        discovered.append(obj.name)
        resolved_map = build_class_map(discovered)

    image_count = 0
    label_count = 0
    skipped_no_xml = 0
    skipped_unknown = 0

    image_paths = sorted(image_dir.glob("*"))
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for image_path in image_paths:
        if image_path.suffix.lower() not in image_extensions:
            continue
        xml_path = xml_dir / (image_path.stem + ".xml")
        if not xml_path.is_file():
            skipped_no_xml += 1
            continue
        annotation = parse_voc_xml(xml_path)
        lines = annotation_to_yolo_lines(annotation, resolved_map)
        unknown = len(annotation.objects) - len(lines)
        skipped_unknown += unknown

        destination = images_out / image_path.name
        # 流式拷贝（不整图读入内存），并始终覆盖，保证重跑后图/标一致。
        shutil.copy2(image_path, destination)
        (labels_out / (image_path.stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        image_count += 1
        label_count += len(lines)

    return {
        "class_map": resolved_map,
        "class_names": [name for name, _ in sorted(resolved_map.items(), key=lambda kv: kv[1])],
        "image_count": image_count,
        "label_count": label_count,
        "skipped_no_xml": skipped_no_xml,
        "skipped_unknown_class": skipped_unknown,
    }
