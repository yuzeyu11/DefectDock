"""Computer-vision dataset tooling: validation, statistics, conversion, splitting."""

from .check import CheckReport, Issue, check_dataset, iter_labeled_images
from .convert import (
    VocAnnotation,
    VocObject,
    annotation_to_yolo_lines,
    box_to_yolo,
    build_class_map,
    convert_voc_dataset,
    parse_voc_xml,
)
from .feedback import FeedbackRecord, FeedbackResult, file_sha256, ingest_feedback
from .gc10 import (
    GC10_CLASS_NAMES,
    GC10_SOURCE_LABELS,
    Gc10Box,
    Gc10Sample,
    discover_gc10_samples,
    import_gc10_dataset,
    split_gc10_samples,
)
from .snapshot import build_training_snapshot
from .stats import ClassStat, DatasetStats, compute_stats

__all__ = [
    "CheckReport",
    "ClassStat",
    "DatasetStats",
    "FeedbackRecord",
    "FeedbackResult",
    "GC10_CLASS_NAMES",
    "GC10_SOURCE_LABELS",
    "Gc10Box",
    "Gc10Sample",
    "Issue",
    "VocAnnotation",
    "VocObject",
    "annotation_to_yolo_lines",
    "box_to_yolo",
    "build_class_map",
    "build_training_snapshot",
    "check_dataset",
    "compute_stats",
    "convert_voc_dataset",
    "discover_gc10_samples",
    "file_sha256",
    "ingest_feedback",
    "import_gc10_dataset",
    "iter_labeled_images",
    "parse_voc_xml",
    "split_gc10_samples",
]
