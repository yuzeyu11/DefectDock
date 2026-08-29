"""Evaluation: industrial detection-rate metrics and threshold optimization."""

from .cv import (
    Box,
    ClassEval,
    EvalSummary,
    ThresholdPoint,
    ThresholdScan,
    evaluate_dataset,
    iou,
    load_ground_truth,
    match_boxes,
    optimize_threshold,
    summarize,
)
from .miss_analysis import (
    MissReport,
    MissSample,
    export_miss_manifest,
    find_misses,
    group_by_class,
)
from .report import build_acceptance_report, write_report

__all__ = [
    "Box",
    "ClassEval",
    "EvalSummary",
    "MissReport",
    "MissSample",
    "ThresholdPoint",
    "ThresholdScan",
    "build_acceptance_report",
    "evaluate_dataset",
    "export_miss_manifest",
    "find_misses",
    "group_by_class",
    "iou",
    "load_ground_truth",
    "match_boxes",
    "optimize_threshold",
    "summarize",
    "write_report",
]
