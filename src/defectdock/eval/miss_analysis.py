"""Miss / false-positive sample analysis for inspection acceptance reports.

Splits a set of predictions into missed detections (ground-truth boxes with no
matching prediction), false positives (predictions with no matching ground
truth), and low-confidence hits. This feeds the acceptance report's "failure
case" gallery and the data-feedback loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .cv import Box, iou


@dataclass
class MissSample:
    kind: str  # "miss" | "false_positive" | "low_confidence"
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    area: float
    image: str | None = None


@dataclass
class MissReport:
    missed: list[MissSample] = field(default_factory=list)
    false_positives: list[MissSample] = field(default_factory=list)
    low_confidence: list[MissSample] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sample(
    kind: str,
    box: Box,
    classes: list[str],
    *,
    image: str | None = None,
    confidence: float | None = None,
) -> MissSample:
    name = classes[box.class_id] if 0 <= box.class_id < len(classes) else str(box.class_id)
    return MissSample(
        kind=kind,
        class_id=box.class_id,
        class_name=name,
        confidence=box.confidence if confidence is None else confidence,
        x1=box.x1,
        y1=box.y1,
        x2=box.x2,
        y2=box.y2,
        area=max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1),
        image=image if image is not None else box.image_id,
    )


def find_misses(
    ground_truths: Iterable[Box],
    predictions: Iterable[Box],
    classes: list[str],
    *,
    iou_threshold: float = 0.5,
    low_conf_threshold: float = 0.5,
    image: str | None = None,
) -> MissReport:
    """Classify predictions into missed / false-positive / low-confidence samples.

    Uses the same greedy confidence-first matching as :func:`defectdock.eval.cv.match_boxes`.
    """
    gt = list(ground_truths)
    preds = sorted(predictions, key=lambda box: box.confidence, reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    for pred_index, prediction in enumerate(preds):
        best_gt: int | None = None
        best_score = -1.0
        for gt_index, truth in enumerate(gt):
            if gt_index in matched_gt:
                continue
            if truth.image_id != prediction.image_id:
                continue
            if truth.class_id != prediction.class_id:
                continue
            score = iou(prediction, truth)
            if score >= iou_threshold and score > best_score:
                best_score = score
                best_gt = gt_index
        if best_gt is not None:
            matched_gt.add(best_gt)
            matched_pred.add(pred_index)

    missed = [
        _sample("miss", gt[index], classes, image=image, confidence=0.0)
        for index in range(len(gt))
        if index not in matched_gt
    ]
    false_positives = [
        _sample("false_positive", preds[index], classes, image=image)
        for index in range(len(preds))
        if index not in matched_pred
    ]
    low_confidence = [
        _sample("low_confidence", preds[index], classes, image=image)
        for index in sorted(matched_pred)
        if preds[index].confidence < low_conf_threshold
    ]

    summary = {
        "ground_truth": len(gt),
        "predictions": len(preds),
        "true_positive": len(matched_gt),
        "missed": len(missed),
        "false_positive": len(false_positives),
        "low_confidence": len(low_confidence),
    }
    return MissReport(
        missed=missed,
        false_positives=false_positives,
        low_confidence=low_confidence,
        summary=summary,
    )


def group_by_class(samples: Iterable[MissSample]) -> dict[str, int]:
    """Count samples per class name."""
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.class_name] = counts.get(sample.class_name, 0) + 1
    return counts


def export_miss_manifest(report: MissReport, path: str | Path) -> Path:
    """Write a JSON manifest of the miss report for downstream tooling."""
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out
