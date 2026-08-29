"""Industrial-grade evaluation metrics for object detection.

Implements the customer-facing metrics used in acceptance tests rather than the
academic mAP: **detection rate** (recall, ``TP / (TP + FN)``) and **false discovery
rate** (FDR, ``FP / (TP + FP)``, i.e. ``1 - precision``). Missing a defect is far more
costly than a false alarm in an inspection line, so the module also exposes a PR
threshold sweep that recommends an operating point.

The core functions (:func:`iou`, :func:`match_boxes`, :func:`summarize`,
:func:`optimize_threshold`) are pure and unit-testable without a model or GPU.
:func:`evaluate_dataset` ties them to an engine-neutral predictor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    image_id: str | None = None


@dataclass
class ClassEval:
    class_id: int
    name: str
    gt: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    false_discovery_rate: float


@dataclass
class EvalSummary:
    classes: list[str]
    gt_total: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    false_discovery_rate: float
    per_class: list[ClassEval] = field(default_factory=list)
    threshold: float | None = None
    iou_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two axis-aligned boxes."""
    inter_width = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    inter_height = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    intersection = inter_width * inter_height
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def match_boxes(
    ground_truths: Iterable[Box],
    predictions: Iterable[Box],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int, set[int]]:
    """Greedy confidence-first matching. Returns ``(tp, fp, fn, matched_gt_indices)``."""
    gt = list(ground_truths)
    predictions = sorted(predictions, key=lambda box: box.confidence, reverse=True)
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    for prediction in predictions:
        best_index: int | None = None
        best_score = -1.0
        for index, truth in enumerate(gt):
            if index in matched_gt:
                continue
            if truth.image_id != prediction.image_id:
                continue
            if truth.class_id != prediction.class_id:
                continue
            score = iou(prediction, truth)
            if score >= iou_threshold and score > best_score:
                best_score = score
                best_index = index
        if best_index is not None:
            matched_gt.add(best_index)
            tp += 1
        else:
            fp += 1
    fn = len(gt) - len(matched_gt)
    return tp, fp, fn, matched_gt


def _rates(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_discovery_rate = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_discovery_rate": round(false_discovery_rate, 4),
    }


def summarize(
    classes: list[str],
    ground_truths: Iterable[Box],
    predictions: Iterable[Box],
    iou_threshold: float = 0.5,
) -> EvalSummary:
    """Aggregate TP/FP/FN overall and per class."""
    gt = list(ground_truths)
    preds = list(predictions)
    tp, fp, fn, _ = match_boxes(gt, preds, iou_threshold)
    rates = _rates(tp, fp, fn)
    per_class: list[ClassEval] = []
    for class_id, name in enumerate(classes):
        class_gt = [box for box in gt if box.class_id == class_id]
        class_preds = [box for box in preds if box.class_id == class_id]
        c_tp, c_fp, c_fn, _ = match_boxes(class_gt, class_preds, iou_threshold)
        c_rates = _rates(c_tp, c_fp, c_fn)
        per_class.append(
            ClassEval(
                class_id=class_id,
                name=name,
                gt=len(class_gt),
                tp=c_tp,
                fp=c_fp,
                fn=c_fn,
                precision=c_rates["precision"],
                recall=c_rates["recall"],
                false_discovery_rate=c_rates["false_discovery_rate"],
            )
        )
    return EvalSummary(
        classes=classes,
        gt_total=len(gt),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=rates["precision"],
        recall=rates["recall"],
        false_discovery_rate=rates["false_discovery_rate"],
        per_class=per_class,
        iou_threshold=iou_threshold,
    )


@dataclass
class ThresholdPoint:
    threshold: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    false_discovery_rate: float


@dataclass
class ThresholdScan:
    points: list[ThresholdPoint] = field(default_factory=list)
    recommended: ThresholdPoint | None = None
    target_recall: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def optimize_threshold(
    ground_truths: Iterable[Box],
    predictions: Iterable[Box],
    iou_threshold: float = 0.5,
    target_recall: float = 0.95,
) -> ThresholdScan:
    """Scan confidence thresholds and recommend the operating point.

    Candidate thresholds are the prediction confidences plus a coarse grid, so
    the scan adapts to the actual score distribution. The recommended point is
    the highest-confidence threshold that still reaches ``target_recall``
    (minimizes false alarms while preserving the required detection rate).
    """
    gt = list(ground_truths)
    preds = list(predictions)
    grid = [round(0.05 * step, 2) for step in range(1, 20)]
    confidences = sorted({round(box.confidence, 4) for box in preds})
    thresholds = sorted(set(grid) | set(confidences), reverse=True)

    points: list[ThresholdPoint] = []
    recommended: ThresholdPoint | None = None
    for threshold in thresholds:
        kept = [box for box in preds if box.confidence >= threshold]
        tp, fp, fn, _ = match_boxes(gt, kept, iou_threshold)
        rates = _rates(tp, fp, fn)
        point = ThresholdPoint(
            threshold=round(threshold, 4),
            tp=tp,
            fp=fp,
            fn=fn,
            precision=rates["precision"],
            recall=rates["recall"],
            false_discovery_rate=rates["false_discovery_rate"],
        )
        points.append(point)
        if point.recall >= target_recall and (recommended is None or point.threshold > recommended.threshold):
            recommended = point
    return ThresholdScan(points=points, recommended=recommended, target_recall=target_recall)


def load_ground_truth(label_path: str | Path, *, confidence: float = 1.0) -> list[Box]:
    """Load normalized YOLO labels (``class cx cy w h``) as ``xyxy`` boxes in [0, 1]."""
    label_path = Path(label_path)
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            continue
        boxes.append(
            Box(
                class_id=class_id,
                x1=cx - width / 2,
                y1=cy - height / 2,
                x2=cx + width / 2,
                y2=cy + height / 2,
                confidence=confidence,
            )
        )
    return boxes


def evaluate_dataset(
    predictor: Any,
    data_yaml: str | Path,
    *,
    split: str = "val",
    conf: float = 0.25,
    iou_threshold: float = 0.5,
    prediction_floor: float = 0.001,
) -> dict[str, Any]:
    """Run an engine-neutral predictor over a split and build the report.

    ``predictor`` must expose ``predict(image_bytes, filename)`` and return
    DefectDock's standard detection dictionaries. This keeps evaluation free
    from any training-framework license or output contract.
    """

    from defectdock.data.cv.check import (
        _iter_image_paths,
        _label_path_for,
        _parse_names,
        _split_sources,
        load_data_yaml,
        resolve_dataset_root,
    )

    data_yaml_path = Path(data_yaml).resolve()
    config = load_data_yaml(data_yaml_path)
    classes, _ = _parse_names(config)
    root = resolve_dataset_root(data_yaml_path, config)

    raw = config.get(split)
    if raw is None:
        raise ValueError(f"data.yaml has no '{split}' split")

    if not 0.0 <= prediction_floor <= conf <= 1.0:
        raise ValueError("expected 0 <= prediction_floor <= conf <= 1")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU thresholds must be in (0, 1]")

    all_gt: list[Box] = []
    all_pred: list[Box] = []
    per_image: list[dict[str, Any]] = []

    image_paths = [
        image_path
        for source in _split_sources(root, raw)
        if source.exists()
        for image_path in _iter_image_paths(source)
    ]
    if not image_paths:
        raise ValueError(f"dataset split '{split}' contains no images")

    for image_path in image_paths:
        from PIL import Image

        image_id = str(image_path.resolve())
        label_path = _label_path_for(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        gt = [
            Box(
                class_id=box.class_id,
                x1=box.x1 * width,
                y1=box.y1 * height,
                x2=box.x2 * width,
                y2=box.y2 * height,
                confidence=1.0,
                image_id=image_id,
            )
            for box in (load_ground_truth(label_path) if label_path is not None and label_path.is_file() else [])
        ]
        result = predictor.predict(image_path.read_bytes(), image_path.name)
        preds = [
            Box(
                class_id=int(item["class_id"]),
                x1=float(item["x1"]),
                y1=float(item["y1"]),
                x2=float(item["x2"]),
                y2=float(item["y2"]),
                confidence=float(item["confidence"]),
                image_id=image_id,
            )
            for item in result.get("detections", [])
            if float(item["confidence"]) >= prediction_floor
        ]
        all_gt.extend(gt)
        all_pred.extend(preds)
        operating_preds = [box for box in preds if box.confidence >= conf]
        image_tp, image_fp, image_fn, _ = match_boxes(gt, operating_preds, iou_threshold)
        per_image.append(
            {
                "image": image_path.name,
                "gt": len(gt),
                "detections": len(operating_preds),
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
                "classes": sorted({classes[int(box.class_id)] for box in operating_preds}),
            }
        )

    operating_predictions = [box for box in all_pred if box.confidence >= conf]
    summary = summarize(classes, all_gt, operating_predictions, iou_threshold)
    summary.threshold = conf
    scan = optimize_threshold(all_gt, all_pred, iou_threshold)
    return {
        "summary": summary.to_dict(),
        "threshold_scan": scan.to_dict(),
        "images": per_image,
    }
