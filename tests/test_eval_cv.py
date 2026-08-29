import tempfile
import unittest
from pathlib import Path

from defectdock.eval import (
    Box,
    iou,
    load_ground_truth,
    match_boxes,
    optimize_threshold,
    summarize,
)


def _box(class_id=0, x1=0.1, y1=0.1, x2=0.5, y2=0.5, confidence=1.0) -> Box:
    return Box(class_id=class_id, x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


class EvalMetricTests(unittest.TestCase):
    def test_iou_full_overlap(self):
        self.assertAlmostEqual(iou(_box(), _box()), 1.0)

    def test_iou_no_overlap(self):
        a = _box(x1=0.0, y1=0.0, x2=0.1, y2=0.1)
        b = _box(x1=0.9, y1=0.9, x2=1.0, y2=1.0)
        self.assertEqual(iou(a, b), 0.0)

    def test_iou_partial_overlap(self):
        a = _box(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
        b = _box(x1=0.25, y1=0.25, x2=0.75, y2=0.75)
        # intersection = 0.25*0.25 = 0.0625; union = 0.25 + 0.25 - 0.0625 = 0.4375
        self.assertAlmostEqual(iou(a, b), 0.0625 / 0.4375, places=6)

    def test_match_boxes_perfect(self):
        gt = [_box()]
        preds = [_box()]
        tp, fp, fn, _ = match_boxes(gt, preds)
        self.assertEqual((tp, fp, fn), (1, 0, 0))

    def test_match_boxes_wrong_class_is_false_positive(self):
        gt = [_box(class_id=0)]
        preds = [_box(class_id=1)]
        tp, fp, fn, _ = match_boxes(gt, preds)
        self.assertEqual((tp, fp, fn), (0, 1, 1))

    def test_match_boxes_missed_detection_is_false_negative(self):
        gt = [_box()]
        preds = [_box(x1=0.8, y1=0.8, x2=0.9, y2=0.9)]
        tp, fp, fn, _ = match_boxes(gt, preds)
        self.assertEqual((tp, fp, fn), (0, 1, 1))

    def test_match_boxes_accepts_iou_equal_to_threshold(self):
        gt = [_box(x1=0.0, y1=0.0, x2=1.0, y2=1.0)]
        prediction = _box(x1=0.0, y1=0.0, x2=1.0, y2=0.5)
        tp, fp, fn, _ = match_boxes(gt, [prediction], iou_threshold=0.5)
        self.assertEqual((tp, fp, fn), (1, 0, 0))

    def test_match_boxes_never_matches_across_images(self):
        gt = [_box()]
        gt[0].image_id = "image-a"
        preds = [_box()]
        preds[0].image_id = "image-b"
        tp, fp, fn, _ = match_boxes(gt, preds)
        self.assertEqual((tp, fp, fn), (0, 1, 1))

    def test_summarize_rates(self):
        gt = [_box(class_id=0)]
        preds = [_box(class_id=0, confidence=0.9), _box(class_id=0, x1=0.7, y1=0.7, x2=0.9, y2=0.9, confidence=0.8)]
        summary = summarize(["cat", "dog"], gt, preds)
        self.assertEqual(summary.tp, 1)
        self.assertEqual(summary.fp, 1)
        self.assertEqual(summary.fn, 0)
        self.assertAlmostEqual(summary.recall, 1.0)
        self.assertAlmostEqual(summary.precision, 0.5)
        self.assertAlmostEqual(summary.false_discovery_rate, 0.5)

    def test_optimize_threshold_recommends_recall_point(self):
        gt = [_box()]
        preds = [
            _box(confidence=0.9),  # true positive
            _box(x1=0.7, y1=0.7, x2=0.9, y2=0.9, confidence=0.3),  # false positive
        ]
        scan = optimize_threshold(gt, preds, target_recall=0.9)
        self.assertIsNotNone(scan.recommended)
        self.assertGreaterEqual(scan.recommended.recall, 0.9)
        self.assertLessEqual(scan.recommended.false_discovery_rate, 1.0)

    def test_load_ground_truth_converts_to_xyxy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            label = Path(temp_dir) / "a.txt"
            label.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            boxes = load_ground_truth(label)
            self.assertEqual(len(boxes), 1)
            box = boxes[0]
            self.assertEqual(box.class_id, 0)
            self.assertAlmostEqual(box.x1, 0.4)
            self.assertAlmostEqual(box.y1, 0.3)
            self.assertAlmostEqual(box.x2, 0.6)
            self.assertAlmostEqual(box.y2, 0.7)


if __name__ == "__main__":
    unittest.main()
