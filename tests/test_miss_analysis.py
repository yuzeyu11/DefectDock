import tempfile
import unittest
from pathlib import Path

from defectdock.eval import Box, find_misses, group_by_class


def _box(class_id=0, x1=0.1, y1=0.1, x2=0.5, y2=0.5, confidence=1.0) -> Box:
    return Box(class_id=class_id, x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


class MissAnalysisTests(unittest.TestCase):
    def test_perfect_match_has_no_misses(self):
        report = find_misses([_box()], [_box()], ["cat"])
        self.assertEqual(report.summary["missed"], 0)
        self.assertEqual(report.summary["false_positive"], 0)
        self.assertEqual(report.summary["true_positive"], 1)

    def test_missed_ground_truth(self):
        gt = [_box()]
        preds = [_box(x1=0.8, y1=0.8, x2=0.9, y2=0.9)]  # 不重叠
        report = find_misses(gt, preds, ["cat"])
        self.assertEqual(report.summary["missed"], 1)
        self.assertEqual(report.summary["false_positive"], 1)
        self.assertEqual(report.missed[0].kind, "miss")
        self.assertEqual(report.missed[0].class_name, "cat")
        self.assertEqual(report.false_positives[0].kind, "false_positive")

    def test_low_confidence_hit(self):
        gt = [_box()]
        preds = [_box(confidence=0.3)]  # 命中但置信度低
        report = find_misses(gt, preds, ["cat"], low_conf_threshold=0.5)
        self.assertEqual(report.summary["low_confidence"], 1)
        self.assertEqual(report.low_confidence[0].kind, "low_confidence")

    def test_never_matches_boxes_from_different_images(self):
        truth = _box()
        truth.image_id = "image-a"
        prediction = _box()
        prediction.image_id = "image-b"
        report = find_misses([truth], [prediction], ["cat"])
        self.assertEqual(report.summary["true_positive"], 0)
        self.assertEqual(report.summary["missed"], 1)
        self.assertEqual(report.summary["false_positive"], 1)
        self.assertEqual(report.missed[0].image, "image-a")
        self.assertEqual(report.false_positives[0].image, "image-b")

    def test_accepts_iou_equal_to_threshold(self):
        truth = _box(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        prediction = _box(x1=0.0, y1=0.0, x2=1.0, y2=0.5)
        report = find_misses([truth], [prediction], ["cat"], iou_threshold=0.5)
        self.assertEqual(report.summary["true_positive"], 1)
        self.assertEqual(report.summary["missed"], 0)

    def test_group_by_class(self):
        report = find_misses(
            [_box(class_id=0), _box(class_id=1, x1=0.7, y1=0.7, x2=0.9, y2=0.9)],
            [],
            ["cat", "dog"],
        )
        counts = group_by_class(report.missed)
        self.assertEqual(counts, {"cat": 1, "dog": 1})

    def test_export_manifest(self):
        from defectdock.eval import export_miss_manifest

        report = find_misses([_box()], [], ["cat"])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "misses.json"
            export_miss_manifest(report, path)
            self.assertTrue(path.is_file())
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["missed"], 1)


if __name__ == "__main__":
    unittest.main()
