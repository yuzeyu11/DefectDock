import unittest

from defectdock.eval import (
    Box,
    build_acceptance_report,
    find_misses,
    optimize_threshold,
    summarize,
)


def _box(class_id=0, x1=0.1, y1=0.1, x2=0.5, y2=0.5, confidence=1.0) -> Box:
    return Box(class_id=class_id, x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


class ReportTests(unittest.TestCase):
    def test_build_report_contains_key_sections(self):
        gt = [_box(class_id=0), _box(class_id=1, x1=0.6, y1=0.6, x2=0.9, y2=0.9)]
        preds = [
            _box(class_id=0, confidence=0.9),
            _box(class_id=1, x1=0.6, y1=0.6, x2=0.9, y2=0.9, confidence=0.8),
        ]
        summary = summarize(["cat", "dog"], gt, preds)
        scan = optimize_threshold(gt, preds, target_recall=0.9)
        miss = find_misses(gt, preds, ["cat", "dog"])
        report = build_acceptance_report(summary, scan=scan, miss_report=miss)
        self.assertIn("总体指标", report)
        self.assertIn("逐类明细", report)
        self.assertIn("阈值工作点推荐", report)
        self.assertIn("误检 / 漏检样本统计", report)
        self.assertIn("验收结论", report)
        self.assertIn("cat", report)
        self.assertIn("dog", report)

    def test_build_report_marks_fail_when_recall_low(self):
        gt = [_box(class_id=0)]
        preds = []  # 全漏检
        summary = summarize(["cat"], gt, preds)
        report = build_acceptance_report(summary, target_recall=0.95)
        self.assertIn("未达标", report)


if __name__ == "__main__":
    unittest.main()
