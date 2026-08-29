import json
import tempfile
import unittest
from pathlib import Path

from defectdock.data.cv import file_sha256, ingest_feedback


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_image(self, name: str, content: bytes = b"img-content") -> Path:
        path = self.root / name
        path.write_bytes(content)
        return path

    def test_ingest_deduplicates_and_writes_manifest(self):
        img = self._make_image("a.jpg", b"same-bytes")
        feedback_dir = self.root / "feedback"
        samples = [
            {"image_path": str(img), "kind": "miss", "class_name": "冲孔", "bbox": [0.1, 0.1, 0.3, 0.3], "source": "field-1"},
            {"image_path": str(img), "kind": "false_positive", "class_name": "水斑", "bbox": [0.5, 0.5, 0.7, 0.7]},
        ]
        result = ingest_feedback(samples, feedback_dir)
        self.assertEqual(len(result.accepted), 2)
        # 同一张图片去重，只复制一次
        self.assertEqual(result.duplicate_images, 1)
        images = list((feedback_dir / "images").glob("*.jpg"))
        self.assertEqual(len(images), 1)
        manifest = json.loads((feedback_dir / "feedback.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["count"], 2)
        self.assertEqual(len(manifest["sha256"]), 1)

    def test_file_sha256_stable(self):
        img = self._make_image("a.jpg", b"hello")
        self.assertEqual(file_sha256(img), file_sha256(img))

    def test_ingest_skips_missing_image(self):
        result = ingest_feedback(
            [{"image_path": str(self.root / "missing.jpg"), "kind": "miss", "class_name": "x"}],
            self.root / "fb",
        )
        self.assertEqual(result.skipped_missing_image, 1)
        self.assertEqual(len(result.accepted), 0)

    def test_invalid_bbox_is_skipped_and_audited(self):
        # M4 回归：提供非法 bbox（3 元组 / 越界 / 非数值 / 零面积）时，
        # 该样本跳过并计入审计，同批合法样本不受影响，不再抛未处理异常。
        img = self._make_image("a.jpg", b"content-a")
        fb = self.root / "fb"
        samples = [
            {"image_path": str(img), "kind": "miss", "class_name": "冲孔", "bbox": [0.1, 0.1, 0.3]},
            {"image_path": str(img), "kind": "miss", "class_name": "水斑", "bbox": [0.1, 0.1, 1.5, 0.3]},
            {"image_path": str(img), "kind": "miss", "class_name": "油斑", "bbox": ["a", 0.1, 0.3, 0.3]},
            {"image_path": str(img), "kind": "miss", "class_name": "夹杂", "bbox": [0.1, 0.1, 0.1, 0.3]},
        ]
        result = ingest_feedback(samples, fb)
        self.assertEqual(result.skipped_invalid_bbox, 4)
        self.assertEqual(len(result.accepted), 0)
        self.assertEqual(len(list((fb / "images").glob("*.jpg"))), 0)

        valid = {"image_path": str(img), "kind": "false_positive", "class_name": "焊缝", "bbox": [0.2, 0.2, 0.8, 0.8]}
        mixed = ingest_feedback([samples[0], valid], self.root / "fb2")
        self.assertEqual(mixed.skipped_invalid_bbox, 1)
        self.assertEqual(len(mixed.accepted), 1)
        self.assertEqual(mixed.accepted[0].bbox, (0.2, 0.2, 0.8, 0.8))

    def test_reingest_is_idempotent(self):
        img = self._make_image("a.jpg", b"content")
        fb = self.root / "fb"
        sample = {"image_path": str(img), "kind": "miss", "class_name": "冲孔"}
        first = ingest_feedback([sample], fb)
        second = ingest_feedback([sample], fb)
        self.assertEqual(len(first.accepted), 1)
        # 重复样本按 (sha256, kind, class) 去重，manifest 不重复膨胀
        self.assertEqual(len(second.accepted), 1)
        # 第二次重复图片，不新增文件
        self.assertEqual(len(list((fb / "images").glob("*.jpg"))), 1)
        manifest = json.loads((fb / "feedback.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["count"], 1)
        self.assertEqual(len(manifest["samples"]), 1)

    def test_manifest_preserves_history_across_batches(self):
        # H2 回归：第二批复写 manifest 时不得丢失第一轮的样本记录。
        img_a = self._make_image("a.jpg", b"content-a")
        img_b = self._make_image("b.jpg", b"content-b")
        fb = self.root / "fb"
        batch1 = [
            {"image_path": str(img_a), "kind": "miss", "class_name": "冲孔", "source": "field-1"},
        ]
        first = ingest_feedback(batch1, fb)
        self.assertEqual(len(first.accepted), 1)

        batch2 = [
            {"image_path": str(img_b), "kind": "false_positive", "class_name": "水斑", "source": "field-2"},
            {"image_path": str(img_a), "kind": "miss", "class_name": "冲孔", "source": "field-2"},
        ]
        second = ingest_feedback(batch2, fb)

        manifest = json.loads((fb / "feedback.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["count"], 2)
        self.assertEqual(len(manifest["samples"]), 2)
        self.assertEqual(len(manifest["sha256"]), 2)
        names = {record["image_name"] for record in manifest["samples"]}
        self.assertEqual(len(names), 2)
        # 第一批的来源信息仍然可追溯
        first_batch = [record for record in manifest["samples"] if record["source"] == "field-1"]
        self.assertEqual(len(first_batch), 1)
        self.assertEqual(second.duplicate_images, 1)


if __name__ == "__main__":
    unittest.main()
