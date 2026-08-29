import tempfile
import unittest
from pathlib import Path

from defectdock.data.cv import (
    annotation_to_yolo_lines,
    box_to_yolo,
    build_class_map,
    convert_voc_dataset,
    parse_voc_xml,
)

SAMPLE_XML = """<annotation>
  <folder>3</folder>
  <filename>img_01.jpg</filename>
  <size><width>2048</width><height>1000</height><depth>1</depth></size>
  <object><name>3_yueyawan</name><bndbox><xmin>1738</xmin><ymin>806</ymin><xmax>1948</xmax><ymax>993</ymax></bndbox></object>
</annotation>
"""


class VocConvertTests(unittest.TestCase):
    def test_parse_voc_xml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            xml = Path(temp_dir) / "a.xml"
            xml.write_text(SAMPLE_XML, encoding="utf-8")
            ann = parse_voc_xml(xml)
            self.assertEqual(ann.filename, "img_01.jpg")
            self.assertEqual((ann.width, ann.height), (2048, 1000))
            self.assertEqual(len(ann.objects), 1)
            self.assertEqual(ann.objects[0].name, "3_yueyawan")

    def test_box_to_yolo_normalizes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            xml = Path(temp_dir) / "a.xml"
            xml.write_text(SAMPLE_XML, encoding="utf-8")
            ann = parse_voc_xml(xml)
            cx, cy, w, h = box_to_yolo(ann.objects[0], ann.width, ann.height)
            self.assertAlmostEqual(cx, (1738 + 1948) / 2 / 2048, places=5)
            self.assertAlmostEqual(cy, (806 + 993) / 2 / 1000, places=5)
            self.assertAlmostEqual(w, (1948 - 1738) / 2048, places=5)
            self.assertAlmostEqual(h, (993 - 806) / 1000, places=5)

    def test_annotation_to_yolo_lines(self):
        class_map = build_class_map(["3_yueyawan"])
        with tempfile.TemporaryDirectory() as temp_dir:
            xml = Path(temp_dir) / "a.xml"
            xml.write_text(SAMPLE_XML, encoding="utf-8")
            lines = annotation_to_yolo_lines(parse_voc_xml(xml), class_map)
        self.assertEqual(len(lines), 1)
        parts = lines[0].split()
        self.assertEqual(parts[0], "0")
        self.assertEqual(len(parts), 5)

    def test_convert_voc_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            xml_dir = root / "xml"
            out_dir = root / "out"
            image_dir.mkdir()
            xml_dir.mkdir()
            (image_dir / "img_01.jpg").write_bytes(b"fake-jpeg")
            (image_dir / "img_02.jpg").write_bytes(b"fake-jpeg")
            (xml_dir / "img_01.xml").write_text(SAMPLE_XML, encoding="utf-8")
            # img_02 has no XML -> skipped
            result = convert_voc_dataset(
                image_dir, xml_dir, out_dir, class_names=["3_yueyawan"]
            )
            self.assertEqual(result["image_count"], 1)
            self.assertEqual(result["label_count"], 1)
            self.assertEqual(result["skipped_no_xml"], 1)
            self.assertEqual(result["class_map"], {"3_yueyawan": 0})
            self.assertTrue((out_dir / "labels" / "img_01.txt").is_file())
            self.assertTrue((out_dir / "images" / "img_01.jpg").is_file())

    def test_rerun_overwrites_stale_outputs(self):
        # M3 回归：源图片/标签变化后重跑，输出必须同步更新，不能残留旧图/旧标。
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            xml_dir = root / "xml"
            out_dir = root / "out"
            image_dir.mkdir()
            xml_dir.mkdir()
            (image_dir / "img_01.jpg").write_bytes(b"old-image-bytes")
            (xml_dir / "img_01.xml").write_text(SAMPLE_XML, encoding="utf-8")
            convert_voc_dataset(image_dir, xml_dir, out_dir, class_names=["3_yueyawan"])
            self.assertEqual((out_dir / "images" / "img_01.jpg").read_bytes(), b"old-image-bytes")

            (image_dir / "img_01.jpg").write_bytes(b"new-image-bytes")
            convert_voc_dataset(image_dir, xml_dir, out_dir, class_names=["3_yueyawan"])
            self.assertEqual((out_dir / "images" / "img_01.jpg").read_bytes(), b"new-image-bytes")


if __name__ == "__main__":
    unittest.main()
