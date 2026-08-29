import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from defectdock.data.cv import (
    GC10_CLASS_NAMES,
    GC10_SOURCE_LABELS,
    check_dataset,
    discover_gc10_samples,
    import_gc10_dataset,
)


class Gc10ImportTests(unittest.TestCase):
    def test_imports_complete_tree_with_stable_splits_and_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "GC10-DET"
            self._build_source(source, per_folder=3, add_unknown=True)
            shutil.copy2(
                source / "1" / "img_01_00.jpg",
                source / "2" / "img_01_00.jpg",
            )
            output = root / "yolo"

            result = import_gc10_dataset(
                source,
                output,
                seed=7,
                val_ratio=0.2,
                test_ratio=0.2,
            )

            self.assertEqual(result["classes"], GC10_CLASS_NAMES)
            self.assertEqual(result["audit"]["source_image_count"], 31)
            self.assertEqual(result["audit"]["unique_image_count"], 30)
            self.assertEqual(result["audit"]["duplicate_image_count"], 1)
            self.assertEqual(result["audit"]["imported_image_count"], 30)
            self.assertEqual(result["audit"]["unknown_labels"], {"d": 1})
            self.assertEqual(
                {name: details["images"] for name, details in result["splits"].items()},
                {"train": 10, "val": 10, "test": 10},
            )
            self.assertTrue((output / "manifest.json").is_file())
            report = check_dataset(output / "data.yaml")
            self.assertTrue(report.ok, report.to_dict())

    def test_missing_xml_is_skipped_unless_explicitly_included(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "GC10-DET"
            self._build_source(source, per_folder=3)
            missing_xml = source / "lable" / "img_01_00.xml"
            missing_xml.unlink()

            samples, audit = discover_gc10_samples(source)
            self.assertEqual(len(samples), 29)
            self.assertEqual(audit["missing_annotation_count"], 1)

            included, included_audit = discover_gc10_samples(
                source, include_unannotated=True
            )
            self.assertEqual(len(included), 30)
            self.assertTrue(included_audit["included_unannotated"])
            self.assertEqual(sum(not sample.boxes for sample in included), 1)

    def test_strict_mode_rejects_unknown_source_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "GC10-DET"
            self._build_source(source, per_folder=1, add_unknown=True)
            with self.assertRaisesRegex(ValueError, "Unknown GC10 label"):
                discover_gc10_samples(source, strict_unknown_labels=True)

    @staticmethod
    def _build_source(source: Path, *, per_folder: int, add_unknown: bool = False) -> None:
        annotation_dir = source / "lable"
        annotation_dir.mkdir(parents=True)
        source_names_by_folder = {
            int(raw_name.split("_", 1)[0]): raw_name
            for raw_name in GC10_SOURCE_LABELS
            if raw_name != "10_yaozhed"
        }
        for folder in range(1, 11):
            image_dir = source / str(folder)
            image_dir.mkdir()
            for index in range(per_folder):
                stem = f"img_{folder:02d}_{index:02d}"
                Image.new("L", (64, 48), color=100 + folder).save(
                    image_dir / f"{stem}.jpg"
                )
                objects = [
                    Gc10ImportTests._object_xml(
                        source_names_by_folder[folder], 4, 5, 24, 25
                    )
                ]
                if add_unknown and folder == 2 and index == 0:
                    objects.append(Gc10ImportTests._object_xml("d", 30, 10, 40, 20))
                xml = (
                    "<annotation>"
                    f"<filename>{stem}.jpg</filename>"
                    "<size><width>64</width><height>48</height><depth>1</depth></size>"
                    + "".join(objects)
                    + "</annotation>"
                )
                (annotation_dir / f"{stem}.xml").write_text(xml, encoding="utf-8")

    @staticmethod
    def _object_xml(name: str, xmin: int, ymin: int, xmax: int, ymax: int) -> str:
        return (
            f"<object><name>{name}</name><bndbox><xmin>{xmin}</xmin>"
            f"<ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax>"
            "</bndbox></object>"
        )


if __name__ == "__main__":
    unittest.main()
