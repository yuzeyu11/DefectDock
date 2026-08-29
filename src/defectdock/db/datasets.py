"""SQLite metadata store for uploaded image datasets."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from defectdock.db.migration import upgrade_database
from defectdock.domain import (
    AnnotationVersionRecord,
    DatasetImageRecord,
    DatasetRecord,
    DatasetStatus,
    new_dataset_id,
    new_image_id,
)
from defectdock.domain.runs import utc_now


class DuplicateImageError(ValueError):
    pass


class DatasetStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        upgrade_database(self.path)

    def create_dataset(
        self,
        name: str,
        scene: str,
        labels: list[str],
        root_dir: str | Path,
        *,
        dataset_id: str | None = None,
    ) -> DatasetRecord:
        dataset_id = dataset_id or new_dataset_id()
        now = utc_now()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO datasets (
                        dataset_id, name, scene, labels_json, status, root_dir,
                        image_count, total_bytes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
                    """,
                    (
                        dataset_id,
                        name,
                        scene,
                        json.dumps(labels, ensure_ascii=False),
                        DatasetStatus.DRAFT.value,
                        str(Path(root_dir).resolve()),
                        now,
                        now,
                    ),
                )
        return self.get_dataset(dataset_id)

    def add_image(
        self,
        dataset_id: str,
        *,
        original_name: str,
        stored_name: str,
        sha256: str,
        size_bytes: int,
        width: int,
        height: int,
    ) -> DatasetImageRecord:
        dataset = self.get_dataset(dataset_id)
        if dataset.status == DatasetStatus.FROZEN:
            raise ValueError("Frozen datasets cannot accept new images")
        image_id = new_image_id()
        now = utc_now()
        try:
            with closing(self._connect()) as connection:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO dataset_images (
                            image_id, dataset_id, original_name, stored_name, sha256,
                            size_bytes, width, height, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            image_id,
                            dataset_id,
                            original_name,
                            stored_name,
                            sha256,
                            size_bytes,
                            width,
                            height,
                            now,
                        ),
                    )
                    connection.execute(
                        """
                        UPDATE datasets
                           SET image_count = image_count + 1,
                               total_bytes = total_bytes + ?, updated_at = ?
                         WHERE dataset_id = ?
                        """,
                        (size_bytes, now, dataset_id),
                    )
        except sqlite3.IntegrityError as exc:
            raise DuplicateImageError(f"Duplicate image in dataset: {original_name}") from exc
        return self.get_image(dataset_id, image_id)

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Dataset not found: {dataset_id}")
        return self._dataset_record(row)

    def list_datasets(self, limit: int = 50) -> list[DatasetRecord]:
        limit = max(1, min(limit, 1000))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM datasets ORDER BY created_at DESC, dataset_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._dataset_record(row) for row in rows]

    def get_image(self, dataset_id: str, image_id: str) -> DatasetImageRecord:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM dataset_images WHERE dataset_id = ? AND image_id = ?",
                (dataset_id, image_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Image not found: {image_id}")
        return self._image_record(row)

    def list_images(self, dataset_id: str, limit: int = 1000) -> list[DatasetImageRecord]:
        self.get_dataset(dataset_id)
        limit = max(1, min(limit, 5000))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM dataset_images
                 WHERE dataset_id = ? ORDER BY created_at, image_id LIMIT ?
                """,
                (dataset_id, limit),
            ).fetchall()
        return [self._image_record(row) for row in rows]

    def attach_cvat_task(self, dataset_id: str, task_id: int) -> DatasetRecord:
        dataset = self.get_dataset(dataset_id)
        if dataset.cvat_task_id is not None:
            raise ValueError(f"Dataset already has CVAT task {dataset.cvat_task_id}")
        if dataset.status == DatasetStatus.FROZEN:
            raise ValueError("Frozen datasets cannot start annotation")
        now = utc_now()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE datasets SET cvat_task_id = ?, status = ?, updated_at = ?
                     WHERE dataset_id = ?
                    """,
                    (task_id, DatasetStatus.ANNOTATING.value, now, dataset_id),
                )
        return self.get_dataset(dataset_id)

    def register_annotation_version(
        self,
        dataset_id: str,
        annotation_version_id: str,
        *,
        source: str,
        format: str,
        root_dir: str | Path,
        manifest_path: str | Path,
        labeled_count: int,
        unlabeled_count: int,
        make_current: bool = True,
    ) -> AnnotationVersionRecord:
        dataset = self.get_dataset(dataset_id)
        if dataset.status == DatasetStatus.FROZEN:
            raise ValueError("Frozen datasets cannot accept annotation versions")
        version_root = Path(root_dir).resolve()
        manifest = Path(manifest_path).resolve()
        dataset_root = Path(dataset.root_dir).resolve()
        if not version_root.is_relative_to(dataset_root):
            raise ValueError("Annotation version must stay inside its dataset directory")
        if not manifest.is_relative_to(version_root) or not manifest.is_file():
            raise ValueError("Annotation manifest is missing or outside its version directory")
        if labeled_count < 0 or unlabeled_count < 0:
            raise ValueError("Annotation counts cannot be negative")
        now = utc_now()
        manifest_sha256 = _file_sha256(manifest)
        try:
            with closing(self._connect()) as connection:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO annotation_versions (
                            annotation_version_id, dataset_id, source, format, root_dir,
                            manifest_path, manifest_sha256, labeled_count, unlabeled_count, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            annotation_version_id,
                            dataset_id,
                            source,
                            format,
                            str(version_root),
                            str(manifest),
                            manifest_sha256,
                            labeled_count,
                            unlabeled_count,
                            now,
                        ),
                    )
                    if make_current:
                        connection.execute(
                            """
                            INSERT INTO dataset_annotation_heads (
                                dataset_id, annotation_version_id, updated_at
                            ) VALUES (?, ?, ?)
                            ON CONFLICT(dataset_id) DO UPDATE SET
                                annotation_version_id = excluded.annotation_version_id,
                                updated_at = excluded.updated_at
                            """,
                            (dataset_id, annotation_version_id, now),
                        )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "Annotation version ID or manifest already exists for this dataset"
            ) from exc
        return self.get_annotation_version(dataset_id, annotation_version_id)

    def get_annotation_version(
        self, dataset_id: str, annotation_version_id: str
    ) -> AnnotationVersionRecord:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT av.*, CASE WHEN dah.annotation_version_id IS NULL THEN 0 ELSE 1 END AS is_current
                  FROM annotation_versions av
                  LEFT JOIN dataset_annotation_heads dah
                    ON dah.dataset_id = av.dataset_id
                   AND dah.annotation_version_id = av.annotation_version_id
                 WHERE av.dataset_id = ? AND av.annotation_version_id = ?
                """,
                (dataset_id, annotation_version_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Annotation version not found: {annotation_version_id}")
        return self._annotation_version_record(row)

    def get_current_annotation_version(self, dataset_id: str) -> AnnotationVersionRecord:
        self.get_dataset(dataset_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT av.*, 1 AS is_current
                  FROM dataset_annotation_heads dah
                  JOIN annotation_versions av
                    ON av.dataset_id = dah.dataset_id
                   AND av.annotation_version_id = dah.annotation_version_id
                 WHERE dah.dataset_id = ?
                """,
                (dataset_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Dataset has no current annotation version: {dataset_id}")
        return self._annotation_version_record(row)

    def list_annotation_versions(
        self, dataset_id: str, limit: int = 100
    ) -> list[AnnotationVersionRecord]:
        self.get_dataset(dataset_id)
        limit = max(1, min(limit, 1000))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT av.*, CASE WHEN dah.annotation_version_id IS NULL THEN 0 ELSE 1 END AS is_current
                  FROM annotation_versions av
                  LEFT JOIN dataset_annotation_heads dah
                    ON dah.dataset_id = av.dataset_id
                   AND dah.annotation_version_id = av.annotation_version_id
                 WHERE av.dataset_id = ?
                 ORDER BY av.created_at DESC, av.annotation_version_id DESC
                 LIMIT ?
                """,
                (dataset_id, limit),
            ).fetchall()
        return [self._annotation_version_record(row) for row in rows]

    def set_current_annotation_version(
        self, dataset_id: str, annotation_version_id: str
    ) -> AnnotationVersionRecord:
        dataset = self.get_dataset(dataset_id)
        version = self.get_annotation_version(dataset_id, annotation_version_id)
        if dataset.status == DatasetStatus.FROZEN:
            try:
                current = self.get_current_annotation_version(dataset_id)
            except KeyError:
                current = None
            if current is None or current.annotation_version_id != annotation_version_id:
                raise ValueError("Frozen datasets cannot change their current annotation version")
            return current
        now = utc_now()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO dataset_annotation_heads (
                        dataset_id, annotation_version_id, updated_at
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(dataset_id) DO UPDATE SET
                        annotation_version_id = excluded.annotation_version_id,
                        updated_at = excluded.updated_at
                    """,
                    (dataset_id, version.annotation_version_id, now),
                )
        return self.get_annotation_version(dataset_id, annotation_version_id)

    def freeze_dataset(self, dataset_id: str) -> DatasetRecord:
        dataset = self.get_dataset(dataset_id)
        if dataset.image_count == 0:
            raise ValueError("Cannot freeze an empty dataset")
        if dataset.status == DatasetStatus.FROZEN:
            return dataset
        now = utc_now()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "UPDATE datasets SET status = ?, updated_at = ? WHERE dataset_id = ?",
                    (DatasetStatus.FROZEN.value, now, dataset_id),
                )
        return self.get_dataset(dataset_id)

    def delete_dataset(self, dataset_id: str) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))

    @staticmethod
    def _dataset_record(row: sqlite3.Row) -> DatasetRecord:
        return DatasetRecord(
            dataset_id=row["dataset_id"],
            name=row["name"],
            scene=row["scene"],
            labels=json.loads(row["labels_json"]),
            status=DatasetStatus(row["status"]),
            root_dir=row["root_dir"],
            image_count=row["image_count"],
            total_bytes=row["total_bytes"],
            cvat_task_id=row["cvat_task_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
    @staticmethod
    def _image_record(row: sqlite3.Row) -> DatasetImageRecord:
        return DatasetImageRecord(
            image_id=row["image_id"],
            dataset_id=row["dataset_id"],
            original_name=row["original_name"],
            stored_name=row["stored_name"],
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
            width=row["width"],
            height=row["height"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _annotation_version_record(row: sqlite3.Row) -> AnnotationVersionRecord:
        return AnnotationVersionRecord(
            annotation_version_id=row["annotation_version_id"],
            dataset_id=row["dataset_id"],
            source=row["source"],
            format=row["format"],
            root_dir=row["root_dir"],
            manifest_path=row["manifest_path"],
            manifest_sha256=row["manifest_sha256"],
            labeled_count=row["labeled_count"],
            unlabeled_count=row["unlabeled_count"],
            created_at=row["created_at"],
            is_current=bool(row["is_current"]),
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
