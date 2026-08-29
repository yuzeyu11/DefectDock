"""SQLite metadata store for uploaded image datasets."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from defectdock.domain import (
    DatasetImageRecord,
    DatasetRecord,
    DatasetStatus,
    new_dataset_id,
    new_image_id,
)
from defectdock.domain.runs import utc_now

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    scene TEXT NOT NULL,
    labels_json TEXT NOT NULL,
    status TEXT NOT NULL,
    root_dir TEXT NOT NULL,
    image_count INTEGER NOT NULL DEFAULT 0,
    total_bytes INTEGER NOT NULL DEFAULT 0,
    cvat_task_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at DESC);
CREATE TABLE IF NOT EXISTS dataset_images (
    image_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    original_name TEXT NOT NULL,
    stored_name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(dataset_id, sha256),
    UNIQUE(dataset_id, stored_name)
);
CREATE INDEX IF NOT EXISTS idx_dataset_images_dataset ON dataset_images(dataset_id, created_at);
"""


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
        with closing(self._connect()) as connection:
            connection.executescript(SCHEMA)
            connection.commit()

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
