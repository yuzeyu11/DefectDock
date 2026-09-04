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
    TrainingSnapshotRecord,
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
        review_status: str = "approved",
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
        if review_status not in {"candidate", "approved"}:
            raise ValueError("Annotation review status must be candidate or approved")
        now = utc_now()
        manifest_sha256 = _file_sha256(manifest)
        try:
            with closing(self._connect()) as connection:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO annotation_versions (
                            annotation_version_id, dataset_id, source, format, root_dir,
                            manifest_path, manifest_sha256, labeled_count, unlabeled_count,
                            review_status, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                            review_status,
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

    def approve_annotation_version(
        self,
        dataset_id: str,
        annotation_version_id: str,
        *,
        actor: str,
    ) -> AnnotationVersionRecord:
        dataset = self.get_dataset(dataset_id)
        if dataset.status == DatasetStatus.FROZEN:
            raise ValueError("Frozen datasets cannot change annotation review state")
        self.get_annotation_version(dataset_id, annotation_version_id)
        now = utc_now()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE annotation_versions
                       SET review_status = 'approved', reviewed_by = ?, reviewed_at = ?
                     WHERE dataset_id = ? AND annotation_version_id = ?
                    """,
                    (actor, now, dataset_id, annotation_version_id),
                )
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
        try:
            annotation = self.get_current_annotation_version(dataset_id)
        except KeyError as exc:
            raise ValueError("Dataset must have a current annotation version before freezing") from exc
        if annotation.review_status != "approved":
            raise ValueError("Model-generated annotations must be approved before freezing")
        if annotation.unlabeled_count:
            raise ValueError("All dataset images must be labeled before freezing")
        now = utc_now()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "UPDATE datasets SET status = ?, updated_at = ? WHERE dataset_id = ?",
                    (DatasetStatus.FROZEN.value, now, dataset_id),
                )
        return self.get_dataset(dataset_id)

    def register_training_snapshot(self, dataset_id: str, snapshot: dict) -> TrainingSnapshotRecord:
        dataset = self.get_dataset(dataset_id)
        if dataset.status != DatasetStatus.FROZEN:
            raise ValueError("Training snapshots require a frozen dataset")
        required = {
            "snapshot_id",
            "annotation_version",
            "snapshot_sha256",
            "data_yaml",
            "image_count",
            "train_count",
            "val_count",
            "seed",
            "val_ratio",
        }
        missing = sorted(required.difference(snapshot))
        if missing:
            raise ValueError(f"Snapshot manifest is missing fields: {', '.join(missing)}")
        annotation = self.get_annotation_version(dataset_id, snapshot["annotation_version"])
        data_yaml = Path(snapshot["data_yaml"]).resolve()
        snapshot_root = data_yaml.parent
        manifest_path = snapshot_root / "manifest.json"
        dataset_root = Path(dataset.root_dir).resolve()
        if not snapshot_root.is_relative_to(dataset_root) or not data_yaml.is_file():
            raise ValueError("Training snapshot is missing or outside its dataset directory")
        if not manifest_path.is_file():
            raise ValueError("Training snapshot manifest is missing")
        if annotation.review_status != "approved":
            raise ValueError("Training snapshots require approved annotations")
        now = utc_now()
        manifest_sha256 = _file_sha256(manifest_path)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO training_snapshots (
                        snapshot_id, dataset_id, annotation_version_id, snapshot_sha256,
                        manifest_path, manifest_sha256, data_yaml, image_count, train_count,
                        val_count, seed, val_ratio, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(snapshot_id) DO NOTHING
                    """,
                    (
                        snapshot["snapshot_id"],
                        dataset_id,
                        annotation.annotation_version_id,
                        snapshot["snapshot_sha256"],
                        str(manifest_path),
                        manifest_sha256,
                        str(data_yaml),
                        snapshot["image_count"],
                        snapshot["train_count"],
                        snapshot["val_count"],
                        snapshot["seed"],
                        snapshot["val_ratio"],
                        now,
                    ),
                )
        record = self.get_training_snapshot(dataset_id, snapshot["snapshot_id"])
        if record.snapshot_sha256 != snapshot["snapshot_sha256"]:
            raise ValueError("Snapshot ID already refers to different content")
        self.verify_training_snapshot(record)
        return record

    def get_training_snapshot(
        self, dataset_id: str, snapshot_id: str
    ) -> TrainingSnapshotRecord:
        self.get_dataset(dataset_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM training_snapshots WHERE dataset_id = ? AND snapshot_id = ?",
                (dataset_id, snapshot_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Training snapshot not found: {snapshot_id}")
        return self._training_snapshot_record(row)

    def list_training_snapshots(
        self, dataset_id: str, limit: int = 100
    ) -> list[TrainingSnapshotRecord]:
        self.get_dataset(dataset_id)
        limit = max(1, min(limit, 1000))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM training_snapshots
                 WHERE dataset_id = ?
                 ORDER BY created_at DESC, snapshot_id DESC LIMIT ?
                """,
                (dataset_id, limit),
            ).fetchall()
        return [self._training_snapshot_record(row) for row in rows]

    @staticmethod
    def verify_training_snapshot(snapshot: TrainingSnapshotRecord) -> None:
        manifest = Path(snapshot.manifest_path)
        data_yaml = Path(snapshot.data_yaml)
        if not manifest.is_file() or _file_sha256(manifest) != snapshot.manifest_sha256:
            raise ValueError(f"Training snapshot manifest integrity failed: {snapshot.snapshot_id}")
        snapshot_root = manifest.parent.resolve()
        if not data_yaml.is_file() or not data_yaml.resolve().is_relative_to(snapshot_root):
            raise ValueError(f"Training snapshot data.yaml is missing: {snapshot.snapshot_id}")
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Training snapshot manifest is invalid: {snapshot.snapshot_id}"
            ) from exc
        expected_snapshot_hash = payload.pop("snapshot_sha256", None)
        computed_snapshot_hash = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if (
            expected_snapshot_hash != snapshot.snapshot_sha256
            or computed_snapshot_hash != snapshot.snapshot_sha256
        ):
            raise ValueError(f"Training snapshot content hash failed: {snapshot.snapshot_id}")
        for item in payload.get("images", []):
            for path_key, hash_key in (("image", "image_sha256"), ("label", "label_sha256")):
                candidate = (snapshot_root / item[path_key]).resolve()
                if (
                    not candidate.is_relative_to(snapshot_root)
                    or not candidate.is_file()
                    or _file_sha256(candidate) != item[hash_key]
                ):
                    raise ValueError(
                        f"Training snapshot file integrity failed: {snapshot.snapshot_id}"
                    )

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
            review_status=row["review_status"],
            reviewed_by=row["reviewed_by"],
            reviewed_at=row["reviewed_at"],
            created_at=row["created_at"],
            is_current=bool(row["is_current"]),
        )

    @staticmethod
    def _training_snapshot_record(row: sqlite3.Row) -> TrainingSnapshotRecord:
        return TrainingSnapshotRecord(
            snapshot_id=row["snapshot_id"],
            dataset_id=row["dataset_id"],
            annotation_version_id=row["annotation_version_id"],
            snapshot_sha256=row["snapshot_sha256"],
            manifest_path=row["manifest_path"],
            manifest_sha256=row["manifest_sha256"],
            data_yaml=row["data_yaml"],
            image_count=row["image_count"],
            train_count=row["train_count"],
            val_count=row["val_count"],
            seed=row["seed"],
            val_ratio=row["val_ratio"],
            created_at=row["created_at"],
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
