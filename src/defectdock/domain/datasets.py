"""Dataset domain models used by upload, annotation, and training flows."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel


class DatasetStatus(str, Enum):
    DRAFT = "draft"
    ANNOTATING = "annotating"
    FROZEN = "frozen"


class DatasetImageRecord(BaseModel):
    image_id: str
    dataset_id: str
    original_name: str
    stored_name: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    created_at: str


class AnnotationVersionRecord(BaseModel):
    annotation_version_id: str
    dataset_id: str
    source: str
    format: str
    root_dir: str
    manifest_path: str
    manifest_sha256: str
    labeled_count: int
    unlabeled_count: int
    created_at: str
    review_status: str = "approved"
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    is_current: bool = False


class TrainingSnapshotRecord(BaseModel):
    snapshot_id: str
    dataset_id: str
    annotation_version_id: str
    snapshot_sha256: str
    manifest_path: str
    manifest_sha256: str
    data_yaml: str
    image_count: int
    train_count: int
    val_count: int
    seed: int
    val_ratio: float
    created_at: str


class DatasetRecord(BaseModel):
    dataset_id: str
    name: str
    scene: str
    labels: list[str]
    status: DatasetStatus
    root_dir: str
    image_count: int
    total_bytes: int
    cvat_task_id: int | None
    created_at: str
    updated_at: str


def new_dataset_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"ds-{timestamp}-{uuid4().hex[:8]}"


def new_image_id() -> str:
    return f"img-{uuid4().hex[:16]}"
