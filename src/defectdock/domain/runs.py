"""Training-run identity and lifecycle rules."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict


class RunStatus(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
ALLOWED_TRANSITIONS = {
    RunStatus.CREATED: {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED},
    RunStatus.QUEUED: {RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED},
    RunStatus.RUNNING: TERMINAL_STATUSES,
    RunStatus.SUCCEEDED: set(),
    RunStatus.FAILED: set(),
    RunStatus.CANCELLED: set(),
}


def ensure_transition(current: RunStatus, target: RunStatus) -> None:
    if current == target:
        return
    if target not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"Invalid run transition: {current.value} -> {target.value}")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id(project: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{project}-{timestamp}-{uuid4().hex[:8]}"


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    project: str
    task: str
    engine: str
    model: str
    dataset: str
    dataset_version: str
    config_hash: str
    config: dict
    status: RunStatus
    output_dir: str
    metrics: dict | None = None
    error: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
