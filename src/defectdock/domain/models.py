"""Registered model versions and activation audit records."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelVersionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_version_id: str
    run_id: str
    project: str
    task: str
    engine: str
    architecture: str
    dataset: str
    dataset_version: str
    config_hash: str
    artifact_path: str
    artifact_sha256: str
    artifact_size: int
    run_manifest_path: str | None = None
    run_manifest_sha256: str | None = None
    metrics: dict | None = None
    created_by: str
    created_at: str
    approval_status: str = "candidate"
    approved_by: str | None = None
    approved_at: str | None = None
    is_active: bool = False


class ModelActivationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activation_id: str
    action: str
    model_version_id: str
    previous_model_version_id: str | None = None
    actor: str
    created_at: str
