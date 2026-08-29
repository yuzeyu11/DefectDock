"""Strict configuration schema for reproducible DefectDock training runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictModel):
    path: str = Field(min_length=1, description="Path to a detection data.yaml file")
    version: str = Field(default="unversioned", min_length=1)


class TrainConfig(StrictModel):
    epochs: int = Field(default=20, ge=1, le=10_000)
    imgsz: int = Field(default=640, ge=128, le=4096)
    batch: int = Field(default=2, ge=1, le=128)
    device: str = "auto"
    workers: int = Field(default=0, ge=0, le=64)
    learning_rate: float = Field(default=0.005, gt=0, le=1)
    momentum: float = Field(default=0.9, ge=0, lt=1)
    weight_decay: float = Field(default=0.0005, ge=0, le=1)
    step_size: int = Field(default=8, ge=1, le=10_000)
    gamma: float = Field(default=0.1, gt=0, le=1)
    seed: int = Field(default=42, ge=0)
    pretrained: bool = True
    score_threshold: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(default=0.5, gt=0, le=1)

    @field_validator("imgsz")
    @classmethod
    def image_size_must_be_stride_aligned(cls, value: int) -> int:
        if value % 32:
            raise ValueError("imgsz must be a multiple of 32")
        return value


class RunConfig(StrictModel):
    schema_version: Literal[1] = 1
    project: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")
    task: Literal["object-detection"] = "object-detection"
    engine: Literal["torchvision"] = "torchvision"
    model: Literal["fasterrcnn-resnet50-fpn-v2"] = "fasterrcnn-resnet50-fpn-v2"
    dataset: DatasetConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    output_root: str = "outputs"

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def resolve_dataset_path(self, project_root: Path) -> str:
        candidate = Path(self.dataset.path)
        if candidate.is_absolute():
            return str(candidate.resolve())
        return str((project_root / candidate).resolve())

    def resolve_output_root(self, project_root: Path) -> Path:
        candidate = Path(self.output_root)
        if candidate.is_absolute():
            return candidate.resolve()
        return (project_root / candidate).resolve()


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Training config root must be a YAML mapping")
    return RunConfig.model_validate(payload)
