"""Stable contracts shared by training engine adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from defectdock.config import RunConfig

EventCallback = Callable[[dict], None]
CancellationCallback = Callable[[], bool]


class TrainingCancelled(RuntimeError):
    """Raised by an engine after a cooperative cancellation request."""


@dataclass(frozen=True)
class EngineResult:
    trainer_output: str
    best_model: str | None
    last_model: str | None
    metrics: dict


class TrainingEngine(Protocol):
    name: str

    def plan(self, config: RunConfig, run_dir: Path) -> dict: ...

    def run(
        self,
        config: RunConfig,
        run_dir: str | Path,
        on_event: EventCallback | None = None,
        should_cancel: CancellationCallback | None = None,
        *,
        project_root: str | Path | None = None,
    ) -> EngineResult: ...
