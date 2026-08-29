"""Single-workstation background training queue with cooperative cancellation."""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from defectdock.config import RunConfig
from defectdock.db import RunStore
from defectdock.domain import RunRecord, RunStatus, new_run_id
from defectdock.domain.runs import TERMINAL_STATUSES, utc_now
from defectdock.engines import EngineResult, TrainingCancelled, run_training
from defectdock.provenance import write_run_manifest

Runner = Callable[..., EngineResult]


@dataclass
class _JobHandle:
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Future | None = None
    cancel_notified: bool = False


class TrainingJobManager:
    """Coordinate durable run metadata with a bounded local worker pool.

    P0 deliberately serializes jobs by default because one workstation usually
    has one accelerator. The public API is independent from this implementation,
    so a durable distributed queue can replace it later.
    """

    def __init__(
        self,
        store: RunStore,
        project_root: str | Path,
        *,
        runner: Runner = run_training,
        max_workers: int = 1,
    ) -> None:
        self.store = store
        self.project_root = Path(project_root).resolve()
        self.runner = runner
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="defectdock-train",
        )
        self._handles: dict[str, _JobHandle] = {}
        self._lock = threading.Lock()
        self.recovered_runs = self.store.recover_incomplete_runs()

    def submit(self, config: RunConfig) -> RunRecord:
        output_root = config.resolve_output_root(self.project_root)
        if not output_root.is_relative_to(self.project_root):
            raise ValueError("API training output_root must stay inside the project directory")

        run_id = new_run_id(config.project)
        run_dir = output_root / config.project / config.task / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.snapshot.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        write_run_manifest(run_dir, config, self.project_root)
        self.store.create_run(config, run_dir, run_id=run_id)
        record = self.store.update_status(run_id, RunStatus.QUEUED)
        self._append_event(run_dir, {"event": "run_queued", "run_id": run_id})

        handle = _JobHandle()
        with self._lock:
            self._handles[run_id] = handle
        handle.future = self._executor.submit(self._execute, run_id, config, run_dir, handle)
        return record

    def cancel(self, run_id: str) -> RunRecord:
        record = self.store.get_run(run_id)
        if record.status in TERMINAL_STATUSES:
            return record
        with self._lock:
            handle = self._handles.get(run_id)
            if handle is None:
                return self.store.update_status(
                    run_id,
                    RunStatus.FAILED,
                    error="training worker is unavailable; restart recovery required",
                )
            handle.cancel_event.set()
            if not handle.cancel_notified:
                handle.cancel_notified = True
                self._append_event(
                    Path(record.output_dir),
                    {"event": "cancellation_requested", "run_id": run_id},
                )
            future = handle.future
            cancelled_before_start = bool(future and future.cancel())
        if cancelled_before_start:
            self._append_event(Path(record.output_dir), {"event": "run_cancelled", "run_id": run_id})
            return self.store.update_status(run_id, RunStatus.CANCELLED, error="cancelled before start")
        return self.store.get_run(run_id)

    def shutdown(self, *, wait: bool = False) -> None:
        with self._lock:
            handles = list(self._handles.values())
        for handle in handles:
            handle.cancel_event.set()
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _execute(
        self,
        run_id: str,
        config: RunConfig,
        run_dir: Path,
        handle: _JobHandle,
    ) -> None:
        try:
            if handle.cancel_event.is_set():
                raise TrainingCancelled("cancelled before start")
            self.store.update_status(run_id, RunStatus.RUNNING)
            result = self.runner(
                config,
                run_dir,
                on_event=None,
                should_cancel=handle.cancel_event.is_set,
                project_root=self.project_root,
            )
            if handle.cancel_event.is_set():
                raise TrainingCancelled("cancelled after engine returned")
            self.store.update_status(run_id, RunStatus.SUCCEEDED, metrics=result.metrics)
        except TrainingCancelled as exc:
            current = self.store.get_run(run_id)
            if current.status not in TERMINAL_STATUSES:
                self.store.update_status(run_id, RunStatus.CANCELLED, error=str(exc))
            self._append_event(run_dir, {"event": "run_cancelled", "run_id": run_id})
        except Exception as exc:
            current = self.store.get_run(run_id)
            if current.status not in TERMINAL_STATUSES:
                self.store.update_status(
                    run_id,
                    RunStatus.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._append_event(
                run_dir,
                {"event": "run_failed", "run_id": run_id, "error": f"{type(exc).__name__}: {exc}"},
            )
        finally:
            with self._lock:
                self._handles.pop(run_id, None)

    @staticmethod
    def _append_event(run_dir: Path, payload: dict) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        event = {"timestamp": utc_now(), **payload}
        with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
