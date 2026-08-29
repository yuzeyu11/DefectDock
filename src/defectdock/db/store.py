"""Small, dependency-free SQLite store for training runs."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from defectdock.config import RunConfig
from defectdock.db.migration import upgrade_database
from defectdock.domain import RunRecord, RunStatus, ensure_transition, new_run_id
from defectdock.domain.runs import TERMINAL_STATUSES, utc_now


class RunStore:
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

    def create_run(self, config: RunConfig, output_dir: str | Path, run_id: str | None = None) -> RunRecord:
        run_id = run_id or new_run_id(config.project)
        now = utc_now()
        values = {
            "run_id": run_id,
            "project": config.project,
            "task": config.task,
            "engine": config.engine,
            "model": config.model,
            "dataset": config.dataset.path,
            "dataset_version": config.dataset.version,
            "config_hash": config.config_hash,
            "config_json": json.dumps(config.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
            "status": RunStatus.CREATED.value,
            "output_dir": str(Path(output_dir).resolve()),
            "created_at": now,
            "updated_at": now,
        }
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, project, task, engine, model, dataset, dataset_version,
                        config_hash, config_json, status, output_dir, created_at, updated_at
                    ) VALUES (
                        :run_id, :project, :task, :engine, :model, :dataset, :dataset_version,
                        :config_hash, :config_json, :status, :output_dir, :created_at, :updated_at
                    )
                    """,
                    values,
                )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Run not found: {run_id}")
        return self._to_record(row)

    def list_runs(self, limit: int = 50, project: str | None = None) -> list[RunRecord]:
        limit = max(1, min(limit, 1000))
        query = "SELECT * FROM runs"
        params: list[object] = []
        if project:
            query += " WHERE project = ?"
            params.append(project)
        query += " ORDER BY created_at DESC, run_id DESC LIMIT ?"
        params.append(limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._to_record(row) for row in rows]

    def recover_incomplete_runs(self, reason: str = "service restarted before completion") -> list[RunRecord]:
        """Move runs left in a non-terminal state to FAILED after a restart."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT run_id FROM runs WHERE status IN (?, ?, ?)",
                (RunStatus.CREATED.value, RunStatus.QUEUED.value, RunStatus.RUNNING.value),
            ).fetchall()
        recovered: list[RunRecord] = []
        for row in rows:
            recovered.append(self.update_status(row["run_id"], RunStatus.FAILED, error=reason))
        return recovered

    def update_status(
        self,
        run_id: str,
        target: RunStatus,
        *,
        metrics: dict | None = None,
        error: str | None = None,
    ) -> RunRecord:
        current = self.get_run(run_id)
        ensure_transition(current.status, target)
        now = utc_now()
        started_at = current.started_at
        finished_at = current.finished_at
        if target == RunStatus.RUNNING and started_at is None:
            started_at = now
        if target in TERMINAL_STATUSES:
            finished_at = now

        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE runs
                       SET status = ?, metrics_json = ?, error = ?, updated_at = ?,
                           started_at = ?, finished_at = ?
                     WHERE run_id = ?
                    """,
                    (
                        target.value,
                        json.dumps(metrics, ensure_ascii=False, sort_keys=True) if metrics is not None else None,
                        error,
                        now,
                        started_at,
                        finished_at,
                        run_id,
                    ),
                )
        return self.get_run(run_id)

    @staticmethod
    def _to_record(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            project=row["project"],
            task=row["task"],
            engine=row["engine"],
            model=row["model"],
            dataset=row["dataset"],
            dataset_version=row["dataset_version"],
            config_hash=row["config_hash"],
            config=json.loads(row["config_json"]),
            status=RunStatus(row["status"]),
            output_dir=row["output_dir"],
            metrics=json.loads(row["metrics_json"]) if row["metrics_json"] else None,
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )
