"""SQLite-backed immutable model registry and activation history."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from defectdock.db.migration import upgrade_database
from defectdock.domain import ModelActivationRecord, ModelVersionRecord, RunRecord, RunStatus


class ModelStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        upgrade_database(self.path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def register_run(self, run: RunRecord, artifact_path: str | Path, *, actor: str) -> ModelVersionRecord:
        if run.status is not RunStatus.SUCCEEDED:
            raise ValueError("Only a succeeded run can be registered")
        artifact = Path(artifact_path).resolve()
        if not artifact.is_file():
            raise FileNotFoundError(f"Model artifact not found: {artifact}")
        existing = self.get_model_for_run(run.run_id, required=False)
        if existing is not None:
            self.verify_artifact(existing)
            return existing

        run_manifest = Path(run.output_dir) / "run.manifest.json"
        manifest_path = str(run_manifest.resolve()) if run_manifest.is_file() else None
        manifest_sha256 = _file_sha256(run_manifest) if run_manifest.is_file() else None
        now = _utc_now()
        model_version_id = f"model-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        values = (
            model_version_id,
            run.run_id,
            run.project,
            run.task,
            run.engine,
            run.model,
            run.dataset,
            run.dataset_version,
            run.config_hash,
            str(artifact),
            _file_sha256(artifact),
            artifact.stat().st_size,
            manifest_path,
            manifest_sha256,
            json.dumps(run.metrics, ensure_ascii=False, sort_keys=True) if run.metrics else None,
            actor,
            now,
        )
        try:
            with closing(self._connect()) as connection:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO model_versions (
                            model_version_id, run_id, project, task, engine, architecture,
                            dataset, dataset_version, config_hash, artifact_path, artifact_sha256,
                            artifact_size, run_manifest_path, run_manifest_sha256, metrics_json,
                            created_by, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        values,
                    )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Run already has a registered model: {run.run_id}") from exc
        return self.get_model(model_version_id)

    def get_model(self, model_version_id: str) -> ModelVersionRecord:
        with closing(self._connect()) as connection:
            row = connection.execute(
                self._select_sql() + " WHERE mv.model_version_id = ?",
                (model_version_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Model version not found: {model_version_id}")
        return self._record(row)

    def get_model_for_run(self, run_id: str, *, required: bool = True) -> ModelVersionRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                self._select_sql() + " WHERE mv.run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None and required:
            raise KeyError(f"Run has no registered model: {run_id}")
        return self._record(row) if row is not None else None

    def list_models(self, limit: int = 100, project: str | None = None) -> list[ModelVersionRecord]:
        limit = max(1, min(limit, 1000))
        query = self._select_sql()
        params: list[object] = []
        if project:
            query += " WHERE mv.project = ?"
            params.append(project)
        query += " ORDER BY mv.created_at DESC, mv.model_version_id DESC LIMIT ?"
        params.append(limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._record(row) for row in rows]

    def approve(self, model_version_id: str, *, actor: str) -> ModelVersionRecord:
        model = self.get_model(model_version_id)
        self.verify_artifact(model)
        if model.approval_status == "approved":
            return model
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE model_versions
                       SET approval_status = 'approved', approved_by = ?, approved_at = ?
                     WHERE model_version_id = ?
                    """,
                    (actor, _utc_now(), model_version_id),
                )
        return self.get_model(model_version_id)

    def get_active_model(self, *, required: bool = True) -> ModelVersionRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                self._select_sql() + " WHERE mah.slot = 'default'"
            ).fetchone()
        if row is None and required:
            raise KeyError("No model version is active")
        return self._record(row) if row is not None else None

    def activate(
        self,
        model_version_id: str,
        *,
        actor: str,
        action: str = "activate",
        expected_previous_id: str | None,
    ) -> ModelActivationRecord:
        if action not in {"activate", "rollback"}:
            raise ValueError(f"Unsupported activation action: {action}")
        model = self.get_model(model_version_id)
        if model.approval_status != "approved":
            raise ValueError("Model version must be approved before activation")
        activation_id = f"activation-{uuid4().hex}"
        now = _utc_now()
        with closing(self._connect()) as connection:
            with connection:
                current_row = connection.execute(
                    "SELECT model_version_id FROM model_activation_head WHERE slot = 'default'"
                ).fetchone()
                current_id = str(current_row[0]) if current_row else None
                if current_id != expected_previous_id:
                    raise ValueError("Active model changed while activation was being prepared")
                connection.execute(
                    """
                    INSERT INTO model_activation_head (slot, model_version_id, actor, updated_at)
                    VALUES ('default', ?, ?, ?)
                    ON CONFLICT(slot) DO UPDATE SET
                        model_version_id = excluded.model_version_id,
                        actor = excluded.actor,
                        updated_at = excluded.updated_at
                    """,
                    (model_version_id, actor, now),
                )
                connection.execute(
                    """
                    INSERT INTO model_activation_events (
                        activation_id, action, model_version_id,
                        previous_model_version_id, actor, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (activation_id, action, model_version_id, current_id, actor, now),
                )
        return self.get_activation(activation_id)

    def rollback_target(self) -> ModelVersionRecord:
        current = self.get_active_model()
        assert current is not None
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT previous_model_version_id
                  FROM model_activation_events
                 WHERE model_version_id = ?
                 ORDER BY rowid DESC
                 LIMIT 1
                """,
                (current.model_version_id,),
            ).fetchone()
        if row is None or row[0] is None:
            raise ValueError("The active model has no rollback target")
        return self.get_model(str(row[0]))

    def get_activation(self, activation_id: str) -> ModelActivationRecord:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM model_activation_events WHERE activation_id = ?",
                (activation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Activation event not found: {activation_id}")
        return self._activation_record(row)

    def activation_history(self, limit: int = 100) -> list[ModelActivationRecord]:
        limit = max(1, min(limit, 1000))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM model_activation_events
                 ORDER BY rowid DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._activation_record(row) for row in rows]

    @staticmethod
    def verify_artifact(record: ModelVersionRecord) -> None:
        artifact = Path(record.artifact_path)
        if not artifact.is_file():
            raise ValueError(f"Registered model artifact is missing: {record.model_version_id}")
        if artifact.stat().st_size != record.artifact_size or not _digest_matches(
            artifact, record.artifact_sha256
        ):
            raise ValueError(f"Registered model artifact failed integrity check: {record.model_version_id}")
        if record.run_manifest_path:
            manifest = Path(record.run_manifest_path)
            if not manifest.is_file() or not record.run_manifest_sha256 or not _digest_matches(
                manifest, record.run_manifest_sha256
            ):
                raise ValueError(f"Registered run manifest failed integrity check: {record.model_version_id}")

    @staticmethod
    def _select_sql() -> str:
        return (
            "SELECT mv.*, CASE WHEN mah.model_version_id IS NULL THEN 0 ELSE 1 END AS is_active "
            "FROM model_versions mv LEFT JOIN model_activation_head mah "
            "ON mah.model_version_id = mv.model_version_id"
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> ModelVersionRecord:
        return ModelVersionRecord(
            model_version_id=row["model_version_id"],
            run_id=row["run_id"],
            project=row["project"],
            task=row["task"],
            engine=row["engine"],
            architecture=row["architecture"],
            dataset=row["dataset"],
            dataset_version=row["dataset_version"],
            config_hash=row["config_hash"],
            artifact_path=row["artifact_path"],
            artifact_sha256=row["artifact_sha256"],
            artifact_size=row["artifact_size"],
            run_manifest_path=row["run_manifest_path"],
            run_manifest_sha256=row["run_manifest_sha256"],
            metrics=json.loads(row["metrics_json"]) if row["metrics_json"] else None,
            created_by=row["created_by"],
            created_at=row["created_at"],
            approval_status=row["approval_status"],
            approved_by=row["approved_by"],
            approved_at=row["approved_at"],
            is_active=bool(row["is_active"]),
        )

    @staticmethod
    def _activation_record(row: sqlite3.Row) -> ModelActivationRecord:
        return ModelActivationRecord(
            activation_id=row["activation_id"],
            action=row["action"],
            model_version_id=row["model_version_id"],
            previous_model_version_id=row["previous_model_version_id"],
            actor=row["actor"],
            created_at=row["created_at"],
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_matches(path: Path, expected: str) -> bool:
    return _file_sha256(path) == expected


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
