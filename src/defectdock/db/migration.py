"""Versioned SQLite migrations, backup, and automatic recovery."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import URL


@dataclass(frozen=True)
class MigrationResult:
    previous_revision: str | None
    current_revision: str
    backup_path: str | None = None


def alembic_config(path: str | Path) -> Config:
    database_path = Path(path).resolve()
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parent / "alembic"),
    )
    url = URL.create("sqlite", database=str(database_path))
    config.set_main_option("sqlalchemy.url", url.render_as_string(hide_password=False).replace("%", "%%"))
    return config


def current_revision(path: str | Path) -> str | None:
    database_path = Path(path).resolve()
    if not database_path.is_file() or database_path.stat().st_size == 0:
        return None
    with closing(sqlite3.connect(database_path)) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'alembic_version'"
        ).fetchone()
        if not exists:
            return None
        row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    return str(row[0]) if row else None


def head_revision(path: str | Path) -> str:
    config = alembic_config(path)
    head = ScriptDirectory.from_config(config).get_current_head()
    if head is None:
        raise RuntimeError("Alembic migration head is missing")
    return head


def backup_database(path: str | Path, destination: str | Path | None = None) -> Path:
    database_path = Path(path).resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")
    if destination is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_dir = database_path.parent / "backups"
        destination_path = backup_dir / f"{database_path.stem}-{timestamp}.sqlite3"
    else:
        destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path == database_path:
        raise ValueError("Backup destination must differ from the live database")
    with (
        closing(sqlite3.connect(database_path)) as source,
        closing(sqlite3.connect(destination_path)) as target,
    ):
        source.backup(target)
    return destination_path


def restore_database(path: str | Path, backup_path: str | Path) -> None:
    database_path = Path(path).resolve()
    source_path = Path(backup_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Database backup not found: {source_path}")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        closing(sqlite3.connect(source_path)) as source,
        closing(sqlite3.connect(database_path)) as target,
    ):
        source.backup(target)


def upgrade_database(path: str | Path, *, create_backup: bool = True) -> MigrationResult:
    database_path = Path(path).resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    previous = current_revision(database_path)
    head = head_revision(database_path)
    if previous == head:
        return MigrationResult(previous_revision=previous, current_revision=head)

    had_database = database_path.is_file() and database_path.stat().st_size > 0
    backup = backup_database(database_path) if create_backup and had_database else None
    try:
        command.upgrade(alembic_config(database_path), head)
    except Exception:
        if backup is not None:
            restore_database(database_path, backup)
        elif database_path.exists():
            database_path.unlink()
        raise
    return MigrationResult(
        previous_revision=previous,
        current_revision=head,
        backup_path=str(backup) if backup else None,
    )
