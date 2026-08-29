"""Runtime paths for an installed or source-tree DefectDock application."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

WORKSPACE_ENV = "DEFECTDOCK_WORKSPACE"


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(frozen=True)
class RuntimeSettings:
    """Resolved writable locations for one DefectDock workspace.

    Constructing settings is side-effect free. Directories are created only by
    the service that owns them, after the application or command is invoked.
    """

    workspace: Path
    state_dir: Path
    db_path: Path
    datasets_root: Path
    active_model_config: Path
    cvat_config: Path

    @classmethod
    def from_sources(
        cls,
        workspace: str | Path | None = None,
        *,
        db_path: str | Path | None = None,
        datasets_root: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> "RuntimeSettings":
        environment = os.environ if environ is None else environ
        current = Path.cwd() if cwd is None else Path(cwd)
        current = current.expanduser().resolve()
        selected = workspace or environment.get(WORKSPACE_ENV) or current
        resolved_workspace = _resolve_path(selected, base=current)
        state_dir = resolved_workspace / ".defectdock"
        return cls(
            workspace=resolved_workspace,
            state_dir=state_dir,
            db_path=(
                _resolve_path(db_path, base=resolved_workspace)
                if db_path is not None
                else state_dir / "defectdock.db"
            ),
            datasets_root=(
                _resolve_path(datasets_root, base=resolved_workspace)
                if datasets_root is not None
                else resolved_workspace / "datasets" / "uploads"
            ),
            active_model_config=state_dir / "active_model.json",
            cvat_config=state_dir / "cvat.json",
        )

    def resolve(self, path: str | Path) -> Path:
        """Resolve a user-supplied path relative to this workspace."""
        return _resolve_path(path, base=self.workspace)
