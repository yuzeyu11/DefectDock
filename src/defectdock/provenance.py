"""Reproducibility manifest for training runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from defectdock.config import RunConfig

TRACKED_DISTRIBUTIONS = (
    "defectdock",
    "fastapi",
    "numpy",
    "opencv-python-headless",
    "pillow",
    "pydantic",
    "pyyaml",
    "torch",
    "torchvision",
    "typer",
    "uvicorn",
)


def write_run_manifest(
    run_dir: str | Path,
    config: RunConfig,
    project_root: str | Path,
    *,
    include_accelerator: bool = False,
    pretrained_weight: dict | None = None,
) -> dict:
    """Atomically write the non-secret inputs needed to reproduce a run."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(project_root).resolve()
    dataset_path = Path(config.resolve_dataset_path(project_root))
    manifest = {
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_hash": config.config_hash,
        "code": _git_state(project_root),
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dependencies": _dependency_versions(),
        },
        "dataset": _dataset_provenance(dataset_path, config.dataset.version),
        "training": {
            "engine": config.engine,
            "model": config.model,
            "seed": config.train.seed,
            "device_requested": config.train.device,
            "pretrained_weight": pretrained_weight
            or {"enabled": config.train.pretrained, "resolved": False},
        },
    }
    if include_accelerator:
        manifest["environment"]["accelerator"] = _accelerator_state()
    path = run_dir / "run.manifest.json"
    staging = run_dir / ".run.manifest.json.tmp"
    staging.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    staging.replace(path)
    return manifest


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(project_root: Path) -> dict:
    repository_root = _find_git_root(project_root)
    if repository_root is None:
        return {"repository": False, "root": None, "commit": None, "dirty": None}
    try:
        commit = _git(repository_root, "rev-parse", "HEAD") or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    try:
        status = _git(repository_root, "status", "--porcelain", "--untracked-files=normal")
        dirty: bool | None = bool(status)
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {
        "repository": True,
        "root": str(repository_root),
        "commit": commit,
        "dirty": dirty,
    }


def _find_git_root(project_root: Path) -> Path | None:
    candidates = [Path(__file__).resolve().parent, project_root]
    visited: set[Path] = set()
    for start in candidates:
        for candidate in (start, *start.parents):
            if candidate in visited:
                continue
            visited.add(candidate)
            if (candidate / ".git").exists():
                return candidate
    return None


def _git(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return result.stdout.strip()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _dataset_provenance(dataset_path: Path, version: str) -> dict:
    result = {
        "version": version,
        "config_path": str(dataset_path),
        "config_sha256": file_sha256(dataset_path) if dataset_path.is_file() else None,
        "snapshot_id": None,
        "snapshot_sha256": None,
    }
    snapshot_manifest = dataset_path.parent / "manifest.json"
    if snapshot_manifest.is_file():
        try:
            payload = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return result
        result["snapshot_id"] = payload.get("snapshot_id")
        result["snapshot_sha256"] = payload.get("snapshot_sha256")
    return result


def _accelerator_state() -> dict:
    try:
        import torch

        available = torch.cuda.is_available()
        return {
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": available,
            "device_count": torch.cuda.device_count() if available else 0,
            "device": torch.cuda.get_device_name(0) if available else None,
        }
    except (ImportError, RuntimeError):
        return {
            "torch_cuda_version": None,
            "cudnn_version": None,
            "cuda_available": False,
            "device_count": 0,
            "device": None,
        }
