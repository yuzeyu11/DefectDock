"""CVAT connectivity and task creation through the official high-level SDK."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from defectdock.domain import DatasetRecord
from defectdock.settings import RuntimeSettings


@dataclass(frozen=True)
class CvatSettings:
    url: str = "http://localhost:8080"
    access_token: str | None = None
    username: str | None = None
    password: str | None = None

    @classmethod
    def from_env(cls, config_path: str | Path | None = None) -> "CvatSettings":
        default_path = RuntimeSettings.from_sources().cvat_config
        path = Path(
            config_path
            or os.getenv("DEFECTDOCK_CVAT_CONFIG", str(default_path))
        ).resolve()
        local = {}
        if path.is_file():
            try:
                local = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                # 损坏的本地配置不能静默吞掉，否则排障时极易困惑。
                print(
                    f"warning: ignoring unreadable CVAT config {path}: {exc}",
                    file=sys.stderr,
                )
                local = {}
        if not isinstance(local, dict):
            print(
                f"warning: CVAT config {path} root must be a JSON object; ignoring",
                file=sys.stderr,
            )
            local = {}
        return cls(
            url=(os.getenv("CVAT_URL") or local.get("url") or "http://localhost:8080").rstrip("/"),
            access_token=os.getenv("CVAT_ACCESS_TOKEN") or local.get("access_token") or None,
            username=os.getenv("CVAT_USERNAME") or local.get("username") or None,
            password=os.getenv("CVAT_PASSWORD") or local.get("password") or None,
        )

    @property
    def configured(self) -> bool:
        return bool(self.access_token or (self.username and self.password))

    def save_local(self, config_path: str | Path | None = None) -> Path:
        path = Path(config_path or RuntimeSettings.from_sources().cvat_config).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "url": self.url,
                    "access_token": self.access_token,
                    "username": self.username,
                    "password": self.password,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path


class CvatClient:
    def __init__(self, settings: CvatSettings | None = None):
        self.settings = settings or CvatSettings.from_env()

    def status(self) -> dict:
        try:
            with httpx.Client(trust_env=False, follow_redirects=True, timeout=3) as client:
                response = client.get(f"{self.settings.url}/api/server/about")
            response.raise_for_status()
        except (httpx.HTTPError, OSError) as exc:
            return {
                "url": self.settings.url,
                "reachable": False,
                "configured": self.settings.configured,
                "detail": str(exc),
            }
        payload = response.json()
        return {
            "url": self.settings.url,
            "reachable": True,
            "configured": self.settings.configured,
            "server": payload,
        }

    def task_url(self, task_id: int) -> str:
        return f"{self.settings.url}/tasks/{task_id}"

    def _client_kwargs(self) -> dict:
        if self.settings.access_token:
            return {"access_token": self.settings.access_token}
        return {"credentials": (self.settings.username, self.settings.password)}

    def create_task(self, dataset: DatasetRecord, image_paths: list[Path]) -> dict:
        if not image_paths:
            raise ValueError("Cannot create a CVAT task for an empty dataset")
        if not self.settings.configured:
            raise RuntimeError(
                "CVAT authentication is not configured; set CVAT_ACCESS_TOKEN or CVAT_USERNAME/CVAT_PASSWORD"
            )
        try:
            from cvat_sdk import make_client
            from cvat_sdk.core.proxies.tasks import ResourceType
        except ImportError as exc:
            raise RuntimeError(
                "CVAT SDK is not installed; install requirements-cvat.txt"
            ) from exc

        spec = {
            "name": f"DefectDock · {dataset.name} · {dataset.dataset_id}",
            "labels": [{"name": label} for label in dataset.labels],
            "segment_size": min(200, max(1, len(image_paths))),
        }
        with make_client(self.settings.url, **self._client_kwargs()) as client:
            task = client.tasks.create_from_data(
                spec=spec,
                resource_type=ResourceType.LOCAL,
                resources=[str(path) for path in image_paths],
            )
            task_id = int(task.id)
        return {
            "task_id": task_id,
            "task_url": self.task_url(task_id),
        }

    def task_status(self, task_id: int) -> dict:
        """Return the aggregate CVAT annotation status for a task."""
        if not self.settings.configured:
            raise RuntimeError("CVAT authentication is not configured")
        try:
            from cvat_sdk import make_client
        except ImportError as exc:
            raise RuntimeError(
                "CVAT SDK is not installed; install requirements-cvat.txt"
            ) from exc

        try:
            with make_client(self.settings.url, **self._client_kwargs()) as client:
                task = client.tasks.retrieve(task_id)
                raw_status = task.status
                size = int(task.size)
        except Exception as exc:
            raise RuntimeError(f"Unable to read CVAT task {task_id}: {exc}") from exc
        status = getattr(raw_status, "value", raw_status)
        return {
            "task_id": task_id,
            "task_url": self.task_url(task_id),
            "status": str(status),
            "size": size,
            "completed": str(status) == "completed",
        }

    def export_dataset(
        self,
        task_id: int,
        destination: str | Path,
        *,
        format_name: str = "YOLO 1.1",
    ) -> dict:
        """Download task annotations using CVAT's high-level SDK."""
        if not self.settings.configured:
            raise RuntimeError("CVAT authentication is not configured")
        try:
            from cvat_sdk import make_client
        except ImportError as exc:
            raise RuntimeError(
                "CVAT SDK is not installed; install requirements-cvat.txt"
            ) from exc

        destination = Path(destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with make_client(self.settings.url, **self._client_kwargs()) as client:
                task = client.tasks.retrieve(task_id)
                task.export_dataset(
                    format_name=format_name,
                    filename=str(destination),
                    include_images=False,
                )
        except Exception as exc:
            raise RuntimeError(f"Unable to export CVAT task {task_id}: {exc}") from exc
        if not destination.is_file():
            raise RuntimeError(f"CVAT export did not create {destination}")
        return {
            "task_id": task_id,
            "format": format_name,
            "archive_path": str(destination),
        }
