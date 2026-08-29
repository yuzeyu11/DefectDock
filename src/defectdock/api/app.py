"""FastAPI surface for DefectDock datasets, inference and run metadata."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from defectdock import __version__
from defectdock.config import RunConfig
from defectdock.data import (
    import_cvat_yolo_export,
    import_uploaded_annotations,
    ingest_images,
    parse_labels,
)
from defectdock.data.cv import build_training_snapshot, check_dataset, compute_stats
from defectdock.db import DatasetStore, RunStore
from defectdock.domain import AnnotationVersionRecord, new_dataset_id
from defectdock.engines import EngineResult, build_plan, run_training
from defectdock.inference import DetectionInferenceService
from defectdock.integrations import CvatClient, CvatSettings
from defectdock.resources import load_example_configs
from defectdock.services import TrainingJobManager
from defectdock.settings import RuntimeSettings

DEFAULT_BOARD_LABELS = "punching_hole,welding_line,crescent_gap,water_spot,oil_spot,silk_spot,inclusion,rolled_pit,crease,waist_folding"
MAX_INFERENCE_UPLOAD_BYTES = 25 * 1024 * 1024


class SnapshotRequest(BaseModel):
    val_ratio: float = Field(default=0.2, ge=0.05, le=0.5)
    seed: int = Field(default=42, ge=0)
    annotation_version_id: str | None = Field(default=None, min_length=1, max_length=160)


def create_app(
    db_path: str | Path | None = None,
    datasets_root: str | Path | None = None,
    cvat_client: CvatClient | None = None,
    inference_service: DetectionInferenceService | None = None,
    project_root: str | Path | None = None,
    training_runner: Callable[..., EngineResult] = run_training,
    *,
    workspace: str | Path | None = None,
    runtime_settings: RuntimeSettings | None = None,
    training_enabled: bool | None = None,
) -> FastAPI:
    if project_root is not None and workspace is not None:
        raise ValueError("Use either project_root or workspace, not both")
    runtime = runtime_settings or RuntimeSettings.from_sources(
        workspace or project_root,
        db_path=db_path,
        datasets_root=datasets_root,
    )
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            application.state.jobs.shutdown()

    app = FastAPI(
        title="DefectDock API",
        version=__version__,
        description="Industrial computer-vision lifecycle API.",
        lifespan=lifespan,
    )
    app.state.settings = runtime
    app.state.training_submission_enabled = (
        _training_stack_available() if training_enabled is None else training_enabled
    )
    app.state.store = RunStore(runtime.db_path)
    app.state.datasets = DatasetStore(runtime.db_path)
    app.state.datasets_root = runtime.datasets_root
    app.state.datasets_root.mkdir(parents=True, exist_ok=True)
    app.state.cvat = cvat_client or CvatClient(CvatSettings.from_env(runtime.cvat_config))
    app.state.project_root = runtime.workspace
    app.state.active_model_config = runtime.active_model_config
    app.state.inference = inference_service or DetectionInferenceService.from_config(
        app.state.active_model_config, runtime.workspace
    )
    app.state.jobs = TrainingJobManager(
        app.state.store,
        runtime.workspace,
        runner=training_runner,
    )

    @app.get("/", include_in_schema=False)
    def index() -> dict:
        return {
            "service": "defectdock",
            "version": __version__,
            "status": "ok",
            "api_docs": "/docs",
            "health": "/api/health",
        }

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "service": "defectdock",
            "version": __version__,
            "training_submission_enabled": app.state.training_submission_enabled,
            "dataset_upload_enabled": True,
            "inference_ready": app.state.inference.available,
        }

    @app.get("/api/inference/status")
    def inference_status() -> dict:
        return app.state.inference.status()

    @app.post("/api/inference/detect")
    async def detect_image(file: UploadFile = File(...)) -> dict:
        image_bytes = await file.read(MAX_INFERENCE_UPLOAD_BYTES + 1)
        await file.close()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")
        if len(image_bytes) > MAX_INFERENCE_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 25 MB limit")
        # 推理是阻塞式 GPU/CPU 调用，必须放入线程池，否则会卡死整个事件循环。
        try:
            return await run_in_threadpool(
                app.state.inference.predict, image_bytes, file.filename or "image"
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/integrations/cvat")
    def cvat_status() -> dict:
        return app.state.cvat.status()

    @app.post("/api/datasets", status_code=201)
    async def create_dataset(
        name: str = Form(..., min_length=2, max_length=100),
        scene: str = Form(default="board", min_length=1, max_length=64),
        labels: str = Form(default=DEFAULT_BOARD_LABELS),
        files: list[UploadFile] = File(...),
    ) -> dict:
        try:
            parsed_labels = parse_labels(labels)
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        dataset_id = new_dataset_id()
        dataset_root = app.state.datasets_root / dataset_id
        image_dir = dataset_root / "images"
        app.state.datasets.create_dataset(
            name=name.strip(),
            scene=scene.strip(),
            labels=parsed_labels,
            root_dir=dataset_root,
            dataset_id=dataset_id,
        )
        try:
            upload = await ingest_images(
                app.state.datasets,
                dataset_id,
                image_dir,
                files,
            )
            dataset = app.state.datasets.get_dataset(dataset_id)
        except ValueError as exc:
            app.state.datasets.delete_dataset(dataset_id)
            if dataset_root.is_relative_to(app.state.datasets_root):
                shutil.rmtree(dataset_root, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"dataset": _dataset_payload(dataset, app.state.cvat), "upload": upload}

    @app.get("/api/datasets")
    def list_datasets(limit: int = Query(default=50, ge=1, le=1000)) -> list[dict]:
        return [
            _dataset_payload(record, app.state.cvat)
            for record in app.state.datasets.list_datasets(limit=limit)
        ]

    @app.get("/api/datasets/{dataset_id}")
    def get_dataset(dataset_id: str) -> dict:
        try:
            dataset = app.state.datasets.get_dataset(dataset_id)
            images = app.state.datasets.list_images(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            current_annotation = app.state.datasets.get_current_annotation_version(dataset_id)
        except KeyError:
            current_annotation = None
        try:
            boxes_by_image = _load_annotation_boxes(current_annotation, images)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        payload = _dataset_payload(dataset, app.state.cvat)
        payload["current_annotation_version"] = (
            current_annotation.model_dump(mode="json") if current_annotation else None
        )
        payload["images"] = [
            {
                **record.model_dump(mode="json"),
                "preview_url": f"/api/datasets/{dataset_id}/images/{record.image_id}",
                "boxes": boxes_by_image.get(record.image_id, []),
            }
            for record in images
        ]
        return payload

    @app.post("/api/datasets/{dataset_id}/annotations", status_code=201)
    async def upload_annotations(
        dataset_id: str,
        files: list[UploadFile] = File(...),
    ) -> dict:
        try:
            dataset = app.state.datasets.get_dataset(dataset_id)
            images = app.state.datasets.list_images(dataset_id, limit=5000)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if dataset.status.value == "frozen":
            raise HTTPException(status_code=409, detail="Frozen datasets cannot accept annotations")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        version_id = f"direct-{timestamp}-{uuid4().hex[:8]}"
        version_dir = Path(dataset.root_dir) / "annotations" / "versions" / version_id
        try:
            imported = await import_uploaded_annotations(dataset, images, files, version_dir)
            version = app.state.datasets.register_annotation_version(
                dataset_id,
                version_id,
                source="direct_upload",
                format="normalized-detection-text-v1",
                root_dir=version_dir,
                manifest_path=imported["manifest_path"],
                labeled_count=imported["labeled_count"],
                unlabeled_count=imported["unlabeled_count"],
            )
        except (OSError, ValueError) as exc:
            shutil.rmtree(version_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "dataset": _dataset_payload(dataset, app.state.cvat),
            "annotation_version": version.model_dump(mode="json"),
        }

    @app.get("/api/datasets/{dataset_id}/annotation-versions")
    def list_annotation_versions(
        dataset_id: str,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict]:
        try:
            versions = app.state.datasets.list_annotation_versions(dataset_id, limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return [version.model_dump(mode="json") for version in versions]

    @app.put("/api/datasets/{dataset_id}/annotation-versions/{version_id}/current")
    def select_annotation_version(dataset_id: str, version_id: str) -> dict:
        try:
            version = app.state.datasets.set_current_annotation_version(dataset_id, version_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return version.model_dump(mode="json")

    @app.get("/api/datasets/{dataset_id}/images/{image_id}", include_in_schema=False)
    def get_dataset_image(dataset_id: str, image_id: str) -> FileResponse:
        try:
            dataset = app.state.datasets.get_dataset(dataset_id)
            record = app.state.datasets.get_image(dataset_id, image_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        path = Path(dataset.root_dir) / "images" / record.stored_name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Image file is missing")
        return FileResponse(path)

    @app.post("/api/datasets/{dataset_id}/freeze")
    def freeze_dataset(dataset_id: str) -> dict:
        try:
            dataset = app.state.datasets.freeze_dataset(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _dataset_payload(dataset, app.state.cvat)

    @app.post("/api/datasets/{dataset_id}/training-snapshot", status_code=201)
    def create_training_snapshot(
        dataset_id: str,
        request: SnapshotRequest | None = None,
    ) -> dict:
        request = request or SnapshotRequest()
        try:
            dataset = app.state.datasets.get_dataset(dataset_id)
            images = app.state.datasets.list_images(dataset_id, limit=5000)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            annotation_version = (
                app.state.datasets.get_annotation_version(
                    dataset_id, request.annotation_version_id
                )
                if request.annotation_version_id
                else app.state.datasets.get_current_annotation_version(dataset_id)
            )
            snapshot = build_training_snapshot(
                dataset,
                images,
                annotation_version,
                val_ratio=request.val_ratio,
                seed=request.seed,
            )
            quality = check_dataset(snapshot["data_yaml"])
            if not quality.ok:
                raise ValueError("Generated snapshot failed its dataset quality gate")
            stats = compute_stats(snapshot["data_yaml"])
        except KeyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "snapshot": snapshot,
            "quality": quality.to_dict(),
            "stats": stats.to_dict(),
        }

    @app.post("/api/datasets/{dataset_id}/cvat-task", status_code=201)
    def create_cvat_task(dataset_id: str) -> dict:
        try:
            dataset = app.state.datasets.get_dataset(dataset_id)
            images = app.state.datasets.list_images(dataset_id, limit=5000)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if dataset.cvat_task_id is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Dataset already has CVAT task {dataset.cvat_task_id}",
            )
        image_dir = Path(dataset.root_dir) / "images"
        image_paths = [image_dir / image.stored_name for image in images]
        try:
            result = app.state.cvat.create_task(dataset, image_paths)
            dataset = app.state.datasets.attach_cvat_task(dataset_id, result["task_id"])
        except (RuntimeError, ValueError, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"dataset": _dataset_payload(dataset, app.state.cvat), **result}

    @app.post("/api/datasets/{dataset_id}/cvat-sync")
    def sync_cvat_annotations(dataset_id: str) -> dict:
        """Export completed CVAT annotations and freeze a versioned local copy."""
        try:
            dataset = app.state.datasets.get_dataset(dataset_id)
            images = app.state.datasets.list_images(dataset_id, limit=5000)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if dataset.cvat_task_id is None:
            raise HTTPException(status_code=409, detail="Dataset has no CVAT task")

        try:
            task_status = app.state.cvat.task_status(dataset.cvat_task_id)
        except (RuntimeError, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not task_status["completed"]:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"CVAT task {dataset.cvat_task_id} is '{task_status['status']}'; "
                    "mark all jobs completed before syncing"
                ),
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sync_id = f"cvat-{dataset.cvat_task_id}-{timestamp}-{uuid4().hex[:8]}"
        annotations_root = Path(dataset.root_dir) / "annotations"
        download_dir = annotations_root / "downloads"
        archive_path = download_dir / f"{sync_id}.zip"
        version_dir = annotations_root / "versions" / sync_id
        try:
            export = app.state.cvat.export_dataset(dataset.cvat_task_id, archive_path)
            imported = import_cvat_yolo_export(archive_path, dataset, images, version_dir)
            version = app.state.datasets.register_annotation_version(
                dataset_id,
                sync_id,
                source="cvat",
                format="YOLO 1.1",
                root_dir=version_dir,
                manifest_path=imported["manifest_path"],
                labeled_count=imported["labeled_count"],
                unlabeled_count=imported["unlabeled_count"],
            )
            dataset = app.state.datasets.freeze_dataset(dataset_id)
        except ValueError as exc:
            shutil.rmtree(version_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (RuntimeError, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "dataset": _dataset_payload(dataset, app.state.cvat),
            "task_status": task_status,
            "export": export,
            "annotation_version": version.model_dump(mode="json"),
        }

    @app.get("/api/datasets/{dataset_id}/cvat-status")
    def get_cvat_task_status(dataset_id: str) -> dict:
        try:
            dataset = app.state.datasets.get_dataset(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if dataset.cvat_task_id is None:
            raise HTTPException(status_code=409, detail="Dataset has no CVAT task")
        try:
            return app.state.cvat.task_status(dataset.cvat_task_id)
        except (RuntimeError, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/configs/examples")
    def example_configs() -> list[dict]:
        items = []
        for name, config in load_example_configs():
            items.append(
                {
                    "name": name,
                    "path": f"packaged://configs/examples/{name}.yaml",
                    "project": config.project,
                    "task": config.task,
                    "model": config.model,
                    "dataset": config.dataset.model_dump(mode="json"),
                    "train": config.train.model_dump(mode="json"),
                    "config_hash": config.config_hash,
                }
            )
        return items

    @app.post("/api/configs/validate")
    def validate_config(config: RunConfig) -> dict:
        placeholder = (
            config.resolve_output_root(app.state.project_root)
            / config.project
            / config.task
            / "<run-id>"
        )
        return {"valid": True, "plan": build_plan(config, placeholder, app.state.project_root)}

    @app.post("/api/runs", status_code=202)
    def submit_run(config: RunConfig) -> dict:
        if not app.state.training_submission_enabled:
            raise HTTPException(
                status_code=503,
                detail="Training dependencies are not installed in this runtime",
            )
        dataset_path = Path(config.resolve_dataset_path(app.state.project_root))
        if not dataset_path.is_file():
            raise HTTPException(status_code=422, detail=f"Dataset config not found: {dataset_path}")
        try:
            report = check_dataset(dataset_path)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not report.ok:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Dataset quality gate failed",
                    "summary": report.to_dict(),
                },
            )
        try:
            record = app.state.jobs.submit(config)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return record.model_dump(mode="json")

    @app.get("/api/runs")
    def list_runs(
        limit: int = Query(default=50, ge=1, le=1000),
        project: str | None = None,
    ) -> list[dict]:
        return [
            record.model_dump(mode="json")
            for record in app.state.store.list_runs(limit=limit, project=project)
        ]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        try:
            record = app.state.store.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return record.model_dump(mode="json")

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict:
        try:
            record = app.state.jobs.cancel(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return record.model_dump(mode="json")

    @app.get("/api/runs/{run_id}/artifacts")
    def get_run_artifacts(run_id: str) -> dict:
        try:
            record = app.state.store.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        run_dir = Path(record.output_dir)
        best_model = run_dir / "trainer_output" / "weights" / "best.ckpt"
        last_model = run_dir / "trainer_output" / "weights" / "last.ckpt"
        metrics = run_dir / "metrics.json"
        return {
            "run_id": run_id,
            "best_model": str(best_model) if best_model.is_file() else None,
            "last_model": str(last_model) if last_model.is_file() else None,
            "metrics": str(metrics) if metrics.is_file() else None,
        }

    @app.post("/api/runs/{run_id}/activate")
    def activate_run_model(run_id: str) -> dict:
        try:
            record = app.state.store.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if record.status.value != "succeeded":
            raise HTTPException(status_code=409, detail="Only a succeeded run can be activated")
        model_path = Path(record.output_dir) / "trainer_output" / "weights" / "best.ckpt"
        if not model_path.is_file():
            raise HTTPException(status_code=404, detail="Run has no best.ckpt artifact")
        try:
            stored_model = model_path.relative_to(app.state.project_root)
        except ValueError:
            stored_model = model_path
        payload = {
            "model": str(stored_model).replace("\\", "/"),
            "confidence": record.config["train"]["score_threshold"],
            "max_detections": 100,
            "device": record.config["train"]["device"],
            "run_id": run_id,
        }
        app.state.active_model_config.parent.mkdir(parents=True, exist_ok=True)
        app.state.active_model_config.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        app.state.inference = DetectionInferenceService.from_config(
            app.state.active_model_config,
            app.state.project_root,
        )
        return {"activated": True, "inference": app.state.inference.status()}

    @app.get("/api/runs/{run_id}/events")
    def get_run_events(run_id: str) -> list[dict]:
        try:
            record = app.state.store.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        path = Path(record.output_dir) / "events.jsonl"
        if not path.is_file():
            return []
        events = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # 训练中断可能留下半行 JSON：跳过坏行而不是让整个接口 500。
                continue
        return events

    return app


def _dataset_payload(dataset, cvat_client: CvatClient) -> dict:
    payload = dataset.model_dump(mode="json")
    if dataset.cvat_task_id is not None:
        payload["annotation_url"] = cvat_client.task_url(dataset.cvat_task_id)
    return payload


def _training_stack_available() -> bool:
    """Check optional training dependencies without importing the heavy stack."""
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchvision") is not None
    )


def _load_annotation_boxes(
    annotation_version: AnnotationVersionRecord | None, images
) -> dict[str, list[dict]]:
    """从数据库明确选中的标注版本读取每张图片的框。

    返回 ``{image_id: [{"class_id": int, "cx": float, "cy": float, "w": float, "h": float}]}``；
    找不到版本或标签文件时返回空列表。
    """
    result: dict[str, list[dict]] = {img.image_id: [] for img in images}
    if annotation_version is None:
        return result
    manifest_path = Path(annotation_version.manifest_path)
    if _file_sha256(manifest_path) != annotation_version.manifest_sha256:
        raise ValueError("Annotation manifest changed after version registration")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_images = {item.get("image_id"): item for item in manifest.get("images", [])}
    version_root = Path(annotation_version.root_dir)
    for img in images:
        item = manifest_images.get(img.image_id, {})
        label_relative = item.get("label")
        if not label_relative:
            continue
        lbl = (version_root / label_relative).resolve()
        if not lbl.is_relative_to(version_root.resolve()) or not lbl.is_file():
            raise ValueError(f"Annotation label is missing or outside its version: {img.original_name}")
        expected_hash = item.get("label_sha256")
        if not expected_hash or _file_sha256(lbl) != expected_hash:
            raise ValueError(f"Annotation label changed after version registration: {img.original_name}")
        boxes: list[dict] = []
        for line in lbl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                boxes.append(
                    {
                        "class_id": int(float(parts[0])),
                        "cx": float(parts[1]),
                        "cy": float(parts[2]),
                        "w": float(parts[3]),
                        "h": float(parts[4]),
                    }
                )
            except ValueError:
                continue
        result[img.image_id] = boxes
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "defectdock.api:create_app",
        host="127.0.0.1",
        port=8000,
        factory=True,
        reload=False,
    )
