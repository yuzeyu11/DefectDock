"""FastAPI surface for DefectDock datasets, inference and run metadata."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from defectdock import __version__
from defectdock.config import RunConfig
from defectdock.data import (
    generate_auto_annotations,
    import_cvat_yolo_export,
    import_uploaded_annotations,
    ingest_images,
    parse_labels,
)
from defectdock.data.cv import build_training_snapshot, check_dataset, compute_stats
from defectdock.db import DatasetStore, ModelStore, RunStore
from defectdock.domain import AnnotationVersionRecord, ModelVersionRecord, new_dataset_id
from defectdock.engines import EngineResult, build_plan, run_training
from defectdock.exports import export_onnx_package
from defectdock.exports.onnx import verify_onnx_package
from defectdock.inference import DetectionInferenceService
from defectdock.integrations import CvatClient, CvatSettings
from defectdock.resources import load_example_configs
from defectdock.security import SecurityBoundaryMiddleware, SecurityMode, SecuritySettings
from defectdock.services import TrainingJobManager
from defectdock.settings import RuntimeSettings

DEFAULT_BOARD_LABELS = "punching_hole,welding_line,crescent_gap,water_spot,oil_spot,silk_spot,inclusion,rolled_pit,crease,waist_folding"
MAX_INFERENCE_UPLOAD_BYTES = 25 * 1024 * 1024


class SnapshotRequest(BaseModel):
    val_ratio: float = Field(default=0.2, ge=0.05, le=0.5)
    seed: int = Field(default=42, ge=0)
    annotation_version_id: str | None = Field(default=None, min_length=1, max_length=160)


class OnnxExportRequest(BaseModel):
    opset: int = Field(default=18, ge=17, le=20)
    warmup_runs: int = Field(default=2, ge=0, le=20)
    benchmark_runs: int = Field(default=10, ge=1, le=100)


class AutoAnnotationRequest(BaseModel):
    model_version_id: str | None = Field(default=None, min_length=1, max_length=160)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1, le=1000)
    device: str = Field(default="auto", min_length=1, max_length=32)


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
    security_settings: SecuritySettings | None = None,
    training_enabled: bool | None = None,
) -> FastAPI:
    if project_root is not None and workspace is not None:
        raise ValueError("Use either project_root or workspace, not both")
    runtime = runtime_settings or RuntimeSettings.from_sources(
        workspace or project_root,
        db_path=db_path,
        datasets_root=datasets_root,
    )
    security = security_settings or SecuritySettings.from_sources(runtime.state_dir)

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
    app.add_middleware(SecurityBoundaryMiddleware, policy=security)
    app.state.settings = runtime
    app.state.security = security
    app.state.training_submission_enabled = (
        _training_stack_available() if training_enabled is None else training_enabled
    )
    app.state.store = RunStore(runtime.db_path)
    app.state.datasets = DatasetStore(runtime.db_path)
    app.state.models = ModelStore(runtime.db_path)
    app.state.datasets_root = runtime.datasets_root
    app.state.datasets_root.mkdir(parents=True, exist_ok=True)
    app.state.cvat = cvat_client or CvatClient(CvatSettings.from_env(runtime.cvat_config))
    app.state.project_root = runtime.workspace
    app.state.active_model_config = runtime.active_model_config
    app.state.activation_lock = threading.Lock()
    app.state.active_model_integrity = "none"
    if inference_service is not None:
        app.state.inference = inference_service
        app.state.active_model_integrity = "injected"
    else:
        active_model = app.state.models.get_active_model(required=False)
        if active_model is not None:
            try:
                app.state.models.verify_artifact(active_model)
                _atomic_write_json(app.state.active_model_config, _model_config_payload(app, active_model))
                app.state.inference = DetectionInferenceService.from_config(
                    app.state.active_model_config, runtime.workspace
                )
                app.state.active_model_integrity = "verified"
            except (KeyError, OSError, ValueError):
                app.state.inference = DetectionInferenceService(None)
                app.state.active_model_integrity = "failed"
        else:
            app.state.inference = DetectionInferenceService.from_config(
                app.state.active_model_config, runtime.workspace
            )
            app.state.active_model_integrity = (
                "legacy" if app.state.inference.available else "none"
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
            "security_mode": app.state.security.mode.value,
            "authentication_required": app.state.security.mode is SecurityMode.NETWORK,
            "max_request_bytes": app.state.security.max_request_bytes,
            "active_model_integrity": app.state.active_model_integrity,
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
        except (ImportError, RuntimeError) as exc:
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

    @app.post("/api/datasets/{dataset_id}/auto-annotations", status_code=201)
    async def create_auto_annotations(
        dataset_id: str,
        auto_request: AutoAnnotationRequest | None = None,
    ) -> dict:
        auto_request = auto_request or AutoAnnotationRequest()
        try:
            dataset = app.state.datasets.get_dataset(dataset_id)
            images = app.state.datasets.list_images(dataset_id, limit=5000)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if dataset.status.value == "frozen":
            raise HTTPException(status_code=409, detail="Frozen datasets cannot accept annotations")
        try:
            model = (
                app.state.models.get_model(auto_request.model_version_id)
                if auto_request.model_version_id
                else app.state.models.get_active_model()
            )
            app.state.models.verify_artifact(model)
            if model.approval_status != "approved":
                raise ValueError("Model version must be approved before automatic annotation")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        predictor = DetectionInferenceService(
            model.artifact_path,
            confidence=auto_request.confidence,
            max_detections=auto_request.max_detections,
            device=auto_request.device,
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        version_id = f"auto-{timestamp}-{uuid4().hex[:8]}"
        version_dir = Path(dataset.root_dir) / "annotations" / "versions" / version_id
        try:
            generated = await run_in_threadpool(
                partial(
                    _generate_checked_auto_annotations,
                    dataset,
                    images,
                    predictor,
                    version_dir,
                    model_version_id=model.model_version_id,
                    model_sha256=model.artifact_sha256,
                    confidence=auto_request.confidence,
                )
            )
            version = app.state.datasets.register_annotation_version(
                dataset_id,
                version_id,
                source="model_prediction",
                format="normalized-detection-text-v1",
                root_dir=version_dir,
                manifest_path=generated["manifest_path"],
                labeled_count=generated["labeled_count"],
                unlabeled_count=generated["unlabeled_count"],
                review_status="candidate",
            )
        except (ImportError, RuntimeError) as exc:
            shutil.rmtree(version_dir, ignore_errors=True)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            shutil.rmtree(version_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "annotation_version": version.model_dump(mode="json"),
            "model_version": model.model_dump(mode="json"),
            "detection_count": generated["detection_count"],
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

    @app.post("/api/datasets/{dataset_id}/annotation-versions/{version_id}/approve")
    def approve_annotation_version(dataset_id: str, version_id: str, request: Request) -> dict:
        try:
            version = app.state.datasets.approve_annotation_version(
                dataset_id,
                version_id,
                actor=request.state.actor,
            )
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
            snapshot_record = app.state.datasets.register_training_snapshot(
                dataset_id, snapshot
            )
        except KeyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "snapshot": snapshot,
            "record": snapshot_record.model_dump(mode="json"),
            "quality": quality.to_dict(),
            "stats": stats.to_dict(),
        }

    @app.get("/api/datasets/{dataset_id}/training-snapshots")
    def list_training_snapshots(
        dataset_id: str,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict]:
        try:
            snapshots = app.state.datasets.list_training_snapshots(dataset_id, limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return [snapshot.model_dump(mode="json") for snapshot in snapshots]

    @app.get("/api/datasets/{dataset_id}/training-snapshots/{snapshot_id}")
    def get_training_snapshot(dataset_id: str, snapshot_id: str) -> dict:
        try:
            snapshot = app.state.datasets.get_training_snapshot(dataset_id, snapshot_id)
            app.state.datasets.verify_training_snapshot(snapshot)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return snapshot.model_dump(mode="json")

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

    @app.post("/api/runs/{run_id}/register", status_code=201)
    def register_run_model(run_id: str, request: Request) -> dict:
        try:
            record = app.state.store.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        artifact = Path(record.output_dir) / "trainer_output" / "weights" / "best.ckpt"
        try:
            model = app.state.models.register_run(record, artifact, actor=request.state.actor)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"model_version": model.model_dump(mode="json")}

    @app.get("/api/models")
    def list_models(
        limit: int = Query(default=100, ge=1, le=1000),
        project: str | None = Query(default=None, min_length=1, max_length=64),
    ) -> list[dict]:
        return [
            model.model_dump(mode="json")
            for model in app.state.models.list_models(limit=limit, project=project)
        ]

    @app.get("/api/models/active")
    def get_active_model() -> dict:
        try:
            model = app.state.models.get_active_model()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return model.model_dump(mode="json")

    @app.get("/api/models/activation-history")
    def get_model_activation_history(
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict]:
        return [
            event.model_dump(mode="json")
            for event in app.state.models.activation_history(limit=limit)
        ]

    @app.get("/api/models/{model_version_id}")
    def get_model(model_version_id: str) -> dict:
        try:
            model = app.state.models.get_model(model_version_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return model.model_dump(mode="json")

    @app.post("/api/models/{model_version_id}/approve")
    def approve_registered_model(model_version_id: str, request: Request) -> dict:
        try:
            model = app.state.models.approve(model_version_id, actor=request.state.actor)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return model.model_dump(mode="json")

    @app.post("/api/models/{model_version_id}/activate")
    def activate_registered_model(model_version_id: str, request: Request) -> dict:
        try:
            model = app.state.models.get_model(model_version_id)
            return _activate_model(app, model, request.state.actor)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/models/{model_version_id}/exports/onnx", status_code=201)
    async def export_registered_model_onnx(
        model_version_id: str,
        export_request: OnnxExportRequest | None = None,
    ) -> dict:
        export_request = export_request or OnnxExportRequest()
        try:
            model = app.state.models.get_model(model_version_id)
            app.state.models.verify_artifact(model)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        output_dir = (
            app.state.project_root
            / "outputs"
            / "deployments"
            / model.model_version_id
            / f"onnx-opset{export_request.opset}"
        )
        if output_dir.exists():
            try:
                manifest = verify_onnx_package(output_dir)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if manifest.get("source", {}).get("checkpoint_sha256") != model.artifact_sha256:
                raise HTTPException(status_code=409, detail="Existing ONNX export has a different source")
            return {"created": False, "export": {**manifest, "package_dir": str(output_dir)}}
        try:
            manifest = await run_in_threadpool(
                partial(
                    export_onnx_package,
                    model.artifact_path,
                    output_dir,
                    opset=export_request.opset,
                    warmup_runs=export_request.warmup_runs,
                    benchmark_runs=export_request.benchmark_runs,
                )
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (FileExistsError, OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"created": True, "export": manifest}

    @app.get("/api/models/{model_version_id}/exports/onnx")
    def get_registered_model_onnx_export(
        model_version_id: str,
        opset: int = Query(default=18, ge=17, le=20),
    ) -> dict:
        try:
            model = app.state.models.get_model(model_version_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        output_dir = (
            app.state.project_root
            / "outputs"
            / "deployments"
            / model.model_version_id
            / f"onnx-opset{opset}"
        )
        try:
            manifest = verify_onnx_package(output_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {**manifest, "package_dir": str(output_dir)}

    @app.post("/api/models/rollback")
    def rollback_model(request: Request) -> dict:
        try:
            target = app.state.models.rollback_target()
            return _activate_model(app, target, request.state.actor, action="rollback")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/activate")
    def activate_run_model(run_id: str, request: Request) -> dict:
        try:
            record = app.state.store.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        model_path = Path(record.output_dir) / "trainer_output" / "weights" / "best.ckpt"
        try:
            model = app.state.models.register_run(record, model_path, actor=request.state.actor)
            model = app.state.models.approve(model.model_version_id, actor=request.state.actor)
            return _activate_model(app, model, request.state.actor)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

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

    if security.mode is SecurityMode.NETWORK:
        _configure_network_openapi(app)
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


def _configure_network_openapi(app: FastAPI) -> None:
    """Expose the middleware Bearer contract in Swagger without protecting health."""

    def custom_openapi() -> dict:
        if app.openapi_schema is not None:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        components = schema.setdefault("components", {})
        schemes = components.setdefault("securitySchemes", {})
        schemes["BearerAuth"] = {"type": "http", "scheme": "bearer"}
        for path, operations in schema.get("paths", {}).items():
            for operation in operations.values():
                if isinstance(operation, dict):
                    operation["security"] = [] if path == "/api/health" else [{"BearerAuth": []}]
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi


def _activate_model(
    app: FastAPI,
    model: ModelVersionRecord,
    actor: str,
    *,
    action: str = "activate",
) -> dict:
    app.state.models.verify_artifact(model)
    payload = _model_config_payload(app, model)

    with app.state.activation_lock:
        previous_model = app.state.models.get_active_model(required=False)
        previous_id = previous_model.model_version_id if previous_model else None
        previous_config = (
            app.state.active_model_config.read_bytes()
            if app.state.active_model_config.is_file()
            else None
        )
        try:
            _atomic_write_json(app.state.active_model_config, payload)
            candidate = DetectionInferenceService.from_config(
                app.state.active_model_config,
                app.state.project_root,
            )
            activation = app.state.models.activate(
                model.model_version_id,
                actor=actor,
                action=action,
                expected_previous_id=previous_id,
            )
        except Exception:
            _restore_file(app.state.active_model_config, previous_config)
            raise
        app.state.inference = candidate
        app.state.active_model_integrity = "verified"

    return {
        "activated": True,
        "model_version": app.state.models.get_model(model.model_version_id).model_dump(mode="json"),
        "activation": activation.model_dump(mode="json"),
        "inference": app.state.inference.status(),
    }


def _model_config_payload(app: FastAPI, model: ModelVersionRecord) -> dict:
    run = app.state.store.get_run(model.run_id)
    model_path = Path(model.artifact_path)
    try:
        stored_model = model_path.relative_to(app.state.project_root)
    except ValueError:
        stored_model = model_path
    return {
        "model": str(stored_model).replace("\\", "/"),
        "confidence": run.config["train"]["score_threshold"],
        "max_detections": 100,
        "device": run.config["train"]["device"],
        "run_id": run.run_id,
        "model_version_id": model.model_version_id,
        "artifact_sha256": model.artifact_sha256,
    }


def _generate_checked_auto_annotations(
    dataset,
    images,
    predictor: DetectionInferenceService,
    version_dir: Path,
    *,
    model_version_id: str,
    model_sha256: str,
    confidence: float,
) -> dict:
    class_names = predictor.class_names
    if class_names != dataset.labels:
        raise ValueError(
            "Registered model classes do not match the target dataset: "
            f"expected {dataset.labels}, got {class_names}"
        )
    return generate_auto_annotations(
        dataset,
        images,
        predictor,
        version_dir,
        model_version_id=model_version_id,
        model_sha256=model_sha256,
        confidence=confidence,
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        staging.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _restore_file(path: Path, previous: bytes | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
        return
    staging = path.with_name(f".{path.name}.{uuid4().hex}.restore")
    try:
        staging.write_bytes(previous)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


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
