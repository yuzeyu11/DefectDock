"""DefectDock command-line interface."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml

from defectdock.config import load_run_config
from defectdock.data.cv import check_dataset, compute_stats, convert_voc_dataset, import_gc10_dataset
from defectdock.db import RunStore
from defectdock.domain import RunStatus, new_run_id
from defectdock.engines import build_plan, run_training
from defectdock.inference import DetectionInferenceService
from defectdock.integrations import CvatSettings
from defectdock.pipeline import prepare_project, recommend_model
from defectdock.provenance import write_run_manifest
from defectdock.settings import RuntimeSettings
from defectdock.stream import stream_camera

app = typer.Typer(no_args_is_help=True, help="DefectDock industrial-vision lifecycle CLI")
data_app = typer.Typer(no_args_is_help=True, help="Inspect and convert detection datasets")
app.add_typer(data_app, name="data")
_runtime_settings: RuntimeSettings | None = None


@app.callback()
def configure_runtime(
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Writable project workspace (or set DEFECTDOCK_WORKSPACE)",
            file_okay=False,
        ),
    ] = None,
) -> None:
    """Configure the writable workspace before running a command."""
    global _runtime_settings
    _runtime_settings = RuntimeSettings.from_sources(workspace)


def _json(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _settings() -> RuntimeSettings:
    return _runtime_settings or RuntimeSettings.from_sources()


def _store(path: Path | None = None) -> RunStore:
    settings = _settings()
    return RunStore(settings.db_path if path is None else settings.resolve(path))


@data_app.command("check")
def data_check(data_yaml: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Validate a normalized object-detection dataset."""
    report = check_dataset(data_yaml)
    _json(report.to_dict())
    if not report.ok:
        raise typer.Exit(code=1)


@data_app.command("stats")
def data_stats(data_yaml: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Compute per-class, box-size and image-size statistics."""
    try:
        _json(compute_stats(data_yaml).to_dict())
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


@data_app.command("convert-voc")
def data_convert_voc(
    image_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    xml_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out_dir: Annotated[Path, typer.Argument()],
    class_names: Annotated[str | None, typer.Option("--names", help="Comma-separated classes")] = None,
) -> None:
    """Convert Pascal VOC XML annotations to the normalized text layout."""
    names = [item.strip() for item in class_names.split(",") if item.strip()] if class_names else None
    _json(convert_voc_dataset(image_dir, xml_dir, out_dir, class_names=names))


@data_app.command("import-gc10")
def data_import_gc10(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out_dir: Annotated[Path, typer.Argument()],
    seed: Annotated[int, typer.Option(min=0)] = 42,
    val_ratio: Annotated[float, typer.Option(min=0.0, max=0.9)] = 0.1,
    test_ratio: Annotated[float, typer.Option(min=0.0, max=0.9)] = 0.1,
    include_unannotated: Annotated[bool, typer.Option("--include-unannotated")] = False,
    strict_unknown_labels: Annotated[bool, typer.Option("--strict-unknown-labels")] = False,
    materialize: Annotated[str, typer.Option(help="copy or hardlink")] = "copy",
) -> None:
    """Import GC10-DET and create deterministic train/val/test splits."""
    try:
        result = import_gc10_dataset(
            source,
            out_dir,
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            include_unannotated=include_unannotated,
            strict_unknown_labels=strict_unknown_labels,
            materialize=materialize,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        typer.echo(f"GC10 import failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _json({key: result[key] for key in ("data_yaml", "manifest_path", "audit", "splits")})


@app.command()
def doctor() -> None:
    """Report the local runtime without requiring the optional training stack."""
    packages: dict[str, str | None] = {}
    for name in ("defectdock", "torch", "torchvision", "fastapi", "pydantic"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda: dict[str, object] = {"available": False}
    try:
        import torch

        cuda["available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            cuda["device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    _json(
        {
            "workspace": str(_settings().workspace),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": packages,
            "cuda": cuda,
        }
    )


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
) -> None:
    """Start the local REST API."""
    import uvicorn

    from defectdock.api import create_app

    uvicorn.run(create_app(runtime_settings=_settings()), host=host, port=port, reload=False)


@app.command("configure-cvat")
def configure_cvat(
    url: Annotated[str, typer.Option(help="CVAT server URL")] = "http://localhost:8080",
) -> None:
    """Store a CVAT token in the ignored local runtime directory."""
    token = typer.prompt("CVAT personal access token", hide_input=True).strip()
    if not token:
        typer.echo("Token cannot be empty", err=True)
        raise typer.Exit(code=2)
    path = CvatSettings(url=url.rstrip("/"), access_token=token).save_local(
        _settings().cvat_config
    )
    typer.echo(f"CVAT token saved to: {path}")


@app.command("recommend")
def recommend(
    num_images: Annotated[int, typer.Argument(help="Number of labeled images")],
    num_classes: Annotated[int, typer.Option(help="Number of classes")] = 1,
    realtime: Annotated[bool, typer.Option("--realtime")] = False,
) -> None:
    """Recommend a conservative built-in training preset."""
    _json(recommend_model(num_images, num_classes=num_classes, realtime=realtime).to_dict())


@app.command("prepare")
def prepare(
    image_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    project_dir: Annotated[Path, typer.Argument()],
) -> None:
    """Prepare a seed/unlabeled annotation workspace from raw images."""
    _json(prepare_project(image_dir, project_dir))


@app.command("deploy")
def deploy(
    model_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.25,
    max_detections: Annotated[int, typer.Option(min=1)] = 100,
    device: Annotated[str, typer.Option()] = "auto",
) -> None:
    """Activate a DefectDock checkpoint for the API and CLI."""
    settings = _settings()
    resolved = model_path.resolve()
    try:
        stored = resolved.relative_to(settings.workspace)
    except ValueError:
        stored = resolved
    payload = {
        "model": str(stored).replace("\\", "/"),
        "confidence": confidence,
        "max_detections": max_detections,
        "device": device,
    }
    settings.active_model_config.parent.mkdir(parents=True, exist_ok=True)
    settings.active_model_config.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _json(payload)


@app.command("detect")
def detect(
    image_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    model_path: Annotated[Path | None, typer.Option("--model", exists=True, dir_okay=False)] = None,
    confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.25,
    max_detections: Annotated[int, typer.Option(min=1)] = 100,
    device: Annotated[str, typer.Option()] = "auto",
) -> None:
    """Run inference on one image and print structured JSON."""
    settings = _settings()
    service = (
        DetectionInferenceService(model_path, confidence=confidence, max_detections=max_detections, device=device)
        if model_path
        else DetectionInferenceService.from_config(
            settings.active_model_config, settings.workspace
        )
    )
    try:
        _json(service.predict(image_path.read_bytes(), image_path.name))
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"detect failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("stream")
def stream(
    source: Annotated[str, typer.Argument(help="RTSP URL, USB index or video path")],
    model_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.25,
    device: Annotated[str, typer.Option()] = "auto",
    output_jsonl: Annotated[Path | None, typer.Option()] = None,
    annotate_dir: Annotated[Path | None, typer.Option()] = None,
    max_frames: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Run detection on an RTSP, USB or file video source."""
    result = stream_camera(
        source,
        model_path,
        [],
        conf=confidence,
        device=device,
        output_jsonl=output_jsonl,
        annotate_dir=annotate_dir,
        max_frames=max_frames,
    )
    _json(result.to_dict())


@app.command("validate")
def validate_config(config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Validate a run configuration."""
    config = load_run_config(config_path)
    _json({"valid": True, "config_hash": config.config_hash, "config": config.model_dump(mode="json")})


@app.command("plan")
def plan_run(config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Resolve a training plan without touching the database or accelerator."""
    settings = _settings()
    config = load_run_config(config_path)
    placeholder = (
        config.resolve_output_root(settings.workspace)
        / config.project
        / config.task
        / "<run-id>"
    )
    _json(build_plan(config, placeholder, settings.workspace))


@app.command("run")
def run_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    db_path: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """Create and execute a reproducible TorchVision training run."""
    settings = _settings()
    config = load_run_config(config_path)
    if dry_run:
        placeholder = (
            config.resolve_output_root(settings.workspace)
            / config.project
            / config.task
            / "<run-id>"
        )
        _json(build_plan(config, placeholder, settings.workspace))
        return

    run_id = new_run_id(config.project)
    run_dir = (
        config.resolve_output_root(settings.workspace) / config.project / config.task / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    write_run_manifest(run_dir, config, settings.workspace)
    store = _store(db_path)
    store.create_run(config, run_dir, run_id=run_id)
    store.update_status(run_id, RunStatus.RUNNING)
    typer.echo(f"run_id: {run_id}")
    typer.echo(f"output_dir: {run_dir}")
    try:
        result = run_training(
            config,
            run_dir,
            on_event=_print_event,
            project_root=settings.workspace,
        )
    except (KeyboardInterrupt, SystemExit):
        store.update_status(run_id, RunStatus.CANCELLED, error="cancelled before completion")
        raise typer.Exit(code=130)
    except Exception as exc:
        store.update_status(run_id, RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
        typer.echo(f"training failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    store.update_status(run_id, RunStatus.SUCCEEDED, metrics=result.metrics)
    _json(result.metrics)


@app.command("list")
def list_runs(
    limit: Annotated[int, typer.Option(min=1, max=1000)] = 20,
    project: Annotated[str | None, typer.Option()] = None,
    db_path: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """List recent training runs."""
    _json([item.model_dump(mode="json") for item in _store(db_path).list_runs(limit=limit, project=project)])


@app.command("show")
def show_run(
    run_id: str,
    db_path: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """Show one training-run record."""
    try:
        _json(_store(db_path).get_run(run_id).model_dump(mode="json"))
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


def _print_event(event: dict) -> None:
    if event.get("event") == "epoch_end":
        typer.echo(f"epoch {event.get('epoch')}/{event.get('epochs')}")
    else:
        typer.echo(f"event: {event.get('event')}")


if __name__ == "__main__":
    app()
