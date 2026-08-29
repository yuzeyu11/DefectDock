"""Permissively licensed TorchVision object-detection training adapter.

PyTorch and TorchVision use permissive licenses. Imports stay lazy so the API,
data checks and unit tests work without the optional training stack installed.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from defectdock.config import RunConfig
from defectdock.data.cv.check import (
    _iter_image_paths,
    _label_path_for,
    _parse_names,
    _split_sources,
    load_data_yaml,
    resolve_dataset_root,
)
from defectdock.engines.base import (
    CancellationCallback,
    EngineResult,
    EventCallback,
    TrainingCancelled,
)
from defectdock.eval import Box, load_ground_truth, optimize_threshold, summarize
from defectdock.provenance import file_sha256, write_run_manifest

SUPPORTED_MODELS = ("fasterrcnn-resnet50-fpn-v2",)


class DetectionDataset:
    """Read the interoperable normalized detection-text layout."""

    def __init__(self, data_yaml: str | Path, split: str) -> None:
        self.data_yaml = Path(data_yaml).resolve()
        config = load_data_yaml(self.data_yaml)
        self.classes, _ = _parse_names(config)
        root = resolve_dataset_root(self.data_yaml, config)
        raw_split = config.get(split)
        if raw_split is None:
            raise ValueError(f"data.yaml has no '{split}' split")
        self.images = [
            image
            for source in _split_sources(root, raw_split)
            if source.exists()
            for image in _iter_image_paths(source)
        ]
        if not self.images:
            raise ValueError(f"dataset split '{split}' contains no images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch
        from torchvision.transforms.functional import pil_to_tensor

        image_path = self.images[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            tensor = pil_to_tensor(image).float().div(255.0)

        label_path = _label_path_for(image_path)
        labels = load_ground_truth(label_path) if label_path and label_path.is_file() else []
        boxes = [
            [box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height]
            for box in labels
        ]
        class_ids = [box.class_id + 1 for box in labels]  # zero is background
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(class_ids, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return tensor, target


def collate_detection_batch(batch):
    return tuple(zip(*batch))


def build_model(
    architecture: str,
    num_classes: int,
    *,
    pretrained: bool,
    input_size: int = 640,
):
    """Build a TorchVision detector with a task-specific classification head."""
    if architecture not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported TorchVision detector: {architecture}")
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        min_size=input_size,
        max_size=input_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def build_plan(
    config: RunConfig,
    run_dir: Path,
    project_root: str | Path | None = None,
) -> dict:
    resolved_project_root = (
        Path.cwd().resolve() if project_root is None else Path(project_root).resolve()
    )
    return {
        "project": config.project,
        "task": config.task,
        "engine": config.engine,
        "model": config.model,
        "dataset": config.resolve_dataset_path(resolved_project_root),
        "dataset_version": config.dataset.version,
        "config_hash": config.config_hash,
        "run_dir": str(run_dir.resolve()),
        "train": config.train.model_dump(mode="json"),
        "license_boundary": {
            "training_framework": "PyTorch/TorchVision",
            "framework_license": "permissive dependency set",
            "agpl_runtime_included": False,
        },
    }


def _resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _emit(events_path: Path, event: dict, callback: EventCallback | None) -> None:
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    if callback:
        callback(event)


def _check_cancelled(should_cancel: CancellationCallback | None) -> None:
    if should_cancel and should_cancel():
        raise TrainingCancelled("training cancelled by operator")


def _save_checkpoint(path: Path, model, config: RunConfig, classes: list[str]) -> None:
    import torch

    torch.save(
        {
            "schema_version": 1,
            "architecture": config.model,
            "classes": classes,
            "input_size": config.train.imgsz,
            "framework": "torchvision",
            "framework_license": "BSD-3-Clause",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_hash": config.config_hash,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path, *, device: str = "cpu"):
    import torch

    checkpoint = torch.load(Path(path).resolve(), map_location=device, weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported DefectDock checkpoint schema")
    classes = list(checkpoint["classes"])
    model = build_model(
        checkpoint["architecture"],
        len(classes),
        pretrained=False,
        input_size=int(checkpoint.get("input_size", 640)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, classes, checkpoint


def _evaluate(model, dataset: DetectionDataset, device, config: RunConfig) -> dict[str, Any]:
    import torch

    ground_truths: list[Box] = []
    predictions: list[Box] = []
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            image_id = str(dataset.images[index].resolve())
            for coordinates, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
                ground_truths.append(Box(label - 1, *map(float, coordinates), image_id=image_id))
            output = model([image.to(device)])[0]
            for coordinates, label, score in zip(
                output["boxes"].detach().cpu().tolist(),
                output["labels"].detach().cpu().tolist(),
                output["scores"].detach().cpu().tolist(),
            ):
                predictions.append(
                    Box(label - 1, *map(float, coordinates), confidence=float(score), image_id=image_id)
                )
    operating = [box for box in predictions if box.confidence >= config.train.score_threshold]
    summary = summarize(dataset.classes, ground_truths, operating, config.train.iou_threshold)
    summary.threshold = config.train.score_threshold
    scan = optimize_threshold(ground_truths, predictions, config.train.iou_threshold)
    return {"standard": summary.to_dict(), "threshold_scan": scan.to_dict()}


def run_training(
    config: RunConfig,
    run_dir: str | Path,
    on_event: EventCallback | None = None,
    should_cancel: CancellationCallback | None = None,
    *,
    project_root: str | Path | None = None,
) -> EngineResult:
    """Train and validate one TorchVision Faster R-CNN model."""
    import torch
    from torch.utils.data import DataLoader

    run_dir = Path(run_dir).resolve()
    trainer_dir = run_dir / "trainer_output"
    weights_dir = trainer_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"

    _check_cancelled(should_cancel)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.train.seed)

    resolved_project_root = (
        Path.cwd().resolve() if project_root is None else Path(project_root).resolve()
    )
    dataset_path = config.resolve_dataset_path(resolved_project_root)
    train_dataset = DetectionDataset(dataset_path, "train")
    val_dataset = DetectionDataset(dataset_path, "val")
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("train and val splits must use the same class list")
    device = _resolve_device(config.train.device)
    model = build_model(
        config.model,
        len(train_dataset.classes),
        pretrained=config.train.pretrained,
        input_size=config.train.imgsz,
    )
    model.to(device)
    write_run_manifest(
        run_dir,
        config,
        resolved_project_root,
        include_accelerator=True,
        pretrained_weight=_pretrained_weight_provenance(config.train.pretrained),
    )
    _check_cancelled(should_cancel)

    loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch,
        shuffle=True,
        num_workers=config.train.workers,
        collate_fn=collate_detection_batch,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=config.train.learning_rate,
        momentum=config.train.momentum,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.train.step_size, gamma=config.train.gamma
    )

    _emit(
        events_path,
        {"event": "training_started", "engine": config.engine, "model": config.model, "device": str(device)},
        on_event,
    )
    best_loss = float("inf")
    best_path = weights_dir / "best.ckpt"
    last_path = weights_dir / "last.ckpt"
    for epoch in range(1, config.train.epochs + 1):
        _check_cancelled(should_cancel)
        model.train()
        total_loss = 0.0
        batches = 0
        for images, targets in loader:
            _check_cancelled(should_cancel)
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        epoch_loss = total_loss / max(1, batches)
        _save_checkpoint(last_path, model, config, train_dataset.classes)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            _save_checkpoint(best_path, model, config, train_dataset.classes)
        _emit(
            events_path,
            {
                "event": "epoch_end",
                "epoch": epoch,
                "epochs": config.train.epochs,
                "metrics": {"train_loss": round(epoch_loss, 6)},
            },
            on_event,
        )

    _check_cancelled(should_cancel)
    metrics = _evaluate(model, val_dataset, device, config)
    _check_cancelled(should_cancel)
    result = EngineResult(
        trainer_output=str(trainer_dir),
        best_model=str(best_path),
        last_model=str(last_path),
        metrics=metrics,
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    _emit(events_path, {"event": "training_completed", "metrics": metrics}, on_event)
    return result


def _pretrained_weight_provenance(enabled: bool) -> dict:
    if not enabled:
        return {"enabled": False, "resolved": True, "identifier": None, "sha256": None}
    import torch
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    cached_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    return {
        "enabled": True,
        "resolved": cached_path.is_file(),
        "identifier": f"FasterRCNN_ResNet50_FPN_V2_Weights.{weights.name}",
        "url": weights.url,
        "cache_path": str(cached_path),
        "sha256": file_sha256(cached_path) if cached_path.is_file() else None,
    }


class TorchvisionEngine:
    name = "torchvision"

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = (
            Path.cwd().resolve() if project_root is None else Path(project_root).resolve()
        )

    def plan(self, config: RunConfig, run_dir: Path) -> dict:
        return build_plan(config, run_dir, self.project_root)

    def run(
        self,
        config: RunConfig,
        run_dir: str | Path,
        on_event: EventCallback | None = None,
        should_cancel: CancellationCallback | None = None,
        *,
        project_root: str | Path | None = None,
    ) -> EngineResult:
        return run_training(
            config,
            run_dir,
            on_event,
            should_cancel,
            project_root=project_root or self.project_root,
        )
