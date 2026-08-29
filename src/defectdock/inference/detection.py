"""Lazy inference for checkpoints produced by DefectDock's training adapters."""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class DetectionInferenceService:
    def __init__(
        self,
        model_path: str | Path | None,
        *,
        confidence: float = 0.25,
        max_detections: int = 100,
        device: str = "auto",
    ) -> None:
        self.model_path = Path(model_path).resolve() if model_path else None
        self.confidence = confidence
        self.max_detections = max_detections
        self.device = device
        self._model: Any | None = None
        self._classes: list[str] = []
        self._resolved_device = "cpu"
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    @classmethod
    def from_config(cls, config_path: str | Path, project_root: str | Path) -> "DetectionInferenceService":
        path = Path(config_path)
        if not path.is_file():
            return cls(None)
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_path = Path(payload["model"])
        if not model_path.is_absolute():
            model_path = Path(project_root) / model_path
        return cls(
            model_path,
            confidence=float(payload.get("confidence", 0.25)),
            max_detections=int(payload.get("max_detections", 100)),
            device=str(payload.get("device", "auto")),
        )

    @property
    def available(self) -> bool:
        return self.model_path is not None and self.model_path.is_file()

    def status(self) -> dict[str, Any]:
        return {
            "configured": self.model_path is not None,
            "available": self.available,
            "loaded": self._model is not None,
            "model": str(self.model_path) if self.model_path else None,
            "engine": "torchvision",
            "confidence": self.confidence,
            "max_detections": self.max_detections,
            "device": self._resolved_device if self._model is not None else self.device,
        }

    def predict(self, image_bytes: bytes, filename: str = "image") -> dict[str, Any]:
        image = self._decode_image(image_bytes)
        result = self._predict_image(image)
        return {
            "filename": Path(filename).name,
            "width": image.width,
            "height": image.height,
            "model": self.model_path.name if self.model_path else None,
            "thresholds": {"confidence": self.confidence},
            **result,
        }

    def predict_array(self, frame: np.ndarray) -> dict[str, Any]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR image array with shape HxWx3")
        image = Image.fromarray(frame[:, :, ::-1])
        return self._predict_image(image)

    def _predict_image(self, image: Image.Image) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("Active inference model is not configured or is missing")
        import torch
        from torchvision.transforms.functional import pil_to_tensor

        model = self._get_model()
        tensor = pil_to_tensor(image).float().div(255.0).to(self._resolved_device)
        started = time.perf_counter()
        with self._predict_lock, torch.inference_mode():
            output = model([tensor])[0]
        elapsed_ms = (time.perf_counter() - started) * 1000

        detections: list[dict[str, Any]] = []
        for coordinates, label, score in zip(
            output["boxes"].detach().cpu().tolist(),
            output["labels"].detach().cpu().tolist(),
            output["scores"].detach().cpu().tolist(),
        ):
            confidence = float(score)
            if confidence < self.confidence:
                continue
            class_id = int(label) - 1
            x1, y1, x2, y2 = map(float, coordinates)
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self._classes[class_id] if 0 <= class_id < len(self._classes) else str(class_id),
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
            )
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        detections = detections[: self.max_detections]
        return {
            "timing_ms": {"inference": round(elapsed_ms, 3), "total": round(elapsed_ms, 3)},
            "detections": detections,
        }

    def _get_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    import torch

                    from defectdock.engines.torchvision import load_checkpoint

                    requested = self.device
                    self._resolved_device = (
                        "cuda"
                        if requested == "auto" and torch.cuda.is_available()
                        else "cpu"
                        if requested == "auto"
                        else requested
                    )
                    self._model, self._classes, _ = load_checkpoint(
                        self.model_path, device=self._resolved_device
                    )
        return self._model

    @staticmethod
    def _decode_image(image_bytes: bytes) -> Image.Image:
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                if source.width * source.height > 60_000_000:
                    raise ValueError("Image dimensions are too large")
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
                return image
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Uploaded file is not a readable image") from exc
