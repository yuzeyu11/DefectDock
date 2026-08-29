"""Real-time camera / video-stream detection (RTSP, USB, local video).

The camera source is abstracted behind ``source`` so the same detector serves a
USB webcam (``0``), an RTSP stream (``rtsp://...``), or a local video file. The
caller only swaps the source string to attach real hardware — no other change.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

DetectionCallback = Callable[[dict], None]


@dataclass
class StreamResult:
    frames: int = 0
    detections: int = 0
    ng_frames: int = 0
    elapsed_s: float = 0.0
    fps: float = 0.0
    source: str = ""
    reason: str = "completed"
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class CameraStream:
    """Open a video source, run detector inference frame-by-frame, emit results.

    Network sources (RTSP/HTTP) get open/read timeouts so a dead stream cannot
    hang the detector; consecutive read failures on a network source abort the
    run with ``reason="stream_read_failed"`` instead of looping forever.
    """

    def __init__(
        self,
        source: str | int,
        model_path: str | Path,
        classes: list[str],
        *,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        device: int | str = 0,
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 3000,
        max_read_failures: int = 10,
    ) -> None:
        self.source = source
        self.model_path = Path(model_path)
        self.classes = classes
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.max_read_failures = max_read_failures
        self._model: Any | None = None
        self._cap: Any | None = None

    def open(self) -> bool:
        import cv2

        source = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
        cap = cv2.VideoCapture()
        # 网络后端（FFmpeg）支持这两个属性；文件/USB 后端不支持时 set 无副作用。
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms)

        # CAP_PROP_OPEN_TIMEOUT_MSEC 在部分 OpenCV 构建的 FFmpeg 后端不生效：
        # 死 RTSP 源会让 cap.open() 一直阻塞到 FFmpeg 自身默认超时（约 30s）。
        # 因此加硬超时兜底——在守护线程里 open，主线程到点即放弃并返回失败，
        # 让死流无法挂死检测器。
        result: dict[str, bool] = {}

        def _do_open() -> None:
            try:
                result["ok"] = bool(cap.open(source))
            except Exception:
                result["ok"] = False

        thread = threading.Thread(target=_do_open, daemon=True)
        thread.start()
        thread.join(self.open_timeout_ms / 1000.0)
        if thread.is_alive():
            # open 未在时限内完成：守护线程结束后会自行清理 cap 引用，
            # 主线程不并发 release，直接判失败退出。
            return False
        if result.get("ok"):
            self._cap = cap
            return True
        cap.release()
        return False

    def _is_local_file_source(self) -> bool:
        if not isinstance(self.source, str) or self.source.isdigit():
            return False
        if "://" in self.source:
            return False
        return Path(self.source).is_file()

    def read_frame(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def _get_model(self):
        if self._model is None:
            from defectdock.inference import DetectionInferenceService

            self._model = DetectionInferenceService(
                self.model_path,
                confidence=self.conf,
                device=str(self.device),
            )
        return self._model

    def detect(self, frame: np.ndarray) -> dict:
        model = self._get_model()
        result = model.predict_array(frame)
        detections = list(result["detections"])
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return {"detections": detections, "count": len(detections)}

    def annotate(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        import cv2

        canvas = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = int(detection["x1"]), int(detection["y1"]), int(detection["x2"]), int(detection["y2"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            cv2.putText(canvas, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return canvas

    def run(
        self,
        *,
        on_detection: DetectionCallback | None = None,
        output_jsonl: str | Path | None = None,
        annotate_dir: str | Path | None = None,
        max_frames: int | None = None,
    ) -> StreamResult:
        if not self.open():
            return StreamResult(reason="source_open_failed", source=str(self.source))

        jsonl_handle = None
        if output_jsonl:
            Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
            jsonl_handle = Path(output_jsonl).open("a", encoding="utf-8")
        if annotate_dir:
            Path(annotate_dir).mkdir(parents=True, exist_ok=True)

        result = StreamResult(source=str(self.source))
        start = time.perf_counter()
        read_failures = 0
        try:
            while max_frames is None or result.frames < max_frames:
                frame = self.read_frame()
                if frame is None:
                    if self._is_local_file_source():
                        # 本地视频正常播完即结束。
                        result.reason = "end_of_stream"
                        break
                    read_failures += 1
                    if read_failures >= self.max_read_failures:
                        # 网络源持续读失败：明确中止，避免无限空转。
                        result.reason = "stream_read_failed"
                        break
                    time.sleep(0.1)
                    continue
                read_failures = 0
                payload = self.detect(frame)
                result.frames += 1
                result.detections += payload["count"]
                if payload["count"] > 0:
                    result.ng_frames += 1
                payload["frame_index"] = result.frames
                payload["timestamp"] = time.time()

                if jsonl_handle:
                    jsonl_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                if annotate_dir:
                    import cv2

                    annotated = self.annotate(frame, payload["detections"])
                    cv2.imwrite(str(Path(annotate_dir) / f"frame_{result.frames:06d}.jpg"), annotated)
                if on_detection:
                    on_detection(payload)
        finally:
            if jsonl_handle:
                jsonl_handle.close()
            self.close()

        result.elapsed_s = time.perf_counter() - start
        result.fps = result.frames / result.elapsed_s if result.elapsed_s > 0 else 0.0
        result.summary = {
            "ng_rate": round(result.ng_frames / result.frames, 4) if result.frames else 0.0,
            "avg_detections_per_frame": round(result.detections / result.frames, 3) if result.frames else 0.0,
        }
        return result

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def stream_camera(
    source: str | int,
    model_path: str | Path,
    classes: list[str],
    *,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
    device: int | str = 0,
    output_jsonl: str | Path | None = None,
    annotate_dir: str | Path | None = None,
    max_frames: int | None = None,
    open_timeout_ms: int = 5000,
    read_timeout_ms: int = 3000,
    max_read_failures: int = 10,
) -> StreamResult:
    """Convenience one-shot wrapper around :class:`CameraStream`."""
    stream = CameraStream(
        source,
        model_path,
        classes,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        open_timeout_ms=open_timeout_ms,
        read_timeout_ms=read_timeout_ms,
        max_read_failures=max_read_failures,
    )
    return stream.run(output_jsonl=output_jsonl, annotate_dir=annotate_dir, max_frames=max_frames)
