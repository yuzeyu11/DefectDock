"""Conservative training presets for the built-in detection engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelRecommendation:
    model: str
    preset: str
    note: str
    epochs: int
    imgsz: int
    batch: int

    def to_dict(self) -> dict:
        return asdict(self)


def recommend_model(
    num_images: int,
    *,
    num_classes: int = 1,
    realtime: bool = False,
) -> ModelRecommendation:
    """Recommend a transparent starting point, not an unverifiable promise."""
    if num_images < 0 or num_classes < 1:
        raise ValueError("num_images must be non-negative and num_classes must be positive")
    if num_images < 300:
        preset, epochs, batch = "bootstrap", 40, 2
        note = "小数据集基线；优先补充独立验证集并监控过拟合"
    elif num_images < 3_000:
        preset, epochs, batch = "balanced", 25, 4
        note = "平衡训练预设；适合首轮可交付基线"
    else:
        preset, epochs, batch = "accuracy", 18, 4
        note = "较大数据集预设；应按显存与吞吐实测调整"
    if num_classes >= 20:
        epochs += 5
        note += "；类别较多，增加训练轮次"
    if realtime:
        note += "；Faster R-CNN 不是低延迟承诺，需用 ONNX/专用实时适配器完成现场验收"
    return ModelRecommendation(
        model="fasterrcnn-resnet50-fpn-v2",
        preset=preset,
        note=note,
        epochs=epochs,
        imgsz=640,
        batch=batch,
    )
