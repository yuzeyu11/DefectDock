"""Field feedback loop: re-ingest missed / false-positive samples.

Deduplicates incoming samples by image SHA-256, stores the images in a feedback
directory, and writes a JSON manifest for downstream re-annotation and
incremental retraining. This is the tooling behind the "gets better with use"
data-closure promise in the customer proposal.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class FeedbackRecord:
    kind: str  # "miss" | "false_positive"
    class_name: str
    sha256: str
    image_name: str
    bbox: tuple[float, float, float, float] | None
    source: str | None = None


@dataclass
class FeedbackResult:
    accepted: list[FeedbackRecord] = field(default_factory=list)
    duplicate_images: int = 0
    skipped_missing_image: int = 0
    skipped_invalid_bbox: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": [asdict(record) for record in self.accepted],
            "duplicate_images": self.duplicate_images,
            "skipped_missing_image": self.skipped_missing_image,
            "skipped_invalid_bbox": self.skipped_invalid_bbox,
        }


def file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bbox(raw: Any) -> tuple[float, float, float, float] | None:
    """Validate a normalized xyxy box; return ``None`` when absent or invalid.

    Invalid means: not a 4-element sequence, non-numeric entries, values
    outside [0, 1], or zero/negative area. Callers decide whether an invalid
    *provided* box skips the sample or is simply dropped.
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = values
    if any(value < 0.0 or value > 1.0 for value in values):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return values


def ingest_feedback(
    samples: Iterable[dict[str, Any]],
    feedback_dir: str | Path,
    *,
    image_dir_name: str = "images",
    manifest_name: str = "feedback.json",
) -> FeedbackResult:
    """Ingest feedback samples into a feedback directory.

    Each sample dict::

        {
          "image_path": str,          # source image on disk (required)
          "kind": "miss" | "false_positive",
          "class_name": str,
          "bbox": [x1, y1, x2, y2],  # normalized xyxy (optional)
          "source": str,             # provenance note (optional)
        }

    Images are stored as ``<image_dir>/<sha256>.jpg`` (deduplicated by content);
    the manifest records every accepted sample with its provenance and box.

    A provided ``bbox`` must be a normalized xyxy 4-tuple with values in
    [0, 1] and positive area; samples carrying an invalid box are skipped and
    counted in ``FeedbackResult.skipped_invalid_bbox`` instead of crashing the
    whole batch.

    The manifest accumulates across calls: samples from previous ingests are
    preserved (deduplicated by image SHA-256 + kind + class) so the feedback
    directory remains a complete audit trail of the data-closure loop.
    """
    feedback_dir = Path(feedback_dir)
    images_dir = feedback_dir / image_dir_name
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = feedback_dir / manifest_name
    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    if not isinstance(existing, dict):
        existing = {}

    records: list[FeedbackRecord] = []
    seen_keys: set[tuple[str, str, str]] = set()

    def add_record(record: FeedbackRecord) -> None:
        key = (record.sha256, record.kind, record.class_name)
        if key not in seen_keys:
            seen_keys.add(key)
            records.append(record)

    # Preserve the audit trail from previous ingests before appending this batch.
    for entry in existing.get("samples", []) or []:
        try:
            add_record(
                FeedbackRecord(
                    kind=str(entry.get("kind", "miss")),
                    class_name=str(entry.get("class_name", "unknown")),
                    sha256=str(entry.get("sha256", "")),
                    image_name=str(entry.get("image_name", "")),
                    bbox=_parse_bbox(entry.get("bbox")),
                    source=entry.get("source"),
                )
            )
        except (AttributeError, TypeError):
            continue

    duplicate_images = 0
    skipped_missing_image = 0
    skipped_invalid_bbox = 0

    for sample in samples:
        image_path = Path(sample.get("image_path", ""))
        if not image_path.is_file():
            skipped_missing_image += 1
            continue
        raw_box = sample.get("bbox")
        bbox = _parse_bbox(raw_box)
        if raw_box is not None and bbox is None:
            # 提供了框但格式/范围非法：整条样本跳过并审计，不污染数据闭环。
            skipped_invalid_bbox += 1
            continue
        digest = file_sha256(image_path)

        extension = image_path.suffix.lower() or ".jpg"
        stored_name = f"{digest}{extension}"
        destination = images_dir / stored_name
        if not destination.is_file():
            destination.write_bytes(image_path.read_bytes())
        else:
            duplicate_images += 1

        add_record(
            FeedbackRecord(
                kind=str(sample.get("kind", "miss")),
                class_name=str(sample.get("class_name", "unknown")),
                sha256=digest,
                image_name=stored_name,
                bbox=bbox,
                source=sample.get("source"),
            )
        )

    manifest = {
        "count": len(records),
        "sha256": sorted({record.sha256 for record in records}),
        "samples": [asdict(record) for record in records],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return FeedbackResult(
        accepted=records,
        duplicate_images=duplicate_images,
        skipped_missing_image=skipped_missing_image,
        skipped_invalid_bbox=skipped_invalid_bbox,
    )
