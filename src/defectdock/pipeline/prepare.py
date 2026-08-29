"""Prepare a project directory from a raw image directory (upload → pipeline entry).

Turns a folder of uploaded images into a ``seed/`` + ``unlabeled/`` workspace.
The operator can select an initial seed set for labeling without changing the
source directory.
"""

from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def prepare_project(image_dir: str | Path, out_dir: str | Path) -> dict:
    """Copy an image directory into the seed/unlabeled project layout.

    Returns ``{unlabeled_images, seed_dir, unlabeled_dir}``.
    """
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    unlabeled = out_dir / "unlabeled"
    seed_images = out_dir / "seed" / "images"
    seed_labels = out_dir / "seed" / "labels"
    for directory in (unlabeled, seed_images, seed_labels):
        directory.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image in iter_images(image_dir):
        destination = unlabeled / image.name
        if not destination.exists():
            destination.write_bytes(image.read_bytes())
        copied += 1

    return {
        "unlabeled_images": copied,
        "seed_dir": str(out_dir / "seed"),
        "unlabeled_dir": str(unlabeled),
    }
