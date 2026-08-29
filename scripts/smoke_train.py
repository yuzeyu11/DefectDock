"""Run a tiny end-to-end TorchVision training smoke test on real hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from defectdock.config import RunConfig
from defectdock.engines import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="Empty directory for smoke-test artifacts")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete device")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    if workspace.exists() and any(workspace.iterdir()):
        raise SystemExit(f"Smoke workspace must be empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    data_yaml = _create_dataset(workspace / "datasets" / "smoke")
    config = RunConfig.model_validate(
        {
            "project": "hardware-smoke",
            "dataset": {"path": str(data_yaml), "version": "synthetic-v1"},
            "train": {
                "epochs": 1,
                "imgsz": 128,
                "batch": 1,
                "device": args.device,
                "workers": 0,
                "pretrained": False,
                "seed": 7,
            },
            "output_root": "outputs",
        }
    )
    run_dir = workspace / "outputs" / "hardware-smoke" / "object-detection" / "smoke-run"
    result = run_training(config, run_dir, project_root=workspace)
    started_event = json.loads(
        (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    print(
        json.dumps(
            {
                "workspace": str(workspace),
                "device_requested": args.device,
                "device_actual": started_event["device"],
                "config_hash": config.config_hash,
                "best_model": result.best_model,
                "last_model": result.last_model,
                "metrics_summary": {
                    key: result.metrics["standard"][key]
                    for key in ("gt_total", "tp", "fp", "fn", "precision", "recall")
                },
                "manifest": str(run_dir / "run.manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _create_dataset(root: Path) -> Path:
    for split, color in (("train", (45, 70, 110)), ("val", (65, 90, 130))):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (128, 128), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle((38, 38, 90, 90), outline=(240, 220, 40), width=4)
        image.save(image_dir / f"{split}.png")
        (label_dir / f"{split}.txt").write_text(
            "0 0.5 0.5 0.40625 0.40625\n",
            encoding="utf-8",
        )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": ["synthetic-defect"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


if __name__ == "__main__":
    raise SystemExit(main())
