"""Read-only resources shipped inside the DefectDock wheel."""

from __future__ import annotations

from importlib import resources

import yaml

from defectdock.config import RunConfig


def load_example_configs() -> list[tuple[str, RunConfig]]:
    """Load packaged example configurations without assuming a source checkout."""
    root = resources.files(__package__).joinpath("configs", "examples")
    examples: list[tuple[str, RunConfig]] = []
    for item in sorted(root.iterdir(), key=lambda candidate: candidate.name):
        if not item.is_file() or not item.name.endswith(".yaml"):
            continue
        payload = yaml.safe_load(item.read_text(encoding="utf-8")) or {}
        examples.append((item.name.removesuffix(".yaml"), RunConfig.model_validate(payload)))
    return examples
