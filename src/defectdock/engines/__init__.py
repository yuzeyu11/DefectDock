"""Training-engine contracts and built-in adapters."""

from .base import EngineResult, TrainingCancelled, TrainingEngine
from .torchvision import TorchvisionEngine, build_plan, run_training

__all__ = [
    "EngineResult",
    "TorchvisionEngine",
    "TrainingCancelled",
    "TrainingEngine",
    "build_plan",
    "run_training",
]
