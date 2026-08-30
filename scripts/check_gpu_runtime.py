"""Fail fast unless the installed Torch stack can execute on an NVIDIA GPU."""

from __future__ import annotations

import json


def main() -> int:
    try:
        import torch
        import torchvision
    except ImportError as exc:
        print(json.dumps({"ok": False, "error": f"training stack unavailable: {exc}"}))
        return 1

    available = torch.cuda.is_available()
    payload = {
        "ok": available,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": available,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if available else None,
        "capability": list(torch.cuda.get_device_capability(0)) if available else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if available else 1


if __name__ == "__main__":
    raise SystemExit(main())

