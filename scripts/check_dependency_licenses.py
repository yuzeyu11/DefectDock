"""Enforce the narrow release license deny-list against scanner JSON output."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DENIED_PACKAGES = {"ultralytics"}
DENIED_LICENSE_MARKERS = ("agpl", "gnu affero general public license")


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def iter_entries(payload: object):
    """Yield package/license pairs from licensecheck or pnpm license JSON."""
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield str(item.get("name") or item.get("package") or ""), str(
                    item.get("license") or item.get("licenses") or "UNKNOWN"
                )
        return
    if isinstance(payload, dict):
        nested = payload.get("packages") or payload.get("dependencies")
        if isinstance(nested, list):
            yield from iter_entries(nested)
            return
        for license_name, packages in payload.items():
            if not isinstance(packages, list):
                continue
            for item in packages:
                if isinstance(item, dict):
                    yield str(item.get("name") or item.get("package") or ""), str(
                        item.get("license") or license_name
                    )


def check(payload: object) -> list[str]:
    violations: list[str] = []
    for package, license_name in iter_entries(payload):
        normalized_package = normalize(package).replace(" ", "-")
        normalized_license = normalize(license_name)
        if normalized_package in DENIED_PACKAGES:
            violations.append(f"forbidden package: {package}")
        if any(marker in normalized_license for marker in DENIED_LICENSE_MARKERS):
            violations.append(f"forbidden license: {package or '<unknown>'}: {license_name}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    violations = check(payload)
    if violations:
        print("Dependency license policy failed:")
        print("\n".join(f"- {item}" for item in violations))
        return 1
    print("Dependency license policy passed: no excluded package or AGPL license found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

