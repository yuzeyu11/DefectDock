"""Fail when the excluded training runtime re-enters code or manifests."""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

FORBIDDEN_PACKAGE = "ultra" + "lytics"
SCAN_TREES = ("src", "tests", "scripts")


def imported_roots(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"cannot parse {path}: {exc}") from exc
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name.split(".")[0]) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.lineno, node.module.split(".")[0]))
    return found


def declared_dependencies(pyproject: Path) -> list[str]:
    payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = payload.get("project", {})
    dependencies = list(project.get("dependencies", []))
    for values in project.get("optional-dependencies", {}).values():
        dependencies.extend(values)
    return dependencies


def check(root: Path) -> list[str]:
    violations: list[str] = []
    for dependency in declared_dependencies(root / "pyproject.toml"):
        normalized = dependency.lower().replace("_", "-").split("[")[0]
        if normalized.startswith(FORBIDDEN_PACKAGE):
            violations.append(f"pyproject.toml: forbidden dependency: {dependency}")
    for tree_name in SCAN_TREES:
        tree_root = root / tree_name
        if not tree_root.exists():
            continue
        for path in tree_root.rglob("*.py"):
            if path.resolve() == Path(__file__).resolve():
                continue
            for line, name in imported_roots(path):
                if name.lower() == FORBIDDEN_PACKAGE:
                    violations.append(f"{path.relative_to(root)}:{line}: forbidden import")
    return violations


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations = check(root)
    if violations:
        print("License boundary check failed:")
        print("\n".join(f"- {item}" for item in violations))
        return 1
    print("License boundary check passed: excluded runtime is absent from manifests and imports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
