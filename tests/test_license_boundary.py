from pathlib import Path

from scripts.check_license_boundary import check


def test_excluded_runtime_is_absent_from_dependencies_and_imports():
    root = Path(__file__).resolve().parents[1]
    assert check(root) == []
