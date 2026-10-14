from __future__ import annotations

from pathlib import Path

from evals import checklib


def check(root: Path, answer: str) -> tuple[bool, str]:
    for path in root.glob("*.py"):
        if "compute_total" in path.read_text(encoding="utf-8"):
            return False, f"old name still in {path.name}"
    ok, out = checklib.run_python(root, "from pricing import calculate_total; assert calculate_total([1, 2]) == 3")
    if not ok:
        return False, out
    return checklib.run_unittest(root)
