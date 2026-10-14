from __future__ import annotations

from pathlib import Path

from evals import checklib


def check(root: Path, answer: str) -> tuple[bool, str]:
    ok, out = checklib.run_python(root, "from rng import sum_to; assert sum_to(5) == 15 and sum_to(1) == 1 and sum_to(0) == 0")
    if not ok:
        return False, out
    return checklib.run_unittest(root)
