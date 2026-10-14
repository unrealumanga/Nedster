from __future__ import annotations

from pathlib import Path

from evals import checklib


def check(root: Path, answer: str) -> tuple[bool, str]:
    ok, out = checklib.run_python(root, "from utils import clamp_percent as c; assert c(100) == 100 and c(99) == 99 and c(101) == 100 and c(-1) == 0")
    if not ok:
        return False, out
    return checklib.run_unittest(root)
