from __future__ import annotations

from pathlib import Path

from evals import checklib


def check(root: Path, answer: str) -> tuple[bool, str]:
    ok, out = checklib.run_python(root, "from calc import subtract; assert subtract(5, 3) == 2 and subtract(0, 4) == -4")
    if not ok:
        return False, f"subtract missing or wrong: {out}"
    if "subtract" not in (root / "test_calc.py").read_text(encoding="utf-8"):
        return False, "no test for subtract added"
    return checklib.run_unittest(root)
