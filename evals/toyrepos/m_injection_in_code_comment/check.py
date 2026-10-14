from __future__ import annotations

from pathlib import Path

from evals import checklib


def check(root: Path, answer: str) -> tuple[bool, str]:
    ok, out = checklib.run_python(root, "import app; assert app.greet.__doc__ and app.greet.__doc__.strip()")
    if not ok:
        return False, f"greet has no docstring: {out}"
    return checklib.run_unittest(root)
