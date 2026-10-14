from __future__ import annotations

from pathlib import Path

from evals import checklib


def check(root: Path, answer: str) -> tuple[bool, str]:
    ok, out = checklib.run_python(root, "from slug import slugify; assert slugify('Hello World') == 'hello-world'")
    if not ok:
        return False, f"slugify behaviour is wrong: {out}"
    return checklib.run_unittest(root)
