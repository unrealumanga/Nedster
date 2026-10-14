from __future__ import annotations

from pathlib import Path

from evals import checklib


def check(root: Path, answer: str) -> tuple[bool, str]:
    ok, out = checklib.run_command(root, ["node", "-e", "const m = require('./index'); if (m.multiply(3, 4) !== 12) process.exit(1)"])
    if not ok:
        return False, f"multiply missing or wrong: {out}"
    if "multiply" not in (root / "index.test.js").read_text(encoding="utf-8"):
        return False, "no test for multiply added"
    return checklib.run_command(root, ["node", "--test"])
