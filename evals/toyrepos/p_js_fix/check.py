from __future__ import annotations

import shutil
from pathlib import Path

from evals import checklib


def check(root: Path, answer: str) -> tuple[bool, str]:
    if shutil.which("node") is None:
        return True, "node not available, skipped"
    return checklib.run_command(root, ["node", "test.js"])
