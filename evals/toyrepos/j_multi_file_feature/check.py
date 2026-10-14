from __future__ import annotations

import json
import sys
from pathlib import Path

from evals import checklib


def check(root: Path, answer: str) -> tuple[bool, str]:
    ok, out = checklib.run_command(
        root, [sys.executable, "-B", "cli.py", "--json", "--name", "box", "--count", "5"]
    )
    if not ok:
        return False, f"cli --json failed: {out}"
    try:
        data = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        return False, f"output is not JSON: {exc}: {out}"
    if not isinstance(data, dict) or data.get("name") != "box" or data.get("count") != 5:
        return False, f"unexpected JSON payload: {data}"
    return True, "cli emitted the expected JSON"
