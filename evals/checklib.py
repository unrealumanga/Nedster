"""Shell-free helpers for fixture check modules. Each returns (ok, detail)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_TIMEOUT = 60


def run_command(root: Path, argv: list[str]) -> tuple[bool, str]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        proc = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=_TIMEOUT, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{argv[0]}: {exc}"
    detail = (proc.stdout + proc.stderr).strip()[-400:]
    return proc.returncode == 0, detail


def run_python(root: Path, code: str) -> tuple[bool, str]:
    return run_command(root, [sys.executable, "-B", "-c", "import sys; sys.path.insert(0, '.'); " + code])


def run_unittest(root: Path) -> tuple[bool, str]:
    return run_command(root, [sys.executable, "-B", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"])
