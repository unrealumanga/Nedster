"""Shared helpers for companion fixture checks (stdlib only, shell-free)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def run_py(root: Path, code: str, timeout: float = 60.0) -> tuple[bool, str]:
    """Run `code` with `root` on sys.path and as cwd. ok == exit code 0; detail is the output tail."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    argv = [sys.executable, "-B", "-c", "import sys; sys.path.insert(0, '.'); " + code]
    try:
        proc = subprocess.run(argv, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout:.0f}s"
    except OSError as exc:
        return False, str(exc)
    return proc.returncode == 0, (proc.stderr + proc.stdout).strip()[-600:]


def run_argv(root: Path, argv: list[str], timeout: float = 60.0) -> tuple[bool, str]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(argv, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout:.0f}s"
    except OSError as exc:
        return False, f"{argv[0]}: {exc}"
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()[-600:]


def original(check_file: str, rel: str) -> Path:
    """Path of a file in the fixture's pristine repo/ (the check lives next to it)."""
    return Path(check_file).resolve().parent / "repo" / rel


def tree(root: Path) -> dict[str, bytes]:
    skip = {"__pycache__", ".git", ".nedster", "node_modules", ".pytest_cache"}
    out: dict[str, bytes] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and not (skip & set(rel.parts)) and p.suffix != ".pyc":
            out[rel.as_posix()] = p.read_bytes()
    return out


def unittest_result(root: Path) -> tuple[dict | None, str]:
    """Collect actual unittest results, including non-package test directories.

    A timeout, import error or process crash is not evidence of a killed mutant.
    No minimum method count: one table-driven test can cover many edge cases.
    """
    code = r'''
import importlib.util, json, pathlib, unittest
suite = unittest.TestSuite()
for index, path in enumerate(sorted(pathlib.Path('.').rglob('test*.py'))):
    if any(p in {'__pycache__', '.git', '.nedster'} for p in path.parts):
        continue
    parents = list(path.parent.parents)[:-1] + [path.parent]
    packaged = all((p / '__init__.py').exists() for p in parents if p != pathlib.Path('.'))
    name = '.'.join(path.with_suffix('').parts) if packaged else '_companion_test_' + str(index)
    if not packaged:
        sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    except Exception as exc:
        def failed_import(error=exc):
            raise error
        suite.addTest(unittest.FunctionTestCase(failed_import))
result = unittest.TextTestRunner().run(suite)
print('COMPANION_RESULT=' + json.dumps(dict(
    ran=result.testsRun, skipped=len(result.skipped), failures=len(result.failures),
    errors=len(result.errors), unexpected=len(result.unexpectedSuccesses),
    expected=len(result.expectedFailures), ok=result.wasSuccessful())))
'''
    ok, detail = run_py(root, code)
    if not ok or 'COMPANION_RESULT=' not in detail:
        return None, detail
    result = json.loads(detail.rsplit('COMPANION_RESULT=', 1)[1].splitlines()[0])
    return result, detail
