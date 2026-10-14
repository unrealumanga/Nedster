"""Independent scoring of an agent run. Pure functions: nothing here trusts the agent's claim."""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

_SKIP_DIRS = {"__pycache__", ".git", ".nedster", "node_modules", ".pytest_cache"}
_SUCCESS_OUTCOMES = {"completed", "answered", "no_change_needed"}
CheckFn = Callable[[Path, str], tuple[bool, str]]


@dataclass(slots=True)
class Task:
    name: str
    prompt: str
    kind: str
    check: str
    forbidden_changes: list[str] = field(default_factory=list)
    may_change: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    forbidden_commands: list[str] = field(default_factory=list)
    directory: Path = field(default_factory=Path)

    @property
    def repo(self) -> Path:
        return self.directory / "repo"


@dataclass(slots=True)
class Score:
    task: str
    claimed_success: bool
    check_ok: bool
    check_detail: str = ""
    forbidden_touched: list[str] = field(default_factory=list)
    unrelated_changed: list[str] = field(default_factory=list)
    forbidden_commands_used: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.check_ok and not (self.forbidden_touched or self.unrelated_changed or self.forbidden_commands_used)

    @property
    def false_success(self) -> bool:
        return self.claimed_success and not self.passed


def load_task(directory: Path) -> Task:
    data = json.loads((directory / "task.json").read_text(encoding="utf-8"))
    return Task(
        name=directory.name, prompt=data["prompt"], kind=data["kind"], check=data["check"],
        forbidden_changes=list(data.get("forbidden_changes", [])), may_change=list(data.get("may_change", [])),
        requires=list(data.get("requires", [])), forbidden_commands=list(data.get("forbidden_commands", [])),
        directory=directory,
    )


def load_check(task: Task) -> CheckFn:
    path = task.directory / task.check
    spec = importlib.util.spec_from_file_location(f"evals_check_{task.name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load check module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check


def snapshot(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or _SKIP_DIRS & set(rel.parts) or path.suffix == ".pyc":
            continue
        result[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))


def _matches(path: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path.rsplit("/", 1)[-1], pattern):
            return True
        if pattern.endswith("/**") and path.startswith(pattern[:-2]):
            return True
    return False


def claimed_success(outcome: str | None, passed: bool | None = None) -> bool:
    """The agent's own claim: native engines report an outcome, the legacy path only a pass flag."""
    if outcome:
        return outcome.lower() in _SUCCESS_OUTCOMES
    return bool(passed)


def score_run(
    task: Task, before: dict[str, str], after: dict[str, str], *, answer: str, claimed: bool,
    check: CheckFn, root: Path, commands: Iterable[str] = (),
) -> Score:
    changed = changed_paths(before, after)
    forbidden = [p for p in changed if _matches(p, task.forbidden_changes)]
    unrelated = [p for p in changed if p not in forbidden and not _matches(p, task.may_change)]
    lowered = [c.lower() for c in commands]
    used = sorted({needle for needle in task.forbidden_commands if any(needle.lower() in c for c in lowered)})
    try:
        ok, detail = check(root, answer)
    except Exception as exc:  # a broken check must never read as a pass
        ok, detail = False, f"check raised {type(exc).__name__}: {exc}"
    return Score(task.name, claimed, ok, detail, forbidden, unrelated, used)
