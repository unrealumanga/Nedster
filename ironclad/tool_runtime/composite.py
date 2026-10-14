"""The native engine's ToolRuntime: file, search, execution and git tools behind one policy gate.

Policy is evaluated on what will actually run: for execution tools that is the argv the Executor resolves
(an absolute program from the user's PATH, or the project's verified venv python for ``run_tests``), never the
model's raw arguments, and the Executor launches exactly that argv. A program that resolves inside the project
is DENIED; malformed arguments are INVALID.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any, Protocol

from ..contracts import JobHandle, ToolCall, ToolResult, ToolSpec, ToolStatus, WorkspaceRevision
from ..types import RiskLevel
from .exec import ResolutionDenied, executable_name

VALIDATING_TOOLS = frozenset({"run_tests"})
# Commands may change the workspace without reporting paths; treat every execution as a potential mutation.
EXECUTION_TOOLS = frozenset({"run_command", "run_tests"})


class SyncTools(Protocol):
    def specs(self) -> list[ToolSpec]: ...

    def run(self, call: ToolCall) -> ToolResult: ...


class JobTools(Protocol):
    def specs(self) -> list[ToolSpec]: ...

    def start(self, call: ToolCall) -> JobHandle: ...


class _Done:
    def __init__(self, result: ToolResult):
        self._result = result

    async def wait(self) -> ToolResult:
        return self._result

    async def cancel(self) -> None:
        return None


class _Threaded:
    """Runs a quick synchronous tool off the event loop. File tools are atomic, so cancel only waits."""

    def __init__(self, runtime: CompositeToolRuntime, tools: SyncTools, call: ToolCall):
        self._args = (tools, call)
        self._runtime = runtime
        self._task: asyncio.Future[ToolResult] | None = None

    def _ensure(self) -> asyncio.Future[ToolResult]:
        if self._task is None:
            self._task = asyncio.ensure_future(asyncio.to_thread(self._runtime._run_sync, *self._args))
        return self._task

    async def wait(self) -> ToolResult:
        return await asyncio.shield(self._ensure())

    async def cancel(self) -> None:
        if self._task is not None:
            await asyncio.wait({self._task})


Snapshot = dict[str, tuple[int, int]]  # relative path -> (size, mtime_ns)
CHANGED_DURING_RUN = "changed_during_validation"
MAX_CHANGED = 50


def _modified(before: Snapshot, after: Snapshot) -> list[str]:
    """Files that existed before and were changed or deleted; new files (test artifacts) are ignored."""
    return sorted(path for path, stat in before.items() if after.get(path) != stat)


class _Tracked:
    """Wraps an execution job so the workspace revision is bumped once it has really finished."""

    def __init__(self, runtime: CompositeToolRuntime, job: JobHandle, before: Snapshot | None = None):
        self._runtime = runtime
        self._job = job
        self._result: ToolResult | None = None
        self._before = before  # taken before the job started

    async def wait(self) -> ToolResult:
        if self._result is None:
            result = await self._job.wait()
            if self._result is None:
                self._result = result
                if result.status not in (ToolStatus.INVALID, ToolStatus.DENIED):
                    self._runtime._bump(result.mutated_paths)
                fingerprint = self._runtime._fingerprint
                if self._before is not None and fingerprint is not None:
                    changed = _modified(self._before, fingerprint())
                    if changed:  # existing files rewritten/deleted while it ran: not trustworthy evidence
                        data = {**result.data, CHANGED_DURING_RUN: changed[:MAX_CHANGED]}
                        self._result = replace(result, data=data)
        return self._result

    async def cancel(self) -> None:
        await self._job.cancel()


class CompositeToolRuntime:
    def __init__(
        self,
        *,
        sync_tools: Iterable[SyncTools] = (),
        job_tools: Iterable[JobTools] = (),
        policy: Any = None,
        resolve_argv: Any = None,
        journal: Any = None,
        fingerprint: Callable[[], Snapshot] | None = None,
    ):
        self._journal = journal
        self._sync_groups = tuple(sync_tools := tuple(sync_tools))
        self._policy = policy
        self._resolve_argv = resolve_argv
        self._owners: dict[str, tuple[str, Any]] = {}
        self._specs: dict[str, ToolSpec] = {}
        for kind, group in (("sync", sync_tools), ("job", job_tools)):
            for tools in group:
                for spec in tools.specs():
                    if spec.name in self._specs:
                        raise ValueError(f"duplicate tool name: {spec.name}")
                    if spec.name in VALIDATING_TOOLS and not spec.validates:
                        spec = replace(spec, validates=True)
                    self._specs[spec.name] = spec
                    self._owners[spec.name] = (kind, tools)
        self._revision = WorkspaceRevision()
        self._fingerprint = fingerprint

    # ------------------------------------------------------------------ ToolRuntime

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def revision(self) -> WorkspaceRevision:
        return self._revision

    def invalidate_after_undo(self, paths: Iterable[str] = ()) -> None:
        """Call after a successful undo (CheckpointStore.on_undone): forgets read-before-edit hashes and bumps
        the workspace revision so earlier validation evidence is no longer fresh."""
        touched = tuple(paths)
        for tools in self._sync_groups:
            invalidate = getattr(tools, "invalidate_read_hashes", None)
            if invalidate is not None:
                invalidate(touched or None)
        self._bump(touched)

    def start(self, call: ToolCall) -> JobHandle:
        owner = self._owners.get(call.name)
        if owner is None:
            known = ", ".join(sorted(self._specs))
            return _Done(ToolResult(call.id, call.name, ToolStatus.INVALID, f"unknown tool {call.name!r}; available: {known}"))
        if "__invalid__" in call.arguments:
            raw = str(call.arguments["__invalid__"])[:300]
            return _Done(ToolResult(call.id, call.name, ToolStatus.INVALID, f"arguments are not valid JSON: {raw}"))
        refused = self._authorize(call)
        if refused is not None:
            return _Done(ToolResult(call.id, call.name, refused[0], refused[1]))
        kind, tools = owner
        if kind == "sync":
            return _Threaded(self, tools, call)
        if self._journal is not None and (call.name in EXECUTION_TOOLS or self._specs[call.name].mutates):
            self._journal.note_command()  # side effects of commands are outside undo coverage
        # Snapshot BEFORE the job starts. Every execution call is covered: the engine strips the model's
        # "validation" flag before the call reaches the runtime, so validations cannot be singled out here.
        before = self._fingerprint() if call.name in EXECUTION_TOOLS and self._fingerprint is not None else None
        job = tools.start(call)
        if call.name in EXECUTION_TOOLS:
            return _Tracked(self, job, before)
        return job

    # ------------------------------------------------------------------ internals

    def _run_sync(self, tools: SyncTools, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        try:
            result = tools.run(call)
        except Exception as exc:  # a tool bug must not escape as an exception
            result = ToolResult(call.id, call.name, ToolStatus.FAILED, f"tool error: {exc}")
        if result.status is ToolStatus.OK and result.mutated_paths:
            self._bump(result.mutated_paths)
        if not result.elapsed_ms:
            result = replace(result, elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
        return result

    def _bump(self, paths: tuple[str, ...]) -> None:
        self._revision = self._revision.bump(tuple(paths))

    def _authorize(self, call: ToolCall) -> tuple[ToolStatus, str] | None:
        policy = self._policy
        if policy is None:
            return None
        spec = self._specs[call.name]
        risk = RiskLevel(spec.risk)
        ceiling = _ceiling(policy)
        denied = ToolStatus.DENIED
        if risk > ceiling:
            return denied, (
                f"{call.name} needs {risk.name.lower()} permission but the policy allows up to "
                f"{ceiling.name.lower()} (set policy.max_auto_risk or enable autonomy)"
            )
        if call.name not in EXECUTION_TOOLS:
            return None
        try:
            argv = list(self._resolve_argv(call)) if self._resolve_argv else list(call.arguments.get("argv") or [])
        except ResolutionDenied as exc:
            return denied, str(exc)
        except ValueError as exc:  # exec.ArgumentError: malformed or unrunnable arguments
            return ToolStatus.INVALID, str(exc)
        except Exception as exc:
            return denied, f"cannot resolve command: {exc}"
        if not argv:
            return denied, "empty command"
        allowed = {name.lower() for name in getattr(policy, "allowed_commands", [])}
        program = executable_name(str(argv[0]))  # the resolved program's basename
        if program.lower() not in allowed:
            return denied, f"executable is not allowlisted: {program} (allowed: {', '.join(sorted(allowed))})"
        rendered = " ".join(argv).lower()
        for fragment in getattr(policy, "denied_fragments", []):
            if fragment.lower() in rendered:
                return denied, f"command contains a denied fragment: {fragment!r}"
        return None


def _ceiling(policy: Any) -> RiskLevel:
    if not getattr(policy, "autonomous", False):
        return RiskLevel.READ
    levels: Mapping[str, RiskLevel] = {level.name.lower(): level for level in RiskLevel}
    return levels.get(str(getattr(policy, "max_auto_risk", "write")).lower(), RiskLevel.WRITE)

