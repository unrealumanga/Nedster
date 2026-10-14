"""The native engine's ToolRuntime: file, search, execution and git tools behind one policy gate.

Policy is evaluated on what will actually run: for execution tools that is the argv the Executor resolves
(for example the project's venv python for ``run_tests``), never the model's raw arguments.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import PurePath
from typing import Any, Protocol

from ..contracts import JobHandle, ToolCall, ToolResult, ToolSpec, ToolStatus, WorkspaceRevision
from ..types import RiskLevel

WINDOWS_SUFFIXES = (".exe", ".cmd", ".bat", ".com")
VALIDATING_TOOLS = frozenset({"run_tests"})
# Commands may change the workspace without reporting paths; treat every execution as a potential mutation.
EXECUTION_TOOLS = frozenset({"run_command", "run_tests"})


class SyncTools(Protocol):
    def specs(self) -> list[ToolSpec]: ...

    def run(self, call: ToolCall) -> ToolResult: ...


class JobTools(Protocol):
    def specs(self) -> list[ToolSpec]: ...

    def start(self, call: ToolCall) -> JobHandle: ...


def executable_name(program: str) -> str:
    """Basename without a Windows launcher suffix; splits both separators on every OS."""
    name = PurePath(program.replace("\\", "/")).name
    lowered = name.lower()
    for suffix in WINDOWS_SUFFIXES:
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name


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


class _Tracked:
    """Wraps an execution job so the workspace revision is bumped once it has really finished."""

    def __init__(self, runtime: CompositeToolRuntime, job: JobHandle):
        self._runtime = runtime
        self._job = job
        self._result: ToolResult | None = None

    async def wait(self) -> ToolResult:
        if self._result is None:
            result = await self._job.wait()
            if self._result is None:
                self._result = result
                if result.status not in (ToolStatus.INVALID, ToolStatus.DENIED):
                    self._runtime._bump(result.mutated_paths)
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
    ):
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

    # ------------------------------------------------------------------ ToolRuntime

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def revision(self) -> WorkspaceRevision:
        return self._revision

    def start(self, call: ToolCall) -> JobHandle:
        owner = self._owners.get(call.name)
        if owner is None:
            known = ", ".join(sorted(self._specs))
            return _Done(ToolResult(call.id, call.name, ToolStatus.INVALID, f"unknown tool {call.name!r}; available: {known}"))
        if "__invalid__" in call.arguments:
            raw = str(call.arguments["__invalid__"])[:300]
            return _Done(ToolResult(call.id, call.name, ToolStatus.INVALID, f"arguments are not valid JSON: {raw}"))
        denied = self._authorize(call)
        if denied is not None:
            return _Done(ToolResult(call.id, call.name, ToolStatus.DENIED, denied))
        kind, tools = owner
        if kind == "sync":
            return _Threaded(self, tools, call)
        job = tools.start(call)
        if call.name in EXECUTION_TOOLS:
            return _Tracked(self, job)
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

    def _authorize(self, call: ToolCall) -> str | None:
        policy = self._policy
        if policy is None:
            return None
        spec = self._specs[call.name]
        risk = RiskLevel(spec.risk)
        ceiling = _ceiling(policy)
        if risk > ceiling:
            return (
                f"{call.name} needs {risk.name.lower()} permission but the policy allows up to "
                f"{ceiling.name.lower()} (set policy.max_auto_risk or enable autonomy)"
            )
        if call.name not in EXECUTION_TOOLS:
            return None
        try:
            argv = list(self._resolve_argv(call)) if self._resolve_argv else list(call.arguments.get("argv") or [])
        except Exception as exc:
            return f"cannot resolve command: {exc}"
        if not argv:
            return "empty command"
        allowed = {name.lower() for name in getattr(policy, "allowed_commands", [])}
        program = executable_name(str(argv[0]))
        if program.lower() not in allowed:
            return f"executable is not allowlisted: {program} (allowed: {', '.join(sorted(allowed))})"
        rendered = " ".join(argv).lower()
        for fragment in getattr(policy, "denied_fragments", []):
            if fragment.lower() in rendered:
                return f"command contains a denied fragment: {fragment!r}"
        return None


def _ceiling(policy: Any) -> RiskLevel:
    if not getattr(policy, "autonomous", False):
        return RiskLevel.READ
    levels: Mapping[str, RiskLevel] = {level.name.lower(): level for level in RiskLevel}
    return levels.get(str(getattr(policy, "max_auto_risk", "write")).lower(), RiskLevel.WRITE)

