"""Deterministic offline doubles for engine tests: scripted transport, scripted tool runtime, memory store."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ironclad.contracts import (
    ChatEvent,
    ChatMessage,
    EngineEvent,
    TextDelta,
    ToolCall,
    ToolCallsReady,
    ToolResult,
    ToolSpec,
    ToolStatus,
    TurnFinished,
    Usage,
    WorkspaceRevision,
)
from ironclad.engine.context import serialize


@dataclass(slots=True)
class Turn:
    text: str = ""
    calls: Sequence[ToolCall] = ()
    done_reason: str = "stop"
    usage: Usage = field(default_factory=lambda: Usage(prompt_tokens=10, completion_tokens=5))


def call(name: str, call_id: str, **arguments: Any) -> ToolCall:
    return ToolCall(call_id, name, arguments)


Script = Sequence[Turn | Callable[[list[ChatMessage]], Turn]]


class ScriptedTransport:
    """Yields scripted turns in order; records a serialized snapshot of every request."""

    def __init__(self, script: Script) -> None:
        self.script = list(script)
        self.requests: list[list[dict[str, Any]]] = []
        self.tool_names: list[list[str]] = []

    async def chat(self, *, model: str, messages: list[ChatMessage], tools: list[ToolSpec], **_: Any) -> AsyncIterator[ChatEvent]:
        self.requests.append(serialize(messages))
        self.tool_names.append([t.name for t in tools])
        if not self.script:
            raise AssertionError("transport script exhausted")
        step = self.script.pop(0)
        turn = step(messages) if callable(step) else step
        if turn.text:
            yield TextDelta(turn.text)
        if turn.calls:
            yield ToolCallsReady(tuple(turn.calls))
        message = ChatMessage(role="assistant", content=turn.text, tool_calls=list(turn.calls))
        yield TurnFinished(message, turn.usage, turn.done_reason)

    async def aclose(self) -> None:
        return None


Handler = Callable[[ToolCall], ToolResult]


def ok(c: ToolCall, output: str = "ok", **data: Any) -> ToolResult:
    return ToolResult(c.id, c.name, ToolStatus.OK, output, data=data)


class FakeJob:
    def __init__(self, runtime: FakeRuntime, c: ToolCall, delay: float) -> None:
        self.runtime, self.call, self.delay = runtime, c, delay
        self.cancelled = False
        self.cancel_awaited = False
        self._done = asyncio.Event()
        self._result: ToolResult | None = None

    async def wait(self) -> ToolResult:
        if self._result is not None:
            return self._result
        if self.cancelled:
            self._result = ToolResult(self.call.id, self.call.name, ToolStatus.CANCELLED, "cancelled")
            return self._result
        mutating = self.runtime.is_mutating(self.call.name)
        rt = self.runtime
        rt.active += 1
        rt.active_mutating += int(mutating)
        rt.max_active = max(rt.max_active, rt.active)
        rt.max_active_mutating = max(rt.max_active_mutating, rt.active_mutating)
        rt.max_overlap_with_mutation = max(rt.max_overlap_with_mutation, rt.active if rt.active_mutating else 0)
        rt.started_order.append(self.call.id)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            result = rt.handlers[self.call.name](self.call)
            if result.status is ToolStatus.OK and result.mutated_paths:
                rt.rev = rt.rev.bump(result.mutated_paths)
            elif self.call.name in rt.bump_always:
                rt.rev = rt.rev.bump()
            self._result = result
            return result
        finally:
            rt.active -= 1
            rt.active_mutating -= int(mutating)
            rt.finished_order.append(self.call.id)
            self._done.set()

    async def cancel(self) -> None:
        self.cancelled = True
        await asyncio.sleep(0)
        self.cancel_awaited = True


class InstantJob:
    def __init__(self, result: ToolResult) -> None:
        self.result = result

    async def wait(self) -> ToolResult:
        return self.result

    async def cancel(self) -> None:
        return None


class FakeRuntime:
    def __init__(
        self,
        handlers: Mapping[str, Handler],
        *,
        mutating: Sequence[str] = ("write_file", "edit_file", "run_command"),
        delays: Mapping[str, float] | None = None,
        bump_always: Sequence[str] = (),
        validators: Mapping[str, Callable[[ToolCall], str | None]] | None = None,
    ) -> None:
        self.handlers = dict(handlers)
        self.mutating = set(mutating)
        self.delays = dict(delays or {})
        self.bump_always = set(bump_always)
        self.validators = dict(validators or {})
        self.rev = WorkspaceRevision()
        self.jobs: list[FakeJob] = []
        self.calls: list[ToolCall] = []
        self.active = self.active_mutating = 0
        self.max_active = self.max_active_mutating = self.max_overlap_with_mutation = 0
        self.started_order: list[str] = []
        self.finished_order: list[str] = []

    def is_mutating(self, name: str) -> bool:
        return name in self.mutating

    def specs(self) -> list[ToolSpec]:
        params = {"type": "object", "properties": {"path": {"type": "string"}}}
        return [
            ToolSpec(name, f"fake {name}", params, risk=1 if name in self.mutating else 0, mutates=name in self.mutating)
            for name in self.handlers
        ]

    def start(self, c: ToolCall) -> FakeJob | InstantJob:
        self.calls.append(c)
        if c.name not in self.handlers:
            return InstantJob(ToolResult(c.id, c.name, ToolStatus.INVALID, f"unknown tool {c.name}"))
        if hasattr(self, "_role_name") and self._role_name is not None:
            allowed = getattr(self, "_allowed_tools", None)
            if allowed is not None and c.name not in allowed:
                return InstantJob(ToolResult(c.id, c.name, ToolStatus.DENIED, f"{c.name} is not allowed in {self._role_name} mode"))
        check = self.validators.get(c.name)
        problem = check(c) if check else None
        if problem:
            return InstantJob(ToolResult(c.id, c.name, ToolStatus.INVALID, problem))
        job = FakeJob(self, c, self.delays.get(c.name, 0.0))
        self.jobs.append(job)
        return job

    def set_role(self, name: str, allowed_tools) -> None:
        """Restrict dispatch to ``allowed_tools`` for role ``name`` (``None`` = no restriction)."""
        self._role_name = name
        self._allowed_tools = frozenset(allowed_tools) if allowed_tools is not None else None

    def revision(self) -> WorkspaceRevision:
        return self.rev


class MemoryStore:
    """Minimal SessionStore with the put_output/get_output/replace_messages extras of package D."""

    def __init__(self) -> None:
        self.msgs: dict[str, list[ChatMessage]] = {}
        self.state: dict[tuple[str, str], Any] = {}
        self.outputs: dict[tuple[str, int], str] = {}
        self.replaced = 0

    def create(self, project_root: str, title: str = "") -> str:
        sid = f"s{len(self.msgs) + 1}"
        self.msgs[sid] = []
        return sid

    def append_message(self, session_id: str, message: ChatMessage) -> None:
        self.msgs.setdefault(session_id, []).append(message)

    def messages(self, session_id: str) -> list[ChatMessage]:
        return list(self.msgs.get(session_id, []))

    def list_sessions(self, project_root: str, limit: int = 20) -> list[Mapping[str, Any]]:
        return [{"id": sid} for sid in self.msgs][:limit]

    def set_state(self, session_id: str, key: str, value: Any) -> None:
        self.state[(session_id, key)] = value

    def get_state(self, session_id: str, key: str, default: Any = None) -> Any:
        return self.state.get((session_id, key), default)

    def put_output(self, session_id: str, tool: str, call_id: str, text: str) -> int:
        handle = 1 + sum(1 for sid, _ in self.outputs if sid == session_id)
        self.outputs[(session_id, handle)] = text
        return handle

    def get_output(self, session_id: str, handle: int) -> str:
        return self.outputs[(session_id, handle)]

    def replace_messages(self, session_id: str, messages: list[ChatMessage]) -> None:
        self.replaced += 1
        self.msgs[session_id] = list(messages)


class Events(list[EngineEvent]):
    def kinds(self) -> list[str]:
        return [e.kind for e in self]

    def of(self, kind: str) -> list[EngineEvent]:
        return [e for e in self if e.kind == kind]
