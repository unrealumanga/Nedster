"""Frozen v4 contracts shared by the engine, tools, transport, session store and UI.

Every wave-1 package codes against these types. Changing a field here is an
integration decision owned by the engine package (A); other packages add
adapters in their own namespaces instead of editing this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Conversation and transport
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True, frozen=True)
class ToolCall:
    """One complete, validated-shape tool call emitted by the model."""

    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set when role == "tool"
    name: str | None = None  # tool name when role == "tool"
    thinking: str = ""


@dataclass(slots=True, frozen=True)
class ToolSpec:
    """A tool advertised to the model. ``parameters`` is a JSON Schema object."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    risk: int  # ironclad.types.RiskLevel value
    mutates: bool = False
    validates: bool = False  # a successful run is validation evidence (e.g. run_tests)


@dataclass(slots=True, frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    load_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(slots=True, frozen=True)
class TextDelta:
    text: str
    kind: Literal["thinking", "content"] = "content"


@dataclass(slots=True, frozen=True)
class ToolCallsReady:
    """Emitted once per assistant turn, only after every call is fully assembled."""

    calls: tuple[ToolCall, ...]


@dataclass(slots=True, frozen=True)
class TurnFinished:
    message: ChatMessage
    usage: Usage
    done_reason: str = "stop"  # stop | length | cancelled | error


ChatEvent = TextDelta | ToolCallsReady | TurnFinished


class TransportError(RuntimeError):
    """Raised for connection/protocol failures. Never raised after a ToolCallsReady was yielded
    for the same turn: a transport must not cause a mutation to be replayed."""


@runtime_checkable
class ChatTransport(Protocol):
    async def chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        think: bool | str = False,
        keep_alive: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ChatEvent]: ...

    async def aclose(self) -> None: ...


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


class ToolStatus(StrEnum):
    OK = "ok"  # tool ran and the operation succeeded
    FAILED = "failed"  # tool ran; the operation reported failure (nonzero exit, failing tests)
    INVALID = "invalid"  # arguments rejected before execution (schema/scope)
    DENIED = "denied"  # policy refused
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass(slots=True, frozen=True)
class ToolResult:
    call_id: str
    tool: str
    status: ToolStatus
    output: str  # bounded, model-facing text
    # Structured extras. Standard keys: exit_code, tests_run (int | None: None = unknown), tests_failed,
    # sha256 (new content hash after write/edit), backend, isolation, duration_ms.
    data: Mapping[str, Any] = field(default_factory=dict)
    mutated_paths: tuple[str, ...] = ()  # project-relative POSIX paths actually changed
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.OK


class JobHandle(Protocol):
    """A running tool operation. ``cancel`` must stop owned processes/containers;
    ``wait`` returns only after the operation has really stopped."""

    async def wait(self) -> ToolResult: ...

    async def cancel(self) -> None: ...


@runtime_checkable
class ToolRuntime(Protocol):
    def specs(self) -> list[ToolSpec]: ...

    def start(self, call: ToolCall) -> JobHandle:
        """Validate, authorize and start ``call``. Invalid/denied calls return a handle whose
        ``wait`` immediately yields an INVALID/DENIED result; nothing executes."""
        ...

    def revision(self) -> WorkspaceRevision: ...


# ---------------------------------------------------------------------------
# Workspace revision and acceptance
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class WorkspaceRevision:
    """Monotonic counter bumped by every (potentially) mutating tool result.
    ``fingerprint`` optionally hashes the touched files for diagnostics."""

    counter: int = 0
    fingerprint: str = ""

    def bump(self, paths: tuple[str, ...] = ()) -> WorkspaceRevision:
        digest = hashlib.sha256(("|".join(paths) + self.fingerprint).encode()).hexdigest()[:16]
        return WorkspaceRevision(self.counter + 1, digest)


class Outcome(StrEnum):
    COMPLETED = "completed"  # requested change made and validated after the latest mutation
    ANSWERED = "answered"  # question answered; no change was requested
    NO_CHANGE_NEEDED = "no_change_needed"  # evidence shows the request is already satisfied
    UNVALIDATED = "unvalidated"  # change made but no post-change validation evidence
    INCOMPLETE = "incomplete"  # budget exhausted / model stopped early / validation failing
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True, frozen=True)
class AcceptanceResult:
    outcome: Outcome
    reasons: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    validated_at: WorkspaceRevision | None = None
    answer: str = ""  # model prose followed by the code-generated summary of changes and validation

    @property
    def passed(self) -> bool:
        return self.outcome in {Outcome.COMPLETED, Outcome.ANSWERED, Outcome.NO_CHANGE_NEEDED}


# ---------------------------------------------------------------------------
# Engine events (consumed by TUI / CLI / JSONL) and sessions
# ---------------------------------------------------------------------------

EventKind = Literal[
    "turn.started",
    "model.delta",  # data: {"text", "kind"}
    "model.usage",  # data: Usage fields
    "tool.started",  # data: {"call_id", "tool", "arguments"}
    "tool.finished",  # data: {"call_id", "tool", "status", "summary", "mutated_paths", "elapsed_ms"}
    "context.compacted",  # data: {"elided", "handles", "tokens_before", "tokens_after"}
    "acceptance",  # data: AcceptanceResult fields
    "turn.finished",  # data: {"answer", "outcome"}
    "error",  # data: {"message", "remedy"}
]


@dataclass(slots=True, frozen=True)
class EngineEvent:
    kind: EventKind
    session_id: str
    run_id: str
    data: Mapping[str, Any] = field(default_factory=dict)
    ts_ns: int = 0


@runtime_checkable
class SessionStore(Protocol):
    def create(self, project_root: str, title: str = "") -> str: ...

    def append_message(self, session_id: str, message: ChatMessage) -> None: ...

    def messages(self, session_id: str) -> list[ChatMessage]: ...

    def list_sessions(self, project_root: str, limit: int = 20) -> list[Mapping[str, Any]]: ...

    def set_state(self, session_id: str, key: str, value: Any) -> None: ...

    def get_state(self, session_id: str, key: str, default: Any = None) -> Any: ...

    def put_output(self, session_id: str, tool: str, call_id: str, text: str) -> int: ...

    def get_output(self, session_id: str, handle: int) -> str: ...

    def replace_messages(self, session_id: str, messages: list[ChatMessage]) -> None: ...
