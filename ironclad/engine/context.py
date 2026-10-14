"""Context management: bounded tool messages, observation masking and the ``show_output`` tool.

Messages are append-only between compactions. A compaction is a discrete event that replaces old tool
outputs with truthful stubs pointing at a durable handle; the full text stays retrievable through
``show_output``. Call/result pairing is never broken: a stub keeps its ``tool_call_id``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ironclad.contracts import ChatMessage, SessionStore, ToolCall, ToolResult, ToolSpec, ToolStatus

Estimator = Callable[[str], int]
KEEP_LAST = 6
THRESHOLD = 0.7
STUB_PREFIX = "[output #"
SHOW_OUTPUT = "show_output"
SHOW_OUTPUT_MAX_LINES = 200

SHOW_OUTPUT_SPEC = ToolSpec(
    name=SHOW_OUTPUT,
    description=(
        "Show the full text of an earlier tool output that was elided or truncated. "
        "Use the handle number from the '[output #N ...]' note."
    ),
    parameters={
        "type": "object",
        "properties": {
            "handle": {"type": "integer", "minimum": 1},
            "start_line": {"type": "integer", "minimum": 1},
            "max_lines": {"type": "integer", "minimum": 1, "maximum": SHOW_OUTPUT_MAX_LINES},
        },
        "required": ["handle"],
        "additionalProperties": False,
    },
    risk=0,
    mutates=False,
)


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def estimate_prompt(messages: Sequence[ChatMessage], specs: Sequence[ToolSpec], estimator: Estimator = estimate_tokens) -> int:
    total = 0
    for message in messages:
        total += estimator(message.content) + estimator(message.thinking) + 4
        for call in message.tool_calls:
            total += estimator(call.name + json.dumps(dict(call.arguments), default=str))
    for spec in specs:
        total += estimator(spec.name + spec.description + json.dumps(dict(spec.parameters), default=str))
    return total


@dataclass(slots=True)
class OutputVault:
    """Durable full outputs by handle: ``store.put_output/get_output`` when available, else memory."""

    session_id: str
    store: SessionStore | None = None
    _memory: dict[int, str] = field(default_factory=dict)

    def put(self, tool: str, call_id: str, text: str) -> int:
        put = getattr(self.store, "put_output", None)
        handle = int(put(self.session_id, tool, call_id, text)) if callable(put) else len(self._memory) + 1
        self._memory[handle] = text
        return handle

    def get(self, handle: int) -> str | None:
        if handle in self._memory:
            return self._memory[handle]
        get = getattr(self.store, "get_output", None)
        if not callable(get):
            return None
        try:
            text = get(self.session_id, handle)
        except (KeyError, LookupError, ValueError):
            return None
        return None if text is None else str(text)


def _first_line(text: str, limit: int = 120) -> str:
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line if len(line) <= limit else line[: limit - 1] + "…"


def tool_message(result: ToolResult, vault: OutputVault, max_chars: int) -> ChatMessage:
    """Bounded model-facing tool message. Oversized output keeps head+tail and a handle to the rest."""
    output = result.output
    if len(output) > max_chars:
        handle = vault.put(result.tool, result.call_id, output)
        half = max(max_chars // 2, 1)
        omitted = len(output) - 2 * half
        output = (
            f"{output[:half]}\n[... {omitted} chars truncated; full text: show_output(handle={handle}) ...]\n"
            f"{output[-half:]}"
        )
    return ChatMessage(role="tool", content=f"[{result.status.value}] {output}", tool_call_id=result.call_id, name=result.tool)


def run_show_output(vault: OutputVault, call: ToolCall, max_chars: int) -> ToolResult:
    args = call.arguments
    handle, start, count = args.get("handle"), args.get("start_line", 1), args.get("max_lines", SHOW_OUTPUT_MAX_LINES)
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in (handle, start, count)):
        return ToolResult(call.id, SHOW_OUTPUT, ToolStatus.INVALID, "handle, start_line, max_lines must be positive integers")
    text = vault.get(handle)
    if text is None:
        return ToolResult(call.id, SHOW_OUTPUT, ToolStatus.INVALID, f"unknown output handle {handle}")
    lines = text.splitlines()
    count = min(count, SHOW_OUTPUT_MAX_LINES)
    chunk = "\n".join(lines[start - 1 : start - 1 + count])[:max_chars]
    end = min(start - 1 + count, len(lines))
    more = f"\n[lines {start}-{end} of {len(lines)}; continue with start_line={end + 1}]" if end < len(lines) else ""
    return ToolResult(call.id, SHOW_OUTPUT, ToolStatus.OK, chunk + more, data={"handle": handle, "lines": len(lines)})


def _status_of(content: str) -> str:
    if content.startswith("[") and "]" in content[:16]:
        return content[1 : content.index("]")]
    return "?"


def _body_of(content: str) -> str:
    status = _status_of(content)
    return content[len(status) + 3 :] if status != "?" else content


@dataclass(slots=True, frozen=True)
class Compaction:
    elided: int
    tokens_before: int
    tokens_after: int
    handles: tuple[int, ...]


@dataclass(slots=True)
class ContextManager:
    budget_tokens: int
    vault: OutputVault
    estimator: Estimator = estimate_tokens
    keep_last: int = KEEP_LAST
    threshold: float = THRESHOLD

    @property
    def limit(self) -> int:
        return int(self.budget_tokens * self.threshold)

    def over(self, messages: Sequence[ChatMessage], specs: Sequence[ToolSpec]) -> bool:
        return estimate_prompt(messages, specs, self.estimator) > self.limit

    def compact(
        self,
        messages: list[ChatMessage],
        specs: Sequence[ToolSpec],
        *,
        protect: Collection[str] = (),
    ) -> Compaction | None:
        """Mask old tool outputs in place (keeping the newest ``keep_last``); ``None`` if nothing changed."""
        before = estimate_prompt(messages, specs, self.estimator)
        handles: list[int] = []
        for keep in dict.fromkeys((self.keep_last, 3, 1)):
            tool_idx = [i for i, m in enumerate(messages) if m.role == "tool"]
            for i in tool_idx[: max(len(tool_idx) - keep, 0)]:
                message = messages[i]
                if message.content.startswith(STUB_PREFIX) or message.tool_call_id in protect:
                    continue
                messages[i] = self._stub(message, handles)
            if not self.over(messages, specs):
                break
        if not handles:
            return None
        return Compaction(len(handles), before, estimate_prompt(messages, specs, self.estimator), tuple(handles))

    def _stub(self, message: ChatMessage, handles: list[int]) -> ChatMessage:
        body = _body_of(message.content)
        handle = self.vault.put(message.name or "tool", message.tool_call_id or "", body)
        handles.append(handle)
        tool = message.name or "tool"
        stub = (
            f"{STUB_PREFIX}{handle} elided: {tool} {_status_of(message.content)} {_first_line(body)}; "
            f"call show_output({handle})]"
        )
        return replace(message, content=stub)


def serialize(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Stable JSON-able view used for prefix comparisons and persistence checks."""
    return [
        {
            "role": m.role,
            "content": m.content,
            "thinking": m.thinking,
            "tool_call_id": m.tool_call_id,
            "name": m.name,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": dict(c.arguments)} for c in m.tool_calls],
        }
        for m in messages
    ]
