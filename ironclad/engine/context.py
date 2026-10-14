"""Context management: bounded tool messages, observation masking and the ``show_output`` tool.

Messages are append-only between compactions. A compaction is a discrete event that replaces old tool
outputs with truthful stubs pointing at a durable handle; the full text stays retrievable through
``show_output``. Call/result pairing is never broken: a stub keeps its ``tool_call_id``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ironclad.contracts import ChatMessage, SessionStore, ToolCall, ToolResult, ToolSpec, ToolStatus
from ironclad.engine.acceptance import validation_verdict
from ironclad.engine.state import ToolRecord

Estimator = Callable[[str], int]
KEEP_LAST = 6
THRESHOLD = 0.7
TARGET = 0.5  # a whole-group compaction aims below this fraction of the window (headroom before the next one)
RESERVE_MIN, RESERVE_MAX = 512, 4_096
SUMMARY_PREFIX = "[compaction summary #"
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


def completion_reserve(budget_tokens: int) -> int:
    """Tokens of the window held back for the model's answer (counted against every request)."""
    return min(RESERVE_MAX, max(RESERVE_MIN, budget_tokens // 8))


@dataclass(slots=True, frozen=True)
class Compaction:
    elided: int
    tokens_before: int
    tokens_after: int
    handles: tuple[int, ...]
    dropped_groups: int = 0  # whole assistant(+results) groups replaced by the summary
    stage: str = "mask"  # "mask" (outputs only) or "groups"


MAX_READS, MAX_CHANGES, MAX_VALIDATIONS, MAX_FAILURES = 30, 30, 6, 8


def _ranges(handles: Sequence[int]) -> str:
    ordered = sorted(set(handles))
    out: list[str] = []
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1] == ordered[end] + 1:
            end += 1
        out.append(f"#{ordered[start]}" if end == start else f"#{ordered[start]}-#{ordered[end]}")
        start = end + 1
    return ", ".join(out)


@dataclass(slots=True, frozen=True)
class ToolPair:
    call: ToolCall
    status: str  # status word of the result message ("?" when unanswered)
    first_line: str


@dataclass(slots=True)
class SummaryData:
    """Evidence about compacted history, built only from structured tool records (no model text)."""

    count: int = 0  # compactions so far
    calls: int = 0
    assistant_messages: int = 0
    reads: dict[str, str] = field(default_factory=dict)  # path -> sha ("" unknown)
    changes: dict[str, str] = field(default_factory=dict)
    validations: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    handles: list[int] = field(default_factory=list)

    def copy(self) -> SummaryData:
        return SummaryData.from_dict(self.to_dict())

    def render(self) -> str:
        lines = [f"{SUMMARY_PREFIX}{self.count}: older steps were removed from this conversation; evidence kept below]"]
        lines.append(f"Removed: {self.calls} tool calls, {self.assistant_messages} assistant messages.")
        if self.reads:
            lines.append("Files read (path sha):")
            lines.extend(f"- {p} {h}".rstrip() for p, h in self.reads.items())
        if self.changes:
            lines.append("Files changed (path sha):")
            lines.extend(f"- {p} {h}".rstrip() for p, h in self.changes.items())
        if self.validations:
            lines.append("Validations:")
            lines.extend(f"- {v}" for v in self.validations)
        if self.failures:
            lines.append("Failures:")
            lines.extend(f"- {f}" for f in self.failures)
        if self.handles:
            lines.append(f"Full outputs: show_output(handle) for {_ranges(self.handles)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "calls": self.calls,
            "assistant_messages": self.assistant_messages,
            "reads": dict(self.reads),
            "changes": dict(self.changes),
            "validations": list(self.validations),
            "failures": list(self.failures),
            "handles": list(self.handles),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> SummaryData:
        if not data:
            return cls()
        return cls(
            count=int(data.get("count", 0)),
            calls=int(data.get("calls", 0)),
            assistant_messages=int(data.get("assistant_messages", 0)),
            reads={str(k): str(v) for k, v in dict(data.get("reads", {})).items()},
            changes={str(k): str(v) for k, v in dict(data.get("changes", {})).items()},
            validations=[str(v) for v in data.get("validations", [])],
            failures=[str(v) for v in data.get("failures", [])],
            handles=[int(h) for h in data.get("handles", [])],
        )

    def absorb(self, pair: ToolPair, record: ToolRecord | None, latest_change: int | None) -> None:
        """Fold one dropped call and its result into the evidence (record preferred, message as fallback)."""
        self.calls += 1
        name, status = pair.call.name, pair.status
        path = str(pair.call.arguments.get("path", ""))
        sha = str(record.result.data.get("sha256", "") or "") if record else ""
        verdict = validation_verdict(record) if record else None
        if record is not None and verdict is not None:
            fresh = latest_change is None or record.revision_before.counter >= latest_change
            self.validations.append(
                f"{name} {verdict} at revision {record.revision_before.counter} "
                f"({'fresh' if fresh else 'stale: a later change exists'}): {pair.first_line}"
            )
        elif record is not None and status == "ok" and record.result.mutated_paths:
            for changed in record.result.mutated_paths:
                self.changes[changed] = sha
        elif status == "ok" and name == "read_file" and path:
            self.reads[path] = sha
        elif status not in ("ok", "?"):
            self.failures.append(f"{name} {status}{' ' + path if path else ''}: {pair.first_line}")
        del self.validations[:-MAX_VALIDATIONS]
        del self.failures[:-MAX_FAILURES]
        for table, limit in ((self.reads, MAX_READS), (self.changes, MAX_CHANGES)):
            while len(table) > limit:
                table.pop(next(iter(table)))


def _is_summary(message: ChatMessage) -> bool:
    return message.role == "user" and message.content.startswith(SUMMARY_PREFIX)


def _stub_status(content: str) -> tuple[str, str]:
    """Status word and first line carried by an already-masked output stub."""
    parts = content.split(" elided: ", 1)[-1].split(" ", 2)
    status = parts[1] if len(parts) > 1 else "?"
    line = parts[2].rsplit("; call show_output", 1)[0] if len(parts) > 2 else ""
    return status, line


def _assistant_groups(history: Sequence[ChatMessage]) -> list[list[int]]:
    """Indexes of each droppable unit: an assistant message plus the tool messages answering its calls.

    User messages and the summary are never part of a group, so the objective and constraints stay verbatim.
    """
    groups: list[list[int]] = []
    owner: dict[str, int] = {}
    for i, message in enumerate(history):
        if message.role == "assistant":
            groups.append([i])
            for c in message.tool_calls:
                owner[c.id] = len(groups) - 1
        elif message.role == "tool" and message.tool_call_id in owner:
            groups[owner[message.tool_call_id]].append(i)
    return groups


@dataclass(slots=True)
class ContextManager:
    budget_tokens: int
    vault: OutputVault
    estimator: Estimator = estimate_tokens
    keep_last: int = KEEP_LAST
    threshold: float = THRESHOLD
    reserve_tokens: int = 0  # completion reserve: counted against every request
    summary: SummaryData = field(default_factory=SummaryData)

    @property
    def hard_limit(self) -> int:
        return self.budget_tokens - self.reserve_tokens

    @property
    def limit(self) -> int:
        """Soft limit that triggers a compaction (never above the hard limit)."""
        return min(int(self.budget_tokens * self.threshold), self.hard_limit)

    def _size(self, messages: Sequence[ChatMessage], specs: Sequence[ToolSpec]) -> int:
        return estimate_prompt(messages, specs, self.estimator)

    def over(self, messages: Sequence[ChatMessage], specs: Sequence[ToolSpec]) -> bool:
        return self._size(messages, specs) > self.limit

    def fits(self, messages: Sequence[ChatMessage], specs: Sequence[ToolSpec]) -> bool:
        """The whole request (messages, tool schemas, pinned state) plus the completion reserve fits the window."""
        return self._size(messages, specs) <= self.hard_limit

    def compact(
        self,
        messages: list[ChatMessage],
        specs: Sequence[ToolSpec],
        *,
        protect: Collection[str] = (),
        records: Sequence[ToolRecord] = (),
        latest_change: int | None = None,
    ) -> Compaction | None:
        """Mask old tool outputs; if still over the soft limit, replace whole old groups by a summary.

        ``messages[:2]`` (system prompt, pinned state) are never touched. ``None`` if nothing changed.
        """
        before = self._size(messages, specs)
        handles: list[int] = []
        self._mask(messages, specs, protect, handles)
        dropped = 0
        if self.over(messages, specs):
            dropped = self._drop_groups(messages, specs, records, latest_change, handles)
        if not handles and not dropped:
            return None
        return Compaction(len(handles), before, self._size(messages, specs), tuple(handles), dropped, "groups" if dropped else "mask")

    def _mask(self, messages: list[ChatMessage], specs: Sequence[ToolSpec], protect: Collection[str], handles: list[int]) -> None:
        for keep in dict.fromkeys((self.keep_last, 3, 1)):
            tool_idx = [i for i, m in enumerate(messages) if m.role == "tool"]
            for i in tool_idx[: max(len(tool_idx) - keep, 0)]:
                message = messages[i]
                if message.content.startswith(STUB_PREFIX) or message.tool_call_id in protect:
                    continue
                messages[i] = self._stub(message, handles)
            if not self.over(messages, specs):
                break

    def _drop_groups(
        self,
        messages: list[ChatMessage],
        specs: Sequence[ToolSpec],
        records: Sequence[ToolRecord],
        latest_change: int | None,
        handles: list[int],
    ) -> int:
        """Drop the oldest whole groups until under TARGET; the newest group always stays."""
        history = messages[2:]
        groups = _assistant_groups(history)
        by_id: dict[str, list[ToolRecord]] = {}
        for record in records:
            by_id.setdefault(record.call.id, []).append(record)
        data = self.summary.copy()
        data.count += 1
        removed: set[int] = set()
        dropped = 0
        for group in groups[:-1]:
            removed.update(group)
            dropped += 1
            self._absorb_group(data, history, group, by_id, latest_change, handles)
            data.handles = sorted({*data.handles, *handles})
            if self._size(self._rebuilt(messages, history, removed, data), specs) <= int(self.budget_tokens * TARGET):
                break
        if not removed:
            return 0
        messages[:] = self._rebuilt(messages, history, removed, data)
        self.summary = data
        return dropped

    @staticmethod
    def _rebuilt(messages: Sequence[ChatMessage], history: Sequence[ChatMessage], removed: set[int], data: SummaryData) -> list[ChatMessage]:
        """Head + history without the removed groups; the summary sits where the first removed group was."""
        out = list(messages[:2])
        placed = False
        for i, message in enumerate(history):
            if i in removed and not placed:
                out.append(ChatMessage(role="user", content=data.render()))
                placed = True
            if i not in removed and not _is_summary(message):
                out.append(message)
        return out

    def _absorb_group(
        self,
        data: SummaryData,
        history: Sequence[ChatMessage],
        group: Sequence[int],
        by_id: Mapping[str, list[ToolRecord]],
        latest_change: int | None,
        handles: list[int],
    ) -> None:
        results = {history[i].tool_call_id: history[i] for i in group if history[i].role == "tool"}
        for i in group:
            message = history[i]
            if message.role == "tool":
                if not message.content.startswith(STUB_PREFIX):  # keep the full text retrievable
                    handles.append(self.vault.put(message.name or "tool", message.tool_call_id or "", _body_of(message.content)))
                continue
            data.assistant_messages += 1
            for call in message.tool_calls:
                result = results.get(call.id)
                status, text = "?", ""
                if result is not None:
                    if result.content.startswith(STUB_PREFIX):
                        status, text = _stub_status(result.content)
                    else:
                        status, text = _status_of(result.content), _first_line(_body_of(result.content))
                queue = by_id.get(call.id, [])
                data.absorb(ToolPair(call, status, text), queue.pop(0) if queue else None, latest_change)

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
