"""Shared streaming and tool-call helpers for the chat transports.

Two review findings live here:

* #13: server responses that omit a tool-call ``id`` used to fall back to
  ``call_{index}``, which repeats every turn (``call_0``) and makes call/result
  pairing and compaction ambiguous. Fallback ids are now random and unique.
* #20: streams were buffered without limit. The transport now caps the total
  bytes per turn and the length of a single line while reading, raising
  :class:`TransportError` before the offending data is buffered in full.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterable

import httpx

from ironclad.contracts import ToolCall, TransportError

DEFAULT_MAX_STREAM_BYTES = 8 * 1024 * 1024  # total bytes read for one turn
DEFAULT_MAX_LINE_BYTES = 1024 * 1024  # bytes in a single streamed line


def new_call_id() -> str:
    """A random, collision-resistant id for a tool call the server did not name."""
    return f"call_{uuid.uuid4().hex[:12]}"


def ensure_unique_call_ids(calls: Iterable[ToolCall]) -> tuple[ToolCall, ...]:
    """Return ``calls`` with every id unique within the turn.

    A missing id is replaced by :func:`new_call_id`; an id that repeats is given a
    numeric suffix (``dup``, ``dup_1``, ``dup_2`` ...).
    """
    result: list[ToolCall] = []
    seen: set[str] = set()
    for call in calls:
        call_id = call.id or new_call_id()
        if call_id in seen:
            base, suffix = call_id, 1
            while f"{base}_{suffix}" in seen:
                suffix += 1
            call_id = f"{base}_{suffix}"
        seen.add(call_id)
        result.append(call if call_id == call.id else ToolCall(call_id, call.name, call.arguments))
    return tuple(result)


async def iter_stream_lines(
    response: httpx.Response,
    *,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_STREAM_BYTES,
) -> AsyncIterator[str]:
    """Yield decoded lines from ``response`` while enforcing the byte caps.

    Raises :class:`TransportError` as soon as a single line or the whole turn
    exceeds its cap, without buffering the offending data in full.
    """
    buffer = bytearray()
    total = 0
    start = 0
    async for chunk in response.aiter_bytes():
        if not chunk:
            continue
        total += len(chunk)
        if total > max_total_bytes:
            raise TransportError(f"Streamed response exceeded {max_total_bytes} bytes for one turn")
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n", start)
            if newline < 0:
                break
            yield buffer[start:newline].decode("utf-8", "replace")
            start = newline + 1
        if start:
            del buffer[:start]
            start = 0
        if len(buffer) > max_line_bytes:
            raise TransportError(f"Streamed line exceeded {max_line_bytes} bytes")
    if buffer:
        yield bytes(buffer).decode("utf-8", "replace")
