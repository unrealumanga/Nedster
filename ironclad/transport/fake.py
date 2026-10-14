"""Scripted in-memory transport for engine tests (no network, fully deterministic).

Each call to :meth:`FakeTransport.chat` replays the next scripted turn and records
the ``(messages, tools)`` it was invoked with in ``self.calls``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ironclad.contracts import (
    ChatEvent,
    ChatMessage,
    ToolSpec,
    TransportError,
)


class FakeTransport:
    def __init__(self, script: list[list[ChatEvent]]) -> None:
        self.script = script
        self.calls: list[tuple[list[ChatMessage], list[ToolSpec]]] = []
        self.closed = False
        self._index = 0

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
    ) -> AsyncIterator[ChatEvent]:
        self.calls.append((list(messages), list(tools)))
        if self._index >= len(self.script):
            raise TransportError("FakeTransport script exhausted")
        turn = self.script[self._index]
        self._index += 1
        for event in turn:
            yield event

    async def aclose(self) -> None:
        self.closed = True
