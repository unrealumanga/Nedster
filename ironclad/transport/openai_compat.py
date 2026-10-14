"""OpenAI-compatible chat transport (SSE streaming), off by default.

Speaks ``POST {base_url}/chat/completions`` with ``stream=true`` and
``stream_options.include_usage=true``. Tool calls arrive as ``delta.tool_calls``
fragments that are keyed by ``index`` and reassembled before one
:class:`ToolCallsReady` is emitted.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from ironclad.contracts import (
    ChatEvent,
    ChatMessage,
    TextDelta,
    ToolCall,
    ToolCallsReady,
    ToolSpec,
    TransportError,
    TurnFinished,
    Usage,
)

_DONE_REASONS = {"stop": "stop", "length": "length"}


def _count(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _usage(raw: Any) -> Usage:
    source = raw if isinstance(raw, Mapping) else {}
    return Usage(
        prompt_tokens=_count(source.get("prompt_tokens")),
        completion_tokens=_count(source.get("completion_tokens")),
    )


def _message_to_payload(message: ChatMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.content,
        }
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments))},
                }
                for call in message.tool_calls
            ],
        }
    return {"role": message.role, "content": message.content}


def _tool_to_payload(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
        },
    }


class _CallBuilder:
    __slots__ = ("id", "name", "args")

    def __init__(self) -> None:
        self.id: str | None = None
        self.name = ""
        self.args = ""


def _accumulate(builders: dict[int, _CallBuilder], raw_calls: Any) -> None:
    if not isinstance(raw_calls, list):
        return
    for position, raw in enumerate(raw_calls):
        if not isinstance(raw, Mapping):
            continue
        index = raw.get("index", position)
        try:
            key = int(index)
        except (TypeError, ValueError):
            key = position
        builder = builders.setdefault(key, _CallBuilder())
        call_id = raw.get("id")
        if call_id:
            builder.id = str(call_id)
        function = raw.get("function")
        function = function if isinstance(function, Mapping) else {}
        name = function.get("name")
        if name:
            builder.name = str(name)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            builder.args += arguments
        elif arguments is not None:
            builder.args += json.dumps(arguments)


def _parse_arguments(raw: str) -> Mapping[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"__invalid__": raw}
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return {"__invalid__": raw}


def _finalize(builders: dict[int, _CallBuilder]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for index, key in enumerate(sorted(builders)):
        builder = builders[key]
        calls.append(
            ToolCall(
                id=builder.id or f"call_{index}",
                name=builder.name,
                arguments=_parse_arguments(builder.args),
            )
        )
    return tuple(calls)


class OpenAICompatTransport:
    """Optional quality-first cloud transport; disabled unless constructed."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 600,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None and self._owns_client:
            await client.aclose()

    def _payload(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_message_to_payload(message) for message in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = [_tool_to_payload(tool) for tool in tools]
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

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
        payload = self._payload(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/chat/completions"
        builders: dict[int, _CallBuilder] = {}
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        usage = Usage()
        finish_reason = ""
        try:
            async with self._http().stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    stripped = line.strip()
                    if not stripped or not stripped.startswith("data:"):
                        continue
                    data = stripped[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise TransportError(f"Malformed SSE data from provider: {data!r}") from exc
                    if not isinstance(chunk, Mapping):
                        raise TransportError(f"Unexpected SSE chunk: {data!r}")
                    if chunk.get("usage"):
                        usage = _usage(chunk["usage"])
                    for choice in chunk.get("choices") or []:
                        if not isinstance(choice, Mapping):
                            continue
                        delta = choice.get("delta")
                        delta = delta if isinstance(delta, Mapping) else {}
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if reasoning:
                            thinking_parts.append(str(reasoning))
                            yield TextDelta(text=str(reasoning), kind="thinking")
                        content = delta.get("content")
                        if content:
                            content_parts.append(str(content))
                            yield TextDelta(text=str(content), kind="content")
                        _accumulate(builders, delta.get("tool_calls"))
                        reason = choice.get("finish_reason")
                        if reason:
                            finish_reason = str(reason)
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as exc:
            raise TransportError(f"OpenAI-compatible request failed: {exc}") from exc

        if not finish_reason:
            # A truncated stream may hold half-assembled tool calls; never execute them.
            raise TransportError("stream ended before a finish_reason was received")
        calls = _finalize(builders)
        if calls:
            yield ToolCallsReady(calls=calls)
        assistant = ChatMessage(
            role="assistant",
            content="".join(content_parts),
            tool_calls=list(calls),
            thinking="".join(thinking_parts),
        )
        reason = _DONE_REASONS.get(finish_reason, "stop")
        yield TurnFinished(message=assistant, usage=usage, done_reason=reason)
