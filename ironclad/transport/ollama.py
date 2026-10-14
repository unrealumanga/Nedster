"""Ollama chat transport: streaming NDJSON with native tool calls.

Implements :class:`ironclad.contracts.ChatTransport`. A single ``httpx.AsyncClient``
is created lazily and reused for every request so connection pools and the model's
prompt cache (stable request prefix) survive across turns.
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

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_NS_PER_MS = 1_000_000.0


def _count(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _duration_ms(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value) / _NS_PER_MS
    except (TypeError, ValueError):
        return 0.0


def _usage(final: Mapping[str, Any]) -> Usage:
    return Usage(
        prompt_tokens=_count(final.get("prompt_eval_count")),
        completion_tokens=_count(final.get("eval_count")),
        load_ms=_duration_ms(final.get("load_duration")),
        prefill_ms=_duration_ms(final.get("prompt_eval_duration")),
        decode_ms=_duration_ms(final.get("eval_duration")),
        total_ms=_duration_ms(final.get("total_duration")),
    )


def _message_to_payload(message: ChatMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "content": message.content,
            "tool_name": message.name or message.tool_call_id or "",
        }
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"function": {"name": call.name, "arguments": dict(call.arguments)}}
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


class _PartialCall:
    __slots__ = ("id", "name", "dict_args", "raw_args")

    def __init__(self) -> None:
        self.id: str | None = None
        self.name = ""
        self.dict_args: dict[str, Any] | None = None
        self.raw_args = ""


def _accumulate_tool_calls(partials: list[_PartialCall], raw_calls: Any) -> None:
    if not isinstance(raw_calls, list):
        return
    for raw in raw_calls:
        if not isinstance(raw, Mapping):
            continue
        function = raw.get("function")
        function = function if isinstance(function, Mapping) else {}
        name = function.get("name")
        arguments = function.get("arguments")
        call_id = raw.get("id")
        if name:
            partial = _PartialCall()
            partial.name = str(name)
            partials.append(partial)
        elif partials:
            partial = partials[-1]
        else:
            partial = _PartialCall()
            partials.append(partial)
        if call_id and not partial.id:
            partial.id = str(call_id)
        if arguments is None:
            continue
        if isinstance(arguments, Mapping):
            partial.dict_args = dict(arguments)
            partial.raw_args = ""
        elif isinstance(arguments, str):
            partial.raw_args += arguments
        else:
            partial.raw_args += json.dumps(arguments)


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


def _finalize_calls(partials: list[_PartialCall]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for index, partial in enumerate(partials):
        arguments = partial.dict_args if partial.dict_args is not None else _parse_arguments(partial.raw_args)
        calls.append(ToolCall(id=partial.id or f"call_{index}", name=partial.name, arguments=arguments))
    return tuple(calls)


class OllamaTransport:
    """Streaming chat against a local (or remote) Ollama server."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 600,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
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
        think: bool | str,
        keep_alive: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_message_to_payload(message) for message in messages],
            "stream": True,
            "think": think,
        }
        if tools:
            payload["tools"] = [_tool_to_payload(tool) for tool in tools]
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        options: dict[str, Any] = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if options:
            payload["options"] = options
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
            think=think,
            keep_alive=keep_alive,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        url = f"{self.base_url}/api/chat"
        partials: list[_PartialCall] = []
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        final: dict[str, Any] = {}
        try:
            async with self._http().stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        chunk = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise TransportError(f"Malformed NDJSON line from Ollama: {stripped!r}") from exc
                    if not isinstance(chunk, Mapping):
                        raise TransportError(f"Unexpected Ollama chunk: {stripped!r}")
                    if chunk.get("error"):
                        raise TransportError(f"Ollama error: {chunk['error']}")
                    message = chunk.get("message")
                    if isinstance(message, Mapping):
                        thinking = message.get("thinking")
                        if thinking:
                            thinking_parts.append(str(thinking))
                            yield TextDelta(text=str(thinking), kind="thinking")
                        content = message.get("content")
                        if content:
                            content_parts.append(str(content))
                            yield TextDelta(text=str(content), kind="content")
                        raw_calls = message.get("tool_calls")
                        if raw_calls:
                            _accumulate_tool_calls(partials, raw_calls)
                    if chunk.get("done"):
                        final = dict(chunk)
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as exc:
            raise TransportError(f"Ollama request failed: {exc}") from exc

        if not final:
            # A truncated stream may hold half-assembled tool calls; never execute them.
            raise TransportError("Ollama stream ended before the final done chunk")
        calls = _finalize_calls(partials)
        if calls:
            yield ToolCallsReady(calls=calls)
        assistant = ChatMessage(
            role="assistant",
            content="".join(content_parts),
            tool_calls=list(calls),
            thinking="".join(thinking_parts),
        )
        reason = final.get("done_reason")
        yield TurnFinished(message=assistant, usage=_usage(final), done_reason=str(reason or "stop"))

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            response = await self._http().get(f"{self.base_url}/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TransportError(f"Ollama list_models failed: {exc}") from exc
        return list(response.json().get("models") or [])

    async def ping(self) -> bool:
        try:
            response = await self._http().get(f"{self.base_url}/api/tags")
        except httpx.HTTPError:
            return False
        return response.is_success
