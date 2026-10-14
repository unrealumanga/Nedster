from __future__ import annotations

import json
import re

import httpx

from ironclad.contracts import (
    ChatMessage,
    TextDelta,
    ToolCall,
    ToolCallsReady,
    ToolSpec,
    TransportError,
    TurnFinished,
)
from ironclad.transport.openai_compat import OpenAICompatTransport

CALL_ID_RE = re.compile(r"^call_[0-9a-f]{12}$")


def sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(chunk)}" for chunk in chunks]
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def make_transport(body: bytes, *, status: int = 200, captured: list[bytes] | None = None) -> OpenAICompatTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(await request.aread())
        return httpx.Response(status, content=body)

    return OpenAICompatTransport(
        "http://localhost:8080/v1",
        "secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def drain(transport: OpenAICompatTransport, **kwargs: object) -> tuple[list[object], Exception | None]:
    events: list[object] = []
    payload = {"model": "m", "messages": [], "tools": []}
    payload.update(kwargs)
    try:
        async for event in transport.chat(**payload):
            events.append(event)
    except TransportError as exc:
        return events, exc
    return events, None


async def test_content_deltas_and_finish():
    body = sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )
    events, error = await drain(make_transport(body))
    assert error is None
    assert events[0] == TextDelta(text="Hel", kind="content")
    assert events[1] == TextDelta(text="lo", kind="content")
    finished = events[-1]
    assert isinstance(finished, TurnFinished)
    assert finished.message.content == "Hello"
    assert finished.done_reason == "stop"
    assert not any(isinstance(event, ToolCallsReady) for event in events)


async def test_reasoning_becomes_thinking():
    body = sse(
        {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )
    events, error = await drain(make_transport(body))
    assert error is None
    assert events[0] == TextDelta(text="hmm", kind="thinking")
    finished = events[-1]
    assert isinstance(finished, TurnFinished)
    assert finished.message.thinking == "hmm"


async def test_tool_call_fragments_reassembled():
    body = sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "read", "arguments": '{"p'},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ath":"x"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )
    events, error = await drain(make_transport(body))
    assert error is None
    ready = next(event for event in events if isinstance(event, ToolCallsReady))
    assert ready.calls == (ToolCall(id="call_abc", name="read", arguments={"path": "x"}),)
    assert events[-1].done_reason == "stop"


async def test_usage_from_final_chunk():
    body = sse(
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}},
    )
    events, error = await drain(make_transport(body))
    assert error is None
    finished = events[-1]
    assert isinstance(finished, TurnFinished)
    assert finished.usage.prompt_tokens == 11
    assert finished.usage.completion_tokens == 7


async def test_http_error_raises_transport_error():
    events, error = await drain(make_transport(b"boom", status=500))
    assert isinstance(error, TransportError)
    assert events == []


async def test_malformed_sse_raises_without_tool_calls_ready():
    body = b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\ndata: {not json}\n\n'
    events, error = await drain(make_transport(body))
    assert isinstance(error, TransportError)
    assert not any(isinstance(event, ToolCallsReady) for event in events)


async def test_request_body_and_stable_bytes():
    body = sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    captured: list[bytes] = []
    transport = make_transport(body, captured=captured)
    history = [
        ChatMessage(role="system", content="be brief"),
        ChatMessage(role="user", content="read it"),
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="call_0", name="read_file", arguments={"path": "a.py"})],
        ),
        ChatMessage(role="tool", content="contents", tool_call_id="call_0"),
    ]
    tools = [ToolSpec(name="read_file", description="read", parameters={"type": "object"}, risk=0)]
    await drain(transport, messages=history, tools=tools, max_tokens=32)
    await drain(transport, messages=history, tools=tools, max_tokens=32)
    assert captured[0] == captured[1]
    request = json.loads(captured[0])
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert request["max_tokens"] == 32
    assert request["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"path": "a.py"}'
    assert request["messages"][3] == {"role": "tool", "tool_call_id": "call_0", "content": "contents"}
    assert request["tools"][0]["type"] == "function"
    assert request["tools"][0]["function"]["parameters"] == {"type": "object"}


async def test_truncated_stream_never_emits_tool_calls():
    chunk = {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "c1", "function": {"name": "write_file", "arguments": "{\"path\": "}}
    ]}, "finish_reason": None}]}
    body = f"data: {json.dumps(chunk)}\n\n".encode()
    events, error = await drain(make_transport(body))
    assert isinstance(error, TransportError)
    assert not any(isinstance(event, ToolCallsReady) for event in events)


# --- regression #13: unique call ids ---------------------------------------


def _id_less_call_body() -> bytes:
    return sse(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "read", "arguments": '{"p":1}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )


async def test_fallback_call_ids_are_unique_across_turns():
    transport = make_transport(_id_less_call_body())
    ids = []
    for _ in range(2):
        events, error = await drain(transport)
        assert error is None
        ready = next(event for event in events if isinstance(event, ToolCallsReady))
        assert CALL_ID_RE.match(ready.calls[0].id)
        ids.append(ready.calls[0].id)
    assert ids[0] != ids[1]


async def test_duplicate_server_call_ids_are_deduped():
    body = sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "dup", "function": {"name": "a", "arguments": "{}"}},
                            {"index": 1, "id": "dup", "function": {"name": "b", "arguments": "{}"}},
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )
    events, error = await drain(make_transport(body))
    assert error is None
    ready = next(event for event in events if isinstance(event, ToolCallsReady))
    assert [call.id for call in ready.calls] == ["dup", "dup_1"]


# --- regression #20: bounded streams ---------------------------------------


def make_capped(body: bytes, **kwargs: object) -> OpenAICompatTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return OpenAICompatTransport(
        "http://localhost:8080/v1",
        "secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


async def test_total_stream_bytes_capped():
    body = sse({"choices": [{"delta": {"content": "x" * 400}}]})
    events, error = await drain(make_capped(body, max_stream_bytes=100))
    assert isinstance(error, TransportError)
    assert "bytes for one turn" in str(error)
    assert not any(isinstance(event, ToolCallsReady) for event in events)


async def test_line_bytes_capped():
    events, error = await drain(make_capped(b"data: " + b"x" * 300, max_line_bytes=100))
    assert isinstance(error, TransportError)
    assert "exceeded 100 bytes" in str(error)


async def test_default_caps_are_8_mib_and_1_mib():
    transport = make_transport(b"")
    assert transport.max_stream_bytes == 8 * 1024 * 1024
    assert transport.max_line_bytes == 1024 * 1024
