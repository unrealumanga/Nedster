from __future__ import annotations

import json
import re

import httpx
import pytest

from ironclad.contracts import (
    ChatMessage,
    TextDelta,
    ToolCall,
    ToolCallsReady,
    ToolSpec,
    TransportError,
    TurnFinished,
    Usage,
)
from ironclad.transport.fake import FakeTransport
from ironclad.transport.ollama import OllamaTransport

CALL_ID_RE = re.compile(r"^call_[0-9a-f]{12}$")


def ndjson(*chunks: dict) -> bytes:
    return ("\n".join(json.dumps(chunk) for chunk in chunks) + "\n").encode()


def make_transport(body: bytes, *, status: int = 200, captured: list[bytes] | None = None) -> OllamaTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(await request.aread())
        return httpx.Response(status, content=body)

    return OllamaTransport(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def drain(transport: OllamaTransport, **kwargs: object) -> tuple[list[object], Exception | None]:
    events: list[object] = []
    payload = {"model": "m", "messages": [], "tools": []}
    payload.update(kwargs)
    try:
        async for event in transport.chat(**payload):
            events.append(event)
    except TransportError as exc:
        return events, exc
    return events, None


async def test_content_and_thinking_deltas():
    body = ndjson(
        {"message": {"role": "assistant", "thinking": "Let me think"}, "done": False},
        {"message": {"role": "assistant", "content": "Hello "}, "done": False},
        {"message": {"role": "assistant", "content": "world"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
    )
    events, error = await drain(make_transport(body))
    assert error is None
    assert events[0] == TextDelta(text="Let me think", kind="thinking")
    assert events[1] == TextDelta(text="Hello ", kind="content")
    assert events[2] == TextDelta(text="world", kind="content")
    finished = events[-1]
    assert isinstance(finished, TurnFinished)
    assert finished.message.role == "assistant"
    assert finished.message.content == "Hello world"
    assert finished.message.thinking == "Let me think"
    assert finished.done_reason == "stop"
    assert not any(isinstance(event, ToolCallsReady) for event in events)


async def test_tool_call_split_across_chunks():
    body = ndjson(
        {
            "message": {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path":'}}],
            },
            "done": False,
        },
        {
            "message": {"role": "assistant", "tool_calls": [{"function": {"arguments": '"a.py"}'}}]},
            "done": True,
            "done_reason": "stop",
        },
    )
    events, error = await drain(make_transport(body))
    assert error is None
    ready = next(event for event in events if isinstance(event, ToolCallsReady))
    assert len(ready.calls) == 1
    assert CALL_ID_RE.match(ready.calls[0].id)
    assert ready.calls[0].name == "read_file"
    assert ready.calls[0].arguments == {"path": "a.py"}
    finished = events[-1]
    assert isinstance(finished, TurnFinished)
    assert finished.message.tool_calls == list(ready.calls)


async def test_arguments_as_json_string():
    body = ndjson(
        {
            "message": {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "foo", "arguments": '{"x": 1}'}}],
            },
            "done": True,
        },
    )
    events, error = await drain(make_transport(body))
    assert error is None
    ready = next(event for event in events if isinstance(event, ToolCallsReady))
    assert ready.calls[0].arguments == {"x": 1}


async def test_invalid_json_arguments_are_marked():
    body = ndjson(
        {
            "message": {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "foo", "arguments": "not-json"}}],
            },
            "done": True,
        },
    )
    events, error = await drain(make_transport(body))
    assert error is None
    ready = next(event for event in events if isinstance(event, ToolCallsReady))
    assert ready.calls[0].arguments == {"__invalid__": "not-json"}


async def test_two_calls_in_one_turn():
    body = ndjson(
        {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "a", "arguments": {"n": 1}}},
                    {"function": {"name": "b", "arguments": '{"n": 2}'}},
                ],
            },
            "done": True,
        },
    )
    events, error = await drain(make_transport(body))
    assert error is None
    ready = next(event for event in events if isinstance(event, ToolCallsReady))
    assert [call.name for call in ready.calls] == ["a", "b"]
    ids = [call.id for call in ready.calls]
    assert all(CALL_ID_RE.match(call_id) for call_id in ids)
    assert len(set(ids)) == 2
    assert ready.calls[1].arguments == {"n": 2}


async def test_usage_nanoseconds_to_milliseconds():
    body = ndjson(
        {
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 10,
            "eval_count": 20,
            "load_duration": 1_000_000,
            "prompt_eval_duration": 2_000_000,
            "eval_duration": 3_000_000,
            "total_duration": 6_000_000,
        },
    )
    events, error = await drain(make_transport(body))
    assert error is None
    finished = events[-1]
    assert isinstance(finished, TurnFinished)
    usage = finished.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.load_ms == 1.0
    assert usage.prefill_ms == 2.0
    assert usage.decode_ms == 3.0
    assert usage.total_ms == 6.0
    assert finished.done_reason == "length"


async def test_http_error_raises_transport_error():
    events, error = await drain(make_transport(b"server error", status=500))
    assert isinstance(error, TransportError)
    assert events == []


async def test_malformed_line_raises_without_tool_calls_ready():
    body = ndjson(
        {
            "message": {
                "role": "assistant",
                "content": "hi",
                "tool_calls": [{"function": {"name": "t", "arguments": {"a": 1}}}],
            },
            "done": False,
        },
    ) + b"not json\n"
    events, error = await drain(make_transport(body))
    assert isinstance(error, TransportError)
    assert not any(isinstance(event, ToolCallsReady) for event in events)
    assert any(isinstance(event, TextDelta) for event in events)


async def test_connection_error_raises_transport_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = OllamaTransport(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    events, error = await drain(transport)
    assert isinstance(error, TransportError)
    assert events == []


async def test_same_history_serializes_to_identical_bytes():
    body = ndjson({"message": {"role": "assistant", "content": "ok"}, "done": True})
    captured: list[bytes] = []
    transport = make_transport(body, captured=captured)
    history = [
        ChatMessage(role="system", content="be brief"),
        ChatMessage(role="user", content="read it"),
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="call_0", name="read_file", arguments={"path": "a.py"})],
        ),
        ChatMessage(role="tool", content="contents", name="read_file", tool_call_id="call_0"),
    ]
    tools = [
        ToolSpec(name="read_file", description="read", parameters={"type": "object"}, risk=0),
    ]
    await drain(transport, messages=history, tools=tools)
    await drain(transport, messages=history, tools=tools)
    assert captured[0] == captured[1]
    request = json.loads(captured[0])
    assert request["stream"] is True
    assert request["messages"][3] == {"role": "tool", "content": "contents", "tool_name": "read_file"}
    assert request["messages"][2]["tool_calls"] == [
        {"function": {"name": "read_file", "arguments": {"path": "a.py"}}}
    ]
    assert request["tools"][0]["function"]["name"] == "read_file"


async def test_options_and_client_reuse():
    body = ndjson({"message": {"role": "assistant", "content": "ok"}, "done": True})
    captured: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(await request.aread())
        return httpx.Response(200, content=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OllamaTransport(client=client)
    assert transport._http() is client
    await drain(transport, max_tokens=64, temperature=0.2)
    assert transport._http() is client
    await drain(transport, max_tokens=64, temperature=0.2)
    assert transport._http() is client
    request = json.loads(captured[0])
    assert request["options"] == {"num_predict": 64, "temperature": 0.2}
    assert "keep_alive" not in request


async def test_fake_transport_replays_script_and_records_calls():
    turn_one = [
        TextDelta(text="working"),
        TurnFinished(message=ChatMessage(role="assistant", content="working"), usage=Usage()),
    ]
    turn_two = [TurnFinished(message=ChatMessage(role="assistant", content="done"), usage=Usage())]
    fake = FakeTransport([turn_one, turn_two])
    messages = [ChatMessage(role="user", content="hi")]
    tools = [ToolSpec(name="t", description="d", parameters={}, risk=0)]

    first = [event async for event in fake.chat(model="m", messages=messages, tools=tools)]
    second = [event async for event in fake.chat(model="m", messages=messages, tools=tools)]

    assert first == turn_one
    assert second == turn_two
    assert fake.calls == [(messages, tools), (messages, tools)]
    with pytest.raises(TransportError):
        [event async for event in fake.chat(model="m", messages=messages, tools=tools)]
    await fake.aclose()
    assert fake.closed is True


async def test_list_models_and_ping():
    body = json.dumps({"models": [{"name": "qwen"}]}).encode()
    transport = make_transport(body)
    assert await transport.ping() is True
    assert await transport.list_models() == [{"name": "qwen"}]

    failing = make_transport(b"nope", status=503)
    assert await failing.ping() is False
    with pytest.raises(TransportError):
        await failing.list_models()


async def test_truncated_stream_never_emits_tool_calls():
    body = ndjson(
        {
            "message": {"role": "assistant", "tool_calls": [{"function": {"name": "write_file", "arguments": {"path": "a"}}}]},
            "done": False,
        },
    )
    events, error = await drain(make_transport(body))
    assert isinstance(error, TransportError)
    assert not any(isinstance(event, ToolCallsReady) for event in events)


async def test_error_chunk_raises_transport_error():
    body = ndjson({"error": "model requires more system memory"})
    events, error = await drain(make_transport(body))
    assert isinstance(error, TransportError)
    assert "system memory" in str(error)


# --- regression #13: unique call ids ---------------------------------------


def _id_less_call_body(name: str = "read_file") -> bytes:
    return ndjson(
        {"message": {"role": "assistant", "tool_calls": [{"function": {"name": name, "arguments": {"path": "a"}}}]},
         "done": True, "done_reason": "stop"},
    )


async def test_fallback_call_ids_are_unique_across_turns():
    transport = make_transport(_id_less_call_body())
    first_ids = []
    second_ids = []
    for target in (first_ids, second_ids):
        events, error = await drain(transport)
        assert error is None
        ready = next(event for event in events if isinstance(event, ToolCallsReady))
        assert CALL_ID_RE.match(ready.calls[0].id)
        target.append(ready.calls[0].id)
    assert first_ids[0] != second_ids[0]


async def test_duplicate_server_call_ids_are_deduped():
    body = ndjson(
        {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {"id": "dup", "function": {"name": "a", "arguments": {"n": 1}}},
                    {"id": "dup", "function": {"name": "b", "arguments": {"n": 2}}},
                ],
            },
            "done": True,
        },
    )
    events, error = await drain(make_transport(body))
    assert error is None
    ready = next(event for event in events if isinstance(event, ToolCallsReady))
    assert [call.id for call in ready.calls] == ["dup", "dup_1"]


# --- regression #20: bounded streams ---------------------------------------


def make_capped(body: bytes, **kwargs: object) -> OllamaTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return OllamaTransport(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kwargs)


async def test_total_stream_bytes_capped():
    body = ndjson({"message": {"role": "assistant", "content": "x" * 400}, "done": True})
    events, error = await drain(make_capped(body, max_stream_bytes=100))
    assert isinstance(error, TransportError)
    assert "bytes for one turn" in str(error)
    assert not any(isinstance(event, ToolCallsReady) for event in events)


async def test_line_bytes_capped():
    body = b"x" * 200
    events, error = await drain(make_capped(body, max_line_bytes=100))
    assert isinstance(error, TransportError)
    assert "exceeded 100 bytes" in str(error)


async def test_default_caps_are_8_mib_and_1_mib():
    transport = make_transport(b"")
    assert transport.max_stream_bytes == 8 * 1024 * 1024
    assert transport.max_line_bytes == 1024 * 1024
