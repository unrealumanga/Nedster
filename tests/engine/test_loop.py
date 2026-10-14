from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ironclad.contracts import ChatMessage, Outcome, ToolCall, ToolResult, ToolStatus, TransportError
from ironclad.engine.loop import Engine
from ironclad.engine.state import Budget

from .fakes import Events, FakeRuntime, MemoryStore, ScriptedTransport, Turn, call, ok


def read(c: ToolCall) -> ToolResult:
    return ok(c, f"contents of {c.arguments.get('path')}")


def edit(c: ToolCall) -> ToolResult:
    path = str(c.arguments.get("path", "calc.py"))
    return ToolResult(c.id, c.name, ToolStatus.OK, f"edited {path}", data={"sha256": "sha256:abc"}, mutated_paths=(path,))


def green(c: ToolCall) -> ToolResult:
    return ok(c, "5 passed", tests_run=5, exit_code=0)


def make(
    script: list[Any], handlers: dict[str, Any] | None = None, **kw: Any
) -> tuple[Engine, ScriptedTransport, FakeRuntime, Events]:
    transport = ScriptedTransport(script)
    runtime = kw.pop("runtime", None) or FakeRuntime(
        handlers or {"read_file": read, "edit_file": edit, "write_file": edit, "run_tests": green}
    )
    events = Events()
    engine = Engine(transport, runtime, model="fake", emit=events.append, budget=kw.pop("budget", Budget()), **kw)
    return engine, transport, runtime, events


async def wait_until(predicate: Any, attempts: int = 400) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached")


async def test_question_with_reads_is_answered() -> None:
    engine, transport, _, events = make([Turn(calls=[call("read_file", "1", path="a.py")]), Turn("It adds numbers.")])
    result = await engine.run("s", "What does a.py do?")
    assert result.outcome is Outcome.ANSWERED
    assert "show_output" in transport.tool_names[0]
    assert events.kinds()[0] == "turn.started" and events.kinds()[-1] == "turn.finished"
    assert events.of("turn.finished")[0].data["outcome"] == "answered"
    assert engine.last_answer.startswith("It adds numbers.")
    assert [e.data["text"] for e in events.of("model.delta")] == ["It adds numbers."]


async def test_edit_then_tests_is_completed_with_code_summary() -> None:
    script = [
        Turn(calls=[call("read_file", "1", path="calc.py")]),
        Turn(calls=[call("edit_file", "2", path="calc.py")]),
        Turn(calls=[call("run_tests", "3")]),
        Turn("Added subtract. All tests pass."),
    ]
    engine, _, _, events = make(script)
    result = await engine.run("s", "add a subtract function to calc.py")
    assert result.outcome is Outcome.COMPLETED
    assert result.changed_paths == ("calc.py",)
    answer = events.of("turn.finished")[0].data["answer"]
    assert answer.startswith("Added subtract.")
    assert "Changed files: calc.py" in answer and "5 tests" in answer
    finished = events.of("tool.finished")
    assert [e.data["status"] for e in finished] == ["ok", "ok", "ok"]
    assert finished[1].data["mutated_paths"] == ["calc.py"]
    assert events.of("acceptance")[0].data["outcome"] == "completed"


async def test_b02_green_baseline_then_rejected_write_is_not_completed() -> None:
    runtime = FakeRuntime(
        {"run_tests": green, "write_file": edit},
        validators={"write_file": lambda c: "missing required property 'content'"},
    )
    script = [
        Turn(calls=[call("run_tests", "1")]),
        Turn(calls=[call("write_file", "2", path="parser.py")]),
        Turn("Fixed the parser bug."),
    ]
    engine, *_ = make(script, runtime=runtime)
    result = await engine.run("s", "Fix parser.py bug")
    assert result.outcome is not Outcome.COMPLETED and not result.passed
    assert "Changed files: none" in engine.last_answer


async def test_b03_pass_then_later_fail_is_incomplete() -> None:
    def failing(c: ToolCall) -> ToolResult:
        return ToolResult(c.id, c.name, ToolStatus.FAILED, "1 failed", data={"tests_run": 5})

    results = iter([green, failing])
    runtime = FakeRuntime({"edit_file": edit, "run_tests": lambda c: next(results)(c)})
    script = [
        Turn(calls=[call("edit_file", "1", path="a.py")]),
        Turn(calls=[call("run_tests", "2")]),
        Turn(calls=[call("run_tests", "3")]),
        Turn("Done."),
    ]
    engine, *_ = make(script, runtime=runtime)
    assert (await engine.run("s", "fix a.py")).outcome is Outcome.INCOMPLETE


async def test_b03_pass_then_mutation_without_rerun_is_unvalidated() -> None:
    script = [
        Turn(calls=[call("edit_file", "1", path="a.py")]),
        Turn(calls=[call("run_tests", "2")]),
        Turn(calls=[call("edit_file", "3", path="b.py")]),
        Turn("Done."),
    ]
    engine, *_ = make(script)
    result = await engine.run("s", "fix it")
    assert result.outcome is Outcome.UNVALIDATED
    assert result.changed_paths == ("a.py", "b.py")


async def test_b03_unknown_test_count_not_counted() -> None:
    runtime = FakeRuntime({"edit_file": edit, "run_tests": lambda c: ok(c, "done", tests_run=None, exit_code=0)})
    script = [Turn(calls=[call("edit_file", "1", path="a.py")]), Turn(calls=[call("run_tests", "2")]), Turn("Done.")]
    engine, *_ = make(script, runtime=runtime)
    assert (await engine.run("s", "fix it")).outcome is Outcome.UNVALIDATED


async def test_step_budget_exhaustion_is_incomplete() -> None:
    script = [Turn(calls=[call("read_file", str(i), path="a.py")]) for i in range(10)]
    engine, transport, *_ = make(script, budget=Budget(max_steps=3))
    result = await engine.run("s", "loop forever")
    assert result.outcome is Outcome.INCOMPLETE and "step budget" in result.reasons[0]
    assert len(transport.requests) == 3


async def test_budget_exhaustion_after_valid_work_is_never_completed() -> None:
    script = [
        Turn(calls=[call("edit_file", "1", path="a.py")]),
        Turn(calls=[call("run_tests", "2")]),
        Turn(calls=[call("read_file", "3", path="a.py")]),
    ]
    engine, *_ = make(script, budget=Budget(max_steps=3))
    assert (await engine.run("s", "fix")).outcome is Outcome.INCOMPLETE


async def test_tool_call_budget_skips_remaining_calls_and_keeps_pairing() -> None:
    calls = [call("read_file", str(i), path=f"{i}.py") for i in range(3)]
    engine, _, runtime, _ = make([Turn(calls=calls)], budget=Budget(max_tool_calls=2))
    result = await engine.run("s", "read three files")
    assert result.outcome is Outcome.INCOMPLETE
    assert len(runtime.calls) == 2
    tool_msgs = [m for m in engine.messages("s") if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["0", "1", "2"]
    assert "budget" in tool_msgs[2].content


async def test_token_budget_exhaustion_is_incomplete() -> None:
    script = [Turn(calls=[call("read_file", str(i), path="a.py")]) for i in range(5)]
    engine, transport, *_ = make(script, budget=Budget(max_tokens=30))
    result = await engine.run("s", "x")
    assert result.outcome is Outcome.INCOMPLETE
    assert len(transport.requests) == 2  # 15 tokens per turn


async def test_deadline_cancels_running_job_and_is_incomplete() -> None:
    runtime = FakeRuntime({"run_command": lambda c: ok(c)}, delays={"run_command": 5.0})
    script = [Turn(calls=[call("run_command", "1", argv=["sleep"])])]
    engine, *_ = make(script, runtime=runtime, budget=Budget(deadline_s=0.05))
    result = await engine.run("s", "x")
    assert result.outcome is Outcome.INCOMPLETE and "deadline" in result.reasons[0]
    assert runtime.jobs[0].cancelled and runtime.jobs[0].cancel_awaited


async def test_cancellation_during_slow_tool_is_cancelled_and_job_awaited() -> None:
    runtime = FakeRuntime({"edit_file": edit, "run_command": lambda c: ok(c)}, delays={"run_command": 5.0})
    script = [Turn(calls=[call("edit_file", "1", path="a.py")]), Turn(calls=[call("run_command", "2", argv=["slow"])])]
    engine, _, _, events = make(script, runtime=runtime)
    task = asyncio.create_task(engine.run("s", "do slow work"))
    await wait_until(lambda: len(runtime.jobs) == 2 and runtime.active == 1)
    engine.cancel()
    result = await asyncio.wait_for(task, 2)
    assert result.outcome is Outcome.CANCELLED and not result.passed
    assert result.changed_paths == ("a.py",)
    job = runtime.jobs[1]
    assert job.cancelled and job.cancel_awaited
    last = engine.messages("s")[-1]
    assert last.role == "tool" and last.tool_call_id == "2" and last.content.startswith("[cancelled]")
    assert events.of("turn.finished")[0].data["outcome"] == "cancelled"


async def test_outer_task_cancellation_propagates_after_cleanup() -> None:
    runtime = FakeRuntime({"run_command": lambda c: ok(c)}, delays={"run_command": 5.0})
    engine, *_ = make([Turn(calls=[call("run_command", "1", argv=["slow"])])], runtime=runtime)
    task = asyncio.create_task(engine.run("s", "x"))
    await wait_until(lambda: runtime.active == 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.jobs[0].cancel_awaited


async def test_three_identical_invalid_calls_stop_incomplete() -> None:
    runtime = FakeRuntime({"write_file": edit}, validators={"write_file": lambda c: "content is required"})
    bad = [Turn(calls=[call("write_file", str(i), path="a.py")]) for i in range(5)]
    engine, transport, *_ = make(bad, runtime=runtime)
    result = await engine.run("s", "write a.py")
    assert result.outcome is Outcome.INCOMPLETE and "repeated" in result.reasons[0]
    assert len(transport.requests) == 3
    tool_msgs = [m for m in engine.messages("s") if m.role == "tool"]
    assert tool_msgs[0].content.startswith("[invalid] content is required")
    assert "stops the task" in tool_msgs[1].content


async def test_different_invalid_calls_are_repairable() -> None:
    def check(c: ToolCall) -> str | None:
        return None if c.arguments.get("content") is not None else "content is required"

    runtime = FakeRuntime({"write_file": edit, "run_tests": green}, validators={"write_file": check})
    script = [
        Turn(calls=[call("write_file", "1", path="a.py")]),
        Turn(calls=[call("write_file", "2", path="a.py", content="x")]),
        Turn(calls=[call("run_tests", "3")]),
        Turn("done"),
    ]
    engine, *_ = make(script, runtime=runtime)
    assert (await engine.run("s", "write a.py")).outcome is Outcome.COMPLETED


async def test_mutations_serialized_in_order_reads_parallel() -> None:
    runtime = FakeRuntime(
        {"read_file": read, "edit_file": edit, "run_tests": green},
        delays={"read_file": 0.05, "edit_file": 0.02},
    )
    calls = [
        call("read_file", "r1", path="a"),
        call("read_file", "r2", path="b"),
        call("read_file", "r3", path="c"),
        call("edit_file", "w1", path="a"),
        call("edit_file", "w2", path="b"),
        call("edit_file", "w3", path="c"),
    ]
    engine, *_ = make([Turn(calls=calls), Turn(calls=[call("run_tests", "t")]), Turn("ok")], runtime=runtime)
    result = await engine.run("s", "edit three files")
    assert runtime.max_active_mutating == 1
    assert runtime.max_overlap_with_mutation == 1  # nothing ran alongside a mutation
    assert runtime.max_active == 3  # the three reads overlapped
    assert [i for i in runtime.started_order if i.startswith("w")] == ["w1", "w2", "w3"]
    assert [i for i in runtime.finished_order if i.startswith("w")] == ["w1", "w2", "w3"]
    tool_ids = [m.tool_call_id for m in engine.messages("s") if m.role == "tool"]
    assert tool_ids[:6] == ["r1", "r2", "r3", "w1", "w2", "w3"]
    assert result.outcome is Outcome.COMPLETED and result.changed_paths == ("a", "b", "c")


async def test_followup_turn_sees_prior_history_b11() -> None:
    engine, transport, *_ = make([Turn("Noted."), Turn("audit-code-17")])
    await engine.run("s", "Remember audit-code-17")
    await engine.run("s", "What code did I give you?")
    contents = [m["content"] for m in transport.requests[1]]
    assert "Remember audit-code-17" in contents and "Noted." in contents
    assert transport.requests[0] == transport.requests[1][: len(transport.requests[0])]


async def test_resume_from_store_restores_history_and_prefix() -> None:
    store = MemoryStore()
    engine, transport, *_ = make([Turn("Noted.")], store=store)
    await engine.run("s1", "Remember audit-code-17")
    engine2, transport2, *_ = make([Turn("audit-code-17")], store=store)
    await engine2.run("s1", "What code did I give you?")
    assert transport2.requests[0][: len(transport.requests[0])] == transport.requests[0]
    assert store.get_state("s1", "task_state")["objective"] == "Remember audit-code-17"


async def test_pinned_state_is_second_message_and_stable() -> None:
    engine, transport, *_ = make([Turn(calls=[call("read_file", "1", path="a")]), Turn("x")])
    await engine.run("s", "Explain a")
    first = transport.requests[0]
    assert first[0]["role"] == "system" and "Nedster" in first[0]["content"]
    assert first[1]["role"] == "system" and "Objective: Explain a" in first[1]["content"]
    assert transport.requests[1][:3] == first[:3]


async def test_transport_error_is_failed() -> None:
    class Broken(ScriptedTransport):
        async def chat(self, **_: Any):  # type: ignore[override]
            raise TransportError("connection refused")
            yield  # pragma: no cover

    events = Events()
    engine = Engine(Broken([]), FakeRuntime({}), model="m", emit=events.append)
    result = await engine.run("s", "hi")
    assert result.outcome is Outcome.FAILED
    assert events.of("error")[0].data["message"] == "connection refused"


async def test_truncated_model_turn_is_incomplete() -> None:
    engine, *_ = make([Turn("partial", done_reason="length")])
    assert (await engine.run("s", "x")).outcome is Outcome.INCOMPLETE


async def test_validation_flag_on_run_command() -> None:
    runtime = FakeRuntime({"edit_file": edit, "run_command": lambda c: ok(c, "built", exit_code=0)})
    script = [
        Turn(calls=[call("edit_file", "1", path="a.py")]),
        Turn(calls=[call("run_command", "2", argv=["make", "check"], validation=True)]),
        Turn("done"),
    ]
    engine, *_ = make(script, runtime=runtime)
    result = await engine.run("s", "fix and check")
    assert result.outcome is Outcome.COMPLETED
    assert "validation" not in runtime.calls[1].arguments
    assert runtime.calls[1].arguments["argv"] == ["make", "check"]


async def test_unknown_tool_is_reported_invalid() -> None:
    engine, *_ = make([Turn(calls=[call("teleport", "1")]), Turn("sorry")])
    result = await engine.run("s", "x")
    assert result.outcome is Outcome.INCOMPLETE  # unknown tools are treated as rejected mutations
    assert any(m.content.startswith("[invalid] unknown tool") for m in engine.messages("s"))


async def test_tool_runtime_exception_becomes_failed_result() -> None:
    def boom(c: ToolCall) -> ToolResult:
        raise RuntimeError("kaboom")

    engine, *_ = make([Turn(calls=[call("read_file", "1")]), Turn("x")], {"read_file": boom})
    result = await engine.run("s", "x")
    assert result.outcome is Outcome.ANSWERED
    assert any("tool runtime error: kaboom" in m.content for m in engine.messages("s"))


async def test_persists_every_message() -> None:
    store = MemoryStore()
    engine, *_ = make([Turn(calls=[call("read_file", "1", path="a")]), Turn("x")], store=store)
    await engine.run("s1", "q")
    roles = [m.role for m in store.messages("s1")]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert isinstance(store.messages("s1")[1], ChatMessage)
