"""Engine-run validation when the model changes files but does not validate them itself."""

from __future__ import annotations

import json
from typing import Any

from ironclad.contracts import Outcome, ToolCall, ToolResult, ToolStatus
from ironclad.engine.loop import AUTO_VERIFY_TOOL, Engine

from .fakes import FakeRuntime, ScriptedTransport, Turn, call, ok


def edit(c: ToolCall) -> ToolResult:
    path = str(c.arguments.get("path", "a.py"))
    return ToolResult(c.id, c.name, ToolStatus.OK, f"edited {path}", mutated_paths=(path,))


def green(c: ToolCall) -> ToolResult:
    return ok(c, "5 passed", tests_run=5, exit_code=0)


def red(c: ToolCall) -> ToolResult:
    return ToolResult(c.id, c.name, ToolStatus.FAILED, "1 failed, 4 passed", data={"tests_run": 5, "tests_failed": 1})


def engine_for(script: list[Turn], handlers: dict[str, Any], **kw: Any) -> tuple[Engine, FakeRuntime, ScriptedTransport]:
    runtime = FakeRuntime(handlers)
    transport = ScriptedTransport(script)
    return Engine(transport, runtime, model="m", **kw), runtime, transport


def auto_calls(runtime: FakeRuntime) -> list[ToolCall]:
    return [c for c in runtime.calls if c.name == AUTO_VERIFY_TOOL and c.id.startswith("auto-verify-")]


async def test_skipped_validation_is_run_by_the_engine_and_completes() -> None:
    engine, runtime, _ = engine_for([Turn(calls=[call("edit_file", "1", path="a.py")]), Turn("Done.")],
                                    {"edit_file": edit, "run_tests": green})  # fmt: skip
    result = await engine.run("s", "fix the bug in a.py")
    assert result.outcome is Outcome.COMPLETED, result.reasons
    assert len(auto_calls(runtime)) == 1


async def test_failing_auto_validation_gives_the_model_one_repair_turn() -> None:
    results = iter([red, green])
    handlers = {"edit_file": edit, "run_tests": lambda c: next(results)(c)}
    script = [
        Turn(calls=[call("edit_file", "1", path="a.py")]),
        Turn("Done."),
        Turn(calls=[call("edit_file", "2", path="a.py")]),  # repair after seeing the failure
        Turn("Fixed."),
    ]
    engine, runtime, transport = engine_for(script, handlers)
    result = await engine.run("s", "fix the bug in a.py")
    assert result.outcome is Outcome.COMPLETED, result.reasons
    assert len(auto_calls(runtime)) == 2
    assert "1 failed" in json.dumps(transport.requests[2])  # the model saw the failing output


async def test_last_attempt_failing_stops_incomplete_without_looping() -> None:
    script = [
        Turn(calls=[call("edit_file", "1", path="a.py")]),
        Turn("Done."),
        Turn(calls=[call("edit_file", "2", path="a.py")]),
        Turn("Still done."),
    ]
    engine, runtime, transport = engine_for(script, {"edit_file": edit, "run_tests": red}, auto_verify=2)
    result = await engine.run("s", "fix the bug in a.py")
    assert result.outcome is Outcome.INCOMPLETE and not result.passed
    assert len(auto_calls(runtime)) == 2 and len(transport.requests) == 4  # no third model turn


async def test_no_rerun_when_the_model_changed_nothing_after_a_failure() -> None:
    script = [Turn(calls=[call("edit_file", "1", path="a.py")]), Turn("Done."), Turn("It is a pre-existing failure.")]
    engine, runtime, _ = engine_for(script, {"edit_file": edit, "run_tests": red})
    result = await engine.run("s", "fix the bug in a.py")
    assert result.outcome is Outcome.INCOMPLETE  # the failing evidence stands
    assert len(auto_calls(runtime)) == 1


async def test_no_auto_validation_for_questions_without_changes_or_when_disabled() -> None:
    engine, runtime, _ = engine_for([Turn("It adds numbers.")], {"run_tests": green})
    assert (await engine.run("s", "what does a.py do?")).outcome is Outcome.ANSWERED
    assert auto_calls(runtime) == []
    engine, runtime, _ = engine_for([Turn(calls=[call("edit_file", "1")]), Turn("Done.")],
                                    {"edit_file": edit, "run_tests": green}, auto_verify=0)  # fmt: skip
    assert (await engine.run("s", "fix it")).outcome is Outcome.UNVALIDATED
    assert auto_calls(runtime) == []


async def test_model_validated_run_is_not_rerun() -> None:
    script = [Turn(calls=[call("edit_file", "1")]), Turn(calls=[call("run_tests", "2")]), Turn("Done.")]
    engine, runtime, _ = engine_for(script, {"edit_file": edit, "run_tests": green})
    assert (await engine.run("s", "fix it")).outcome is Outcome.COMPLETED
    assert auto_calls(runtime) == []


async def test_without_a_test_tool_the_outcome_stays_unvalidated() -> None:
    engine, runtime, _ = engine_for([Turn(calls=[call("edit_file", "1")]), Turn("Done.")], {"edit_file": edit})
    assert (await engine.run("s", "fix it")).outcome is Outcome.UNVALIDATED


async def test_denied_or_undetectable_suite_is_not_evidence() -> None:
    def denied(c: ToolCall) -> ToolResult:
        return ToolResult(c.id, c.name, ToolStatus.DENIED, "execution is not allowed")

    engine, runtime, _ = engine_for([Turn(calls=[call("edit_file", "1")]), Turn("Done.")],
                                    {"edit_file": edit, "run_tests": denied})  # fmt: skip
    result = await engine.run("s", "fix it")
    assert result.outcome is Outcome.UNVALIDATED and not result.passed
    assert len(auto_calls(runtime)) == 1
