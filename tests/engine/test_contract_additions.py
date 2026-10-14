from __future__ import annotations

from dataclasses import replace

from ironclad.contracts import Outcome, ToolCall, ToolResult, ToolSpec

from .fakes import FakeRuntime, Turn, call, ok
from .test_loop import edit, make, read


class ValidatingRuntime(FakeRuntime):
    """Advertises `lint` with ToolSpec.validates=True (no `validation` argument needed)."""

    def specs(self) -> list[ToolSpec]:
        return [replace(s, validates=True) if s.name == "lint" else s for s in super().specs()]


def lint_ok(c: ToolCall) -> ToolResult:
    return ok(c, "lint clean", exit_code=0, tests_run=3)


async def test_spec_validates_flag_counts_as_validation_and_answer_is_returned() -> None:
    script = [
        Turn(calls=[call("read_file", "1", path="calc.py")]),
        Turn(calls=[call("edit_file", "2", path="calc.py")]),
        Turn(calls=[call("lint", "3")]),
        Turn("Done."),
    ]
    runtime = ValidatingRuntime({"read_file": read, "edit_file": edit, "lint": lint_ok})
    engine, _, _, _ = make(script, runtime=runtime)
    result = await engine.run("s", "tidy calc.py")
    assert result.outcome is Outcome.COMPLETED
    assert result.answer.startswith("Done.") and "calc.py" in result.answer
    assert result.answer == engine.last_answer


async def test_without_validates_flag_same_tool_is_unvalidated() -> None:
    script = [
        Turn(calls=[call("edit_file", "2", path="calc.py")]),
        Turn(calls=[call("lint", "3")]),
        Turn("Done."),
    ]
    engine, _, _, _ = make(script, {"edit_file": edit, "lint": lint_ok})
    result = await engine.run("s", "tidy calc.py")
    assert result.outcome is Outcome.UNVALIDATED


async def test_answer_without_any_successful_tool_is_flagged_unverified() -> None:
    engine, _, _, _ = make([Turn("It returns {'debug': False}.")], {"read_file": read})
    result = await engine.run("s", "what does parse_config return?")
    assert result.outcome is Outcome.ANSWERED
    assert "Evidence: none" in result.answer

