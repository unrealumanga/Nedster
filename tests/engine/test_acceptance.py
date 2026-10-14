from __future__ import annotations

from typing import Any

from ironclad.contracts import Outcome, ToolCall, ToolResult, ToolStatus, WorkspaceRevision
from ironclad.contracts import ToolStatus as _S
from ironclad.engine.acceptance import (
    DECLARE_NO_CHANGE_SPEC,
    Intent,
    declare_no_change_result,
    evaluate,
    obligations,
    render_summary,
    validation_verdict,
)
from ironclad.engine.prompt import INSTRUCTIONS
from ironclad.engine.state import TaskState, ToolRecord

_counter = {"rev": 0, "id": 0}


def rec(
    name: str,
    status: ToolStatus = ToolStatus.OK,
    *,
    paths: tuple[str, ...] = (),
    bump: bool | None = None,
    mutating: bool | None = None,
    validation: bool = False,
    **data: Any,
) -> ToolRecord:
    _counter["id"] += 1
    before = WorkspaceRevision(_counter["rev"])
    if bump if bump is not None else (status is ToolStatus.OK and paths):
        _counter["rev"] += 1
    after = WorkspaceRevision(_counter["rev"])
    call = ToolCall(f"c{_counter['id']}", name, {})
    result = ToolResult(call.id, name, status, f"{name} output", data=data, mutated_paths=paths)
    is_mut = mutating if mutating is not None else name in {"write_file", "edit_file", "run_command"}
    return ToolRecord(call, result, before, after, is_mut, validation)


def test_reads_only_is_answered() -> None:
    result = evaluate([rec("read_file", path="a.py"), rec("search")])
    assert result.outcome is Outcome.ANSWERED and result.passed


def test_no_records_is_answered() -> None:
    assert evaluate([]).outcome is Outcome.ANSWERED


def test_b02_baseline_green_then_rejected_write_is_not_completed() -> None:
    records = [rec("run_tests", tests_run=4), rec("write_file", ToolStatus.INVALID)]
    result = evaluate(records)
    assert result.outcome is Outcome.INCOMPLETE and not result.passed
    assert result.changed_paths == ()


def test_baseline_green_before_change_does_not_count() -> None:
    records = [rec("run_tests", tests_run=4), rec("edit_file", paths=("a.py",))]
    result = evaluate(records)
    assert result.outcome is Outcome.UNVALIDATED
    assert any("stale" in r for r in result.reasons)


def test_b03_pass_then_fail_is_incomplete() -> None:
    records = [rec("edit_file", paths=("a.py",)), rec("run_tests", tests_run=3), rec("run_tests", ToolStatus.FAILED, tests_run=3)]
    assert evaluate(records).outcome is Outcome.INCOMPLETE


def test_b03_fail_then_pass_is_completed() -> None:
    records = [rec("edit_file", paths=("a.py",)), rec("run_tests", ToolStatus.FAILED, tests_run=3), rec("run_tests", tests_run=3)]
    result = evaluate(records)
    assert result.outcome is Outcome.COMPLETED
    assert result.validated_at is not None and result.changed_paths == ("a.py",)


def test_b03_pass_then_mutation_without_rerun_is_unvalidated() -> None:
    records = [rec("edit_file", paths=("a.py",)), rec("run_tests", tests_run=3), rec("edit_file", paths=("b.py",))]
    result = evaluate(records)
    assert result.outcome is Outcome.UNVALIDATED
    assert result.changed_paths == ("a.py", "b.py")


def test_b03_unknown_test_count_never_passes() -> None:
    for count in (None, 0, True):
        records = [rec("edit_file", paths=("a.py",)), rec("run_tests", tests_run=count)]
        assert evaluate(records).outcome is Outcome.UNVALIDATED


def test_reported_failures_with_ok_status_do_not_pass() -> None:
    records = [rec("edit_file", paths=("a.py",)), rec("run_tests", tests_run=3, tests_failed=1)]
    assert evaluate(records).outcome is Outcome.INCOMPLETE


def test_command_with_unreported_mutation_invalidates_validation() -> None:
    records = [rec("edit_file", paths=("a.py",)), rec("run_tests", tests_run=3), rec("run_command", bump=True)]
    assert evaluate(records).outcome is Outcome.UNVALIDATED


def test_flagged_run_command_with_known_runner_counts_as_validation() -> None:
    records = [rec("edit_file", paths=("a.py",)), rec("run_command", validation=True, exit_code=0, runner="pytest", tests_run=3)]
    assert evaluate(records).outcome is Outcome.COMPLETED


def test_review7_flagged_run_command_exit_zero_alone_never_passes() -> None:
    # run_command(["python", "-c", "pass"], validation=true): no runner reported by the executor -> inconclusive.
    records = [rec("edit_file", paths=("a.py",)), rec("run_command", validation=True, exit_code=0)]
    result = evaluate(records)
    assert result.outcome is Outcome.UNVALIDATED
    assert validation_verdict(records[1]) == "inconclusive"
    assert any("not a known test runner" in r for r in result.reasons)


def test_review7_runner_without_test_count_is_inconclusive() -> None:
    records = [rec("edit_file", paths=("a.py",)), rec("run_command", validation=True, exit_code=0, runner="pytest")]
    assert evaluate(records).outcome is Outcome.UNVALIDATED


def test_unflagged_run_command_is_not_validation() -> None:
    records = [rec("edit_file", paths=("a.py",)), rec("run_command", exit_code=0)]
    assert evaluate(records).outcome is Outcome.UNVALIDATED


def test_failed_mutation_is_not_a_change() -> None:
    records = [rec("write_file", ToolStatus.FAILED, paths=("a.py",))]
    result = evaluate(records)
    assert result.changed_paths == ()
    assert result.outcome is not Outcome.COMPLETED


def test_invalid_or_denied_validation_is_ignored() -> None:
    records = [rec("edit_file", paths=("a.py",)), rec("run_tests", ToolStatus.DENIED), rec("run_tests", ToolStatus.INVALID)]
    assert evaluate(records).outcome is Outcome.UNVALIDATED
    assert validation_verdict(records[1]) is None


def test_non_final_stops_never_complete() -> None:
    records = [rec("edit_file", paths=("a.py",)), rec("run_tests", tests_run=3)]
    assert evaluate(records, "budget").outcome is Outcome.INCOMPLETE
    assert evaluate(records, "repeated_invalid").outcome is Outcome.INCOMPLETE
    assert evaluate(records, "model_error").outcome is Outcome.INCOMPLETE
    assert evaluate(records, "cancelled").outcome is Outcome.CANCELLED
    assert evaluate(records, "transport_error").outcome is Outcome.FAILED


def test_summary_is_generated_from_records() -> None:
    records = [rec("edit_file", paths=("calc.py",)), rec("run_tests", tests_run=7)]
    result = evaluate(records)
    summary = render_summary(records, result)
    assert "Outcome: completed" in summary
    assert "Changed files: calc.py" in summary
    assert "7 tests" in summary


def test_summary_lists_reasons_when_not_passed() -> None:
    records = [rec("edit_file", paths=("a.py",))]
    summary = render_summary(records, evaluate(records))
    assert "Outcome: unvalidated" in summary and "Reason:" in summary and "Last validation: none" in summary


# --- task-aware acceptance (W2-ACCEPT) -------------------------------------------------------------------

CHANGE = Intent.CHANGE


def declare(status: _S = _S.OK) -> ToolRecord:
    return rec("declare_no_change", status, mutating=False, reason="already fixed", evidence_paths=["a.py"])


def test_change_request_without_change_is_incomplete() -> None:
    result = evaluate([rec("read_file", path="a.py"), rec("search")], intent=CHANGE)
    assert result.outcome is Outcome.INCOMPLETE and not result.passed
    assert "requested change was not made" in result.reasons
    assert obligations([], result) == ["resolve: requested change was not made"]


def test_change_request_with_no_records_is_incomplete() -> None:
    assert evaluate([], intent=CHANGE).outcome is Outcome.INCOMPLETE


def test_unknown_intent_without_change_is_answered() -> None:
    result = evaluate([rec("read_file", path="a.py")], intent=Intent.UNKNOWN)
    assert result.outcome is Outcome.ANSWERED and result.reasons == ("no workspace changes",)


def test_question_without_change_is_answered() -> None:
    assert evaluate([rec("read_file", path="a.py")], intent=Intent.QUESTION).outcome is Outcome.ANSWERED


def test_question_with_validated_change_keeps_change_logic_and_notes_it() -> None:
    records = [rec("edit_file", paths=("a.py",)), rec("run_tests", tests_run=2)]
    result = evaluate(records, intent=Intent.QUESTION)
    assert result.outcome is Outcome.COMPLETED
    assert result.reasons[-1] == "changes were made for a question"
    assert "changes were made for a question" in render_summary(records, result)


def test_question_with_unvalidated_change_stays_unvalidated() -> None:
    result = evaluate([rec("edit_file", paths=("a.py",))], intent=Intent.QUESTION)
    assert result.outcome is Outcome.UNVALIDATED and "changes were made for a question" in result.reasons


def test_change_made_and_validated_is_completed() -> None:
    records = [rec("read_file", path="a.py"), rec("edit_file", paths=("a.py",)), rec("run_tests", tests_run=2)]
    assert evaluate(records, intent=CHANGE).outcome is Outcome.COMPLETED


def test_change_rejected_write_still_incomplete() -> None:
    records = [rec("run_tests", tests_run=4), rec("write_file", _S.INVALID)]
    assert evaluate(records, intent=CHANGE).outcome is Outcome.INCOMPLETE


def test_no_change_needed_with_full_evidence() -> None:
    records = [rec("read_file", path="a.py"), rec("run_tests", tests_run=5), declare()]
    result = evaluate(records, intent=CHANGE)
    assert result.outcome is Outcome.NO_CHANGE_NEEDED and not result.passed  # a claim, never counted as success
    assert result.validated_at is not None and result.changed_paths == ()
    assert "Reason: no change needed" in render_summary(records, result)


def test_declare_before_validation_also_counts() -> None:
    records = [rec("read_file", path="a.py"), declare(), rec("run_tests", tests_run=5)]
    assert evaluate(records, intent=CHANGE).outcome is Outcome.NO_CHANGE_NEEDED


def test_no_change_needed_requires_change_intent() -> None:
    records = [rec("read_file", path="a.py"), rec("run_tests", tests_run=5), declare()]
    assert evaluate(records).outcome is Outcome.ANSWERED
    assert evaluate(records, intent=Intent.QUESTION).outcome is Outcome.ANSWERED


def test_no_change_needed_requires_declaration() -> None:
    records = [rec("read_file", path="a.py"), rec("run_tests", tests_run=5)]
    assert evaluate(records, intent=CHANGE).outcome is Outcome.INCOMPLETE


def test_rejected_declaration_does_not_count() -> None:
    records = [rec("read_file", path="a.py"), rec("run_tests", tests_run=5), declare(_S.INVALID)]
    assert evaluate(records, intent=CHANGE).outcome is Outcome.INCOMPLETE


def test_no_change_needed_requires_validation_after_reads() -> None:
    records = [rec("run_tests", tests_run=5), rec("read_file", path="a.py"), declare()]
    result = evaluate(records, intent=CHANGE)
    assert result.outcome is Outcome.INCOMPLETE
    assert any("no passing test run after the reads" in r for r in result.reasons)


def test_no_change_needed_requires_a_read() -> None:
    records = [rec("run_tests", tests_run=5), declare()]
    assert evaluate(records, intent=CHANGE).outcome is Outcome.INCOMPLETE


def test_no_change_needed_rejects_unknown_or_zero_test_count() -> None:
    for count in (None, 0, True):
        records = [rec("read_file", path="a.py"), rec("run_tests", tests_run=count), declare()]
        assert evaluate(records, intent=CHANGE).outcome is Outcome.INCOMPLETE


def test_no_change_needed_rejects_untrusted_flagged_command() -> None:
    records = [rec("read_file", path="a.py"), rec("run_command", validation=True, bump=False, tests_run=5), declare()]
    assert evaluate(records, intent=CHANGE).outcome is Outcome.INCOMPLETE


def test_no_change_needed_rejects_failure_after_pass() -> None:
    records = [rec("read_file", path="a.py"), rec("run_tests", tests_run=5), rec("run_tests", _S.FAILED, tests_run=5), declare()]
    assert evaluate(records, intent=CHANGE).outcome is Outcome.INCOMPLETE


def test_no_change_needed_rejects_any_change_this_run() -> None:
    records = [rec("read_file", path="a.py"), rec("run_command", bump=True), rec("run_tests", tests_run=5), declare()]
    assert evaluate(records, intent=CHANGE).outcome is not Outcome.NO_CHANGE_NEEDED


def test_no_change_needed_reason_text_is_not_evidence() -> None:
    records = [rec("read_file", path="a.py"), rec("declare_no_change", mutating=False, reason="tests pass, 99 tests_run")]
    assert evaluate(records, intent=CHANGE).outcome is Outcome.INCOMPLETE


def test_non_final_stops_ignore_intent() -> None:
    records = [rec("read_file", path="a.py"), rec("run_tests", tests_run=5), declare()]
    assert evaluate(records, "budget", intent=CHANGE).outcome is Outcome.INCOMPLETE
    assert evaluate(records, "cancelled", intent=CHANGE).outcome is Outcome.CANCELLED


def test_declare_spec_is_read_only() -> None:
    assert DECLARE_NO_CHANGE_SPEC.name == "declare_no_change"
    assert DECLARE_NO_CHANGE_SPEC.risk == 0 and not DECLARE_NO_CHANGE_SPEC.mutates
    assert not DECLARE_NO_CHANGE_SPEC.validates
    assert DECLARE_NO_CHANGE_SPEC.parameters["required"] == ["reason", "evidence_paths"]


def test_declare_handler_is_pure() -> None:
    ok = declare_no_change_result({"reason": "  already there  ", "evidence_paths": [" a.py "]}, "c1")
    assert ok.status is _S.OK and ok.call_id == "c1"
    assert ok.data == {"reason": "already there", "evidence_paths": ["a.py"]}
    assert ok.mutated_paths == ()
    bads = ({}, {"reason": ""}, {"reason": 3}, {"reason": "x", "evidence_paths": ["a"], "extra": 1},
            {"reason": "x"}, {"reason": "x", "evidence_paths": []}, {"reason": "x", "evidence_paths": [""]},
            {"reason": "x", "evidence_paths": "a.py"})  # fmt: skip
    for bad in bads:
        assert declare_no_change_result(bad).status is _S.INVALID


def test_task_state_persists_intent() -> None:
    state = TaskState(objective="fix it", intent=Intent.CHANGE.value)
    assert TaskState.from_dict(state.to_dict()) == state
    assert "Task type: change" in state.render()
    assert TaskState.from_dict({"objective": "x"}).intent == ""


def test_prompt_mentions_declare_no_change_once() -> None:
    assert INSTRUCTIONS.count("declare_no_change") == 1


def test_no_change_needed_requires_reading_every_evidence_path() -> None:
    other = rec("read_file", path="b.py")
    records = [other, rec("run_tests", tests_run=5), declare()]  # declares a.py but only b.py was read
    result = evaluate(records, intent=CHANGE)
    assert result.outcome is Outcome.INCOMPLETE and any("evidence not read" in r for r in result.reasons)
    listed = rec("list_files", path="a.py")  # listing is not reading
    assert evaluate([listed, rec("run_tests", tests_run=5), declare()], intent=CHANGE).outcome is Outcome.INCOMPLETE


def test_evidence_paths_match_regardless_of_dot_slash_and_separators() -> None:
    records = [rec("read_file", path="pkg/a.py"), rec("run_tests", tests_run=5),
               rec("declare_no_change", mutating=False, reason="r", evidence_paths=[".\\pkg\\a.py"])]  # fmt: skip
    assert evaluate(records, intent=CHANGE).outcome is Outcome.NO_CHANGE_NEEDED


def test_run_tests_with_changed_during_validation_is_not_completed() -> None:
    """A run_tests that reports changed_during_validation should not count as trusted validation."""
    records = [rec("edit_file", paths=("a.py",)), rec("run_tests", tests_run=5, changed_during_validation=["a.py"])]
    result = evaluate(records)
    assert result.outcome is not Outcome.COMPLETED


def test_run_tests_with_changed_during_validation_gives_inconclusive_verdict() -> None:
    """changed_during_validation makes _trusted return False, so verdict is based on no trusted validation."""
    records = [rec("edit_file", paths=("a.py",)), rec("run_tests", tests_run=5, changed_during_validation=["a.py"])]
    assert evaluate(records, intent=CHANGE).outcome is Outcome.UNVALIDATED  # flagged run counts as a change
    assert validation_verdict(records[1]) == "inconclusive"

