from __future__ import annotations

from typing import Any

from ironclad.contracts import Outcome, ToolCall, ToolResult, ToolStatus, WorkspaceRevision
from ironclad.engine.acceptance import evaluate, render_summary, validation_verdict
from ironclad.engine.state import ToolRecord

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
    result = evaluate([rec("read_file"), rec("search")])
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


def test_flagged_run_command_counts_as_validation() -> None:
    records = [rec("edit_file", paths=("a.py",)), rec("run_command", validation=True, exit_code=0)]
    assert evaluate(records).outcome is Outcome.COMPLETED


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
