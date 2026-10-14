"""Outcome decisions from tool records — never from model prose.

Rules (``evaluate``):
- cancelled -> CANCELLED; transport failure -> FAILED; budget / repeated-invalid / truncated model -> INCOMPLETE.
- A reported change is an OK record with ``mutated_paths``. An *unreported* change is any other record that bumped
  the workspace revision (e.g. ``run_command``, which cannot say what it touched), unless it is a trusted
  validation. FAILED/INVALID/DENIED mutations never count as reported changes.
- Trusted validations: ``run_tests`` (and other ``validates`` tools), or a model-flagged ``run_command`` whose
  record says the resolved program is a known test runner (``data["runner"]``, set by the executor). A flagged
  command without it is inconclusive. Exit 0 alone never passes: a pass needs ``tests_run > 0`` and no failures.
- No successful change: if a mutating call failed or was rejected the requested change did not happen ->
  INCOMPLETE, otherwise ANSWERED.
- Change made: only validation records that started at or after the latest change count; a trusted validation's
  own revision bump does not invalidate it. The latest decisive one wins: pass -> COMPLETED, fail -> INCOMPLETE;
  none -> UNVALIDATED. Baseline runs before the latest change are stale.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ironclad.contracts import AcceptanceResult, Outcome, ToolStatus
from ironclad.engine.state import ToolRecord

__all__ = [
    "FLAGGABLE_TOOLS",
    "StopReason",
    "ToolRecord",
    "changed_paths",
    "evaluate",
    "is_validation",
    "obligations",
    "render_summary",
    "unreported_changes",
    "validation_verdict",
]

StopReason = Literal["final", "budget", "repeated_invalid", "model_error", "cancelled", "transport_error"]
Verdict = Literal["pass", "fail", "inconclusive"]

FLAGGABLE_TOOLS = frozenset({"run_command"})  # the model may flag these as validation; trusted only with a runner
_NOT_RUN = {ToolStatus.INVALID, ToolStatus.DENIED, ToolStatus.CANCELLED}
_REJECTED = {ToolStatus.INVALID, ToolStatus.DENIED}


def is_validation(record: ToolRecord) -> bool:
    return record.call.name == "run_tests" or record.validation


def _trusted(record: ToolRecord) -> bool:
    """A validation whose program is a known test runner (so its own side effects are not a 'change')."""
    if not is_validation(record):
        return False
    return record.call.name not in FLAGGABLE_TOOLS or bool(record.result.data.get("runner"))


def _tests_run(record: ToolRecord) -> int | None:
    value = record.result.data.get("tests_run")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def validation_verdict(record: ToolRecord) -> Verdict | None:
    """``None`` when the record is not a validation that actually ran."""
    if not is_validation(record) or record.result.status in _NOT_RUN:
        return None
    if record.result.status is not ToolStatus.OK:
        return "fail"
    if not _trusted(record):
        return "inconclusive"
    count = _tests_run(record)
    failed = record.result.data.get("tests_failed")
    if count is None or count <= 0:
        return "inconclusive"
    return "fail" if isinstance(failed, int) and not isinstance(failed, bool) and failed > 0 else "pass"


def changed_paths(records: Sequence[ToolRecord]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for record in records:
        if record.result.status is ToolStatus.OK:
            for path in record.result.mutated_paths:
                seen.setdefault(path, None)
    return tuple(seen)


def _reported(record: ToolRecord) -> bool:
    return record.result.status is ToolStatus.OK and bool(record.result.mutated_paths)


def _is_change(record: ToolRecord) -> bool:
    """The record changed (or may have changed) the workspace and is not a trusted validation."""
    if _reported(record):
        return True
    return record.bumped and record.result.status not in _REJECTED and not _trusted(record)


def unreported_changes(records: Sequence[ToolRecord]) -> list[ToolRecord]:
    return [r for r in records if _is_change(r) and not _reported(r)]


def _latest_change(records: Sequence[ToolRecord]) -> int | None:
    bumps = [r.revision_after.counter for r in records if _is_change(r)]
    return max(bumps) if bumps else None


def _fresh_validations(records: Sequence[ToolRecord]) -> list[tuple[ToolRecord, Verdict]]:
    latest = _latest_change(records)
    fresh: list[tuple[ToolRecord, Verdict]] = []
    for record in records:
        verdict = validation_verdict(record)
        if verdict is None:
            continue
        if latest is not None and record.revision_before.counter < latest:
            continue
        fresh.append((record, verdict))
    return fresh


def _describe(record: ToolRecord) -> str:
    data = record.result.data
    bits = [record.call.name, record.result.status.value]
    count = _tests_run(record)
    if record.call.name == "run_tests" or count is not None:
        bits.append(f"{count} tests" if count is not None else "test count unknown")
    if isinstance(data.get("exit_code"), int):
        bits.append(f"exit {data['exit_code']}")
    if record.call.name in FLAGGABLE_TOOLS and is_validation(record) and not _trusted(record):
        bits.append("(not a known test runner)")
    return " ".join(bits)


def _names(records: Sequence[ToolRecord]) -> str:
    return ", ".join(sorted({f"{r.call.name} {r.result.status.value}" for r in records}))


def evaluate(records: Sequence[ToolRecord], stop: StopReason = "final", detail: str = "") -> AcceptanceResult:
    changed = changed_paths(records)
    if stop == "cancelled":
        return AcceptanceResult(Outcome.CANCELLED, ("cancelled by user",), changed)
    if stop == "transport_error":
        return AcceptanceResult(Outcome.FAILED, (detail or "model transport failed",), changed)
    if stop != "final":
        reason = detail or {"budget": "budget exhausted", "repeated_invalid": "repeated invalid tool call"}.get(
            stop, "model stopped before finishing"
        )
        return AcceptanceResult(Outcome.INCOMPLETE, (reason,), changed)

    unreported = unreported_changes(records)
    if not any(_is_change(r) and r.result.status is ToolStatus.OK for r in records):
        failed = [r for r in records if r.mutating and not is_validation(r) and r.result.status is not ToolStatus.OK]
        if failed and all(r.result.status in _REJECTED for r in failed):
            reason = f"tool calls were rejected and nothing was changed ({_names(failed)})"
            return AcceptanceResult(Outcome.INCOMPLETE, (reason,), changed)
        if failed:
            return AcceptanceResult(Outcome.INCOMPLETE, (f"requested change failed ({_names(failed)})",), changed)
        if not unreported:
            return AcceptanceResult(Outcome.ANSWERED, ("no workspace changes",))

    fresh = _fresh_validations(records)
    decisive = [(r, v) for r, v in fresh if v != "inconclusive"]
    if decisive:
        record, verdict = decisive[-1]
        if verdict == "pass":
            return AcceptanceResult(
                Outcome.COMPLETED, (f"validated after latest change: {_describe(record)}",), changed,
                record.revision_after,
            )
        return AcceptanceResult(Outcome.INCOMPLETE, (f"validation failing after latest change: {_describe(record)}",), changed)
    reasons = ["no passing validation after the latest change"]
    if unreported:
        reasons.append(f"unreported changes by {_names(unreported)}")
    reasons += [f"inconclusive: {_describe(r)}" for r, _ in fresh]
    fresh_ids = {id(r) for r, _ in fresh}
    stale = [r for r in records if validation_verdict(r) is not None and id(r) not in fresh_ids]
    if stale:
        reasons.append(f"stale (before latest change): {_describe(stale[-1])}")
    return AcceptanceResult(Outcome.UNVALIDATED, tuple(reasons), changed)


def obligations(records: Sequence[ToolRecord], result: AcceptanceResult) -> list[str]:
    if result.outcome is Outcome.UNVALIDATED:
        return ["run validation (run_tests) after the latest change"]
    if result.outcome is Outcome.INCOMPLETE:
        return [f"resolve: {reason}" for reason in result.reasons]
    return []


def validation_status(records: Sequence[ToolRecord]) -> str:
    latest = [r for r in records if validation_verdict(r) is not None]
    if not latest:
        return "not run"
    record = latest[-1]
    verdict = validation_verdict(record)
    return f"{verdict}: {_describe(record)} (revision {record.revision_before.counter})"


def render_summary(records: Sequence[ToolRecord], result: AcceptanceResult) -> str:
    """Code-generated facts appended to the model's prose."""
    lines = ["---", f"Outcome: {result.outcome.value}"]
    lines.append("Changed files: " + (", ".join(result.changed_paths) if result.changed_paths else "none"))
    unreported = unreported_changes(records)
    if unreported:
        lines.append(f"Unreported changes: {_names(unreported)} (paths unknown)")
    validations = [r for r in records if validation_verdict(r) is not None]
    if validations:
        lines.append(f"Last validation: {_describe(validations[-1])}")
    elif result.changed_paths or unreported:
        lines.append("Last validation: none")
    if not any(r.result.status is ToolStatus.OK for r in records):
        lines.append("Evidence: none - no tool call succeeded, so this answer is not verified against the project")
    if not result.passed:
        lines.extend(f"Reason: {reason}" for reason in result.reasons)
    return "\n".join(lines)
