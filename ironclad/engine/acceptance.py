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
  INCOMPLETE. Otherwise it depends on the task intent (``ironclad.engine.intent``): CHANGE -> INCOMPLETE
  ("requested change was not made") unless there is NO_CHANGE_NEEDED evidence; QUESTION / UNKNOWN -> ANSWERED.
- NO_CHANGE_NEEDED needs all of: intent CHANGE; no change this run; an OK ``declare_no_change`` call; at least one
  successful read; and a trusted validation verdict "pass" recorded after every read, with no mutation or failed
  record after it. The declared reason is explanatory only and never evidence.
- QUESTION + a change: the change rules below apply unchanged and the reasons note "changes were made for a
  question".
- Change made: only validation records that started at or after the latest change count; a trusted validation's
  own revision bump does not invalidate it. The latest decisive one wins: pass -> COMPLETED, fail -> INCOMPLETE;
  none -> UNVALIDATED. Baseline runs before the latest change are stale.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

from ironclad.contracts import AcceptanceResult, Outcome, ToolResult, ToolSpec, ToolStatus
from ironclad.engine.intent import Intent
from ironclad.engine.state import ToolRecord

__all__ = [
    "DECLARE_NO_CHANGE",
    "DECLARE_NO_CHANGE_SPEC",
    "FLAGGABLE_TOOLS",
    "Intent",
    "StopReason",
    "ToolRecord",
    "changed_paths",
    "declare_no_change_result",
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

NOT_MADE = "requested change was not made"
QUESTION_CHANGED = "changes were made for a question"
DECLARE_NO_CHANGE = "declare_no_change"
MAX_REASON_CHARS = 1000
MAX_EVIDENCE_PATHS = 20
DECLARE_NO_CHANGE_SPEC = ToolSpec(
    name=DECLARE_NO_CHANGE,
    description=(
        "Declare that the requested change is already in place, so nothing needs editing. Only call this after "
        "read_file on every file in evidence_paths and a passing run_tests; otherwise it is not accepted."
    ),
    parameters={
        "type": "object",
        "properties": {
            "reason": {"type": "string", "minLength": 1, "maxLength": MAX_REASON_CHARS},
            "evidence_paths": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": MAX_EVIDENCE_PATHS,
                "description": "Files you read that show the request is already satisfied",
            },
        },
        "required": ["reason", "evidence_paths"],
        "additionalProperties": False,
    },
    risk=0,  # RiskLevel.READ; no side effects
    mutates=False,
)


def declare_no_change_result(args: Mapping[str, Any], call_id: str = "") -> ToolResult:
    """Pure handler: records the claim; acceptance decides from tool evidence whether it holds."""
    reason = args.get("reason") if isinstance(args, Mapping) else None
    paths = args.get("evidence_paths") if isinstance(args, Mapping) else None
    extra = set(args) - {"reason", "evidence_paths"} if isinstance(args, Mapping) else set()
    valid_paths = (
        isinstance(paths, list)
        and 0 < len(paths) <= MAX_EVIDENCE_PATHS
        and all(isinstance(path, str) and path.strip() for path in paths)
    )
    if not isinstance(reason, str) or not reason.strip() or not valid_paths or extra:
        text = (
            "declare_no_change needs 'reason' (non-empty string) and 'evidence_paths' "
            f"(1-{MAX_EVIDENCE_PATHS} files you read) and nothing else."
        )
        return ToolResult(call_id, DECLARE_NO_CHANGE, ToolStatus.INVALID, text)
    reason = reason.strip()[:MAX_REASON_CHARS]
    text = (
        "Recorded. 'No change needed' is accepted only if every evidence path was read this run, a passing test "
        "run followed those reads, and nothing was changed. It is reported to the user as a claim, not a success."
    )
    data = {"reason": reason, "evidence_paths": [path.strip() for path in paths]}  # type: ignore[union-attr]
    return ToolResult(call_id, DECLARE_NO_CHANGE, ToolStatus.OK, text, data=data)


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
    if record.result.data.get("no_tests") is True and _trusted(record):
        return "inconclusive"  # the runner collected nothing (e.g. unittest on pytest-style tests)
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


def _no_change_evidence(records: Sequence[ToolRecord]) -> tuple[ToolRecord | None, str]:
    """The passing validation that justifies NO_CHANGE_NEEDED, or ``None`` and why not."""
    declarations = [r for r in records if r.call.name == DECLARE_NO_CHANGE and r.result.status is ToolStatus.OK]
    if not declarations:
        return None, ""
    if any(_is_change(r) for r in records):
        return None, "declare_no_change ignored: the workspace changed this run"
    wanted = {_norm_path(path) for path in declarations[-1].result.data.get("evidence_paths", ())}
    read_at: dict[str, int] = {}
    for i, r in enumerate(records):
        if r.call.name == "read_file" and r.result.status is ToolStatus.OK:
            read_at[_norm_path(str(r.result.data.get("path") or r.call.arguments.get("path", "")))] = i
    missing = sorted(path for path in wanted if path not in read_at)
    if not wanted or missing:
        return None, f"declare_no_change ignored: evidence not read this run ({', '.join(missing) or 'none given'})"
    last_read = max(read_at[path] for path in wanted)
    passes = [i for i, r in enumerate(records) if validation_verdict(r) == "pass" and i > last_read]
    if not passes:
        return None, "declare_no_change ignored: no passing test run after the reads"
    index = passes[-1]
    later = records[index + 1:]
    if any(r.mutating or r.bumped or r.result.status is not ToolStatus.OK for r in later):
        return None, "declare_no_change ignored: a mutation or failure followed the passing test run"
    return records[index], ""


def _norm_path(path: str) -> str:
    text = path.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.casefold()



def _with_reason(result: AcceptanceResult, reason: str) -> AcceptanceResult:
    return replace(result, reasons=(*result.reasons, reason))


def evaluate(
    records: Sequence[ToolRecord],
    stop: StopReason = "final",
    detail: str = "",
    *,
    intent: Intent = Intent.UNKNOWN,
) -> AcceptanceResult:
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

    if not any(_is_change(r) and r.result.status is ToolStatus.OK for r in records):
        failed = [r for r in records if r.mutating and not is_validation(r) and r.result.status is not ToolStatus.OK]
        if failed and all(r.result.status in _REJECTED for r in failed):
            reason = f"tool calls were rejected and nothing was changed ({_names(failed)})"
            return AcceptanceResult(Outcome.INCOMPLETE, (reason,), changed)
        if failed:
            return AcceptanceResult(Outcome.INCOMPLETE, (f"requested change failed ({_names(failed)})",), changed)
        if intent is Intent.CHANGE:
            return _unchanged_for_change(records, changed)
        if not unreported_changes(records):
            return AcceptanceResult(Outcome.ANSWERED, ("no workspace changes",))

    result = _change_outcome(records, changed)
    return _with_reason(result, QUESTION_CHANGED) if intent is Intent.QUESTION else result


def _unchanged_for_change(records: Sequence[ToolRecord], changed: tuple[str, ...]) -> AcceptanceResult:
    evidence, why_not = _no_change_evidence(records)
    if evidence is not None:
        reason = f"no change needed; tests passed after reading the code: {_describe(evidence)}"
        return AcceptanceResult(Outcome.NO_CHANGE_NEEDED, (reason,), changed, evidence.revision_after)
    return AcceptanceResult(Outcome.INCOMPLETE, tuple(r for r in (NOT_MADE, why_not) if r), changed)


def _change_outcome(records: Sequence[ToolRecord], changed: tuple[str, ...]) -> AcceptanceResult:
    unreported = unreported_changes(records)
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
    if not result.passed or result.outcome is Outcome.NO_CHANGE_NEEDED or QUESTION_CHANGED in result.reasons:
        lines.extend(f"Reason: {reason}" for reason in result.reasons)
    return "\n".join(lines)
