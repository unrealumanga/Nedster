"""Gate 6: a 60-step scripted run with repeated compactions, a mid-run restart and a hard window bound."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ironclad.contracts import (
    ChatEvent,
    ChatMessage,
    Outcome,
    TextDelta,
    ToolCall,
    ToolCallsReady,
    ToolResult,
    ToolSpec,
    ToolStatus,
    TurnFinished,
    WorkspaceRevision,
)
from ironclad.engine.context import (
    SUMMARY_PREFIX,
    ContextManager,
    OutputVault,
    completion_reserve,
    estimate_prompt,
    serialize,
)
from ironclad.engine.loop import Engine
from ironclad.engine.state import Budget, ToolRecord

from .fakes import Events, FakeRuntime, MemoryStore, ScriptedTransport, Turn, call, ok


def est(text: str) -> int:
    return len(text) // 3  # deliberately different (more pessimistic) than the default estimator


class Crash(BaseException):
    """Simulates the process dying while a tool runs (not an Exception, so the engine cannot swallow it)."""


class CheckedTransport(ScriptedTransport):
    """Scripted turns (with thinking); asserts the window and pairing on EVERY request."""

    def __init__(self, script: Any, budget: int, reserve: int) -> None:
        super().__init__(script)
        self.budget, self.reserve = budget, reserve
        self.sizes: list[int] = []
        self.system_prompts: list[str] = []

    async def chat(self, *, model: str, messages: list[ChatMessage], tools: list[ToolSpec], **_: Any) -> AsyncIterator[ChatEvent]:
        size = estimate_prompt(messages, tools, est)
        self.sizes.append(size)
        assert size + self.reserve <= self.budget, f"request {len(self.requests)} is {size}+{self.reserve} > {self.budget}"
        self._check_pairing(messages)
        self.system_prompts.append(messages[0].content)
        self.requests.append(serialize(messages))
        self.tool_names.append([t.name for t in tools])
        step = self.script.pop(0)
        turn = step(messages) if callable(step) else step
        thinking = getattr(turn, "thinking", "")
        if turn.text:
            yield TextDelta(turn.text)
        if turn.calls:
            yield ToolCallsReady(tuple(turn.calls))
        message = ChatMessage(role="assistant", content=turn.text, tool_calls=list(turn.calls), thinking=thinking)
        yield TurnFinished(message, turn.usage, turn.done_reason)

    @staticmethod
    def _check_pairing(messages: list[ChatMessage]) -> None:
        pending: list[str] = []
        seen: set[str] = set()
        for m in messages:
            if m.role == "tool":
                assert m.tool_call_id in pending, f"orphan tool message {m.tool_call_id}"
                pending.remove(m.tool_call_id or "")
            else:
                assert not pending, f"calls {pending} unanswered before a {m.role} message"
                if m.role == "assistant":
                    pending.extend(c.id for c in m.tool_calls)
                    seen.update(c.id for c in m.tool_calls)
        assert not pending, f"request ends with unanswered calls {pending}"


@dataclass(slots=True)
class ThinkTurn(Turn):
    thinking: str = ""


def turn(text: str = "", calls: list[ToolCall] | None = None, thinking: str = "") -> ThinkTurn:
    return ThinkTurn(text, calls or (), thinking=thinking)


BIG = 2_800
OBJECTIVE = "Refactor the parser module and keep behaviour. " + "Detail sentence for the objective. " * 20  # > 400 chars


def handlers() -> dict[str, Any]:
    def read_file(c: ToolCall) -> ToolResult:
        path = str(c.arguments.get("path"))
        if path.startswith("missing"):
            return ToolResult(c.id, c.name, ToolStatus.FAILED, f"no such file {path}")
        return ok(c, f"file {path} line 1\n" + "x" * BIG, sha256=f"sha-{path}")

    def write_file(c: ToolCall) -> ToolResult:
        path = str(c.arguments.get("path"))
        return ToolResult(c.id, c.name, ToolStatus.OK, f"wrote {path}", data={"sha256": f"new-{path}"}, mutated_paths=(path,))

    def run_tests(c: ToolCall) -> ToolResult:
        failing = bool(c.arguments.get("fail"))
        status = ToolStatus.FAILED if failing else ToolStatus.OK
        return ToolResult(c.id, c.name, status, ("FAILED test_x\n" if failing else "all passed\n") + "y" * 1500,
                          data={"tests_run": 4, "tests_failed": 1 if failing else 0})

    def crash(c: ToolCall) -> ToolResult:
        raise Crash

    return {"read_file": read_file, "write_file": write_file, "run_tests": run_tests, "crash_tool": crash}


def step(i: int) -> Any:
    think = "thinking " * (150 if i % 5 == 0 else 5)
    text = "narration " * (120 if i % 6 == 0 else 3)
    if i == 22:
        return turn(text, [call("show_output", f"c{i}", handle=1)], think)
    if i == 30:
        return turn(text, [call("crash_tool", f"c{i}")], think)
    if i == 45:
        return turn(text, [call("show_output", f"c{i}", handle=2)], think)
    if i % 7 == 0:
        return turn(text, [call("write_file", f"c{i}", path=f"w{i}.py", content="w" * 1500)], think)
    if i % 11 == 3:
        return turn(text, [call("run_tests", f"c{i}", fail=i < 50)], think)
    if i % 9 == 4:
        return turn(text, [call("read_file", f"c{i}", path=f"missing{i}.py")], think)
    return turn(text, [call("read_file", f"c{i}", path=f"f{i}.py"), call("read_file", f"d{i}", path=f"g{i}.py")], think)


def make_runtime() -> FakeRuntime:
    return FakeRuntime(handlers(), mutating=("write_file", "crash_tool"))


async def test_sixty_step_run_with_restart_respects_window_everywhere() -> None:
    budget = Budget(context_tokens=6_000, max_steps=200, max_tool_calls=400, max_output_chars=3_000)
    reserve = completion_reserve(budget.context_tokens)
    store, events = MemoryStore(), Events()
    transport = CheckedTransport([step(i) for i in range(60)] + [turn("done")], budget.context_tokens, reserve)
    runtime = make_runtime()

    def new_engine() -> Engine:
        return Engine(transport, runtime, model="m", store=store, budget=budget, emit=events.append, estimator=est)

    crashed = False
    try:
        await new_engine().run("s", OBJECTIVE)
    except Crash:
        crashed = True
    assert crashed and len(transport.requests) == 31
    before_restart = len(events.of("context.compacted"))
    assert before_restart >= 2

    engine = new_engine()  # restart from the store
    result = await engine.run("s", "Also: do not touch tests/")
    assert result.outcome in (Outcome.UNVALIDATED, Outcome.COMPLETED, Outcome.ANSWERED, Outcome.INCOMPLETE)
    assert len(transport.requests) == 61 and transport.script == []

    compactions = events.of("context.compacted")
    assert [e.data["compaction"] for e in compactions] == list(range(1, len(compactions) + 1))  # counter survives restart
    assert len(compactions) > before_restart
    assert sum(1 for e in compactions if e.data["stage"] == "groups") >= 3
    assert all(e.data["tokens_after"] <= e.data["tokens_before"] for e in compactions)

    # the system prompt and tool list never change; objective + constraint stay verbatim in the last request
    assert len(set(transport.system_prompts)) == 1
    assert len({tuple(n) for n in transport.tool_names}) == 1
    last = transport.requests[-1]
    users = [m["content"] for m in last if m["role"] == "user"]
    assert OBJECTIVE in users and "Also: do not touch tests/" in users
    assert any(u.startswith(SUMMARY_PREFIX) for u in users)

    # output-handle retrieval worked before and after the restart (call ids c22 / c45)
    for cid, handle in (("c22", 1), ("c45", 2)):
        first = next(m["content"] for r in transport.requests for m in r if m["tool_call_id"] == cid)
        assert first.startswith("[ok] ") and first[5:].startswith(store.outputs[("s", handle)].splitlines()[0])
        assert handle in {h for e in compactions for h in e.data["handles"]}


async def test_prefix_is_byte_identical_between_compactions() -> None:
    budget = Budget(context_tokens=6_000, max_steps=200, max_output_chars=3_000)
    events = Events()
    marks: list[int] = []
    transport = CheckedTransport([], budget.context_tokens, completion_reserve(budget.context_tokens))

    def emit(event: Any) -> None:
        events.append(event)
        if event.kind == "context.compacted":
            marks.append(len(transport.requests))

    transport.script = [step(i) for i in range(1, 29)] + [turn("done")]
    engine = Engine(transport, make_runtime(), model="m", budget=budget, emit=emit, estimator=est)
    await engine.run("s", OBJECTIVE)
    assert marks
    for i in range(len(transport.requests) - 1):
        if i + 1 in marks:
            continue
        prev, nxt = transport.requests[i], transport.requests[i + 1]
        assert nxt[: len(prev)] == prev, f"prefix changed at request {i + 1}"
    # across a compaction the system prompt (first message) is still identical
    assert all(r[0] == transport.requests[0][0] for r in transport.requests)


def _record(c: ToolCall, result: ToolResult, before: int, after: int, validation: bool = False) -> ToolRecord:
    return ToolRecord(c, result, WorkspaceRevision(before), WorkspaceRevision(after), False, validation)


def _history() -> tuple[list[ChatMessage], list[ToolRecord]]:
    msgs = [ChatMessage(role="system", content="sys"), ChatMessage(role="system", content="pinned"),
            ChatMessage(role="user", content=OBJECTIVE)]
    records: list[ToolRecord] = []
    plan = [
        (call("read_file", "a", path="p.py"), ToolResult("a", "read_file", ToolStatus.OK, "head\n" + "z" * 800, data={"sha256": "h1"}), 0, 0),
        (call("write_file", "b", path="p.py"), ToolResult("b", "write_file", ToolStatus.OK, "wrote", data={"sha256": "h2"}, mutated_paths=("p.py",)), 0, 1),
        (call("run_tests", "c"), ToolResult("c", "run_tests", ToolStatus.FAILED, "FAILED t\n" + "z" * 800, data={"tests_run": 2, "tests_failed": 1}), 1, 1),
        (call("write_file", "d", path="q.py"), ToolResult("d", "write_file", ToolStatus.OK, "wrote", data={"sha256": "h3"}, mutated_paths=("q.py",)), 1, 2),
        (call("read_file", "e", path="gone.py"), ToolResult("e", "read_file", ToolStatus.FAILED, "no such file gone.py", ), 2, 2),
        (call("read_file", "f", path="last.py"), ToolResult("f", "read_file", ToolStatus.OK, "last\n" + "z" * 800, data={"sha256": "h4"}), 2, 2),
    ]
    for c, r, b, a in plan:
        msgs.append(ChatMessage(role="assistant", content="SECRET-PROSE " * 5, tool_calls=[c]))
        msgs.append(ChatMessage(role="tool", content=f"[{r.status.value}] {r.output}", tool_call_id=c.id, name=c.name))
        records.append(_record(c, r, b, a, validation=c.name == "run_tests"))
    return msgs, records


def test_summary_is_deterministic_structured_and_keeps_user_text() -> None:
    results = []
    for _ in range(2):
        msgs, records = _history()
        ctx = ContextManager(budget_tokens=900, vault=OutputVault("s"), keep_last=2)
        compaction = ctx.compact(msgs, [], records=records, latest_change=2)
        assert compaction is not None and compaction.stage == "groups" and compaction.dropped_groups >= 1
        summary = next(m.content for m in msgs if m.content.startswith(SUMMARY_PREFIX))
        results.append((summary, [m.content for m in msgs]))
    assert results[0] == results[1]
    summary = results[0][0]
    assert "SECRET-PROSE" not in summary  # no model prose
    assert "- p.py h1" in summary and "- p.py h2" in summary and "- q.py h3" in summary
    assert "run_tests fail at revision 1 (stale: a later change exists)" in summary
    assert "read_file failed gone.py: no such file gone.py" in summary
    assert "show_output(handle)" in summary
    assert OBJECTIVE in results[0][1]  # user message untouched, verbatim


def test_groups_are_dropped_whole_and_newest_group_stays() -> None:
    msgs, records = _history()
    ctx = ContextManager(budget_tokens=500, vault=OutputVault("s"), keep_last=1)
    ctx.compact(msgs, [], records=records, latest_change=2)
    ids_calls = [c.id for m in msgs for c in m.tool_calls]
    ids_results = [m.tool_call_id for m in msgs if m.role == "tool"]
    assert ids_calls == ids_results and ids_calls[-1] == "f"
    assert msgs[0].content == "sys" and msgs[1].content == "pinned"


def test_dropped_outputs_stay_retrievable() -> None:
    msgs, records = _history()
    vault = OutputVault("s")
    ctx = ContextManager(budget_tokens=500, vault=vault, keep_last=1)
    ctx.compact(msgs, [], records=records, latest_change=2)
    assert vault.get(1) is not None and "head" in vault.get(1)  # type: ignore[operator]


async def test_exhausted_window_stops_incomplete_without_crash() -> None:
    budget = Budget(context_tokens=1_200, max_steps=10)
    transport = ScriptedTransport([turn("never reached")])
    engine = Engine(transport, make_runtime(), model="m", budget=budget, estimator=est)
    result = await engine.run("s", "x" * 20_000)
    assert result.outcome is Outcome.INCOMPLETE
    assert any("context budget exhausted" in r for r in result.reasons)
    assert transport.requests == []  # nothing oversized was ever sent


async def test_single_oversized_result_stops_instead_of_overflowing() -> None:
    budget = Budget(context_tokens=1_500, max_steps=10, max_output_chars=50_000)
    transport = ScriptedTransport([turn("", [call("read_file", "c1", path="big.py")]), turn("never")])
    runtime = FakeRuntime({"read_file": lambda c: ok(c, "x" * 30_000)})
    engine = Engine(transport, runtime, model="m", budget=budget, estimator=est)
    result = await engine.run("s", "read it")
    assert result.outcome is Outcome.INCOMPLETE
    assert any("context budget exhausted" in r for r in result.reasons)
    assert len(transport.requests) == 1
