from __future__ import annotations

from typing import Any

from ironclad.contracts import ChatMessage, Outcome, ToolCall, ToolResult, ToolStatus
from ironclad.engine.context import (
    SHOW_OUTPUT_SPEC,
    ContextManager,
    OutputVault,
    estimate_prompt,
    run_show_output,
    tool_message,
)
from ironclad.engine.loop import Engine
from ironclad.engine.prompt import build_system_prompt, project_facts
from ironclad.engine.state import Budget, TaskState

from .fakes import Events, FakeRuntime, MemoryStore, ScriptedTransport, Turn, call, ok

BIG = 2_000  # chars per scripted tool output (~500 tokens)


def big_read(c: ToolCall) -> ToolResult:
    head = f"file {c.arguments.get('path')} line 1"
    return ok(c, head + "\n" + "x" * BIG)


def failing_tests(c: ToolCall) -> ToolResult:
    return ToolResult(c.id, c.name, ToolStatus.FAILED, "FAILED test_calc.py::test_sub\n" + "y" * BIG, data={"tests_run": 3})


def long_script(n: int) -> list[Any]:
    turns: list[Any] = [Turn(calls=[call("run_tests", "t0")])]
    turns += [Turn(calls=[call("read_file", f"r{i}", path=f"f{i}.py")]) for i in range(n - 1)]
    return turns


async def test_sixty_tool_turns_stay_under_budget_with_retrievable_handles() -> None:
    budget = Budget(context_tokens=8_000, max_steps=80)
    events = Events()
    marks: list[int] = []  # request count at each compaction
    transport = ScriptedTransport([])

    def emit(event: Any) -> None:
        events.append(event)
        if event.kind == "context.compacted":
            marks.append(len(transport.requests))

    store = MemoryStore()

    def final(messages: list[ChatMessage]) -> Turn:
        return Turn(calls=[call("show_output", "show", handle=handle)])

    transport.script = [*long_script(60), final, Turn("done")]
    runtime = FakeRuntime({"read_file": big_read, "run_tests": failing_tests})
    engine = Engine(transport, runtime, model="m", store=store, budget=budget, emit=emit)
    handle = 1  # first elided output; set before the final turn runs
    result = await engine.run("s", "survey the project")

    assert result.outcome is Outcome.ANSWERED
    assert len(transport.requests) == 62
    assert marks, "expected at least one compaction"
    specs = [*runtime.specs(), SHOW_OUTPUT_SPEC]
    for request in transport.requests:
        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in request]
        assert estimate_prompt(messages, specs) < budget.context_tokens

    # prefix identical between compactions: request i+1 extends request i unless a compaction preceded i+1
    for i in range(len(transport.requests) - 1):
        if (i + 1) in marks:
            continue
        nxt = transport.requests[i + 1]
        assert transport.requests[i] == nxt[: len(transport.requests[i])], f"prefix changed at request {i + 1}"

    # handles retrievable: the show_output call returned the full first elided output
    show_msg = next(m for m in engine.messages("s") if m.tool_call_id == "show")
    assert show_msg.content.startswith("[ok] ")
    assert "file f0.py line 1" in show_msg.content and len(show_msg.content) > BIG
    stub = next(m for m in engine.messages("s") if m.role == "tool" and m.content.startswith("[output #"))
    assert "elided:" in stub.content and "call show_output(" in stub.content
    assert store.replaced >= 1 and store.outputs

    # latest failing validation output kept verbatim
    failing = next(m for m in engine.messages("s") if m.tool_call_id == "t0")
    assert failing.content.startswith("[failed] FAILED test_calc.py::test_sub") and len(failing.content) > BIG

    # call/result pairing kept: every assistant tool call has exactly one tool message
    msgs = engine.messages("s")
    call_ids = [c.id for m in msgs for c in m.tool_calls]
    tool_ids = [m.tool_call_id for m in msgs if m.role == "tool"]
    assert call_ids == tool_ids

    # resume after compaction: a fresh engine on the same store sees the masked history and the handles
    engine2 = Engine(ScriptedTransport([Turn(calls=[call("show_output", "again", handle=1)]), Turn("ok")]), runtime,
                     model="m", store=store, budget=budget)
    await engine2.run("s", "show me the first file again")
    again = next(m for m in engine2.messages("s") if m.tool_call_id == "again")
    assert "file f0.py line 1" in again.content
    assert any(m.content.startswith("[output #") for m in store.messages("s"))


async def test_compaction_refreshes_pinned_state_and_emits_event() -> None:
    budget = Budget(context_tokens=3_000)
    events = Events()
    transport = ScriptedTransport([*long_script(12), Turn("done")])
    runtime = FakeRuntime({"read_file": big_read, "run_tests": failing_tests})
    engine = Engine(transport, runtime, model="m", budget=budget, emit=events.append)
    await engine.run("s", "survey")
    compactions = events.of("context.compacted")
    assert compactions and compactions[0].data["elided"] >= 1
    assert compactions[0].data["tokens_after"] < compactions[0].data["tokens_before"]
    pinned = engine.messages("s")[1].content
    assert pinned.startswith("[task state") and "Validation: fail" in pinned


def test_stub_format_and_keep_last() -> None:
    vault = OutputVault("s")
    ctx = ContextManager(budget_tokens=100, vault=vault, keep_last=2)
    messages = [ChatMessage(role="system", content="sys")]
    for i in range(4):
        messages.append(ChatMessage(role="assistant", tool_calls=[ToolCall(str(i), "read_file", {})]))
        messages.append(ChatMessage(role="tool", content=f"[ok] head {i}\n" + "z" * 400, tool_call_id=str(i), name="read_file"))
    compaction = ctx.compact(messages, [])
    assert compaction is not None
    tools = [m for m in messages if m.role == "tool"]
    assert tools[0].content == "[output #1 elided: read_file ok head 0; call show_output(1)]"
    assert tools[0].tool_call_id == "0"
    assert vault.get(1) == "head 0\n" + "z" * 400


def test_protected_message_is_not_masked() -> None:
    ctx = ContextManager(budget_tokens=10, vault=OutputVault("s"), keep_last=1)
    messages = [
        ChatMessage(role="tool", content="[failed] boom " + "z" * 200, tool_call_id="keep", name="run_tests"),
        ChatMessage(role="tool", content="[ok] " + "z" * 200, tool_call_id="a", name="read_file"),
        ChatMessage(role="tool", content="[ok] " + "z" * 200, tool_call_id="b", name="read_file"),
    ]
    ctx.compact(messages, [], protect={"keep"})
    assert messages[0].content.startswith("[failed] boom")
    assert messages[1].content.startswith("[output #")


def test_tool_message_bounds_output_and_keeps_full_text() -> None:
    vault = OutputVault("s")
    result = ToolResult("c", "run_command", ToolStatus.OK, "A" * 50_000)
    message = tool_message(result, vault, max_chars=1_000)
    assert len(message.content) < 1_200
    assert "show_output(handle=1)" in message.content
    assert vault.get(1) == "A" * 50_000


def test_show_output_paging_and_errors() -> None:
    vault = OutputVault("s")
    vault.put("t", "c", "\n".join(f"line {i}" for i in range(1, 501)))
    page = run_show_output(vault, ToolCall("x", "show_output", {"handle": 1, "start_line": 10, "max_lines": 5}), 10_000)
    assert page.status is ToolStatus.OK
    assert page.output.splitlines()[0] == "line 10" and "start_line=15" in page.output
    assert run_show_output(vault, ToolCall("x", "show_output", {"handle": 9}), 100).status is ToolStatus.INVALID
    assert run_show_output(vault, ToolCall("x", "show_output", {"handle": "1"}), 100).status is ToolStatus.INVALID


def test_vault_uses_store_handles() -> None:
    store = MemoryStore()
    vault = OutputVault("s", store)
    assert vault.put("t", "c", "one") == 1 and vault.put("t", "c", "two") == 2
    assert OutputVault("s", store).get(2) == "two"
    assert OutputVault("s", store).get(7) is None


def test_task_state_render_and_roundtrip() -> None:
    state = TaskState()
    state.note_user("Fix the parser")
    state.note_user("Do not touch tests/")
    state.files["parser.py"] = "sha256:abc"
    state.obligations = ["run validation"]
    text = state.render()
    assert "Objective: Fix the parser" in text and "- Do not touch tests/" in text
    assert "- parser.py sha256:abc" in text and "- run validation" in text
    assert TaskState.from_dict(state.to_dict()) == state


def test_system_prompt_is_deterministic(tmp_path: Any) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "src").mkdir()
    facts = project_facts(tmp_path)
    assert facts["toolchains"] == "python (pytest)" and "src/" in facts["top_level"]
    assert build_system_prompt(facts) == build_system_prompt(project_facts(tmp_path))
    assert build_system_prompt(None).startswith("You are Nedster")
    assert project_facts(tmp_path / "missing")["os"]
