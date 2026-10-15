"""Tests for built-in role presets (plan, review, explain) -- read-only modes deny write tools."""
from __future__ import annotations

import pytest

from ironclad.contracts import Outcome, ToolCall, ToolResult, ToolStatus
from ironclad.engine.loop import Engine
from ironclad.engine.roles import DEFAULT_ROLE, READ_ONLY_TOOLS, ROLES, get_role
from ironclad.engine.state import Budget

from .fakes import Events, FakeRuntime, ScriptedTransport, Turn, call, ok


def read(c: ToolCall) -> ToolResult:
    return ok(c, f"contents of {c.arguments.get(chr(39)+'path'+chr(39))}")

def edit(c: ToolCall) -> ToolResult:
    path = str(c.arguments.get("path", "calc.py"))
    return ToolResult(c.id, c.name, ToolStatus.OK, f"edited {path}", data={"sha256": "sha256:abc"}, mutated_paths=(path,))

def run_cmd(c: ToolCall) -> ToolResult:
    return ok(c, "3 passed", exit_code=0, tests_run=3)

def green(c: ToolCall) -> ToolResult:
    return ok(c, "5 passed", tests_run=5, exit_code=0)

def make_engine(script, handlers=None, role=DEFAULT_ROLE, **kw):
    transport = ScriptedTransport(script)
    runtime = kw.pop("runtime", None) or FakeRuntime(
        handlers or {"read_file": read, "edit_file": edit, "write_file": edit, "run_tests": green}
    )
    events = Events()
    engine = Engine(transport, runtime, model="fake", emit=events.append, budget=kw.pop("budget", Budget()), role=role, **kw)
    return engine, transport, runtime, events

@pytest.mark.parametrize("role_name", ["plan", "review", "explain"])
async def test_readonly_role_denies_edit_file_and_run_command(role_name):
    runtime = FakeRuntime(
        {"read_file": read, "edit_file": edit, "write_file": edit, "run_tests": green},
        mutating=["edit_file", "write_file"],
    )
    script = [
        Turn(calls=[call("read_file", "1", path="a.py")]),
        Turn(calls=[call("edit_file", "2", path="a.py")]),
        Turn("Plan written."),
    ]
    executed: list[str] = []
    runtime.handlers["edit_file"] = lambda c: executed.append(c.name) or edit(c)
    engine, *_ = make_engine(script, runtime=runtime, role=role_name)
    result = await engine.run("s", "fix a.py")
    denied = [m for m in engine.messages("s") if f"not allowed in {role_name} mode" in str(m.content)]
    assert len(denied) == 1  # the edit_file call, refused at dispatch
    assert executed == []  # the runtime never ran it
    assert result.outcome is not Outcome.COMPLETED
    assert result.changed_paths == ()

@pytest.mark.parametrize("role_name", ["plan", "review", "explain"])
async def test_readonly_role_denies_run_command(role_name):
    runtime = FakeRuntime(
        {"read_file": read, "edit_file": edit, "run_tests": green, "run_command": run_cmd},
        mutating=["edit_file", "run_command"],
    )
    script = [
        Turn(calls=[call("run_command", "1", argv=["ls"])]),
        Turn("Done."),
    ]
    executed: list[str] = []
    runtime.handlers["run_command"] = lambda c: executed.append(c.name) or run_cmd(c)
    engine, *_ = make_engine(script, runtime=runtime, role=role_name)
    result = await engine.run("s", "list files")
    assert any(f"run_command is not allowed in {role_name} mode" in str(m.content) for m in engine.messages("s"))
    assert executed == []
    assert result.outcome is not Outcome.COMPLETED

async def test_code_role_allows_edit_file():
    script = [
        Turn(calls=[call("edit_file", "1", path="a.py")]),
        Turn(calls=[call("run_tests", "2")]),
        Turn("Done."),
    ]
    engine, _, runtime, _ = make_engine(script, role=DEFAULT_ROLE)
    result = await engine.run("s", "edit a.py")
    assert result.outcome is Outcome.COMPLETED
    allowed_calls = [c for c in runtime.calls if c.name == "edit_file"]
    assert len(allowed_calls) >= 1

async def test_set_role_mid_session_updates_prompt_not_steps():
    engine, transport, _, _ = make_engine(
        [Turn("Noted."), Turn("audit-code-17")],
    )
    result = await engine.run("s", "Remember audit-code-17")
    assert result.outcome is Outcome.ANSWERED
    old_messages_before = len(engine.messages("s"))
    engine.set_role("plan")
    old_messages_after = len(engine.messages("s"))
    assert old_messages_after == old_messages_before
    system_prompt = engine.messages("s")[0].content
    assert "read-only" in system_prompt.lower() or "not allowed" in system_prompt.lower()

async def test_role_cannot_add_tools_beyond_policy():
    runtime = FakeRuntime(
        {"read_file": read, "run_command": run_cmd},
        mutating=["run_command"],
    )
    script = [Turn(calls=[call("run_command", "1", argv=["ls"])]), Turn("Done.")]
    engine = Engine(
        ScriptedTransport(script),
        runtime,
        model="fake",
        emit=Events().append,
        budget=Budget(),
        role="plan",
    )
    await engine.run("s", "run ls")
    denied_texts = [str(m.content) for m in engine.messages("s") if "[denied]" in str(m.content).lower()]
    assert len(denied_texts) >= 1

def test_role_names_returns_tuple_of_known_keys():
    from ironclad.engine.roles import role_names
    names = role_names()
    assert names == ("code", "plan", "review", "explain")

def test_get_role_default_is_code():
    assert get_role(None).name == DEFAULT_ROLE
    assert get_role("").name == DEFAULT_ROLE

def test_get_role_raises_on_unknown():
    with pytest.raises(ValueError, match="unknown role"):
        get_role("wizard")

def test_read_only_roles_have_tools_set():
    for name in ("plan", "review", "explain"):
        role = ROLES[name]
        assert role.read_only is True
        assert role.tools == READ_ONLY_TOOLS
        assert isinstance(role.step_budget, int) and role.step_budget > 0

def test_code_role_is_not_read_only():
    code_role = ROLES["code"]
    assert code_role.read_only is False
    assert code_role.tools is None
