"""Regression tests for the final-review role findings B1, S2, S3, S4 (scripted transports, no models)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from ironclad.config import resolve_config
from ironclad.contracts import ToolCall, ToolResult, ToolStatus
from ironclad.engine.loop import ROLE_STATE, Engine
from ironclad.engine.state import Budget

from .fakes import FakeRuntime, ScriptedTransport, Turn, call, ok


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("NEDSTER_") or name == "OLLAMA_HOST":
            monkeypatch.delenv(name)
    user = tmp_path / "user"
    project = tmp_path / "proj"
    user.mkdir()
    (project / ".nedster").mkdir(parents=True)
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(user))
    monkeypatch.setenv("NEDSTER_DATA_DIR", str(tmp_path / "data"))
    return user, project


def _config(project: Path, overrides: dict | None = None):
    config, prov = resolve_config(overrides or {}, project)
    config.runtime.project_root = project
    return config, prov


def _native(config, **kw):
    from ironclad.native import NativeSession

    return NativeSession(config, **kw)


# --- B1: the untrusted project layer cannot change the role ----------------------------------


def test_project_config_cannot_override_user_plan_role(dirs):
    user, project = dirs
    (user / "config.toml").write_text('[runtime]\nrole = "plan"\n', encoding="utf-8")
    (project / ".nedster" / "config.toml").write_text('[runtime]\nrole = "code"\n', encoding="utf-8")
    config, prov = _config(project)
    assert config.runtime.role == "plan"
    assert prov["runtime.role"] == "user"
    assert any("runtime.role" in w and "not allowed" in w for w in config.warnings)


def test_project_config_cannot_set_role_even_without_user_role(dirs):
    _, project = dirs
    (project / ".nedster" / "config.toml").write_text('[runtime]\nrole = "review"\n', encoding="utf-8")
    config, prov = _config(project)
    assert config.runtime.role == "code" and prov["runtime.role"] == "default"


def test_invalid_role_keeps_the_trusted_role(dirs, monkeypatch):
    user, project = dirs
    (user / "config.toml").write_text('[runtime]\nrole = "plan"\n', encoding="utf-8")
    (project / ".nedster" / "config.toml").write_text('[runtime]\nrole = "bogus"\n', encoding="utf-8")
    monkeypatch.setenv("NEDSTER_RUNTIME__ROLE", "nonsense")
    config, prov = _config(project)
    assert config.runtime.role == "plan" and prov["runtime.role"] == "user"
    assert any("unknown runtime.role" in w for w in config.warnings)


def test_cli_role_stays_authoritative(dirs):
    user, project = dirs
    (user / "config.toml").write_text('[runtime]\nrole = "plan"\n', encoding="utf-8")
    config, prov = _config(project, {"runtime.role": "code"})
    assert config.runtime.role == "code" and prov["runtime.role"] == "cli"


async def test_project_role_cannot_enable_a_real_write(dirs):
    user, project = dirs
    (user / "config.toml").write_text('[runtime]\nrole = "plan"\n', encoding="utf-8")
    (project / ".nedster" / "config.toml").write_text('[runtime]\nrole = "code"\n', encoding="utf-8")
    config, _ = _config(project)
    session = _native(config)
    try:
        assert session.engine.role_name == "plan"
        session.engine.transport = ScriptedTransport(
            [Turn(calls=[call("write_file", "1", path="role-bypass.txt", content="pwned")]), Turn("Done.")]
        )
        await session.run("write role-bypass.txt")
        assert not (project / "role-bypass.txt").exists()
        assert any("not allowed in plan mode" in str(m.content) for m in session.engine.messages(session.session_id))
    finally:
        await session.aclose()


# --- S2: the role is persisted with the session and restored on resume -----------------------


async def test_resume_restores_the_saved_role(dirs):
    _, project = dirs
    config, _ = _config(project)
    first = _native(config, role="plan")
    try:
        first.engine.transport = ScriptedTransport([Turn("A plan.")])
        await first.run("plan something")
        sid = first.session_id
        assert first.store.get_state(sid, ROLE_STATE) == "plan"
    finally:
        await first.aclose()

    resumed = _native(config, session_id=sid)  # ordinary code config, no --role
    try:
        assert resumed.engine.role_name == "plan"
    finally:
        await resumed.aclose()
    latest = _native(config, resume_latest=True)
    try:
        assert latest.engine.role_name == "plan"
    finally:
        await latest.aclose()
    explicit = _native(config, session_id=sid, role="code")  # an explicit trusted override still wins
    try:
        assert explicit.engine.role_name == "code"
    finally:
        await explicit.aclose()


async def test_role_switch_is_recorded_and_old_sessions_keep_config_role(dirs):
    _, project = dirs
    config, _ = _config(project)
    session = _native(config)
    try:
        sid = session.ensure_session_id()
        assert session.store.get_state(sid, ROLE_STATE) is None  # an "old" session: nothing saved
    finally:
        await session.aclose()
    old = _native(config, session_id=sid)
    try:
        assert old.engine.role_name == "code"  # falls back to the configured role
        old.set_role("review")
        assert old.store.get_state(sid, ROLE_STATE) == "review"
    finally:
        await old.aclose()
    again = _native(config, session_id=sid)
    try:
        assert again.engine.role_name == "review"
    finally:
        await again.aclose()


# --- S3: no role switches while a turn or tool job is active ---------------------------------


def _edit(c: ToolCall) -> ToolResult:
    return ToolResult(c.id, c.name, ToolStatus.OK, "edited", data={"sha256": "sha256:abc"}, mutated_paths=("a.py",))


async def test_set_role_during_an_active_job_raises_and_role_is_frozen():
    runtime = FakeRuntime(
        {"write_file": _edit, "run_tests": lambda c: ok(c, "1 passed", tests_run=1, exit_code=0)},
        delays={"write_file": 0.2},
    )
    transport = ScriptedTransport(
        [Turn(calls=[call("write_file", "1", path="a.py")]), Turn(calls=[call("run_tests", "2")]), Turn("Done.")]
    )
    engine = Engine(transport, runtime, model="fake", budget=Budget(), role="code")
    task = asyncio.create_task(engine.run("s", "edit a.py"))
    for _ in range(100):
        if engine._running:
            break
        await asyncio.sleep(0.01)
    assert engine._running, "the delayed write job should be active"
    assert engine.busy
    with pytest.raises(RuntimeError, match="role"):
        engine.set_role("plan")
    assert engine.role_name == "code"
    await task
    assert engine.role_name == "code"
    assert not engine.busy
    steps_before = len(transport.requests)
    engine.set_role("plan")  # allowed once idle; history/budget are not reset
    assert engine.role_name == "plan"
    assert len(transport.requests) == steps_before


async def test_set_role_during_a_model_turn_raises():
    gate = asyncio.Event()

    class SlowTransport(ScriptedTransport):
        async def chat(self, **kw):
            await gate.wait()
            async for event in super().chat(**kw):
                yield event

    engine = Engine(SlowTransport([Turn("ok")]), FakeRuntime({}), model="fake", budget=Budget(), role="plan")
    task = asyncio.create_task(engine.run("s", "hi"))
    await asyncio.sleep(0.01)
    with pytest.raises(RuntimeError):
        engine.set_role("code")
    gate.set()
    await task
    assert engine.role_name == "plan"


# --- S4: the TUI adapter shows the real engine's role -------------------------------------------


async def test_native_adapter_shows_configured_role_and_first_toggle_goes_to_code(dirs):
    from nedster_agent.tui import native_session

    user, project = dirs
    (user / "config.toml").write_text('[runtime]\nrole = "plan"\n', encoding="utf-8")
    config, _ = _config(project)
    adapter = native_session(config, lambda event: None)
    try:
        info = adapter.info()
        assert info["role"] == "plan"
        assert info["role_label"]
        # The Shift+Tab rule in tui.action_toggle_role: anything but plan -> plan, plan -> code.
        target = "plan" if str(info["role"]).lower() != "plan" else "code"
        assert target == "code"
        adapter.set_role(target)
        assert adapter.info()["role"] == "code"
        assert adapter.core.engine.role_name == "code"
    finally:
        await adapter.close()
