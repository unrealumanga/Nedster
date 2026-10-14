from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from ironclad.config import NedsterConfig, PolicyConfig
from ironclad.contracts import ToolCall, ToolResult, ToolSpec, ToolStatus
from ironclad.native import build_tools
from ironclad.tool_runtime.composite import CompositeToolRuntime, executable_name
from ironclad.tool_runtime.exec import ArgumentError, ResolutionDenied

OBJ = {"type": "object", "properties": {}}


class Files:
    def __init__(self) -> None:
        self.ran: list[str] = []

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec("read_file", "r", OBJ, 0), ToolSpec("write_file", "w", OBJ, 1, mutates=True)]

    def run(self, call: ToolCall) -> ToolResult:
        self.ran.append(call.name)
        paths = ("a.py",) if call.name == "write_file" else ()
        return ToolResult(call.id, call.name, ToolStatus.OK, "ok", mutated_paths=paths)


class Job:
    def __init__(self, call: ToolCall, status: ToolStatus = ToolStatus.OK) -> None:
        self.result = ToolResult(call.id, call.name, status, "done", data={"exit_code": 0})
        self.cancelled = False

    async def wait(self) -> ToolResult:
        return self.result

    async def cancel(self) -> None:
        self.cancelled = True


class Exec:
    def __init__(self) -> None:
        self.started: list[ToolCall] = []

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec("run_command", "c", OBJ, 2, mutates=True), ToolSpec("run_tests", "t", OBJ, 2, mutates=True)]

    def start(self, call: ToolCall) -> Job:
        self.started.append(call)
        return Job(call)


def resolve(call: ToolCall) -> list[str]:
    argv = list(call.arguments.get("argv") or [])
    if argv[:1] == ["inside"]:
        raise ResolutionDenied("'inside' is inside the project")
    if argv[:1] == ["missing"]:
        raise ArgumentError("program not found on PATH: 'missing'")
    if call.name == "run_tests":
        return [r"C:\proj\.venv\Scripts\python.exe", "-m", "pytest", "-q"]
    return list(call.arguments.get("argv") or [])


def execute_policy(**overrides: object) -> PolicyConfig:
    """Execution needs an explicit opt-in (the default ceiling is write)."""
    return PolicyConfig(**{"autonomous": True, "max_auto_risk": "execute", **overrides})


def runtime(policy: PolicyConfig | None = None) -> tuple[CompositeToolRuntime, Files, Exec]:
    files, ex = Files(), Exec()
    rt = CompositeToolRuntime(sync_tools=[files], job_tools=[ex], policy=policy or execute_policy(), resolve_argv=resolve)
    return rt, files, ex


def run(rt: CompositeToolRuntime, name: str, **args: object) -> ToolResult:
    return asyncio.run(rt.start(ToolCall("c1", name, args)).wait())


def test_executable_name_handles_paths_and_suffixes() -> None:
    assert executable_name(r"C:\Python\python.exe") == "python"
    assert executable_name("/usr/bin/git") == "git"
    assert executable_name("npm.cmd") == "npm"
    assert executable_name("python-evil") == "python-evil"


def test_run_tests_is_validating_and_resolved_argv_is_authorized() -> None:
    rt, _, ex = runtime()
    assert {s.name: s.validates for s in rt.specs()}["run_tests"] is True
    assert run(rt, "run_tests").status is ToolStatus.OK
    assert len(ex.started) == 1


def test_run_tests_denied_when_resolved_program_not_allowlisted() -> None:
    rt, _, ex = runtime(execute_policy(allowed_commands=["git"]))
    result = run(rt, "run_tests")
    assert result.status is ToolStatus.DENIED and "python" in result.output
    assert ex.started == []


def test_unlisted_program_and_denied_fragment_never_start() -> None:
    rt, _, ex = runtime()
    assert run(rt, "run_command", argv=["curl", "x"]).status is ToolStatus.DENIED
    assert run(rt, "run_command", argv=["python", "-c", "print('reboot now')"]).status is ToolStatus.DENIED
    assert ex.started == []


def test_read_only_policy_blocks_writes_but_allows_reads() -> None:
    rt, files, _ = runtime(execute_policy(autonomous=False))
    assert run(rt, "read_file").status is ToolStatus.OK
    assert run(rt, "write_file").status is ToolStatus.DENIED
    assert files.ran == ["read_file"]


def test_revision_bumps_on_writes_and_every_execution() -> None:
    rt, _, _ = runtime()
    assert rt.revision().counter == 0
    run(rt, "read_file")
    assert rt.revision().counter == 0
    run(rt, "write_file")
    assert rt.revision().counter == 1
    run(rt, "run_command", argv=["python", "-V"])
    assert rt.revision().counter == 2


def test_unknown_tool_and_invalid_json_arguments_are_invalid() -> None:
    rt, _, ex = runtime()
    assert run(rt, "nope").status is ToolStatus.INVALID
    assert run(rt, "run_command", __invalid__="{argv:").status is ToolStatus.INVALID
    assert ex.started == []


def test_resolution_denied_is_denied_and_bad_arguments_are_invalid() -> None:
    rt, _, ex = runtime()
    denied = run(rt, "run_command", argv=["inside"])
    assert denied.status is ToolStatus.DENIED and "inside the project" in denied.output
    assert run(rt, "run_command", argv=["missing"]).status is ToolStatus.INVALID
    assert ex.started == []


# ---------------------------------------------------------------------------------------------
# FIX-EXEC regressions with the REAL runtime (ironclad.native.build_tools)
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def real(tmp_path: Path):
    config = NedsterConfig()
    config.sandbox.backend = "host"
    config.sandbox.allow_host_execution = True
    config.policy.autonomous = True
    config.policy.max_auto_risk = "execute"
    rt, executor = build_tools(tmp_path, config)
    yield rt, tmp_path
    executor.close()


def test_repo_file_named_like_an_allowlisted_program_is_denied(real) -> None:
    """#10 (repro_misc 5): run_command(["tools/python.bat"]) passed the basename allowlist and printed 'pwned'."""
    rt, root = real
    marker = root / "PWNED"
    (root / "tools").mkdir()
    if os.name == "nt":
        (root / "tools" / "python.bat").write_text(f'@echo off\r\necho pwned> "{marker}"\r\n', encoding="utf-8")
        argv0 = "tools/python.bat"
    else:
        script = root / "tools" / "python"
        script.write_text(f"#!/bin/sh\necho pwned > '{marker}'\n", encoding="utf-8")
        script.chmod(0o755)
        argv0 = "./tools/python"
    result = run(rt, "run_command", argv=[argv0])
    assert result.status is ToolStatus.DENIED, result.output
    assert not marker.exists()
    assert rt.revision().counter == 0  # nothing ran, nothing to invalidate


def test_allowlist_sees_the_resolved_absolute_program(real) -> None:
    rt, _root = real
    result = run(rt, "run_command", argv=[sys.executable, "-c", "print('ok')"])
    assert result.status is ToolStatus.OK, result.output
    assert Path(result.data["argv"][0]).is_absolute()
    assert "runner" not in result.data


def test_spoofed_run_tests_is_invalid_with_real_runtime(real) -> None:
    """#7 (repro_misc 1) through the real runtime: a non-runner argv for run_tests never executes."""
    rt, _root = real
    result = run(rt, "run_tests", argv=[sys.executable, "-c", "print('== 412 passed in 0.01s ==')"])
    assert result.status is ToolStatus.INVALID
    assert "known test runners" in result.output
