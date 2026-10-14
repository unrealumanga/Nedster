from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import psutil
import pytest

from ironclad.config import SandboxConfig
from ironclad.contracts import ToolCall, ToolStatus
from ironclad.tool_runtime.exec import (
    IS_WINDOWS,
    ArgumentError,
    BoundedOutput,
    Executor,
    ResolutionDenied,
    batch_command_line,
    build_env,
    find_program,
    host_search_path,
    is_within,
    parse_tests_run,
    runner_for,
    venv_python,
)

TREE_SCRIPT = """
import os, subprocess, sys, time
depth, marker = int(sys.argv[1]), sys.argv[2]
if depth < 2:
    subprocess.Popen([sys.executable, __file__, str(depth + 1), marker])
with open(f"{marker}.{depth}", "w") as handle:
    handle.write(str(os.getpid()))
time.sleep(60)
"""


@pytest.fixture
def executor(tmp_path: Path):
    config = SandboxConfig(backend="host", allow_host_execution=True, max_seconds=60)
    ex = Executor(tmp_path, config)
    yield ex
    ex.close()


def call(argv, name: str = "run_command", **extra) -> ToolCall:
    arguments = {"argv": argv, **extra} if argv is not None else extra
    return ToolCall(id="c1", name=name, arguments=arguments)


def py(code: str, *args: str) -> list[str]:
    return [sys.executable, "-c", code, *args]


async def test_success_reports_exit_code_backend_and_isolation(executor: Executor):
    result = await executor.start(call(py("print('hello')"))).wait()
    assert result.status is ToolStatus.OK
    assert "hello" in result.output
    assert result.data["exit_code"] == 0
    assert result.data["backend"] == "host"
    assert result.data["isolation"] == "none"
    assert result.data["bytes"] > 0
    assert "duration_ms" in result.data
    assert "tests_run" not in result.data  # unknown and not a test run


async def test_nonzero_exit_is_failed(executor: Executor):
    result = await executor.start(call(py("import sys; print('bad'); sys.exit(3)"))).wait()
    assert result.status is ToolStatus.FAILED
    assert result.data["exit_code"] == 3


async def test_timeout(executor: Executor):
    started = time.monotonic()
    result = await executor.start(call(py("import time; time.sleep(30)"), timeout_s=0.5)).wait()
    assert result.status is ToolStatus.TIMEOUT
    assert time.monotonic() - started < 8


async def test_output_is_bounded_for_5mb(executor: Executor):
    result = await executor.start(call(py("import sys; sys.stdout.write('x' * 5_000_000)"))).wait()
    assert result.status is ToolStatus.OK
    assert len(result.output) <= 64 * 1024
    assert result.data["bytes"] >= 5_000_000
    assert result.data["truncated"] is True
    assert "bytes omitted" in result.output


def _pids(marker: Path, timeout: float = 15.0) -> list[int]:
    deadline = time.monotonic() + timeout
    files = [Path(f"{marker}.{depth}") for depth in range(3)]
    while time.monotonic() < deadline:
        if all(item.exists() and item.read_text() for item in files):
            return [int(item.read_text()) for item in files]
        time.sleep(0.05)
    raise AssertionError("process tree did not start")


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


async def test_cancel_kills_child_and_grandchild(executor: Executor, tmp_path: Path):
    script = tmp_path / "tree.py"
    script.write_text(TREE_SCRIPT, encoding="utf-8")
    marker = tmp_path / "pid"
    job = executor.start(call([sys.executable, str(script), "0", str(marker)]))
    pids = await asyncio.to_thread(_pids, marker)
    assert all(_alive(pid) for pid in pids)
    started = time.monotonic()
    await job.cancel()
    deadline = time.monotonic() + 3
    while any(_alive(pid) for pid in pids) and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert not any(_alive(pid) for pid in pids)
    assert time.monotonic() - started < 3
    result = await job.wait()
    assert result.status is ToolStatus.CANCELLED
    await job.cancel()  # idempotent


async def test_cancel_after_completion_is_noop(executor: Executor):
    job = executor.start(call(py("print(1)")))
    result = await job.wait()
    await job.cancel()
    assert (await job.wait()) is result
    assert result.status is ToolStatus.OK


async def test_env_is_allowlisted(executor: Executor, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("NEDSTER_RANDOM_VAR", "1")
    result = await executor.start(call(py("import os, json; print(json.dumps(sorted(os.environ)))"))).wait()
    names = {name.upper() for name in json.loads(result.output.strip().splitlines()[-1])}
    assert "OPENAI_API_KEY" not in names
    assert "NEDSTER_RANDOM_VAR" not in names
    if IS_WINDOWS:
        assert "SYSTEMROOT" in names


def test_build_env_private_temp_secrets_and_no_project_path(tmp_path: Path):
    venv_dir = tmp_path / ".venv" / ("Scripts" if IS_WINDOWS else "bin")
    venv_dir.mkdir(parents=True)
    temp = tmp_path / "tmp"
    outside = str(Path(os.path.abspath(os.sep)) / "usr" / "bin")
    path = os.pathsep.join([str(venv_dir), str(tmp_path / "bin"), "relative/bin", outside])
    base = {"PATH": path, "HOME": "/h", "LC_ALL": "C", "LC_API_TOKEN": "x", "GITHUB_TOKEN": "y",
            "SYSTEMROOT": r"C:\Windows", "AWS_SECRET_ACCESS_KEY": "z"}  # fmt: skip
    env = build_env(tmp_path, temp, base)
    assert not any(marker in key for key in env for marker in ("TOKEN", "SECRET", "KEY"))
    # #2: neither the project venv nor any other project directory (or a relative entry) is on PATH.
    assert env["PATH"] == outside and "VIRTUAL_ENV" not in env
    interpreter = venv_dir / ("python.exe" if IS_WINDOWS else "python")
    with_venv = build_env(tmp_path, temp, base, venv=interpreter)
    assert with_venv["PATH"].split(os.pathsep)[0] == str(venv_dir)
    if IS_WINDOWS:
        assert env["TEMP"] == env["TMP"] == str(temp)
        assert env["SYSTEMROOT"] == r"C:\Windows"
    else:
        assert env["TMPDIR"] == str(temp)
        assert env["LC_ALL"] == "C"
        assert "SYSTEMROOT" not in env


async def test_invalid_arguments_never_execute(executor: Executor):
    for bad in ("echo hi", [], [1, 2], ["a\x00b"]):
        result = await executor.start(call(bad)).wait()
        assert result.status is ToolStatus.INVALID
    too_long = await executor.start(call(py("print(1)"), timeout_s=10_000)).wait()
    assert too_long.status is ToolStatus.INVALID
    missing = await executor.start(call(["definitely-not-a-program-xyz"])).wait()
    assert missing.status is ToolStatus.INVALID
    unknown = await executor.start(ToolCall("c1", "rm_rf", {})).wait()
    assert unknown.status is ToolStatus.INVALID


def test_batch_command_line_rejects_metacharacters():
    script = Path("C:/x/run.bat")
    for bad in ("a&b", "a|b", "a>b", "a<b", "a^b", "%PATH%", "a!b", 'a"b', "a\nb"):
        with pytest.raises(ArgumentError):
            batch_command_line("C:/Windows/System32/cmd.exe", script, [bad])
    line = batch_command_line("C:/Windows/System32/cmd.exe", script, ["plain", "with space", ""])
    assert line.startswith('"C:/Windows/System32/cmd.exe" /d /s /c "')
    assert '"with space"' in line and line.endswith('"""')


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows .bat handling")
async def test_bat_argument_with_ampersand_is_invalid(executor: Executor, tmp_path_factory: pytest.TempPathFactory):
    tools = tmp_path_factory.mktemp("tools")  # outside the project: project files may not be run
    script = tools / "echoer.bat"
    script.write_text("@echo off\r\necho ARG=%~1\r\n", encoding="utf-8")
    bad = await executor.start(call([str(script), "x&whoami"])).wait()
    assert bad.status is ToolStatus.INVALID
    good = await executor.start(call([str(script), "hello world"])).wait()
    assert good.status is ToolStatus.OK, good.output
    assert "ARG=hello world" in good.output


@pytest.mark.parametrize(
    ("runner", "text", "expected"),
    [
        ("pytest", "....\n5 passed in 0.12s\n", 5),
        ("pytest", "===== 1 failed, 4 passed, 2 warnings in 1.0s =====", 5),
        ("pytest", "== 2 passed, 1 error in 0.3s ==", 3),
        ("pytest", "no tests ran in 0.01s", 0),
        ("pytest", "== 3 passed in 61.20s (0:01:01) ==", 3),
        (
            "cargo",
            "test result: ok. 3 passed; 0 failed; 1 ignored\ntest result: FAILED. 1 passed; 2 failed; 0 ignored",
            6,
        ),
        ("npm", "Tests:       1 failed, 5 passed, 6 total", 6),
        ("npm", "      Tests  1 failed | 5 passed (6)", 6),
        ("npm", "\u2139 tests 4\n\u2139 pass 4", 4),
        ("node", "# tests 7\n# pass 7", 7),
        ("unittest", "Ran 4 tests in 0.002s\n\nOK", 4),
        ("go", "=== RUN   TestA\n--- PASS: TestA (0.00s)\n=== RUN   TestB\n--- FAIL: TestB (0.00s)\nFAIL", 2),
        ("go", "ok  \texample.com/pkg\t0.01s", None),
        ("pytest", "hello world", None),
        ("pytest", "", None),
        (None, "5 passed in 0.1s", None),  # not a known runner: never a count
        ("unittest", "5 passed in 0.1s", None),  # another runner's format is ignored
    ],
)
def test_parse_tests_run(runner: str | None, text: str, expected: int | None):
    assert parse_tests_run(text, runner) == expected


def test_parse_uses_only_the_runners_format_and_last_summary():
    """#7: a test printing a cargo summary (or an early fake pytest summary) cannot override the real one."""
    spoofed = "test result: ok. 999 passed; 0 failed\n== 412 passed in 0.01s ==\n...\n== 1 failed, 2 passed in 0.5s =="
    assert parse_tests_run(spoofed, "pytest") == 3


def test_bounded_output_small_and_large():
    small = BoundedOutput()
    small.feed(b"abc")
    assert small.text() == "abc" and not small.truncated
    big = BoundedOutput(head=4, tail=4)
    for _ in range(10):
        big.feed(b"0123456789")
    assert big.total == 100 and big.truncated


def test_run_tests_resolves_runner_before_execution(tmp_path: Path):
    ex = Executor(tmp_path, SandboxConfig())
    try:
        with pytest.raises(ArgumentError):
            ex.resolve_argv(ToolCall("c", "run_tests", {}))
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        assert ex.detect_test_argv()[1:] == ["-m", "unittest", "discover"]  # no evidence the project uses pytest
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n[tool.pytest.ini_options]\n", encoding="utf-8")
        assert ex.detect_test_argv()[1:] == ["-m", "pytest", "-q"]
        argv = ex.resolve_argv(ToolCall("c", "run_tests", {}))
        assert Path(argv[0]).is_absolute() and argv[1:] == ["-m", "pytest", "-q"]
        # An unverifiable "venv interpreter" (no pyvenv.cfg, not an executable image) is never used.
        venv = tmp_path / ".venv" / ("Scripts" if IS_WINDOWS else "bin")
        venv.mkdir(parents=True)
        (venv / ("python.exe" if IS_WINDOWS else "python")).write_text("", encoding="utf-8")
        assert not ex.resolve_argv(ToolCall("c", "run_tests", {}))[0].startswith(str(venv))
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        assert ex.detect_test_argv() == ["npm", "test"]
        (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
        assert ex.detect_test_argv() == ["go", "test", "./..."]
        (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        assert ex.detect_test_argv() == ["cargo", "test"]
        with pytest.raises(ArgumentError, match="known test runners"):
            ex.resolve_argv(ToolCall("c", "run_tests", {"argv": [sys.executable, "-c", "pass"]}))
    finally:
        ex.close()


async def test_run_tests_reports_tests_run(executor: Executor, tmp_path: Path):
    (tmp_path / "test_sample.py").write_text("def test_a():\n    pass\n\ndef test_b():\n    pass\n", "utf-8")
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_sample.py"]
    result = await executor.start(call(argv, name="run_tests")).wait()
    assert result.status is ToolStatus.OK, result.output
    assert result.data["tests_run"] == 2


async def test_run_tests_reports_runner_and_unknown_count_is_none(executor: Executor):
    argv = [sys.executable, "-m", "pytest", "--version"]  # pytest runs, but prints no summary
    result = await executor.start(call(argv, name="run_tests")).wait()
    assert result.data["runner"] == "pytest"
    assert result.data["tests_run"] is None


def test_specs_shape(executor: Executor):
    specs = {spec.name: spec for spec in executor.specs()}
    assert set(specs) == {"run_command", "run_tests"}
    assert specs["run_command"].parameters["required"] == ["argv"]
    assert specs["run_command"].mutates


# ---------------------------------------------------------------------------------------------
# FIX-EXEC regressions (review findings #2, #7, #10, #14)
# ---------------------------------------------------------------------------------------------


def _fake_program(directory: Path, name: str, marker: Path) -> Path:
    """A repo-supplied 'program' that records that it ran."""
    directory.mkdir(parents=True, exist_ok=True)
    if IS_WINDOWS:
        path = directory / f"{name}.bat"
        path.write_text(f'@echo off\r\necho pwned> "{marker}"\r\necho pwned\r\n', encoding="utf-8")
    else:
        path = directory / name
        path.write_text(f"#!/bin/sh\necho pwned > '{marker}'\necho pwned\n", encoding="utf-8")
        path.chmod(0o755)
    return path


async def test_path_qualified_project_program_is_denied(executor: Executor, tmp_path: Path):
    """#10 (repro_misc 5): ``tools/python.bat`` passed the allowlist by basename and ran a project file."""
    marker = tmp_path / "PWNED"
    _fake_program(tmp_path / "tools", "python", marker)
    for argv0 in ("tools/python.bat" if IS_WINDOWS else "./tools/python", str(tmp_path / "tools" / "python")):
        with pytest.raises(ResolutionDenied):
            executor.resolve_argv(call([argv0]))
        result = await executor.start(call([argv0])).wait()
        assert result.status is ToolStatus.DENIED, result.output
    assert not marker.exists()


async def test_bare_name_never_resolves_into_the_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """#2/#10: a project directory on the user's PATH (e.g. an activated .venv) cannot shadow a program."""
    marker = tmp_path / "PWNED"
    venv_bin = tmp_path / ".venv" / ("Scripts" if IS_WINDOWS else "bin")
    _fake_program(venv_bin, "python", marker)
    monkeypatch.setenv("PATH", os.pathsep.join([str(venv_bin), str(Path(sys.executable).parent), os.environ["PATH"]]))
    ex = Executor(tmp_path, SandboxConfig(backend="host", allow_host_execution=True))
    try:
        argv = ex.resolve_argv(call(["python", "-c", "print('real')"]))
        assert Path(argv[0]).is_absolute() and not is_within(argv[0], tmp_path)
        assert str(venv_bin) not in ex.env()["PATH"].split(os.pathsep)
        result = await ex.start(call(["python", "-c", "print('real')"])).wait()
    finally:
        ex.close()
    assert result.status is ToolStatus.OK and "real" in result.output
    assert not marker.exists()


def test_find_program_skips_project_and_relative_entries(tmp_path: Path):
    root = tmp_path / "proj"
    marker = tmp_path / "m"
    _fake_program(root / "bin", "tool", marker)
    outside = _fake_program(tmp_path / "outside", "tool", marker)
    search = host_search_path(root, os.pathsep.join([str(root / "bin"), "bin", str(tmp_path / "outside")]))
    assert search == str(tmp_path / "outside")
    assert find_program("tool", search, root) == Path(os.path.abspath(outside))
    assert find_program("tool", str(root / "bin"), root) is None


def _structural_venv(root: Path, *, home: Path, magic: bytes) -> Path:
    venv = root / ".venv"
    interpreter = venv / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(magic + b"\0" * 60)
    if not IS_WINDOWS:
        interpreter.chmod(0o755)
    (venv / "pyvenv.cfg").write_text(f"home = {home}\nversion_info = 3.13\n", encoding="utf-8")
    return interpreter


def _base_python_dir() -> Path:
    return Path(getattr(sys, "_base_executable", sys.executable)).parent


def test_venv_interpreter_must_be_structurally_real(tmp_path: Path):
    good_root, bad_root, inside_root = tmp_path / "good", tmp_path / "bad", tmp_path / "inside"
    native = b"MZ" if IS_WINDOWS else b"\x7fELF"
    interpreter = _structural_venv(good_root, home=_base_python_dir(), magic=native)
    assert venv_python(good_root) == interpreter
    _structural_venv(bad_root, home=_base_python_dir(), magic=b"@echo pwned" if IS_WINDOWS else b"#!/bin/sh")
    assert venv_python(bad_root) is None
    (inside_root / "fakehome").mkdir(parents=True)
    _structural_venv(inside_root, home=inside_root / "fakehome", magic=native)
    assert venv_python(inside_root) is None  # pyvenv.cfg home inside the project
    assert venv_python(tmp_path / "missing") is None


def test_verified_venv_for_python_but_other_project_programs_denied(tmp_path: Path):
    """The verified project venv backs bare python/pytest in run_tests and run_command; nothing else in the project runs."""
    native = b"MZ" if IS_WINDOWS else b"\x7fELF"
    interpreter = _structural_venv(tmp_path, home=_base_python_dir(), magic=native)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    ex = Executor(tmp_path, SandboxConfig(backend="host", allow_host_execution=True))
    try:
        assert ex.resolve_argv(ToolCall("t", "run_tests", {}))[0] == str(interpreter)
        assert ex.resolve_argv(ToolCall("t", "run_tests", {"argv": ["pytest", "-x"]})) == [
            str(interpreter), "-m", "pytest", "-x"]  # fmt: skip
        assert ex.resolve_argv(ToolCall("t", "run_tests", {"argv": [str(interpreter), "-m", "pytest"]}))[0] == str(
            interpreter
        )
        assert ex.resolve_argv(call(["python", "-V"]))[0] == str(interpreter)
        assert ex.resolve_argv(call([str(interpreter), "-V"]))[0] == str(interpreter)
        tool = tmp_path / "tools" / ("python.bat" if IS_WINDOWS else "python")
        tool.parent.mkdir()
        tool.write_text("echo pwned", encoding="utf-8")
        with pytest.raises(ResolutionDenied):
            ex.resolve_argv(call([str(tool)]))
    finally:
        ex.close()


async def test_run_tests_rejects_spoofed_runner(executor: Executor):
    """#7 (repro_misc 1): run_tests(argv=[python, -c, print('== 412 passed ==')]) reported tests_run=412."""
    result = await executor.start(call(py("print('== 412 passed in 0.01s ==')"), name="run_tests")).wait()
    assert result.status is ToolStatus.INVALID
    assert "tests_run" not in result.data and "runner" not in result.data


async def test_run_command_sets_runner_only_for_known_runners(executor: Executor, tmp_path: Path):
    """#7: data["runner"] marks a known runner invocation; anything else has no runner and no count."""
    plain = await executor.start(call(py("print('== 412 passed in 0.01s ==')"))).wait()
    assert plain.status is ToolStatus.OK
    assert "runner" not in plain.data and "tests_run" not in plain.data
    (tmp_path / "test_one.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_one.py"]
    real = await executor.start(call(argv)).wait()
    assert real.status is ToolStatus.OK, real.output
    assert real.data["runner"] == "pytest" and real.data["tests_run"] == 1


def test_runner_shadowed_by_project_module_is_not_a_runner(tmp_path: Path):
    """``python -m pytest`` from the project root would import a project pytest.py that prints anything."""
    argv = [sys.executable, "-m", "pytest", "-q"]
    assert runner_for(argv, tmp_path) == "pytest"
    (tmp_path / "pytest.py").write_text("print('99 passed in 0.1s')\n", encoding="utf-8")
    assert runner_for(argv, tmp_path) is None
    assert runner_for(["npm", "test"], tmp_path) == "npm"
    assert runner_for(["npm", "run", "build"], tmp_path) is None
    assert runner_for(["cargo", "test"], tmp_path) == "cargo"
    assert runner_for(["go", "test", "./..."], tmp_path) == "go"
    assert runner_for(["node", "--test"], tmp_path) == "node"
    assert runner_for(["make", "check"], tmp_path) is None


async def test_start_launches_exactly_the_authorized_argv(executor: Executor, monkeypatch: pytest.MonkeyPatch):
    """The policy layer authorizes resolve_argv's result; start must not resolve again differently."""
    request = call(["python", "-c", "print('same')"])
    authorized = executor.resolve_argv(request)
    monkeypatch.setattr(executor, "_search_path", "")  # a later lookup would now fail
    result = await executor.start(request).wait()
    assert result.status is ToolStatus.OK, result.output
    assert result.data["argv"] == authorized


async def test_start_host_refuses_relative_or_project_programs(executor: Executor, tmp_path: Path):
    marker = tmp_path / "PWNED"
    fake = _fake_program(tmp_path / "bin", "git", marker)
    for argv0 in ("git", str(fake)):
        result = await executor.start_host("g", "git_status", [argv0, "status"], 10).wait()
        assert result.status is ToolStatus.DENIED
    assert not marker.exists()


async def test_host_cancel_is_honest_about_detached_processes(executor: Executor):
    """#14: on the host backend a cancel cannot reach processes detached from the tree; the result says so."""
    job = executor.start(call(py("import time; time.sleep(30)")))
    await asyncio.sleep(0.3)
    await job.cancel()
    result = await job.wait()
    assert result.status is ToolStatus.CANCELLED
    assert "not tracked on the host backend" in result.output
    assert "docker" in result.data["remedy"]


def test_python_runner_is_chosen_from_static_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ex = Executor(tmp_path, SandboxConfig())
    monkeypatch.setattr(ex, "_container", lambda: False)
    try:
        monkeypatch.setattr("ironclad.tool_runtime.exec.find_program", lambda *a: None)
        (tmp_path / "test_a.py").write_text("", encoding="utf-8")
        assert ex.detect_test_argv()[1:] == ["-m", "unittest", "discover"]
        (tmp_path / "conftest.py").write_text("", encoding="utf-8")
        assert ex.detect_test_argv()[1:] == ["-m", "pytest", "-q"]
        (tmp_path / "conftest.py").unlink()
        (tmp_path / "test_a.py").unlink()
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_b.py").write_text("", encoding="utf-8")
        assert ex.detect_test_argv()[1:] == ["-m", "unittest", "discover", "-s", "tests", "-t", "."]
    finally:
        ex.close()
