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
    batch_command_line,
    build_env,
    parse_tests_run,
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


def test_build_env_private_temp_secrets_and_venv(tmp_path: Path):
    (tmp_path / ".venv" / ("Scripts" if IS_WINDOWS else "bin")).mkdir(parents=True)
    temp = tmp_path / "tmp"
    base = {"PATH": "/usr/bin", "HOME": "/h", "LC_ALL": "C", "LC_API_TOKEN": "x", "GITHUB_TOKEN": "y",
            "SYSTEMROOT": r"C:\Windows", "AWS_SECRET_ACCESS_KEY": "z"}  # fmt: skip
    env = build_env(tmp_path, temp, base)
    assert not any(marker in key for key in env for marker in ("TOKEN", "SECRET", "KEY"))
    assert env["PATH"].split(os.pathsep)[0].endswith("Scripts" if IS_WINDOWS else "bin")
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
async def test_bat_argument_with_ampersand_is_invalid(executor: Executor, tmp_path: Path):
    (tmp_path / "echoer.bat").write_text("@echo off\r\necho ARG=%~1\r\n", encoding="utf-8")
    bad = await executor.start(call(["./echoer.bat", "x&whoami"])).wait()
    assert bad.status is ToolStatus.INVALID
    good = await executor.start(call(["./echoer.bat", "hello world"])).wait()
    assert good.status is ToolStatus.OK, good.output
    assert "ARG=hello world" in good.output


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("....\n5 passed in 0.12s\n", 5),
        ("===== 1 failed, 4 passed, 2 warnings in 1.0s =====", 5),
        ("== 2 passed, 1 error in 0.3s ==", 3),
        ("no tests ran in 0.01s", 0),
        (
            "test result: ok. 3 passed; 0 failed; 1 ignored\ntest result: FAILED. 1 passed; 2 failed; 0 ignored",
            6,
        ),
        ("Tests:       1 failed, 5 passed, 6 total", 6),
        ("      Tests  1 failed | 5 passed (6)", 6),
        ("Ran 4 tests in 0.002s\n\nOK", 4),
        ("=== RUN   TestA\n--- PASS: TestA (0.00s)\n=== RUN   TestB\n--- FAIL: TestB (0.00s)\nFAIL", 2),
        ("ok  \texample.com/pkg\t0.01s", None),
        ("hello world", None),
        ("", None),
    ],
)
def test_parse_tests_run(text: str, expected: int | None):
    assert parse_tests_run(text) == expected


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
        argv = ex.resolve_argv(ToolCall("c", "run_tests", {}))
        assert argv[1:] == ["-m", "pytest", "-q"]
        venv = tmp_path / ".venv" / ("Scripts" if IS_WINDOWS else "bin")
        venv.mkdir(parents=True)
        (venv / ("python.exe" if IS_WINDOWS else "python")).write_text("", encoding="utf-8")
        assert ex.resolve_argv(ToolCall("c", "run_tests", {}))[0].startswith(str(venv))
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        assert ex.resolve_argv(ToolCall("c", "run_tests", {})) == ["npm", "test"]
        (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
        assert ex.resolve_argv(ToolCall("c", "run_tests", {})) == ["go", "test", "./..."]
        (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        assert ex.resolve_argv(ToolCall("c", "run_tests", {})) == ["cargo", "test"]
        explicit = ex.resolve_argv(ToolCall("c", "run_tests", {"argv": ["make", "check"]}))
        assert explicit == ["make", "check"]
    finally:
        ex.close()


async def test_run_tests_reports_tests_run(executor: Executor, tmp_path: Path):
    (tmp_path / "test_sample.py").write_text("def test_a():\n    pass\n\ndef test_b():\n    pass\n", "utf-8")
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_sample.py"]
    result = await executor.start(call(argv, name="run_tests")).wait()
    assert result.status is ToolStatus.OK, result.output
    assert result.data["tests_run"] == 2


async def test_run_tests_unknown_count_is_none(executor: Executor):
    result = await executor.start(call(py("print('done')"), name="run_tests")).wait()
    assert result.data["tests_run"] is None


def test_specs_shape(executor: Executor):
    specs = {spec.name: spec for spec in executor.specs()}
    assert set(specs) == {"run_command", "run_tests"}
    assert specs["run_command"].parameters["required"] == ["argv"]
    assert specs["run_command"].mutates
