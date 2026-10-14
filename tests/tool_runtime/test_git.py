from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ironclad.config import NedsterConfig
from ironclad.contracts import ToolCall, ToolStatus
from ironclad.tool_runtime.exec import Executor, executable_name, is_within
from ironclad.tool_runtime.git import SafeGit

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"})
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "Tester")
    _git(root, "config", "commit.gpgsign", "false")
    (root / "a.txt").write_text("one\n", encoding="utf-8")
    _git(root, "add", "a.txt")
    assert _git(root, "commit", "-q", "-m", "first").returncode == 0
    (root / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    (root / "new.txt").write_text("fresh\n", encoding="utf-8")
    return root


@pytest.fixture
def safe(repo: Path):
    git = SafeGit(repo)
    yield git
    git.executor.close()


def _evil(tmp_path: Path, name: str, passthrough: bool = False) -> tuple[str, Path]:
    marker = tmp_path / f"{name}.marker"
    script = tmp_path / f"{name}.sh"
    body = f'#!/bin/sh\necho hit >> "{marker.as_posix()}"\n'
    if passthrough:
        body += 'if [ -n "$1" ] && [ -f "$1" ]; then cat "$1"; else cat; fi\n'
    script.write_text(body, encoding="utf-8", newline="\n")
    script.chmod(0o755)
    return script.as_posix(), marker


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id="g1", name=name, arguments=arguments)


async def test_diff_status_log_show(safe: SafeGit):
    diff = await safe.run(call("git_diff"))
    assert diff.status is ToolStatus.OK, diff.output
    assert "+two" in diff.output
    assert diff.data["backend"] == "host"
    status = await safe.run(call("git_status"))
    assert " M a.txt" in status.output and "?? new.txt" in status.output
    log = await safe.run(call("git_log", max_count=5))
    assert "first" in log.output
    show = await safe.run(call("git_show", rev="HEAD", paths=["a.txt"]))
    assert show.status is ToolStatus.OK and "+one" in show.output


async def test_repo_local_diff_external_never_runs(repo: Path, safe: SafeGit, tmp_path: Path):
    script, marker = _evil(tmp_path, "external")
    _git(repo, "config", "diff.external", script)
    _git(repo, "diff")  # control: plain git really runs it
    assert marker.exists(), "control failed: plain git diff did not run diff.external"
    marker.unlink()
    for name, arguments in (("git_diff", {}), ("git_log", {"patch": True}), ("git_show", {"rev": "HEAD"})):
        result = await safe.run(call(name, **arguments))
        assert result.status is ToolStatus.OK, result.output
    assert not marker.exists()


async def test_textconv_and_diff_driver_never_run(repo: Path, safe: SafeGit, tmp_path: Path):
    textconv, textconv_marker = _evil(tmp_path, "textconv", passthrough=True)
    command, command_marker = _evil(tmp_path, "command")
    (repo / ".gitattributes").write_text("*.txt diff=evil\n", encoding="utf-8")
    _git(repo, "config", "diff.evil.textconv", textconv)
    _git(repo, "config", "diff.evil.command", command)
    _git(repo, "diff")
    assert command_marker.exists(), "control failed: diff driver command did not run"
    command_marker.unlink()
    textconv_marker.unlink(missing_ok=True)
    result = await safe.run(call("git_diff"))
    assert result.status is ToolStatus.OK and "+two" in result.output
    assert not command_marker.exists() and not textconv_marker.exists()


async def test_fsmonitor_never_runs(repo: Path, safe: SafeGit, tmp_path: Path):
    script, marker = _evil(tmp_path, "fsmonitor")
    _git(repo, "config", "core.fsmonitor", script)
    _git(repo, "status")
    if not marker.exists():
        pytest.skip("this git build does not run core.fsmonitor hooks")
    marker.unlink()
    result = await safe.run(call("git_status"))
    assert result.status is ToolStatus.OK, result.output
    assert not marker.exists()


async def test_repo_local_filter_driver_never_runs(repo: Path, safe: SafeGit, tmp_path: Path):
    script, marker = _evil(tmp_path, "clean", passthrough=True)
    (repo / ".gitattributes").write_text("*.txt filter=evil\n", encoding="utf-8")
    _git(repo, "config", "filter.evil.clean", script)
    _git(repo, "diff")
    assert marker.exists(), "control failed: clean filter did not run"
    marker.unlink()
    assert safe.untrusted_filters() == ["evil"]
    for name in ("git_diff", "git_status"):
        result = await safe.run(call(name))
        assert result.status is ToolStatus.OK, result.output
    assert not marker.exists()


async def test_hostile_environment_is_dropped(safe: SafeGit, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    script, marker = _evil(tmp_path, "envdiff")
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", script)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "nowhere"))
    env = safe.env()
    assert "GIT_EXTERNAL_DIFF" not in env and "GIT_DIR" not in env
    assert (
        env["GIT_CONFIG_NOSYSTEM"] == "1" and env["GIT_TERMINAL_PROMPT"] == "0" and env["GIT_PAGER"] == "cat"
    )
    result = await safe.run(call("git_diff"))
    assert result.status is ToolStatus.OK
    assert not marker.exists()


def test_argv_hardening(safe: SafeGit):
    argv = safe.argv_for(call("git_diff", paths=["a.txt"]))
    assert Path(argv[0]).is_absolute() and executable_name(argv[0]).lower() == "git"
    assert not is_within(argv[0], safe.project_root) and argv[1] == "--no-pager"
    joined = " ".join(argv)
    for flag in ("core.fsmonitor=false", "diff.external=", "core.pager=cat", "core.hooksPath=",
                 "--no-optional-locks", "--no-ext-diff", "--no-textconv", "log.showSignature=false"):  # fmt: skip
        assert flag in joined
    assert argv[-2:] == ["--", "a.txt"]
    log = safe.argv_for(call("git_log", patch=True))
    assert "--no-ext-diff" in log and "--no-textconv" in log and "-p" in log


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("git_diff", {"paths": ["../outside"]}),
        ("git_diff", {"paths": ["--output=x"]}),
        ("git_diff", {"paths": [":(attr:x)"]}),
        ("git_diff", {"paths": "a.txt"}),
        ("git_diff", {"staged": "yes"}),
        ("git_show", {"rev": "--output=/tmp/x"}),
        ("git_show", {"rev": "HEAD; rm -rf /"}),
        ("git_log", {"max_count": 0}),
        ("git_log", {"max_count": True}),
        ("git_push", {}),
    ],
)
async def test_invalid_arguments(safe: SafeGit, name: str, arguments: dict):
    result = await safe.run(call(name, **arguments))
    assert result.status is ToolStatus.INVALID


def test_specs_are_read_only(safe: SafeGit):
    specs = safe.specs()
    assert {spec.name for spec in specs} == {"git_diff", "git_status", "git_log", "git_show"}
    assert all(not spec.mutates and spec.risk == 0 for spec in specs)


# ---------------------------------------------------------------------------------------------
# FIX-EXEC regressions (#2): git is never taken from the project
# ---------------------------------------------------------------------------------------------


def _planted_git(repo: Path, marker: Path) -> Path:
    bindir = repo / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    bindir.mkdir(parents=True)
    if os.name == "nt":
        script = bindir / "git.bat"
        script.write_text(f'@echo off\r\necho owned> "{marker}"\r\n', encoding="utf-8")
    else:
        script = bindir / "git"
        script.write_text(f"#!/bin/sh\necho owned > '{marker}'\n", encoding="utf-8")
        script.chmod(0o755)
    return bindir


async def test_repo_supplied_venv_git_never_runs(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """repro_host_exec A: a project .venv git ran on the host for git_status, even with execution disabled."""
    marker = tmp_path / "PWNED_A"
    bindir = _planted_git(repo, marker)
    # Even when the user's shell has the project venv activated (first on PATH).
    monkeypatch.setenv("PATH", os.pathsep.join([str(bindir), os.environ["PATH"]]))
    executor = Executor(repo, NedsterConfig())  # default: no sandbox opt-in needed for read-only git
    try:
        safe = SafeGit(repo, executor)
        assert not is_within(safe.git_path(), repo)
        assert safe.untrusted_filters() == []  # runs git config through the absolute git too
        assert str(bindir) not in safe.env()["PATH"].split(os.pathsep)
        result = await safe.run(call("git_status"))
    finally:
        executor.close()
    assert result.status is ToolStatus.OK, result.output
    assert not marker.exists()


async def test_missing_git_fails_cleanly(repo: Path, monkeypatch: pytest.MonkeyPatch):
    executor = Executor(repo, NedsterConfig())
    monkeypatch.setattr(executor, "_git", None)
    try:
        result = await SafeGit(repo, executor).run(call("git_status"))
    finally:
        executor.close()
    assert result.status is ToolStatus.FAILED and "git was not found" in result.output


def test_hooks_dir_is_not_sandbox_writable(safe: SafeGit):
    """core.hooksPath must not point into the temp dir that sandboxed commands can write."""
    assert not is_within(safe.hooks_dir, safe.executor.temp_dir)
