from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ironclad.contracts import ToolCall, ToolStatus
from ironclad.tool_runtime.backends import probe_env, resolve_binary
from ironclad.tool_runtime.backends.docker import DockerBackend
from ironclad.tool_runtime.files import FileTools
from ironclad.tool_runtime.search import SearchTools

WINDOWS = os.name == "nt"


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    return project


def link_dir(link: Path, target: Path) -> None:
    """Symlink, or a junction on Windows without symlink rights."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        if not WINDOWS:
            pytest.skip("symlinks unavailable")
        done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False)
        if done.returncode != 0:
            pytest.skip("cannot create a junction")


# -- #1 backend binaries -------------------------------------------------------------------


def plant(directory: Path, name: str) -> Path:
    directory.mkdir(exist_ok=True)
    fake = directory / (name + (".exe" if WINDOWS else ""))
    fake.write_bytes(b"MZ" if WINDOWS else b"#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    return fake


@pytest.mark.parametrize("name", ["docker", "bwrap", "bubblewrap"])
def test_project_path_backend_binary_is_never_resolved(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
):
    fake_dir = root / "bin"
    plant(fake_dir, name)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", os.pathsep.join([str(fake_dir), str(empty)]))
    monkeypatch.setenv("PATHEXT", ".EXE;.CMD")
    assert resolve_binary(root, name) is None


def test_relative_path_entry_is_ignored(root: Path, monkeypatch: pytest.MonkeyPatch):
    plant(root / "bin", "docker")
    monkeypatch.chdir(root)
    monkeypatch.setenv("PATH", "bin")
    assert resolve_binary(root, "docker") is None


def test_trusted_binary_outside_project_is_resolved(root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trusted = tmp_path / "trusted"
    fake = plant(trusted, "docker")
    monkeypatch.setenv("PATH", str(trusted))
    monkeypatch.setenv("PATHEXT", ".EXE;.CMD")
    assert resolve_binary(root, "docker") == str(fake)


def test_docker_backend_does_not_pick_planted_binary(root: Path, monkeypatch: pytest.MonkeyPatch):
    plant(root / "bin", "docker")
    monkeypatch.setenv("PATH", str(root / "bin"))
    backend = DockerBackend(root)
    assert backend.binary is None
    assert not backend.probe().ready


def test_docker_probe_uses_allowlisted_env(root: Path, monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 1, b"", b"")

    monkeypatch.setenv("NEDSTER_SECRET_TOKEN", "hunter2")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = DockerBackend(root)
    backend.binary = "docker"
    backend.probe()
    env = seen["env"]
    assert isinstance(env, dict)
    assert "OPENAI_API_KEY" not in env and "NEDSTER_SECRET_TOKEN" not in env


def test_probe_env_drops_project_path_entries(root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", os.pathsep.join([str(root / "bin"), str(Path(sys.executable).parent)]))
    assert str(root) not in probe_env(root)["PATH"]


# -- #4 protected paths --------------------------------------------------------------------


def write(tools: FileTools, path: str, content: str = "x"):
    return tools.run(call("write_file", path=path, content=content))


@pytest.fixture
def tools(root: Path) -> FileTools:
    return FileTools(root)


def test_plain_git_dir_still_protected(root: Path, tools: FileTools):
    (root / ".git").mkdir()
    assert write(tools, ".git/config").status is ToolStatus.INVALID


def test_lexical_protected_component_through_link_denied(root: Path, tools: FileTools):
    real = root / "metadata"
    real.mkdir()
    link_dir(root / ".git", real)
    via_name = write(tools, ".git/hooks-pre-commit")
    assert via_name.status is ToolStatus.INVALID
    assert not (real / "hooks-pre-commit").exists()


def test_direct_target_of_protected_link_denied(root: Path, tools: FileTools):
    real = root / "metadata"
    real.mkdir()
    link_dir(root / ".git", real)
    direct = write(tools, "metadata/config")
    assert direct.status is ToolStatus.INVALID
    assert "protected" in direct.output
    assert not (real / "config").exists()


def test_nedster_link_target_denied(root: Path, tools: FileTools):
    real = root / "cfg"
    real.mkdir()
    link_dir(root / ".nedster", real)
    assert write(tools, "cfg/config.toml").status is ToolStatus.INVALID
    assert write(tools, ".nedster/config.toml").status is ToolStatus.INVALID


def test_gitdir_file_target_denied_by_both_spellings(root: Path, tools: FileTools):
    gitdir = root / "meta" / "repo"
    gitdir.mkdir(parents=True)
    (root / ".git").write_text("gitdir: meta/repo\n", encoding="utf-8")
    assert write(tools, "meta/repo/config").status is ToolStatus.INVALID
    assert write(tools, ".git").status is ToolStatus.INVALID
    assert not (gitdir / "config").exists()
    assert write(tools, "meta/other.txt").status is ToolStatus.OK


def test_gitdir_file_absolute_target_denied(root: Path, tools: FileTools):
    gitdir = root / "abs"
    gitdir.mkdir()
    (root / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    assert write(tools, str(gitdir / "HEAD")).status is ToolStatus.INVALID


def test_dotdot_spelling_through_protected_name_denied(root: Path, tools: FileTools):
    (root / ".git").mkdir()
    (root / "src").mkdir()
    assert write(tools, ".git/../src/a.txt").status is ToolStatus.INVALID
    assert write(tools, "src/../.git/config").status is ToolStatus.INVALID


def test_absolute_lexical_spelling_denied(root: Path, tools: FileTools):
    real = root / "metadata"
    real.mkdir()
    link_dir(root / ".git", real)
    assert write(tools, str(root / ".git" / "config")).status is ToolStatus.INVALID


def test_ordinary_writes_still_work(root: Path, tools: FileTools):
    (root / ".git").mkdir()
    assert write(tools, "src/app.py").status is ToolStatus.OK


# -- #5 search fallback --------------------------------------------------------------------


@pytest.fixture
def python_search(root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("ironclad.tool_runtime.search.find_ripgrep", lambda _root: None)

    def make(entries: list[str]) -> SearchTools:
        return SearchTools(root, inventory=lambda _base: entries)

    return make


def test_search_fallback_does_not_follow_file_link_outside_root(root: Path, tmp_path: Path, python_search):
    outside = tmp_path / "outside.txt"
    outside.write_text("needle secret\n", encoding="utf-8")
    try:
        os.symlink(outside, root / "leak.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    (root / "ok.txt").write_text("needle fine\n", encoding="utf-8")
    result = python_search(["leak.txt", "ok.txt"]).run(call("search_text", pattern="needle"))
    assert result.status is ToolStatus.OK
    assert "secret" not in result.output
    assert "ok.txt:1" in result.output


def test_search_fallback_skips_parent_link_escape(root: Path, tmp_path: Path, python_search):
    outside = tmp_path / "outdir"
    outside.mkdir()
    (outside / "f.txt").write_text("needle outside\n", encoding="utf-8")
    link_dir(root / "linked", outside)
    result = python_search(["linked/f.txt"]).run(call("search_text", pattern="needle"))
    assert "outside" not in result.output


def test_search_fallback_rejects_traversal_entries(root: Path, tmp_path: Path, python_search):
    (tmp_path / "up.txt").write_text("needle up\n", encoding="utf-8")
    result = python_search(["../up.txt"]).run(call("search_text", pattern="needle"))
    assert "up.txt" not in result.output


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
def test_search_fallback_never_opens_fifo(root: Path, python_search):
    os.mkfifo(root / "pipe")
    result = python_search(["pipe"]).run(call("search_text", pattern="needle"))  # would block if opened
    assert result.status is ToolStatus.OK


def test_search_fallback_skips_directory_entry_and_oversized(root: Path, python_search, monkeypatch):
    (root / "d").mkdir()
    (root / "big.txt").write_text("needle\n" * 10, encoding="utf-8")
    monkeypatch.setattr("ironclad.tool_runtime.search.MAX_SEARCH_FILE_BYTES", 5)
    result = python_search(["d", "big.txt"]).run(call("search_text", pattern="needle"))
    assert result.status is ToolStatus.OK
    assert "big.txt" not in result.output


# -- #10 write_file no-op ------------------------------------------------------------------


class CountingJournal:
    def __init__(self) -> None:
        import threading

        self.mutation_lock = threading.RLock()
        self.recorded: list[str] = []

    def record_pre_image(self, rel: str, path: Path) -> None:
        self.recorded.append(rel)


def test_identical_write_is_ok_noop(root: Path):
    journal = CountingJournal()
    tools = FileTools(root, journal=journal)
    target = root / "a.txt"
    target.write_bytes(b"same\n")
    sha = tools.run(call("read_file", path="a.txt")).data["sha256"]
    before = target.stat().st_mtime_ns
    result = tools.run(call("write_file", path="a.txt", content="same\n", expected_sha=sha))
    assert result.status is ToolStatus.OK
    assert not result.mutated_paths
    assert "no change" in result.output
    assert journal.recorded == []
    assert target.read_bytes() == b"same\n"
    assert target.stat().st_mtime_ns == before


def test_identical_write_respects_crlf_normalisation(root: Path):
    tools = FileTools(root)
    (root / "w.txt").write_bytes(b"a\r\nb\r\n")
    sha = tools.run(call("read_file", path="w.txt")).data["sha256"]
    result = tools.run(call("write_file", path="w.txt", content="a\nb\n", expected_sha=sha))
    assert result.status is ToolStatus.OK and not result.mutated_paths


def test_changed_write_still_mutates_and_journals(root: Path):
    journal = CountingJournal()
    tools = FileTools(root, journal=journal)
    (root / "a.txt").write_text("one", encoding="utf-8")
    sha = tools.run(call("read_file", path="a.txt")).data["sha256"]
    result = tools.run(call("write_file", path="a.txt", content="two", expected_sha=sha))
    assert result.mutated_paths == ("a.txt",)
    assert journal.recorded == ["a.txt"]


def test_identical_write_with_stale_sha_still_refused(root: Path):
    tools = FileTools(root)
    (root / "a.txt").write_text("one", encoding="utf-8")
    result = tools.run(call("write_file", path="a.txt", content="one", expected_sha="0" * 64))
    assert result.status is ToolStatus.INVALID
