from __future__ import annotations

import asyncio
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from ironclad.config import NedsterConfig, SandboxConfig
from ironclad.contracts import ToolCall, ToolStatus
from ironclad.tool_runtime import backends as backends_mod
from ironclad.tool_runtime.backends import BackendStatus, select_backend
from ironclad.tool_runtime.backends.bwrap import BwrapBackend, protected_mounts, protected_paths
from ironclad.tool_runtime.backends.docker import DockerBackend
from ironclad.tool_runtime.backends.host import HostBackend
from ironclad.tool_runtime.backends.seatbelt import SeatbeltBackend
from ironclad.tool_runtime.exec import IS_WINDOWS, Executor

TREE_SCRIPT = """
import os, subprocess, sys, time
depth, marker = int(sys.argv[1]), sys.argv[2]
if depth < 2:
    subprocess.Popen([sys.executable, __file__, str(depth + 1), marker])
with open(f"{marker}.{depth}", "w") as handle:
    handle.write(str(os.getpid()))
time.sleep(60)
"""


def _status(name: str, ready: bool) -> BackendStatus:
    return BackendStatus(name, ready, "full", "fake")


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch):
    """Control which backends report ready, without touching the real system."""
    ready: dict[str, bool] = {"bwrap": False, "seatbelt": False, "docker": False}
    for cls in (BwrapBackend, SeatbeltBackend, DockerBackend):
        monkeypatch.setattr(cls, "probe", lambda self: _status(self.name, ready[self.name]))
    return ready


def test_no_backend_and_no_opt_in_is_denied(probes, tmp_path: Path):
    selection = select_backend(tmp_path, tmp_path, SandboxConfig())
    assert selection.backend is None
    assert "allow_host_execution" in selection.remedy
    assert [status.name for status in selection.probes] == ["bwrap", "seatbelt", "docker"]


async def test_executor_denies_and_runs_nothing(probes, tmp_path: Path):
    marker = tmp_path / "ran.txt"
    executor = Executor(tmp_path, NedsterConfig())
    try:
        code = f"open({str(marker)!r}, 'w').write('x')"
        result = await executor.start(
            ToolCall("c1", "run_command", {"argv": [sys.executable, "-c", code]})
        ).wait()
    finally:
        executor.close()
    assert result.status is ToolStatus.DENIED
    assert "allow_host_execution" in result.output
    assert not marker.exists()


def test_auto_prefers_first_ready_in_order(probes, tmp_path: Path):
    probes["seatbelt"] = probes["docker"] = True
    assert select_backend(tmp_path, tmp_path, SandboxConfig()).backend.name == "seatbelt"
    probes["seatbelt"] = False
    assert select_backend(tmp_path, tmp_path, SandboxConfig()).backend.name == "docker"


def test_auto_uses_host_only_with_opt_in(probes, tmp_path: Path):
    selection = select_backend(tmp_path, tmp_path, SandboxConfig(allow_host_execution=True))
    assert selection.backend.name == "host"
    assert selection.status.isolation == "none"
    assert "no filesystem/network isolation" in selection.status.detail


def test_explicit_backend_never_falls_back(probes, tmp_path: Path):
    config = SandboxConfig(backend="docker", allow_host_execution=True)
    selection = select_backend(tmp_path, tmp_path, config)
    assert selection.backend is None
    assert "docker" in selection.remedy


def test_explicit_host_requires_opt_in(probes, tmp_path: Path):
    assert select_backend(tmp_path, tmp_path, SandboxConfig(backend="host")).backend is None
    config = SandboxConfig(backend="process", allow_host_execution=True)  # legacy alias
    assert select_backend(tmp_path, tmp_path, config).backend.name == "host"


def test_explicit_alias_and_unknown(probes, tmp_path: Path):
    probes["bwrap"] = True
    assert select_backend(tmp_path, tmp_path, SandboxConfig(backend="bubblewrap")).backend.name == "bwrap"
    assert select_backend(tmp_path, tmp_path, SandboxConfig(backend="firecracker")).backend is None


def test_protected_paths_follow_gitdir_file(tmp_path: Path):
    real_gitdir = tmp_path / "elsewhere" / "gitdir"
    real_gitdir.mkdir(parents=True)
    root = tmp_path / "proj"
    (root / ".nedster").mkdir(parents=True)
    (root / ".git").write_text(f"gitdir: {real_gitdir}\n", encoding="utf-8")
    found = protected_paths(root)
    assert root / ".nedster" in found and root / ".git" in found and real_gitdir.resolve() in found


def test_bwrap_wrap_layout(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    backend = BwrapBackend(tmp_path, tmp_path / "t")
    backend.binary = "/usr/bin/bwrap"
    argv = backend.wrap(["pytest", "-q"], backend.root, {})
    assert argv[0] == "/usr/bin/bwrap"
    assert {"--unshare-all", "--die-with-parent", "--new-session"} <= set(argv)
    assert "--share-net" not in argv
    root_bind = argv.index(str(backend.root))
    git_ro = argv.index(str(backend.root / ".git"))
    assert argv[git_ro - 1] == "--ro-bind" and git_ro > root_bind  # read-only re-bind comes later
    assert argv[-3:] == ["--", "pytest", "-q"]


def test_seatbelt_profile(tmp_path: Path):
    backend = SeatbeltBackend(tmp_path, tmp_path / "t")
    profile = backend.profile()
    assert "(deny default)" in profile and "(deny network*)" in profile
    assert profile.index("(allow file-write*") < profile.index("(deny file-write*")
    assert '"/.git"' in profile and '"/.nedster"' in profile
    assert str(tmp_path) not in profile  # paths travel as -D parameters
    argv = backend.wrap(["true"], tmp_path, {})
    assert argv[0] == "/usr/bin/sandbox-exec"
    assert f"ROOT={backend.root}" in argv and argv[-1] == "true"
    if sys.platform != "darwin":
        assert not backend.probe().ready


def test_docker_wrap_and_kill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / "sub").mkdir()
    backend = DockerBackend(tmp_path, "img:1")
    backend.binary = "docker"
    argv = backend.wrap(["pytest"], tmp_path / "sub", {"PATH": "C:/host", "LANG": "C.UTF-8", "TEMP": "x"})
    name = DockerBackend.container_name(argv)
    assert name and name.startswith("nedster-")
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--workdir") + 1] == "/workspace/sub"
    assert any(item.endswith("dst=/workspace/.git,readonly") for item in argv)
    assert "PATH" not in argv and "TEMP" not in argv and "LANG" in argv
    assert argv[-2:] == ["img:1", "pytest"]
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(backend, "_run", lambda *args, timeout=10: calls.append(args))
    backend.on_cancel(argv)
    assert calls == [("kill", name)]


def test_docker_not_installed_is_not_ready(tmp_path: Path):
    backend = DockerBackend(tmp_path)
    backend.binary = None
    assert not backend.probe().ready


class RecordingBackend(HostBackend):
    name = "fake-container"
    isolation = "full"

    def __init__(self) -> None:
        self.cancelled: list[list[str]] = []

    def on_cancel(self, wrapped: list[str]) -> None:
        self.cancelled.append(wrapped)


async def test_cancel_invokes_backend_cancel(tmp_path: Path):
    executor = Executor(tmp_path, SandboxConfig())
    backend = RecordingBackend()
    try:
        job = executor.launch("c1", "run_command", [sys.executable, "-c", "import time; time.sleep(30)"], 30,
                              backend, "full")  # fmt: skip
        await asyncio.sleep(0.2)
        await job.cancel()
        result = await job.wait()
    finally:
        executor.close()
    assert result.status is ToolStatus.CANCELLED
    assert len(backend.cancelled) == 1 and backend.cancelled[0][-1].endswith("time.sleep(30)")


def test_select_backend_module_exports():
    assert backends_mod.AUTO_ORDER == ("bwrap", "seatbelt", "docker")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="bubblewrap is Linux only")
async def test_bwrap_denies_network_and_git_writes(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    executor = Executor(tmp_path, SandboxConfig(backend="bwrap"))
    try:
        if executor.selection().backend is None:
            pytest.skip("bwrap not usable here")
        code = (
            "import socket, pathlib\n"
            "pathlib.Path('ok.txt').write_text('x')\n"
            "try:\n    pathlib.Path('.git/evil').write_text('x'); print('GIT-WRITE')\nexcept OSError: pass\n"
            "try:\n    socket.create_connection(('1.1.1.1', 53), 2); print('NET')\nexcept OSError: pass\n"
        )
        result = await executor.start(ToolCall("c", "run_command", {"argv": ["python3", "-c", code]})).wait()
    finally:
        executor.close()
    assert result.status is ToolStatus.OK, result.output
    assert (tmp_path / "ok.txt").exists()
    assert "GIT-WRITE" not in result.output and "NET" not in result.output


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS only")
async def test_seatbelt_denies_git_writes(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    executor = Executor(tmp_path, SandboxConfig(backend="seatbelt"))
    try:
        if executor.selection().backend is None:
            pytest.skip("sandbox-exec not usable here")
        code = (
            "import pathlib\npathlib.Path('ok.txt').write_text('x')\n"
            "try:\n    pathlib.Path('.git/evil').write_text('x'); print('GIT-WRITE')\nexcept OSError: pass\n"
        )
        result = await executor.start(ToolCall("c", "run_command", {"argv": ["python3", "-c", code]})).wait()
    finally:
        executor.close()
    assert (tmp_path / "ok.txt").exists(), result.output
    assert "GIT-WRITE" not in result.output


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows Job Objects")
def test_winjob_structure_layouts():
    from ironclad.tool_runtime.backends import winjob

    pointer = ctypes.sizeof(ctypes.c_void_p)
    expected = (64, 144) if pointer == 8 else (48, 112)
    assert ctypes.sizeof(winjob.JOBOBJECT_BASIC_LIMIT_INFORMATION) == expected[0]
    assert ctypes.sizeof(winjob.JOBOBJECT_EXTENDED_LIMIT_INFORMATION) == expected[1]
    assert ctypes.sizeof(winjob.JOBOBJECT_BASIC_ACCOUNTING_INFORMATION) == 48
    assert winjob.probe().ready


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows Job Objects")
def test_winjob_kills_the_tree(tmp_path: Path):
    from ironclad.tool_runtime.backends.winjob import WinJob, resume_process

    script = tmp_path / "tree.py"
    script.write_text(TREE_SCRIPT, encoding="utf-8")
    marker = tmp_path / "pid"
    python = getattr(sys, "_base_executable", sys.executable)  # skip the venv launcher's own job
    job = WinJob()
    proc = subprocess.Popen([python, str(script), "0", str(marker)], creationflags=0x00000004 | 0x00000200)
    try:
        job.assign(proc)
        resume_process(proc)
        files = [Path(f"{marker}.{depth}") for depth in range(3)]
        deadline = time.monotonic() + 15
        while not all(item.exists() and item.read_text() for item in files):
            assert time.monotonic() < deadline, "tree did not start"
            time.sleep(0.05)
        pids = [int(item.read_text()) for item in files]
        assert job.active_processes() >= 3
        members = [psutil.Process(pid) for pid in pids]
        started = time.monotonic()
        assert job.terminate()
        assert time.monotonic() - started < 3
        _gone, alive = psutil.wait_procs(members, timeout=1)  # waits on the process object itself
        assert not alive
        assert job.active_processes() == 0
    finally:
        job.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------------------------
# FIX-EXEC regressions (#5, #11, #14, #19)
# ---------------------------------------------------------------------------------------------


def _ro_binds(argv: list[str]) -> dict[str, str]:
    """target -> source for every ``--ro-bind source target`` in a bwrap argv."""
    return {argv[i + 2]: argv[i + 1] for i, item in enumerate(argv[:-2]) if item == "--ro-bind"}


def test_bwrap_protects_missing_git_and_nedster(tmp_path: Path):
    """#5: protected_paths() skipped missing dirs, so a sandboxed command could create .nedster/config.toml."""
    backend = BwrapBackend(tmp_path, tmp_path / "t")
    backend.binary = "/usr/bin/bwrap"
    argv = backend.wrap(["true"], backend.root, {})
    binds = _ro_binds(argv)
    git, nedster = str(backend.root / ".git"), str(backend.root / ".nedster")
    assert git in binds and nedster in binds
    assert binds[nedster] == nedster and (backend.root / ".nedster").is_dir()  # pre-created, then read-only
    empty = Path(binds[git])
    assert empty.is_dir() and not any(empty.iterdir()) and not empty.is_relative_to(backend.root)
    assert not empty.is_relative_to(backend.temp)  # not reachable through the writable temp bind
    root_bind = argv.index("--bind")
    assert all(argv.index(target) > root_bind for target in (git, nedster))  # ro re-binds come later


def test_protected_mounts_keep_existing_and_refuse_dangling_links(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".nedster").write_text("not a dir", encoding="utf-8")
    empty = tmp_path / "empty"
    mounts = protected_mounts(tmp_path, empty)
    assert (tmp_path / ".git", tmp_path / ".git") in mounts
    assert (tmp_path / ".nedster", tmp_path / ".nedster") in mounts  # a file is protected too
    link_root = tmp_path / "linked"
    link_root.mkdir()
    try:
        (link_root / ".nedster").symlink_to(link_root / "sub" / "x")
    except OSError:
        pytest.skip("symlinks are not available")
    with pytest.raises(ValueError, match="dangling"):
        protected_mounts(link_root, empty)


def test_docker_protects_missing_git_and_nedster(tmp_path: Path):
    """#5 for Docker: .git and .nedster are read-only mounts even when they did not exist."""
    backend = DockerBackend(tmp_path, "img:1")
    backend.binary = "docker"
    argv = backend.wrap(["pytest"], tmp_path, {})
    mounts = [argv[i + 1] for i, item in enumerate(argv[:-1]) if item == "--mount"]
    assert any(item.endswith("dst=/workspace/.git,readonly") for item in mounts)
    assert any(item.endswith("dst=/workspace/.nedster,readonly") for item in mounts)
    assert (tmp_path / ".nedster").is_dir()


@pytest.mark.parametrize("bad", ["x,readonly=false,dst=/workspace/.git", 'x"y'])
def test_docker_refuses_mount_injection(tmp_path: Path, bad: str):
    """#19: a project path with ',' could append --mount options (e.g. undo readonly)."""
    if os.name == "nt" and '"' in bad:
        pytest.skip('Windows paths cannot contain "')
    root = tmp_path / bad
    root.mkdir(parents=True)
    backend = DockerBackend(root, "img:1")
    backend.binary = "docker"
    with pytest.raises(ValueError, match="--mount"):
        backend.wrap(["pytest"], backend.root, {})
    status = backend.probe()
    assert not status.ready and "--mount" in status.detail


def test_docker_user_and_resource_limits(tmp_path: Path):
    """#19: no root-owned files on Linux hosts; memory/CPU capped; limits configurable."""
    backend = DockerBackend(tmp_path, "img:1")
    backend.binary = "docker"
    argv = backend.wrap(["pytest"], tmp_path, {})
    assert argv[argv.index("--memory") + 1] == "4g" and argv[argv.index("--cpus") + 1] == "2"
    if hasattr(os, "getuid"):
        assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    else:
        assert "--user" not in argv
    settings = SimpleNamespace(docker_image="img:2", docker_memory="1g", docker_cpus="0.5")
    configured = backends_mod.build_backend("docker", tmp_path, tmp_path, settings)
    configured.binary = "docker"
    argv = configured.wrap(["pytest"], tmp_path, {})
    assert argv[argv.index("--memory") + 1] == "1g" and argv[argv.index("--cpus") + 1] == "0.5"


async def test_launch_with_unsafe_docker_path_is_invalid_not_a_crash(tmp_path: Path):
    root = tmp_path / "a,b"
    root.mkdir()
    backend = DockerBackend(root, "img:1")
    backend.binary = "docker"
    executor = Executor(root, SandboxConfig())
    try:
        result = await executor.launch("c1", "run_command", ["pytest"], 10, backend, "full").wait()
    finally:
        executor.close()
    assert result.status is ToolStatus.INVALID and "--mount" in result.output


def test_seatbelt_limits_mach_lookup_and_home_reads(tmp_path: Path):
    """#11: (allow mach-lookup) reached LaunchServices; $HOME (~/.git-credentials, ~/.npmrc, ...) was readable."""
    profile = SeatbeltBackend(tmp_path, tmp_path / "t").profile()
    lines = profile.splitlines()
    assert "(allow mach-lookup)" not in lines
    mach = next(line for line in lines if line.startswith("(allow mach-lookup"))
    assert "(global-name " in mach and "launchservices" not in mach.lower() and "coreservices" not in mach.lower()
    deny_home = lines.index('(deny file-read* (subpath (param "HOME")))')
    allow_back = next(i for i, line in enumerate(lines) if line.startswith('(allow file-read* (subpath (param "ROOT"))'))
    sensitive = next(i for i, line in enumerate(lines) if line.startswith("(deny file-read*") and "/.ssh" in line)
    assert lines.index("(allow file-read*)") < deny_home < allow_back < sensitive  # later rules win
    assert '(param "TMP")' in lines[allow_back] and '"/.git-credentials"' not in lines[allow_back]


def test_host_probe_is_honest_about_untracked_processes():
    """#14: the host backend cannot contain processes detached from the tree (WMI, launchd, ...)."""
    detail = HostBackend().probe().detail
    assert "no filesystem/network isolation" in detail and "not tracked" in detail
