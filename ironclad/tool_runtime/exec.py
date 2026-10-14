"""Cancellable, bounded, cross-platform command execution (run_command / run_tests).

Invariants:
- argv only; never a shell. Windows .cmd/.bat go through ``cmd.exe /d /s /c`` only after
  rejecting cmd metacharacters in every argument.
- No execution without a ready backend or an explicit host opt-in (DENIED + remedy).
- Programs are resolved against the user's PATH with every directory inside the project
  removed; a program that resolves inside the project is DENIED. The only exception is the
  project's verified venv interpreter, and only for ``run_tests``. Policy sees the resolved
  argv and the launch runs exactly that argv.
- ``run_tests`` runs only known test runners; test counts are parsed with that runner's own
  format (last summary line). ``data["runner"]`` is set only for a known runner invocation.
- ``Job.cancel()`` returns after the tracked process tree is gone (POSIX process group,
  Windows Job Object, Docker container). On the host backend a process deliberately detached
  from the tree (WMI, launchd, systemd-run, ...) is not tracked; results say so.
- Output is bounded to head 32 KB + tail 32 KB; the total byte count is always reported.
- Unknown test counts are ``None``, never 0.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.machinery
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any

import psutil

from ironclad.contracts import ToolCall, ToolResult, ToolSpec, ToolStatus
from ironclad.types import RiskLevel

from .backends import Backend, BackendStatus, Selection, sandbox_settings, select_backend
from .backends.host import HostBackend

HEAD_BYTES = 32 * 1024
TAIL_BYTES = 32 * 1024
KILL_GRACE_S = 2.0
READER_JOIN_S = 2.0
POLL_S = 0.02
TRACK_INTERVAL_S = 0.25
DEFAULT_TIMEOUT_S = 120.0

IS_WINDOWS = os.name == "nt"
POSIX_ENV = ("PATH", "HOME", "LANG", "TERM")
WINDOWS_ENV = (
    "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
    "PROGRAMDATA", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
)  # fmt: skip
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")
BATCH_UNSAFE = frozenset('"%!^&|<>\r\n')
WINDOWS_DIRECT = {".exe", ".com"}
WINDOWS_BATCH = {".bat", ".cmd"}
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
CREATE_SUSPENDED = 0x00000004
WINDOWS_SUFFIXES = (".exe", ".cmd", ".bat", ".com")
EXECUTABLE_MAGIC = (b"MZ", b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe",
                    b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce")  # fmt: skip
PYTHON_NAME = re.compile(r"^python(?:\d+(?:\.\d+)?)?$")
KNOWN_RUNNERS = ("pytest", "python -m pytest", "python -m unittest", "npm test", "cargo test", "go test",
                 "node --test")  # fmt: skip
HOST_UNTRACKED = (
    "processes detached from the tree (e.g. started through WMI, launchd, systemd-run or a service) "
    "are not tracked on the host backend"
)
HOST_REMEDY = "use the docker backend (or bwrap/seatbelt) for full containment"
RESOLUTION_CACHE = 64


class ArgumentError(ValueError):
    """Arguments rejected before anything executes (ToolStatus.INVALID)."""


class ResolutionDenied(Exception):  # noqa: N818 - mirrors ToolStatus.DENIED
    """The resolved program may not run, e.g. it lives inside the project (ToolStatus.DENIED)."""


# ---------------------------------------------------------------------------
# Environment and program resolution
# ---------------------------------------------------------------------------


def _running(proc: psutil.Process) -> bool:
    try:
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, OSError):
        return False


def _is_secret(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def executable_name(program: str) -> str:
    """Basename without a Windows launcher suffix; splits both separators on every OS."""
    name = PurePath(program.replace("\\", "/")).name
    lowered = name.lower()
    for suffix in WINDOWS_SUFFIXES:
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _path_key(path: str | Path, *, real: bool) -> str:
    text = os.path.realpath(path) if real else os.path.abspath(path)
    text = os.path.normcase(text)
    return text.casefold() if IS_WINDOWS or sys.platform == "darwin" else text  # case-insensitive filesystems


def is_within(path: str | Path, root: str | Path) -> bool:
    """True when ``path`` is ``root`` or below it, by its literal location or after following links."""
    for real in (False, True):
        key, base = _path_key(path, real=real), _path_key(root, real=True)
        if key == base or key.startswith(base.rstrip(os.sep) + os.sep):
            return True
    return False


def is_path_qualified(program: str) -> bool:
    return "/" in program or "\\" in program or os.path.isabs(program)


def host_search_path(root: Path, path_value: str) -> str:
    """The user's PATH without relative entries and without any directory inside the project."""
    kept: list[str] = []
    for raw in path_value.split(os.pathsep):
        entry = raw.strip().strip('"')
        if entry and os.path.isabs(entry) and not is_within(entry, root):
            kept.append(entry)
    return os.pathsep.join(dict.fromkeys(kept))


def _pathext() -> list[str]:
    return [ext.lower() for ext in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";") if ext]


def _executable_file(path: Path) -> bool:
    if not os.path.exists(path) or os.path.isdir(path):
        return False
    return IS_WINDOWS or os.access(path, os.X_OK)


def find_program(name: str, search_path: str, root: Path) -> Path | None:
    """Look ``name`` up in ``search_path`` only (never the current directory); never a project file."""
    candidates = [name]
    if IS_WINDOWS and PurePath(name).suffix.lower() not in _pathext():
        candidates = [name + ext for ext in _pathext()]
    for directory in search_path.split(os.pathsep):
        if not directory:
            continue
        for candidate in candidates:
            path = Path(directory) / candidate
            if _executable_file(path) and not is_within(path, root):
                return Path(os.path.abspath(path))
    return None


def venv_python(root: Path) -> Path | None:
    """The project's ``.venv`` interpreter, only when it is a structurally real venv interpreter.

    Without executing anything: ``pyvenv.cfg`` must name a ``home`` outside the project that
    holds a Python, and the interpreter must be a symlink that leaves the project or a native
    executable image. A repo-supplied script named ``python`` fails these checks.
    """
    venv = root / ".venv"
    interpreter = venv / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    try:
        lines = (venv / "pyvenv.cfg").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    home = next((line.split("=", 1)[1].strip() for line in lines
                 if line.split("=", 1)[0].strip().lower() == "home" and "=" in line), "")  # fmt: skip
    if not home or not os.path.isabs(home) or not os.path.isdir(home) or is_within(home, root):
        return None
    names = ("python.exe", "python3.exe") if IS_WINDOWS else ("python3", "python")
    if not any(os.path.exists(Path(home) / name) for name in names):
        return None
    try:
        if interpreter.is_symlink():
            target = Path(os.path.realpath(interpreter))
            return interpreter if target.is_file() and not is_within(target, root) else None
        if not interpreter.is_file():
            return None
        with interpreter.open("rb") as handle:
            magic = handle.read(4)
    except OSError:
        return None
    return interpreter if magic.startswith(EXECUTABLE_MAGIC) else None


def _shadowed_module(root: Path, module: str) -> bool:
    """``python -m <module>`` from the project root would import a project file instead."""
    suffixes = importlib.machinery.all_suffixes()
    return any((root / f"{module}{suffix}").exists() for suffix in suffixes) or any(
        (root / module / f"__init__{suffix}").exists() for suffix in suffixes
    )


def runner_for(argv: list[str], root: Path) -> str | None:
    """Name of the known test runner that ``argv`` invokes, or ``None``."""
    if not argv:
        return None
    name, rest = executable_name(argv[0]).lower(), argv[1:]
    if name in ("pytest", "py.test"):
        return "pytest"
    if PYTHON_NAME.match(name) and len(rest) >= 2 and rest[0] == "-m" and rest[1] in ("pytest", "unittest"):
        return None if _shadowed_module(root, rest[1]) else rest[1]
    if name == "npm" and rest[:1] in (["test"], ["t"]):
        return "npm"
    if name in ("cargo", "go") and rest[:1] == ["test"]:
        return name
    if name == "node" and "--test" in rest:
        return "node"
    return None


def build_env(
    root: Path, temp: Path, base: Mapping[str, str] | None = None, *, venv: Path | None = None
) -> dict[str, str]:
    """Per-OS allowlisted environment with a private temp dir; secrets never pass.

    PATH never contains a directory inside the project. ``venv`` (a verified venv interpreter)
    puts that venv's bin directory first, for test runs that use it.
    """
    source = dict(os.environ if base is None else base)
    if IS_WINDOWS:
        upper = {key.upper(): value for key, value in source.items()}
        env = {key: upper[key] for key in WINDOWS_ENV if key in upper}
        env["TEMP"] = env["TMP"] = str(temp)
    else:
        env = {key: source[key] for key in POSIX_ENV if key in source}
        env.update({key: value for key, value in source.items() if key.startswith("LC_")})
        env["TMPDIR"] = str(temp)
    env = {key: value for key, value in env.items() if not _is_secret(key)}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PATH"] = host_search_path(root, env.get("PATH", ""))
    if venv is not None:
        env["PATH"] = os.pathsep.join(item for item in (str(venv.parent), env["PATH"]) if item)
        env["VIRTUAL_ENV"] = str(venv.parent.parent)
    return env


def _which(name: str, env: Mapping[str, str], cwd: Path) -> Path | None:
    if is_path_qualified(name):
        # abspath, not resolve(): a venv interpreter symlink must keep its venv location.
        base = Path(os.path.abspath(name if os.path.isabs(name) else cwd / name))
        if base.is_file():
            return base
        if IS_WINDOWS and not base.suffix:
            for ext in _pathext():
                if base.with_suffix(ext).is_file():
                    return base.with_suffix(ext)
        return None
    return find_program(name, env.get("PATH", ""), cwd)


def _cmd_quote(arg: str) -> str:
    return f'"{arg}"' if not arg or any(ch in arg for ch in " \t,;=()") else arg


def batch_command_line(comspec: str, script: Path, args: list[str]) -> str:
    """``cmd.exe /d /s /c ""script" args"`` with every metacharacter rejected up front."""
    for value in (str(script), *args):
        bad = sorted(set(value) & BATCH_UNSAFE)
        if bad:
            raise ArgumentError(
                f"argument {value!r} contains characters cmd.exe would interpret ({''.join(bad)!r}); "
                "they are not allowed for .bat/.cmd programs"
            )
    inner = " ".join([_cmd_quote(str(script)), *(_cmd_quote(arg) for arg in args)])
    return f'"{comspec}" /d /s /c "{inner}"'


def _comspec(env: Mapping[str, str]) -> str:
    comspec = env.get("COMSPEC", "")
    if comspec and Path(comspec).is_absolute():
        return comspec
    return str(Path(env.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "cmd.exe")


def resolve_program(argv: list[str], cwd: Path, env: Mapping[str, str]) -> list[str] | str:
    """Return Popen args: an absolute-program argv, or (Windows batch only) a command line."""
    program = _which(argv[0], env, cwd)
    if program is None:
        raise ArgumentError(f"program not found on the sandbox PATH: {argv[0]!r}")
    if not IS_WINDOWS:
        return [str(program), *argv[1:]]
    suffix = program.suffix.lower()
    if suffix in WINDOWS_BATCH:
        return batch_command_line(_comspec(env), program, argv[1:])
    if suffix not in WINDOWS_DIRECT:
        raise ArgumentError(f"{program.name} cannot be executed directly; run it through its interpreter")
    return [str(program), *argv[1:]]


# ---------------------------------------------------------------------------
# Output capture and test-count parsing
# ---------------------------------------------------------------------------


class BoundedOutput:
    """Keeps the first and last 32 KB of a byte stream and counts every byte."""

    def __init__(self, head: int = HEAD_BYTES, tail: int = TAIL_BYTES):
        self._head_limit, self._tail_limit = head, tail
        self._head = bytearray()
        self._tail = bytearray()
        self.total = 0
        self._lock = threading.Lock()

    def feed(self, chunk: bytes) -> None:
        with self._lock:
            self.total += len(chunk)
            room = self._head_limit - len(self._head)
            if room > 0:
                self._head += chunk[:room]
                chunk = chunk[room:]
            if chunk:
                self._tail += chunk
                if len(self._tail) > self._tail_limit:
                    del self._tail[: len(self._tail) - self._tail_limit]

    @property
    def truncated(self) -> bool:
        return self.total > len(self._head) + len(self._tail)

    def text(self) -> str:
        with self._lock:
            head, tail, total = bytes(self._head), bytes(self._tail), self.total
        if total <= len(head) + len(tail):
            return (head + tail).decode("utf-8", errors="replace")
        omitted = total - len(head) - len(tail)
        marker = f"\n[... {omitted} bytes omitted; {total} bytes total ...]\n".encode()
        tail = tail[len(marker) :]
        return (head + marker + tail).decode("utf-8", errors="replace")


_CARGO = re.compile(r"^test result: \w+\. (\d+) passed; (\d+) failed", re.MULTILINE)
_JEST = re.compile(r"^Tests:\s+.*?(\d+) total", re.MULTILINE)
_VITEST = re.compile(r"^\s*Tests\s+.*?\((\d+)\)\s*$", re.MULTILINE)
_NODE = re.compile(r"^(?:ℹ|#) tests (\d+)\s*$", re.MULTILINE)  # spec reporter (info sign) or TAP
_PYTEST_LINE = re.compile(r"^[=\s]*(?:(?:\d+ \w+(?:, )?)+|no tests ran)\b.* in [\d.]+s\b.*$", re.MULTILINE)
_PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|errors?|xfailed|xpassed)\b")
_UNITTEST = re.compile(r"^Ran (\d+) tests? in ", re.MULTILINE)
_GO_RESULT = re.compile(r"^\s*--- (?:PASS|FAIL): ", re.MULTILINE)
_GO_VERBOSE = re.compile(r"^=== RUN ", re.MULTILINE)


def _last(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def _pytest_count(text: str) -> int | None:
    line = _last(_PYTEST_LINE, text)
    if line is None:
        return None
    return sum(int(number) for number, _kind in _PYTEST_COUNT.findall(line.group(0)))


def _js_count(text: str) -> int | None:
    """npm test runs whatever the project configured: jest, vitest or node --test; last summary wins."""
    found = [match for match in (_last(_JEST, text), _last(_VITEST, text), _last(_NODE, text)) if match]
    return int(max(found, key=lambda match: match.start()).group(1)) if found else None


def _single(pattern: re.Pattern[str]) -> Any:
    def count(text: str) -> int | None:
        match = _last(pattern, text)
        return int(match.group(1)) if match else None

    return count


def _cargo_count(text: str) -> int | None:
    # One summary line per test binary (unit, integration, doc tests): they add up.
    results = _CARGO.findall(text)
    return sum(int(passed) + int(failed) for passed, failed in results) if results else None


def _go_count(text: str) -> int | None:
    return len(_GO_RESULT.findall(text)) if _GO_VERBOSE.search(text) else None


_PARSERS: dict[str, Any] = {
    "pytest": _pytest_count,
    "unittest": _single(_UNITTEST),
    "npm": _js_count,
    "node": _single(_NODE),
    "cargo": _cargo_count,
    "go": _go_count,
}


def parse_tests_run(text: str, runner: str | None) -> int | None:
    """Tests executed per ``runner``'s own summary format only; ``None`` when unknown.

    Other runners' formats are ignored, so output printed by a test ("test result: ok. 999
    passed") cannot override the real summary.
    """
    parser = _PARSERS.get(runner or "")
    return parser(text) if parser else None


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LaunchSpec:
    call_id: str
    tool: str
    argv: list[str]  # resolved, unwrapped argv (what policy saw)
    popen_args: list[str] | str  # wrapped, ready for Popen
    cwd: Path
    env: dict[str, str]
    timeout_s: float
    backend: Backend
    isolation: str
    count_tests: bool = False
    runner: str | None = None
    extra_data: dict[str, Any] = field(default_factory=dict)


class Job:
    """A running (or already finished) tool operation. Implements contracts.JobHandle."""

    def __init__(self, spec: LaunchSpec | None = None, *, result: ToolResult | None = None):
        self._spec = spec
        self._result = result
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._kill_lock = threading.Lock()
        self._reason: str | None = None  # "timeout" | "cancelled"
        self._output = BoundedOutput()
        self._proc: subprocess.Popen[bytes] | None = None
        self._winjob: Any = None
        self._winjob_used = False
        self._root: psutil.Process | None = None
        self._known: dict[int, psutil.Process] = {}
        self._started = time.monotonic()
        if result is not None:
            self._done.set()
            return
        assert spec is not None
        try:
            self._spawn(spec)
        except OSError as exc:
            self._finish_now(ToolStatus.FAILED, f"could not start {spec.argv[0]!r}: {exc}")
            return
        self._reader = threading.Thread(target=self._read, name=f"job-read-{spec.call_id}", daemon=True)
        self._reader.start()
        threading.Thread(target=self._monitor, name=f"job-wait-{spec.call_id}", daemon=True).start()

    @classmethod
    def completed(cls, result: ToolResult) -> Job:
        return cls(result=result)

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    @property
    def done(self) -> bool:
        return self._done.is_set()

    # -- lifecycle -------------------------------------------------------------------

    def _spawn(self, spec: LaunchSpec) -> None:
        kwargs: dict[str, Any] = {
            "cwd": str(spec.cwd),
            "env": spec.env,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "shell": False,
        }
        job = self._new_winjob() if IS_WINDOWS else None
        if IS_WINDOWS:
            suspended = CREATE_SUSPENDED if job is not None else 0
            kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | suspended
        else:
            kwargs["start_new_session"] = True
        self._started = time.monotonic()
        self._proc = subprocess.Popen(spec.popen_args, **kwargs)
        if job is not None:
            self._assign_and_resume(job)
        with contextlib.suppress(psutil.Error, OSError):
            self._root = psutil.Process(self._proc.pid)

    def _new_winjob(self) -> Any:
        from .backends.winjob import WinJob

        try:
            return WinJob()
        except OSError:
            return None

    def _assign_and_resume(self, job: Any) -> None:
        """The child starts suspended, so it cannot spawn anything before it is in the job."""
        from .backends.winjob import resume_process

        assert self._proc is not None
        try:
            job.assign(self._proc)
            self._winjob = job
            self._winjob_used = True
        except OSError:
            job.close()
        try:
            resume_process(self._proc)
        except OSError:
            self._proc.kill()
            self._proc.wait()
            if self._proc.stdout is not None:
                self._proc.stdout.close()
            if self._winjob is not None:
                self._winjob.close()
                self._winjob = None
            raise

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stream = self._proc.stdout
        try:
            while True:
                chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
                if not chunk:
                    break
                self._output.feed(chunk)
        except (OSError, ValueError):
            pass

    def _monitor(self) -> None:
        assert self._spec is not None and self._proc is not None
        deadline = self._started + self._spec.timeout_s
        while True:
            self._track()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self._lock:
                    if self._reason is None:
                        self._reason = "timeout"
                self._kill_tree()
                self._proc.wait()
                break
            try:
                self._proc.wait(timeout=min(TRACK_INTERVAL_S, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        self._kill_tree(after_exit=True)  # leftover background descendants
        self._reader.join(READER_JOIN_S)
        self._finalize()

    def _track(self) -> None:
        """Remember every descendant (psutil guards against pid reuse). Catches processes
        that escape the process group (setsid) or the Job Object (nested silent-breakaway
        jobs, e.g. the Windows venv launcher), as long as they were seen once."""
        roots = [self._root] if self._root is not None else []
        roots.extend(self._known.values())
        for root in roots:
            try:
                children = root.children(recursive=True)
            except (psutil.Error, OSError):
                continue
            for child in children:
                self._known.setdefault(child.pid, child)

    def _kill_tree(self, after_exit: bool = False) -> None:
        if self._proc is None:
            return
        with self._kill_lock:
            if not after_exit and self._spec is not None:
                wrapped = self._spec.popen_args
                self._spec.backend.on_cancel(wrapped if isinstance(wrapped, list) else [])
            self._track()
            if IS_WINDOWS:
                self._freeze_known()
                if self._winjob is not None:
                    self._winjob.terminate(wait_s=KILL_GRACE_S * 2)
            else:
                self._kill_posix()
            self._kill_known()
            if self._proc.poll() is None:
                self._proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=KILL_GRACE_S)

    def _freeze_known(self) -> None:
        """Suspend tracked processes so none can spawn more, then rescan once."""
        for member in list(self._known.values()):
            with contextlib.suppress(psutil.Error, OSError):
                member.suspend()
        self._track()

    def _kill_known(self) -> None:
        members = list(self._known.values())
        for member in members:
            with contextlib.suppress(psutil.Error, OSError):
                member.kill()
        with contextlib.suppress(psutil.Error, OSError):
            psutil.wait_procs(members, timeout=KILL_GRACE_S)
        self._known = {pid: proc for pid, proc in self._known.items() if _running(proc)}

    def _kill_posix(self) -> None:
        assert self._proc is not None
        pgid = self._proc.pid  # start_new_session=True => the child leads its own group
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                return
            deadline = time.monotonic() + KILL_GRACE_S
            while time.monotonic() < deadline:
                self._proc.poll()
                try:
                    os.killpg(pgid, 0)
                except (ProcessLookupError, PermissionError):
                    return
                time.sleep(POLL_S)

    # -- results ---------------------------------------------------------------------

    def _finish_now(self, status: ToolStatus, message: str) -> None:
        spec = self._spec
        assert spec is not None
        data = {"exit_code": None, "duration_ms": 0.0, "bytes": 0, "backend": spec.backend.name,
                "isolation": spec.isolation, "argv": list(spec.argv), **spec.extra_data}  # fmt: skip
        self._result = ToolResult(spec.call_id, spec.tool, status, message, data)
        self._done.set()

    def _finalize(self) -> None:
        spec, proc = self._spec, self._proc
        assert spec is not None and proc is not None
        duration_ms = round((time.monotonic() - self._started) * 1000, 1)
        if self._winjob is not None:
            self._winjob.close()
            self._winjob = None
        text = self._output.text()
        exit_code = proc.returncode
        with self._lock:
            reason = self._reason
        on_host = spec.backend.name == "host"
        killed = "tracked process tree terminated" + (f"; {HOST_UNTRACKED}" if on_host else "")
        if reason == "cancelled":
            status = ToolStatus.CANCELLED
            text += f"\n[cancelled: {killed}]"
        elif reason == "timeout":
            status = ToolStatus.TIMEOUT
            text += f"\n[timed out after {spec.timeout_s:g}s: {killed}]"
        else:
            status = ToolStatus.OK if exit_code == 0 else ToolStatus.FAILED
        data: dict[str, Any] = {
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "bytes": self._output.total,
            "truncated": self._output.truncated,
            "backend": spec.backend.name,
            "isolation": spec.isolation,
            "argv": list(spec.argv),
            "job_object": self._winjob_used if IS_WINDOWS else None,
            **spec.extra_data,
        }
        if reason is not None and on_host:
            data["remedy"] = f"{HOST_UNTRACKED}; {HOST_REMEDY}"
        if spec.runner is not None:
            data["runner"] = spec.runner
        if spec.count_tests or spec.runner is not None:
            data["tests_run"] = parse_tests_run(text, spec.runner)
        self._result = ToolResult(spec.call_id, spec.tool, status, text, data, elapsed_ms=duration_ms)
        self._done.set()

    async def wait(self) -> ToolResult:
        while not self._done.is_set():
            await asyncio.sleep(POLL_S)
        assert self._result is not None
        return self._result

    async def cancel(self) -> None:
        """Idempotent; returns after the tracked process tree is gone (see HOST_UNTRACKED)."""
        if self._done.is_set():
            return
        with self._lock:
            if self._reason is None:
                self._reason = "cancelled"
        await asyncio.to_thread(self._kill_tree)
        while not self._done.is_set():
            await asyncio.sleep(POLL_S)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

RUN_COMMAND = "run_command"
RUN_TESTS = "run_tests"


def _argv_argument(value: Any, name: str = "argv") -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ArgumentError(f"{name} must be a non-empty array of strings (argv, not a shell string)")
    if any("\x00" in item for item in value):
        raise ArgumentError(f"{name} must not contain NUL characters")
    if not value[0].strip():
        raise ArgumentError(f"{name}[0] must name a program")
    return list(value)


class Executor:
    """Owns backend selection, the private temp dir and job launching for one project."""

    def __init__(self, project_root: str | Path, config: Any = None):
        self.project_root = Path(project_root).resolve()
        self.config = config
        self._settings = sandbox_settings(config)
        base = Path(tempfile.mkdtemp(prefix="nedster-exec-"))
        self._finalizer = weakref.finalize(self, shutil.rmtree, str(base), True)
        # The sandbox may write to ``tmp``; ``private`` (e.g. SafeGit's empty hooks dir) is never exposed to it.
        self._temp, self._private = base / "tmp", base / "private"
        self._temp.mkdir()
        self._private.mkdir()
        # Resolved once, from the user's own PATH, before the model can influence anything.
        self._search_path = host_search_path(self.project_root, os.environ.get("PATH", ""))
        self._git: Path | None = find_program("git", self._search_path, self.project_root)
        self._resolved: OrderedDict[str, tuple[Any, list[str]]] = OrderedDict()
        self._selection: Selection | None = None
        self._selection_lock = threading.Lock()

    @property
    def temp_dir(self) -> Path:
        """Private temp directory handed to commands (sandboxes may write here)."""
        return self._temp

    @property
    def private_dir(self) -> Path:
        """Executor-owned directory that sandboxed commands can neither see nor write."""
        return self._private

    @property
    def search_path(self) -> str:
        return self._search_path

    def git_program(self) -> Path | None:
        """Absolute git from the user's PATH (project directories excluded), or None."""
        return self._git

    @property
    def max_timeout_s(self) -> float:
        return float(getattr(self._settings, "max_seconds", DEFAULT_TIMEOUT_S) or DEFAULT_TIMEOUT_S)

    def close(self) -> None:
        self._finalizer()

    def selection(self, refresh: bool = False) -> Selection:
        with self._selection_lock:
            if self._selection is None or refresh:
                self._selection = select_backend(self.project_root, self._temp, self.config)
            return self._selection

    def backend_status(self) -> BackendStatus | None:
        return self.selection().status

    def env(self, venv: Path | None = None) -> dict[str, str]:
        return build_env(self.project_root, self._temp, venv=venv)

    def venv_python(self) -> Path | None:
        return venv_python(self.project_root)

    def runner_for(self, argv: list[str]) -> str | None:
        return runner_for(argv, self.project_root)

    def specs(self) -> list[ToolSpec]:
        timeout = {"type": "number", "exclusiveMinimum": 0, "maximum": self.max_timeout_s,
                   "description": f"Timeout in seconds (default and maximum {self.max_timeout_s:g})."}  # fmt: skip
        argv = {"type": "array", "items": {"type": "string"}, "minItems": 1,
                "description": "Program and arguments as a list. No shell: pipes, redirects and globbing do not work."}  # fmt: skip
        return [
            ToolSpec(
                RUN_COMMAND,
                "Run one program in the project root inside the sandbox and return its exit code and output.",
                {
                    "type": "object",
                    "properties": {"argv": argv, "timeout_s": timeout},
                    "required": ["argv"],
                    "additionalProperties": False,
                },
                int(RiskLevel.EXECUTE),
                mutates=True,
            ),  # fmt: skip
            ToolSpec(
                RUN_TESTS,
                "Run the project's test suite (auto-detected: pytest, npm test, cargo test, go test) "
                "or the given test argv, and report pass/fail and the number of tests run.",
                {
                    "type": "object",
                    "properties": {
                        "argv": {**argv, "description": "Optional explicit test argv."},
                        "timeout_s": timeout,
                    },
                    "additionalProperties": False,
                },
                int(RiskLevel.EXECUTE),
                mutates=True,
            ),  # fmt: skip
        ]

    # -- resolution (pure; used by the policy layer before authorization) ---------------

    def detect_test_argv(self) -> list[str]:
        root = self.project_root
        if (root / "Cargo.toml").is_file():
            return ["cargo", "test"]
        if (root / "go.mod").is_file():
            return ["go", "test", "./..."]
        if (root / "package.json").is_file():
            return ["npm", "test"]
        markers = ("pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini", "conftest.py")
        has_python_tests = (
            any((root / name).is_file() for name in markers)
            or (root / "tests").is_dir()
            or any(root.glob("test_*.py"))
            or any(root.glob("*/test_*.py"))
        )
        if has_python_tests:
            # resolve_argv swaps in the project's venv interpreter when it is a verified one.
            return ["python" if IS_WINDOWS else "python3", "-m", "pytest", "-q"]
        raise ArgumentError(
            "could not detect a test suite (no Cargo.toml, go.mod, package.json or Python tests)"
        )

    def _container(self) -> bool:
        backend = self.selection().backend
        return backend is not None and not backend.host_paths

    def _prefer_venv(self, argv: list[str]) -> list[str]:
        """Bare python/pytest use the project's verified venv interpreter (run_tests and run_command)."""
        venv = self.venv_python()
        if venv is None or self._container() or is_path_qualified(argv[0]):
            return argv
        name = executable_name(argv[0]).lower()
        if PYTHON_NAME.match(name):
            return [str(venv), *argv[1:]]
        if name in ("pytest", "py.test"):
            return [str(venv), "-m", "pytest", *argv[1:]]
        return argv

    def resolve_program_argv(self, argv: list[str], *, allow_venv: bool = False) -> list[str]:
        """``argv`` with argv[0] replaced by the absolute program that will run.

        Raises ResolutionDenied for a program inside the project (except the verified venv
        interpreter when ``allow_venv``) and ArgumentError when nothing is found.
        """
        program, rest = argv[0], argv[1:]
        venv = self.venv_python() if allow_venv else None
        if is_path_qualified(program):
            location = Path(program) if os.path.isabs(program) else self.project_root / program
            if venv is not None and _path_key(location, real=False) == _path_key(venv, real=False):
                return ["python" if self._container() else str(venv), *rest]
            if is_within(location, self.project_root):
                raise ResolutionDenied(
                    f"{program!r} is inside the project; project files cannot be run as programs. "
                    "Run scripts through an allowlisted interpreter, e.g. ['python', 'script.py']."
                )
            if self._container():
                raise ResolutionDenied(
                    f"{program!r}: host paths do not exist in the container; use a program name from the image"
                )
            found = _which(program, {"PATH": ""}, self.project_root)
            if found is None:
                raise ArgumentError(f"program not found: {program!r}")
            if is_within(found, self.project_root):
                raise ResolutionDenied(f"{program!r} resolves into the project; project files cannot be run")
            return [str(found), *rest]
        if self._container():
            return list(argv)  # looked up on the image's PATH, which the project cannot change
        found = find_program(program, self._search_path, self.project_root)
        if found is None:
            raise ArgumentError(
                f"program not found on PATH: {program!r} (directories inside the project are never searched)"
            )
        return [str(found), *rest]

    def resolve_argv(self, call: ToolCall) -> list[str]:
        """The resolved argv that ``start(call)`` will run; raises ArgumentError/ResolutionDenied."""
        args = call.arguments or {}
        if call.name == RUN_COMMAND:
            # Bare python/pytest use the verified project venv (same interpreter run_tests uses), so project
            # scripts see the project's dependencies; any other program inside the project stays denied.
            argv = self.resolve_program_argv(self._prefer_venv(_argv_argument(args.get("argv"))), allow_venv=True)
        elif call.name == RUN_TESTS:
            raw = args.get("argv")
            requested = _argv_argument(raw) if raw is not None else self.detect_test_argv()
            argv = self.resolve_program_argv(self._prefer_venv(requested), allow_venv=True)
            if self.runner_for(argv) is None:
                raise ArgumentError(
                    f"run_tests only runs known test runners ({', '.join(KNOWN_RUNNERS)}); got {requested[0]!r}. "
                    "A project file named like the runner module (e.g. pytest.py) also disqualifies it."
                )
        else:
            raise ArgumentError(f"unknown execution tool {call.name!r}")
        self._resolved[call.id] = (dict(args), argv)
        while len(self._resolved) > RESOLUTION_CACHE:
            self._resolved.popitem(last=False)
        return list(argv)

    def _argv_for_start(self, call: ToolCall) -> list[str]:
        """Exactly the argv the policy layer authorized for this call, when it resolved it first."""
        cached = self._resolved.pop(call.id, None)
        if cached is not None and cached[0] == dict(call.arguments or {}):
            return list(cached[1])
        self._resolved.pop(call.id, None)
        return self.resolve_argv(call)

    def resolve_timeout(self, call: ToolCall) -> float:
        raw = (call.arguments or {}).get("timeout_s")
        if raw is None:
            return self.max_timeout_s
        if isinstance(raw, bool) or not isinstance(raw, int | float) or raw <= 0:
            raise ArgumentError("timeout_s must be a positive number")
        if raw > self.max_timeout_s:
            raise ArgumentError(f"timeout_s may not exceed {self.max_timeout_s:g}")
        return float(raw)

    # -- launching ----------------------------------------------------------------------

    def _result(self, call: ToolCall, status: ToolStatus, message: str, **data: Any) -> Job:
        return Job.completed(ToolResult(call.id, call.name, status, message, data))

    def start(self, call: ToolCall) -> Job:
        try:
            argv = self._argv_for_start(call)
            timeout = self.resolve_timeout(call)
        except ResolutionDenied as exc:
            return self._result(
                call, ToolStatus.DENIED, str(exc), argv=list((call.arguments or {}).get("argv") or [])
            )
        except ArgumentError as exc:
            return self._result(call, ToolStatus.INVALID, str(exc))
        selection = self.selection()
        if selection.backend is None or selection.status is None:
            return self._result(call, ToolStatus.DENIED, selection.remedy, remedy=selection.remedy)
        venv = self.venv_python()
        uses_venv = venv is not None and selection.backend.host_paths and argv[0] == str(venv)
        return self.launch(
            call.id, call.name, argv, timeout, selection.backend, selection.status.isolation,
            env=self.env(venv if uses_venv else None), count_tests=call.name == RUN_TESTS,
            runner=self.runner_for(argv),
        )  # fmt: skip

    def start_host(
        self, call_id: str, tool: str, argv: list[str], timeout_s: float, env: Mapping[str, str] | None = None
    ) -> Job:
        """Host execution of a code-constructed (never model-provided) read-only argv, e.g. git.

        argv[0] must be an absolute program outside the project (see ``git_program``).
        """
        if not os.path.isabs(argv[0]) or is_within(argv[0], self.project_root):
            result = ToolResult(call_id, tool, ToolStatus.DENIED, f"refusing host program {argv[0]!r}: "
                                "host programs must be absolute and outside the project")  # fmt: skip
            return Job.completed(result)
        return self.launch(call_id, tool, argv, timeout_s, HostBackend(), "none", env=env)

    def launch(
        self,
        call_id: str,
        tool: str,
        argv: list[str],
        timeout_s: float,
        backend: Backend,
        isolation: str,
        *,
        env: Mapping[str, str] | None = None,
        count_tests: bool = False,
        runner: str | None = None,
    ) -> Job:
        """Run ``argv`` as given (already resolved by ``resolve_argv`` for model calls)."""
        environment = dict(env) if env is not None else self.env()
        cwd = self.project_root
        try:
            resolved = resolve_program(argv, cwd, environment) if backend.host_paths else list(argv)
            popen_args = resolved if isinstance(resolved, str) else backend.wrap(resolved, cwd, environment)
        except (ArgumentError, ValueError) as exc:
            return Job.completed(
                ToolResult(call_id, tool, ToolStatus.INVALID, str(exc), {"argv": list(argv)})
            )
        spec = LaunchSpec(
            call_id, tool, list(argv), popen_args, cwd, environment, timeout_s, backend, isolation,
            count_tests=count_tests, runner=runner,
        )  # fmt: skip
        return Job(spec)


__all__ = [
    "ArgumentError",
    "BoundedOutput",
    "Executor",
    "Job",
    "LaunchSpec",
    "ResolutionDenied",
    "batch_command_line",
    "build_env",
    "executable_name",
    "find_program",
    "host_search_path",
    "is_within",
    "parse_tests_run",
    "resolve_program",
    "runner_for",
    "venv_python",
]
