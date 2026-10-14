"""Linux bubblewrap backend.

Network and every other namespace are unshared; system directories are read-only; the
project root and the private temp directory are writable; ``.git`` (including a resolved
gitdir) and ``.nedster`` are re-bound read-only on top, whether or not they exist: ``.nedster``
is created first, and a missing ``.git`` is covered by an empty read-only directory, so a
sandboxed command can never create them (a planted ``.nedster/config.toml`` or ``.git/config``
would run code on the host later). Cancellation relies on
``--die-with-parent`` plus the PID namespace: killing bwrap kills the namespace init,
which takes every sandboxed descendant with it.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import weakref
from pathlib import Path

from . import BackendStatus, Isolation, probe_env, resolve_binary

SYSTEM_DIRS = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/etc", "/opt")
PROTECTED = (".git", ".nedster")


def protected_paths(root: Path) -> list[Path]:
    """Metadata that sandboxed commands must not write, including a gitdir behind a .git file."""
    found: list[Path] = []
    for name in PROTECTED:
        path = root / name
        if path.is_dir():
            found.append(path)
        elif path.is_file() and name == ".git":
            found.append(path)
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text.startswith("gitdir:"):
                target = Path(text.split(":", 1)[1].strip())
                target = (root / target).resolve() if not target.is_absolute() else target.resolve()
                if target.exists():
                    found.append(target)
    return found


class EmptyDir:
    """A private empty directory (outside every sandbox-writable path) used as a read-only mount source."""

    def __init__(self) -> None:
        self._path: Path | None = None

    @property
    def path(self) -> Path:
        if self._path is None:
            self._path = Path(tempfile.mkdtemp(prefix="nedster-ro-"))
            weakref.finalize(self, shutil.rmtree, str(self._path), True)
        return self._path


def protected_mounts(root: Path, empty: Path) -> list[tuple[Path, Path]]:
    """(source, target) pairs to mount read-only so ``.git``/``.nedster`` stay protected even when missing."""
    nedster = root / PROTECTED[1]
    if not os.path.lexists(nedster):
        with contextlib.suppress(OSError):
            nedster.mkdir()
    mounts = [(path, path) for path in protected_paths(root)]
    covered = {target for _source, target in mounts}
    for name in PROTECTED:
        target = root / name
        if target in covered:
            continue
        if target.exists():
            mounts.append((target, target))  # e.g. a plain file named .nedster
        elif os.path.lexists(target):
            # A dangling link would let the sandbox create its target and with it host-read config.
            raise ValueError(f"{target} is a dangling symlink; remove it before running commands")
        else:
            mounts.append((empty, target))
    return mounts


class BwrapBackend:
    name = "bwrap"
    isolation: Isolation = "full"
    host_paths = True

    def __init__(self, root: Path, temp: Path):
        self.root, self.temp = Path(root).resolve(), Path(temp).resolve()
        self.binary = resolve_binary(self.root, "bwrap", "bubblewrap")
        self.empty = EmptyDir()

    def probe(self) -> BackendStatus:
        if not sys.platform.startswith("linux"):
            return BackendStatus(self.name, False, self.isolation, "bubblewrap is Linux only")
        if not self.binary:
            return BackendStatus(self.name, False, self.isolation, "bwrap is not installed")
        env = probe_env(self.root)
        try:
            result = subprocess.run(
                self.wrap(["true"], self.root, env),
                capture_output=True,
                timeout=5,
                check=False,
                env=env,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return BackendStatus(self.name, False, self.isolation, f"bwrap probe failed: {exc}")
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()[:300]
            return BackendStatus(
                self.name, False, self.isolation, f"bwrap probe exited {result.returncode}: {stderr}"
            )
        return BackendStatus(
            self.name, True, self.isolation, "namespaces + network denied; project/temp writable"
        )

    def wrap(self, argv: list[str], cwd: Path, env: dict[str, str]) -> list[str]:
        if not self.binary:
            raise RuntimeError("bwrap is not installed")
        cmd = [
            self.binary,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
        ]
        bound: list[str] = []
        for path in SYSTEM_DIRS:
            if Path(path).exists():
                cmd.extend(["--ro-bind", path, path])
                bound.append(path)
        prefix = os.path.realpath(sys.base_prefix)
        if not any(prefix == item or prefix.startswith(item + "/") for item in bound):
            cmd.extend(["--ro-bind", prefix, prefix])
        cmd.extend(["--bind", str(self.root), str(self.root), "--bind", str(self.temp), str(self.temp)])
        for source, target in protected_mounts(self.root, self.empty.path):
            cmd.extend(["--ro-bind", str(source), str(target)])
        cmd.extend(["--chdir", str(cwd), "--"])
        return [*cmd, *argv]

    def on_cancel(self, wrapped: list[str]) -> None:
        return None
