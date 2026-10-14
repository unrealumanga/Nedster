"""Linux bubblewrap backend.

Network and every other namespace are unshared; system directories are read-only; the
project root and the private temp directory are writable; ``.git`` (including a resolved
gitdir) and ``.nedster`` are re-bound read-only on top. Cancellation relies on
``--die-with-parent`` plus the PID namespace: killing bwrap kills the namespace init,
which takes every sandboxed descendant with it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import BackendStatus, Isolation

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


class BwrapBackend:
    name = "bwrap"
    isolation: Isolation = "full"
    host_paths = True

    def __init__(self, root: Path, temp: Path):
        self.root, self.temp = Path(root).resolve(), Path(temp).resolve()
        self.binary = shutil.which("bwrap") or shutil.which("bubblewrap")

    def probe(self) -> BackendStatus:
        if not sys.platform.startswith("linux"):
            return BackendStatus(self.name, False, self.isolation, "bubblewrap is Linux only")
        if not self.binary:
            return BackendStatus(self.name, False, self.isolation, "bwrap is not installed")
        env = {"PATH": "/usr/bin:/bin"}
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
        for path in protected_paths(self.root):
            cmd.extend(["--ro-bind", str(path), str(path)])
        cmd.extend(["--chdir", str(cwd), "--"])
        return [*cmd, *argv]

    def on_cancel(self, wrapped: list[str]) -> None:
        return None
