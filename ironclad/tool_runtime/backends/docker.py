"""Docker backend: a throwaway, network-less container named ``nedster-<uuid>``.

Cancellation kills the container itself (``docker kill <name>``), not only the CLI.
Host PATH/TEMP make no sense inside a Linux image, so only locale/terminal variables
are forwarded, by name (values come from the CLI's own filtered environment).

``--mount`` values are CSV-parsed by Docker, so host paths containing ``,``, ``"`` or a line
break are refused instead of being allowed to inject mount options. ``.git`` and ``.nedster``
are mounted read-only even when missing (see ``bwrap.protected_mounts``). On POSIX hosts the
container runs as the invoking user; memory and CPUs are capped.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from . import BackendStatus, Isolation
from .bwrap import EmptyDir, protected_mounts

WORKSPACE = "/workspace"
FORWARDED = ("LANG", "TERM", "PYTHONIOENCODING")
MOUNT_UNSAFE = frozenset(',"\r\n')
DEFAULT_MEMORY = "4g"
DEFAULT_CPUS = "2"


def mount_safe(path: str | Path) -> str:
    """The path as a ``--mount`` field value; raises ValueError if it could inject options."""
    text = str(path)
    bad = sorted(set(text) & MOUNT_UNSAFE)
    if bad:
        raise ValueError(f"path {text!r} contains {''.join(bad)!r}, which docker --mount cannot carry safely")
    return text


class DockerBackend:
    name = "docker"
    isolation: Isolation = "full"
    host_paths = False

    def __init__(
        self,
        root: Path,
        image: str = "python:3.13-slim",
        *,
        memory: str = DEFAULT_MEMORY,
        cpus: str = DEFAULT_CPUS,
    ):
        self.root, self.image = Path(root).resolve(), image
        self.memory, self.cpus = str(memory), str(cpus)
        self.binary = shutil.which("docker")
        self.empty = EmptyDir()

    def _run(self, *args: str, timeout: float = 10) -> subprocess.CompletedProcess[bytes]:
        assert self.binary
        return subprocess.run(
            [self.binary, *args], capture_output=True, timeout=timeout, check=False, stdin=subprocess.DEVNULL
        )

    def probe(self) -> BackendStatus:
        if not self.binary:
            return BackendStatus(self.name, False, self.isolation, "docker is not installed")
        try:
            mount_safe(self.root)
        except ValueError as exc:
            return BackendStatus(self.name, False, self.isolation, str(exc))
        try:
            info = self._run("info", "--format", "{{.ServerVersion}}")
            if info.returncode != 0:
                return BackendStatus(self.name, False, self.isolation, "docker daemon is not reachable")
            image = self._run("image", "inspect", "--format", "{{.Id}}", self.image)
            if image.returncode != 0:
                return BackendStatus(
                    self.name,
                    False,
                    self.isolation,
                    f"image {self.image!r} is not present (docker pull it first)",
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return BackendStatus(self.name, False, self.isolation, f"docker probe failed: {exc}")
        return BackendStatus(self.name, True, self.isolation, f"container from {self.image}; network none")

    def wrap(self, argv: list[str], cwd: Path, env: dict[str, str]) -> list[str]:
        if not self.binary:
            raise RuntimeError("docker is not installed")
        rel = Path(cwd).resolve().relative_to(self.root).as_posix()
        workdir = WORKSPACE if rel == "." else f"{WORKSPACE}/{rel}"
        cmd = [
            self.binary, "run", "--rm", "-i", "--name", f"nedster-{uuid.uuid4().hex}",
            "--network", "none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--pids-limit", "512", "--memory", self.memory, "--cpus", self.cpus,
            "--tmpfs", "/tmp:rw,nosuid,size=256m",
            "--mount", f"type=bind,src={mount_safe(self.root)},dst={WORKSPACE}",
        ]  # fmt: skip
        if hasattr(os, "getuid"):
            cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        for source, target in protected_mounts(self.root, self.empty.path):
            try:
                rel_protected = target.relative_to(self.root).as_posix()
            except ValueError:
                continue  # a gitdir outside the project is not mounted at all
            dst = mount_safe(f"{WORKSPACE}/{rel_protected}")
            cmd.extend(["--mount", f"type=bind,src={mount_safe(source)},dst={dst},readonly"])
        cmd.extend(["--workdir", workdir, "--env", "TMPDIR=/tmp", "--env", "HOME=/tmp"])
        for key in env:
            if key in FORWARDED or key.startswith("LC_"):
                cmd.extend(["--env", key])
        return [*cmd, self.image, *argv]

    @staticmethod
    def container_name(wrapped: list[str]) -> str | None:
        for index, item in enumerate(wrapped[:-1]):
            if item == "--name" and wrapped[index + 1].startswith("nedster-"):
                return wrapped[index + 1]
        return None

    def on_cancel(self, wrapped: list[str]) -> None:
        name = self.container_name(wrapped)
        if not name or not self.binary:
            return
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            self._run("kill", name, timeout=15)
