"""Docker backend: a throwaway, network-less container named ``nedster-<uuid>``.

Cancellation kills the container itself (``docker kill <name>``), not only the CLI.
Host PATH/TEMP make no sense inside a Linux image, so only locale/terminal variables
are forwarded, by name (values come from the CLI's own filtered environment).
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import uuid
from pathlib import Path

from . import BackendStatus, Isolation
from .bwrap import protected_paths

WORKSPACE = "/workspace"
FORWARDED = ("LANG", "TERM", "PYTHONIOENCODING")


class DockerBackend:
    name = "docker"
    isolation: Isolation = "full"
    host_paths = False

    def __init__(self, root: Path, image: str = "python:3.13-slim"):
        self.root, self.image = Path(root).resolve(), image
        self.binary = shutil.which("docker")

    def _run(self, *args: str, timeout: float = 10) -> subprocess.CompletedProcess[bytes]:
        assert self.binary
        return subprocess.run(
            [self.binary, *args], capture_output=True, timeout=timeout, check=False, stdin=subprocess.DEVNULL
        )

    def probe(self) -> BackendStatus:
        if not self.binary:
            return BackendStatus(self.name, False, self.isolation, "docker is not installed")
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
            "--pids-limit", "512", "--tmpfs", "/tmp:rw,nosuid,size=256m",
            "--mount", f"type=bind,src={self.root},dst={WORKSPACE}",
        ]  # fmt: skip
        for path in protected_paths(self.root):
            try:
                rel_protected = path.relative_to(self.root).as_posix()
            except ValueError:
                continue
            cmd.extend(["--mount", f"type=bind,src={path},dst={WORKSPACE}/{rel_protected},readonly"])
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
