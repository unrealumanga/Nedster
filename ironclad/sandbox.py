from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import SandboxConfig


@dataclass(slots=True)
class SandboxResult:
    backend: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class SandboxExecutor:
    def __init__(self, config: SandboxConfig, project_root: str | Path):
        self.config = config
        self.project_root = Path(project_root).resolve()
        self.backend = self._choose_backend()

    def _choose_backend(self) -> str:
        if self.config.backend != "auto":
            return self.config.backend
        if platform.system() == "Linux":
            if shutil.which("bwrap") or shutil.which("bubblewrap"):
                return "bubblewrap"
            if shutil.which("docker") and shutil.which("runsc"):
                return "gvisor"
        if shutil.which("podman"):
            return "podman"
        if shutil.which("docker"):
            return "docker"
        return "process"

    def _bubblewrap(self, command: list[str], cwd: Path) -> list[str]:
        binary = shutil.which("bwrap") or shutil.which("bubblewrap") or "bwrap"
        args = [binary, "--die-with-parent", "--new-session", "--unshare-all"]
        if self.config.network != "deny":
            args.append("--share-net")
        for directory in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
            if Path(directory).exists():
                args.extend(["--ro-bind", directory, directory])
        relative = cwd.relative_to(self.project_root)
        sandbox_cwd = Path("/workspace") / relative
        args.extend(
            [
                "--bind",
                str(self.project_root),
                "/workspace",
                "--tmpfs",
                "/tmp",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--chdir",
                sandbox_cwd.as_posix(),
                "--",
                *command,
            ]
        )
        return args

    def _container(self, command: list[str], cwd: Path, engine: str, runtime: str | None = None) -> list[str]:
        relative = cwd.relative_to(self.project_root)
        args = [engine, "run", "--rm", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges"]
        if runtime:
            args.extend(["--runtime", runtime])
        if self.config.network == "deny":
            args.extend(["--network", "none"])
        args.extend(
            [
                "--mount",
                f"type=bind,src={self.project_root},dst=/workspace",
                "--workdir",
                (Path("/workspace") / relative).as_posix(),
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=256m",
                self.config.docker_image,
                *command,
            ]
        )
        return args

    def build_command(self, command: list[str], cwd: str | Path | None = None) -> list[str]:
        workdir = Path(cwd or self.project_root).resolve()
        workdir.relative_to(self.project_root)
        if self.backend == "bubblewrap":
            return self._bubblewrap(command, workdir)
        if self.backend == "gvisor":
            return self._container(command, workdir, "docker", "runsc")
        if self.backend in {"docker", "podman"}:
            return self._container(command, workdir, self.backend)
        if self.backend == "firecracker":
            if not self.config.firecracker_wrapper:
                raise RuntimeError("firecracker backend requires sandbox.firecracker_wrapper")
            return [self.config.firecracker_wrapper, "--project", str(self.project_root), "--", *command]
        if not self.config.allow_unsafe_fallback:
            raise RuntimeError(
                "No kernel/container sandbox is available. Install bubblewrap on Linux, "
                "configure gVisor/Podman, or explicitly enable the unsafe process fallback."
            )
        return command

    def execute(self, command: list[str], cwd: str | Path | None = None) -> SandboxResult:
        workdir = Path(cwd or self.project_root).resolve()
        built = self.build_command(command, workdir)
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            result = subprocess.run(
                built,
                cwd=workdir if self.backend == "process" else self.project_root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=self.config.max_seconds,
                shell=False,
                check=False,
            )
            return SandboxResult(
                self.backend,
                built,
                result.returncode,
                result.stdout[-self.config.max_output_chars :],
                result.stderr[-self.config.max_output_chars :],
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                self.backend,
                built,
                124,
                str(exc.stdout or "")[-self.config.max_output_chars :],
                str(exc.stderr or "")[-self.config.max_output_chars :],
                timed_out=True,
            )
