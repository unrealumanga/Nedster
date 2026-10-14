"""macOS Seatbelt backend via the absolute /usr/bin/sandbox-exec.

The profile denies by default, denies all network, allows reads (except common credential
stores), and allows writes only below the project root and the private temp directory.
``.git`` (and a resolved gitdir) and ``.nedster`` are denied for writes; later SBPL rules
take precedence, so the denials come last. Paths are passed as ``-D`` parameters, never
interpolated into the profile text.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import BackendStatus, Isolation
from .bwrap import protected_paths

BINARY = "/usr/bin/sandbox-exec"
SENSITIVE_HOME_DIRS = (".ssh", ".aws", ".gnupg", ".config/gcloud", ".docker", ".kube", ".netrc")


def _sbpl_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class SeatbeltBackend:
    name = "seatbelt"
    isolation: Isolation = "full"
    host_paths = True
    binary = BINARY

    def __init__(self, root: Path, temp: Path):
        self.root = Path(os.path.realpath(root))
        self.temp = Path(os.path.realpath(temp))

    def params(self) -> dict[str, str]:
        params = {"ROOT": str(self.root), "TMP": str(self.temp), "HOME": os.path.realpath(Path.home())}
        for index, path in enumerate(protected_paths(self.root)):
            params[f"PROTECTED_{index}"] = os.path.realpath(path)
        return params

    def profile(self) -> str:
        params = self.params()
        protected = [f'(subpath (param "{key}"))' for key in params if key.startswith("PROTECTED_")]
        protected.append('(subpath (string-append (param "ROOT") "/.git"))')
        protected.append('(subpath (string-append (param "ROOT") "/.nedster"))')
        sensitive = " ".join(
            f'(subpath (string-append (param "HOME") {_sbpl_string("/" + item)}))'
            for item in SENSITIVE_HOME_DIRS
        )
        return "\n".join(
            [
                "(version 1)",
                "(deny default)",
                "(allow process-exec process-fork)",
                "(allow signal (target same-sandbox))",
                "(allow sysctl-read)",
                "(allow mach-lookup)",
                "(allow ipc-posix-shm)",
                "(allow file-read*)",
                f"(deny file-read* {sensitive})",
                "(allow file-write*",
                '  (subpath (param "ROOT")) (subpath (param "TMP"))',
                '  (literal "/dev/null") (literal "/dev/zero") (regex #"^/dev/tty") (regex #"^/dev/fd/"))',
                f"(deny file-write* {' '.join(protected)})",
                "(deny network*)",
            ]
        )

    def _command(self) -> list[str]:
        cmd = [self.binary, "-p", self.profile()]
        for key, value in self.params().items():
            cmd.extend(["-D", f"{key}={value}"])
        return cmd

    def probe(self) -> BackendStatus:
        if sys.platform != "darwin":
            return BackendStatus(self.name, False, self.isolation, "Seatbelt is macOS only")
        if not Path(self.binary).exists():
            return BackendStatus(self.name, False, self.isolation, f"{self.binary} is missing")
        try:
            result = subprocess.run(
                [*self._command(), "/usr/bin/true"],
                capture_output=True,
                timeout=5,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return BackendStatus(self.name, False, self.isolation, f"sandbox-exec probe failed: {exc}")
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()[:300]
            return BackendStatus(
                self.name, False, self.isolation, f"sandbox-exec exited {result.returncode}: {stderr}"
            )
        return BackendStatus(
            self.name, True, self.isolation, "Seatbelt profile: network denied; writes limited"
        )

    def wrap(self, argv: list[str], cwd: Path, env: dict[str, str]) -> list[str]:
        return [*self._command(), *argv]

    def on_cancel(self, wrapped: list[str]) -> None:
        return None
