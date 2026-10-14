"""Host backend: no wrapping and no filesystem/network isolation. Opt-in only."""

from __future__ import annotations

import os
from pathlib import Path

from . import BackendStatus, Isolation


class HostBackend:
    name = "host"
    isolation: Isolation = "none"
    host_paths = True

    def probe(self) -> BackendStatus:
        detail = "host process: no filesystem/network isolation"
        if os.name == "nt":
            detail += "; process tree contained by a Windows Job Object"
        else:
            detail += "; process tree contained by a new session/process group"
        return BackendStatus(self.name, True, self.isolation, detail)

    def wrap(self, argv: list[str], cwd: Path, env: dict[str, str]) -> list[str]:
        return list(argv)

    def on_cancel(self, wrapped: list[str]) -> None:
        return None
