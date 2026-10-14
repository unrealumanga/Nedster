"""Host backend: no wrapping and no filesystem/network isolation. Opt-in only.

Cancellation kills the tracked process tree (Job Object / process group plus every descendant
seen while running). A process deliberately detached from that tree, e.g. created through WMI
``Win32_Process.Create``, the Task Scheduler, launchd or ``systemd-run``, is not tracked and can
survive a cancel; results say so. Use a container backend for full containment.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import BackendStatus, Isolation

UNTRACKED_NOTE = "processes detached from the tree (WMI, launchd, systemd-run, services) are not tracked"


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
        return BackendStatus(self.name, True, self.isolation, f"{detail}; {UNTRACKED_NOTE}")

    def wrap(self, argv: list[str], cwd: Path, env: dict[str, str]) -> list[str]:
        return list(argv)

    def on_cancel(self, wrapped: list[str]) -> None:
        return None
