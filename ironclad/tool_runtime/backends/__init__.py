"""Execution backends: probe what can really isolate a command, wrap argv for it.

Selection never falls back silently: an explicitly configured backend that is not
ready, or "auto" with no isolating backend and no host opt-in, yields no backend and a
remedy message. The caller must then refuse execution (ToolStatus.DENIED).
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

Isolation = Literal["full", "process-only", "none"]

AUTO_ORDER = ("bwrap", "seatbelt", "docker")
ALIASES = {"bubblewrap": "bwrap", "sandbox-exec": "seatbelt", "process": "host", "native": "host"}


@dataclass(slots=True, frozen=True)
class BackendStatus:
    name: str
    ready: bool
    isolation: Isolation
    detail: str = ""


class Backend(Protocol):
    name: str
    isolation: Isolation
    host_paths: bool  # True when the wrapped command sees the host filesystem layout

    def probe(self) -> BackendStatus: ...

    def wrap(self, argv: list[str], cwd: Path, env: dict[str, str]) -> list[str]: ...

    def on_cancel(self, wrapped: list[str]) -> None: ...


@dataclass(slots=True, frozen=True)
class Selection:
    backend: Backend | None
    status: BackendStatus | None
    remedy: str = ""
    probes: tuple[BackendStatus, ...] = ()


def resolve_binary(root: Path, *names: str) -> str | None:
    """First of ``names`` found with the executor's trusted resolver (absolute PATH entries outside ``root`` only).

    Never ``shutil.which``: that honours relative PATH entries and project directories, so a planted
    ``docker``/``bwrap`` in the repository would be executed by the readiness probe.
    """
    from ..exec import find_program, host_search_path  # lazy: exec imports this package

    search_path = host_search_path(Path(root), os.environ.get("PATH", ""))
    for name in names:
        found = find_program(name, search_path, Path(root))
        if found is not None:
            return str(found)
    return None


def probe_env(root: Path) -> dict[str, str]:
    """Allowlisted environment (``exec.build_env``) for backend probes and helper commands."""
    import tempfile

    from ..exec import build_env  # lazy: exec imports this package

    return build_env(Path(root), Path(tempfile.gettempdir()))


def sandbox_settings(config: Any) -> Any:
    """Accept a NedsterConfig, a SandboxConfig, or None."""
    return getattr(config, "sandbox", config)


def build_backend(name: str, root: Path, temp: Path, config: Any) -> Backend:
    settings = sandbox_settings(config)
    if name == "bwrap":
        from .bwrap import BwrapBackend

        return BwrapBackend(root, temp)
    if name == "seatbelt":
        from .seatbelt import SeatbeltBackend

        return SeatbeltBackend(root, temp)
    if name == "docker":
        from .docker import DockerBackend

        return DockerBackend(
            root,
            getattr(settings, "docker_image", "python:3.13-slim"),
            memory=getattr(settings, "docker_memory", None) or "4g",
            cpus=getattr(settings, "docker_cpus", None) or "2",
        )
    if name == "host":
        from .host import HostBackend

        return HostBackend()
    raise ValueError(f"unknown execution backend: {name}")


def _remedy() -> str:
    system = platform.system()
    hint = {
        "Linux": "install bubblewrap (bwrap) with unprivileged user namespaces enabled",
        "Darwin": "make /usr/bin/sandbox-exec usable",
        "Windows": "start Docker Desktop with the configured image pulled",
    }.get(system, "install a supported sandbox")
    host_note = "host process: no filesystem/network isolation"
    if os.name == "nt":
        host_note += " (process tree is still contained with a Windows Job Object)"
    return (
        f"Command execution is disabled: no sandbox backend is ready. To enable it, {hint}, "
        f"or start Docker with the configured image, or opt in to host execution with "
        f"sandbox.allow_host_execution = true ({host_note})."
    )


def select_backend(root: Path, temp: Path, config: Any) -> Selection:
    settings = sandbox_settings(config)
    requested = str(getattr(settings, "backend", "auto") or "auto").strip().lower()
    requested = ALIASES.get(requested, requested)
    allow_host = bool(getattr(settings, "allow_host_execution", False))
    if requested != "auto":
        if requested == "host" and not allow_host:
            return Selection(
                None, None, "sandbox.backend = host requires sandbox.allow_host_execution = true."
            )
        try:
            backend = build_backend(requested, root, temp, config)
        except ValueError as exc:
            return Selection(None, None, f"{exc}. Use auto, bwrap, seatbelt, docker or host.")
        status = backend.probe()
        if not status.ready:
            return Selection(
                None, status, f"Configured backend {requested!r} is not ready: {status.detail}", (status,)
            )
        return Selection(backend, status, probes=(status,))
    probes: list[BackendStatus] = []
    for name in AUTO_ORDER:
        backend = build_backend(name, root, temp, config)
        status = backend.probe()
        probes.append(status)
        if status.ready:
            return Selection(backend, status, probes=tuple(probes))
    if allow_host:
        backend = build_backend("host", root, temp, config)
        status = backend.probe()
        probes.append(status)
        return Selection(backend, status, probes=tuple(probes))
    return Selection(None, None, _remedy(), tuple(probes))


__all__ = [
    "AUTO_ORDER",
    "Backend",
    "BackendStatus",
    "Isolation",
    "Selection",
    "build_backend",
    "probe_env",
    "resolve_binary",
    "sandbox_settings",
    "select_backend",
]
