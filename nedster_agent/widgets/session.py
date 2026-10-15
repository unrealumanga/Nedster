"""The adapter between the TUI and an engine.

The TUI only ever sees :class:`EngineEvent` (plus :class:`SessionNote` for free-form progress
lines). Anything that runs turns implements :class:`EngineSession`.

Required: ``submit`` / ``cancel`` / ``info``. Optional (looked up with ``getattr``; the TUI explains
when one is missing): ``start()``, ``close()``, ``new()``, ``list_sessions()``, ``resume(id)``,
``set_model(name)``, ``undo_last()``.

``cancel`` only *requests* a stop. The engine confirms it by emitting ``turn.finished`` (outcome
``cancelled``) or by ``submit`` returning; until then the TUI keeps showing "cancelling…".
"""

from __future__ import annotations

import asyncio
import ipaddress
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

from ironclad.contracts import EngineEvent, ToolStatus


@dataclass(slots=True, frozen=True)
class SessionNote:
    """A progress line that is not an engine event."""

    text: str
    level: str = "muted"  # muted | info | warn | fail | accent | raw (raw-event view only)


SessionEvent = EngineEvent | SessionNote
EventCallback = Callable[[SessionEvent], None]


@runtime_checkable
class EngineSession(Protocol):
    async def submit(self, text: str) -> None:
        """Run one user turn; events arrive through the callback given at construction."""
        ...

    async def cancel(self) -> None:
        """Request a stop of the running turn. Must return promptly."""
        ...

    def info(self) -> dict[str, Any]:
        """{model, sandbox, isolation, context_used, context_max, session_id} (+ optional provider)."""
        ...


SessionFactory = Callable[[EventCallback], EngineSession]


def provider_label(endpoint: str, model: str = "") -> str:
    """Truthful LOCAL / REMOTE / CLOUD label for where inference runs."""
    lowered = model.lower()
    if lowered.endswith((":cloud", "-cloud")):
        return "CLOUD"
    host = urlparse(endpoint if "//" in endpoint else f"http://{endpoint}").hostname or ""
    if host in {"", "localhost"} or host.endswith(".localhost"):
        return "LOCAL"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return "REMOTE"
    return "LOCAL" if address.is_loopback or address.is_unspecified else "REMOTE"


def failure_reason(status: str, data: Mapping[str, Any]) -> str | None:
    """None when a finished tool really succeeded, else a short human reason.

    Dispatch success is not user-facing success: a nonzero exit or a test run that ran nothing
    is a failure even when the tool itself reported ``ok``.
    """
    if status and status != ToolStatus.OK.value:
        summary = str(data.get("summary") or data.get("error") or "").strip()
        return f"{status}: {summary}" if summary else status
    for key in ("exit_code", "returncode"):
        code = data.get(key)
        if isinstance(code, int) and not isinstance(code, bool) and code != 0:
            return f"exit {code}"
    tests_run = data.get("tests_run")
    if isinstance(tests_run, int) and not isinstance(tests_run, bool) and tests_run == 0:
        return "no tests ran"
    failed = data.get("tests_failed") or data.get("failures")
    if isinstance(failed, int) and not isinstance(failed, bool) and failed > 0:
        return f"{failed} tests failed"
    return None


# ---------------------------------------------------------------------------
# Native engine
# ---------------------------------------------------------------------------


class NativeSession:
    """Adapter over ``ironclad.engine.loop.Engine`` (duck-typed: ``run``, ``cancel``, ``model``).

    ``build_engine(emit)`` must return an engine whose events are passed to ``emit``.
    """

    def __init__(
        self,
        build_engine: Callable[[Callable[[EngineEvent], None]], Any],
        emit: EventCallback,
        *,
        store: Any = None,
        project_root: str = "",
        sandbox: str = "unknown",
        isolation: str = "unknown",
        endpoint: str = "",
    ) -> None:
        self._emit = emit
        self.engine = build_engine(self._on_event)
        self.store = store
        self.project_root = project_root
        self.sandbox, self.isolation, self.endpoint = sandbox, isolation, endpoint
        self.instructions: Mapping[str, Any] | None = None
        self.context_used = 0
        self.session_id = self._create()

    def _create(self) -> str:
        if self.store is not None:
            return str(self.store.create(self.project_root))
        return uuid.uuid4().hex

    def _on_event(self, event: EngineEvent) -> None:
        if event.kind == "model.usage":
            self.context_used = int(event.data.get("prompt_tokens", 0)) + int(event.data.get("completion_tokens", 0))
        self._emit(event)

    async def submit(self, text: str) -> None:
        await self.engine.run(self.session_id, text)

    async def cancel(self) -> None:
        self.engine.cancel()

    def info(self) -> dict[str, Any]:
        model = str(getattr(self.engine, "model", ""))
        budget = getattr(self.engine, "budget", None)
        return {
            "model": model,
            "provider": provider_label(self.endpoint, model) if self.endpoint else "LOCAL",
            "sandbox": self.sandbox,
            "isolation": self.isolation,
            "context_used": self.context_used,
            "context_max": int(getattr(budget, "context_tokens", 0) or 0),
            "max_steps": getattr(budget, "max_steps", None),  # None = no limit
            "session_id": self.session_id,
            "engine": "native",
            "instructions": self.instructions,
            "role": getattr(self.engine, "_role_name", "") or "",
            "role_label": getattr(self.engine, "_role_label", "") or "",
        }

    def set_role(self, name: str) -> None:
        """Switch the engine to a named role preset."""
        if hasattr(self.engine, "set_role"):
            self.engine.set_role(name)
        # Track role locally for info() in case the engine doesn't expose it
        from ironclad.engine.roles import get_role

        role = get_role(name)
        self.engine._role_name = role.name
        self.engine._role_label = role.status_label()

    def new(self) -> str:
        self.session_id = self._create()
        self.context_used = 0
        return self.session_id

    def list_sessions(self) -> list[Mapping[str, Any]]:
        if self.store is None:
            return []
        return list(self.store.list_sessions(self.project_root))

    def resume(self, session_id: str | None = None) -> str:
        sessions = self.list_sessions()
        if not sessions:
            raise LookupError("no saved sessions for this project")
        if session_id is None:
            chosen = sessions[0]
        else:
            resolver = getattr(self.store, "resolve_session", None)
            if callable(resolver):
                # id, unique prefix or exact title, over every session (not just a list page)
                chosen = resolver(str(self.project_root), session_id)
            else:
                matches = [s for s in sessions if str(s.get("id", "")).startswith(session_id)]
                if len(matches) != 1:
                    raise LookupError(f"no unique session matches {session_id!r}")
                chosen = matches[0]
        self.session_id = str(chosen["id"])
        return self.session_id

    def set_model(self, name: str) -> None:
        self.engine.model = name

    async def undo_last(self) -> str:
        undo = getattr(self.engine, "undo_last", None)
        if undo is None:
            raise NotImplementedError("this engine does not support undo yet")
        result = undo()
        if asyncio.iscoroutine(result):
            result = await result
        return str(result or "undone")
