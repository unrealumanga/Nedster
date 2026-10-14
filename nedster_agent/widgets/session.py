"""The adapter between the TUI and an engine.

The TUI only ever sees :class:`EngineEvent` (plus :class:`SessionNote` for free-form progress
lines from the legacy orchestrator). Anything that runs turns implements :class:`EngineSession`.

Required: ``submit`` / ``cancel`` / ``info``. Optional (looked up with ``getattr``; the TUI explains
when one is missing): ``start()``, ``close()``, ``new()``, ``list_sessions()``, ``resume(id)``,
``set_model(name)``, ``undo_last()``.

``cancel`` only *requests* a stop. The engine confirms it by emitting ``turn.finished`` (outcome
``cancelled``) or by ``submit`` returning; until then the TUI keeps showing "cancelling…".
"""

from __future__ import annotations

import asyncio
import ipaddress
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

from ironclad.contracts import EngineEvent, EventKind, Outcome, ToolStatus


@dataclass(slots=True, frozen=True)
class SessionNote:
    """A progress line that is not an engine event (legacy plan/step activity)."""

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


def _probe_sandbox(config: Any) -> tuple[str, str]:
    try:
        from nedster_agent.model_manager import probe_execution

        probe = probe_execution(config)
    except Exception:  # noqa: BLE001 - status display must never crash the TUI
        return str(config.sandbox.backend), "unknown"
    if not probe.get("ready"):
        return "unavailable", "unknown"
    return str(probe.get("backend") or "unknown"), str(probe.get("isolation") or "unknown")


# ---------------------------------------------------------------------------
# Legacy orchestrator (engine = "legacy")
# ---------------------------------------------------------------------------


class LegacySession:
    """Wraps ``ironclad.orchestrator.NedsterRuntime`` and translates its dict events."""

    def __init__(self, config: Any, emit: EventCallback, runtime_class: Any = None) -> None:
        self.config = config
        self._emit = emit
        self._runtime_class = runtime_class
        self.runtime: Any = None
        self.session_id = uuid.uuid4().hex[:12]
        self.run_id = ""
        self.model = ""
        self.sandbox, self.isolation = str(config.sandbox.backend), "unknown"
        self._task: asyncio.Task[Any] | None = None
        self._steps: dict[str, str] = {}
        self._calls = 0
        self._open_calls: dict[str, str] = {}

    async def start(self) -> None:
        runtime_class = self._runtime_class
        if runtime_class is None:
            from ironclad.orchestrator import NedsterRuntime as runtime_class
        self.runtime = runtime_class(self.config, on_event=self._legacy_event)
        fleet = await self.runtime.initialize()
        from ironclad.types import ModelRole

        self.model = str(fleet.for_role(ModelRole.CODER).name)
        self.sandbox, self.isolation = await asyncio.to_thread(_probe_sandbox, self.config)

    def info(self) -> dict[str, Any]:
        models = self.config.models
        return {
            "model": self.model or "starting",
            "provider": provider_label(models.ollama_url, self.model),
            "cloud_oracle": models.cloud_model if models.cloud_oracle_enabled else "",
            "sandbox": self.sandbox,
            "isolation": self.isolation,
            "context_used": 0,
            "context_max": 0,
            "session_id": self.session_id,
            "engine": "legacy",
        }

    def models(self) -> list[tuple[str, str, str]]:
        fleet = getattr(self.runtime, "fleet", None)
        return list(fleet.as_rows()) if fleet else []

    def new(self) -> str:
        self.session_id = uuid.uuid4().hex[:12]
        return self.session_id

    async def submit(self, text: str) -> None:
        if self.runtime is None:
            raise RuntimeError("runtime is not started; run `nedster setup`")
        self.run_id = uuid.uuid4().hex[:12]
        self._steps.clear()
        self._event("turn.started", {"user_text": text})
        self._task = asyncio.ensure_future(self.runtime.respond(text))
        try:
            result = await self._task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            self._finish(Outcome.CANCELLED, ("cancelled by user",), "")
            return
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as an error event
            self._event("error", {"message": str(exc), "remedy": "run /doctor"})
            self._finish(Outcome.FAILED, (str(exc),), "")
            return
        finally:
            self._task = None
        self._finish(*self._outcome(result), str(result.get("answer") or "").strip())

    async def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    def close(self) -> None:
        if self.runtime is not None:
            self.runtime.close()

    # ------------------------------------------------------------------ translation

    @staticmethod
    def _outcome(result: Mapping[str, Any]) -> tuple[Outcome, tuple[str, ...]]:
        completed = int(result.get("completed_steps") or 0)
        total = int(result.get("total_steps") or 0)
        reasons = (f"{completed}/{total} steps",) if total else ()
        if not result.get("passed"):
            return Outcome.INCOMPLETE, (*reasons, "the orchestrator did not pass its own checks")
        if not total:
            return Outcome.ANSWERED, reasons
        if result.get("grounded"):
            return Outcome.COMPLETED, reasons
        return Outcome.UNVALIDATED, (*reasons, "no grounded evidence was recorded")

    def _finish(self, outcome: Outcome, reasons: tuple[str, ...], answer: str) -> None:
        for call_id, tool in list(self._open_calls.items()):
            self._event("tool.finished", {"call_id": call_id, "tool": tool, "status": "cancelled", "summary": ""})
        self._open_calls.clear()
        self._event("acceptance", {"outcome": outcome.value, "reasons": list(reasons), "changed_paths": []})
        self._event("turn.finished", {"answer": answer, "outcome": outcome.value})

    def _event(self, kind: EventKind, data: Mapping[str, Any]) -> None:
        self._emit(EngineEvent(kind, self.session_id, self.run_id, dict(data), time.time_ns()))

    def _note(self, text: str, level: str = "muted") -> None:
        self._emit(SessionNote(text, level))

    def _legacy_event(self, event: dict[str, Any]) -> None:
        kind = str(event.get("kind") or "")
        self._note(f"{kind} {dict(event)}", "raw")  # raw-event view (F2) only
        if kind == "tool.authorized":
            self._calls += 1
            call_id = f"legacy-{self._calls}"
            tool = str(event.get("tool") or "tool")
            self._open_calls[call_id] = tool
            arguments = event.get("arguments") or event.get("args") or {}
            self._event("tool.started", {"call_id": call_id, "tool": tool, "arguments": dict(arguments)})
        elif kind == "tool.completed":
            self._tool_finished(str(event.get("actor") or "tool"), event.get("result"), None)
        elif kind in {"tool.failed", "tool.denied"}:
            status = "denied" if kind == "tool.denied" else "failed"
            reason = str(event.get("error") or event.get("reason") or status)
            self._tool_finished(str(event.get("tool") or event.get("actor") or "tool"), {"error": reason}, status)
        else:
            self._progress(kind, event)

    def _tool_finished(self, tool: str, result: Any, status: str | None) -> None:
        call_id = next((cid for cid, name in self._open_calls.items() if name == tool), "")
        if not call_id:
            self._calls += 1
            call_id = f"legacy-{self._calls}"
        self._open_calls.pop(call_id, None)
        data = dict(result) if isinstance(result, Mapping) else {}
        summary = str(data.get("error") or "") or _legacy_summary(data)
        payload: dict[str, Any] = {"call_id": call_id, "tool": tool, "summary": summary}
        for key in ("returncode", "exit_code", "tests_run", "tests_failed", "path", "diff"):
            if key in data:
                payload[key] = data[key]
        for key in ("stdout", "output", "content"):
            if data.get(key):
                payload["output"] = str(data[key])
                break
        payload["mutated_paths"] = [str(data["path"])] if data.get("changed") and data.get("path") else []
        reason = None if status else failure_reason("ok", payload)
        if reason:
            payload["summary"] = f"{reason} · {summary}" if summary and summary != reason else reason
        payload["status"] = status or ("failed" if reason else "ok")
        self._event("tool.finished", payload)

    def _progress(self, kind: str, event: Mapping[str, Any]) -> None:
        if kind == "response.escalated":
            self._note(f"◇ Switching to tools · {event.get('reason') or 'verification required'}", "accent")
        elif kind == "plan.created":
            plan = event.get("plan") or {}
            steps = [s for s in (plan.get("steps") or []) if isinstance(s, Mapping)] if isinstance(plan, Mapping) else []
            self._steps.update({str(s.get("id") or ""): str(s.get("title") or "Work") for s in steps})
            self._note(f"Plan · {len(steps)} steps", "accent")
            for index, step in enumerate(steps, 1):
                self._note(f"  {index:>2}  {step.get('title') or step.get('objective') or 'Work'}")
        elif kind == "planner.fallback":
            self._note("◇ Using a simplified plan", "warn")
        elif kind == "step.started":
            step = event.get("step") or {}
            title = str(step.get("title") or step.get("objective") or "Working")
            self._steps[str(step.get("id") or "")] = title
            self._note(f"◆ {title}", "info")
        elif kind == "worker.stalled":
            self._note(f"  ↻ Retrying · {event.get('reason') or 'retrying with more evidence'}", "warn")
        elif kind == "step.finished":
            step = event.get("step") or {}
            verdict = event.get("verdict") or {}
            title = str(step.get("title") or self._steps.get(str(step.get("id") or ""), "Step"))
            # A critic pass is a review, not validation evidence: never say "verified" here.
            state = "reviewed" if verdict.get("passed") else "needs review"
            self._note(f"  {title} · {state}", "muted" if verdict.get("passed") else "warn")
        elif kind == "run.blocked":
            self._note(f"Run blocked · {event.get('reason') or 'blocked'}", "fail")


def _legacy_summary(result: Mapping[str, Any]) -> str:
    if "tracked_count" in result or "untracked_count" in result:
        return f"{result.get('tracked_count', 0)} tracked, {result.get('untracked_count', 0)} untracked"
    for key, noun in (("tests_run", "tests"), ("total_matches", "matches"), ("count", "items")):
        if key in result:
            return f"{result[key]} {noun}"
    if result.get("path"):
        return f"{result['path']}{' · changed' if result.get('changed') else ''}"
    if "returncode" in result:
        return f"exit {result['returncode']}"
    return "complete"


# ---------------------------------------------------------------------------
# Native engine (engine = "native")
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
            "session_id": self.session_id,
            "engine": "native",
        }

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
