"""Assemble the v4 native engine from configuration: transport, tools, session store and policy."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .contracts import AcceptanceResult, EngineEvent
from .session_store.titles import normalize_title

DEFAULT_WORKER = "nedster-worker"
FALLBACK_CONTEXT_TOKENS = 32_768
FINGERPRINT_MAX_FILES = 20_000


def worker_model(config: Any, override: str | None = None) -> str:
    if override:
        return override
    coder = str(getattr(config.models, "coder", "") or "")
    return coder if coder and coder != "auto" else DEFAULT_WORKER


def context_budget(config: Any, model: str) -> int:
    """``runtime.context_tokens`` when set; otherwise the window Ollama actually serves ``model`` with (num_ctx).

    Compacting against a smaller budget than the served window wastes context; a larger one gets silently truncated.
    """
    configured = int(getattr(config.runtime, "context_tokens", 0) or 0)
    if configured > 0:
        return configured
    import httpx

    url = str(config.models.ollama_url).rstrip("/")
    try:
        response = httpx.post(f"{url}/api/show", json={"model": model}, timeout=3)
        response.raise_for_status()
        for line in str(response.json().get("parameters") or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "num_ctx" and parts[1].isdigit():
                return max(2_048, int(parts[1]))
    except (httpx.HTTPError, ValueError, AttributeError):
        pass
    return FALLBACK_CONTEXT_TOKENS


def engine_budget(config: Any, model: str) -> Any:
    """Run limits from ``runtime.max_steps`` / ``runtime.max_tool_calls`` (0 = no limit)."""
    from .engine.state import Budget

    steps = int(getattr(config.runtime, "max_steps", 40))
    tool_calls = int(getattr(config.runtime, "max_tool_calls", 120))
    return Budget(
        max_steps=steps if steps > 0 else None,
        max_tool_calls=tool_calls if tool_calls > 0 else None,
        context_tokens=context_budget(config, model),
    )


def git_inventory(git: Any) -> Callable[[Path], list[str]]:
    """Inventory via hardened ``git ls-files`` (respects .gitignore); falls back to a directory walk."""
    import subprocess

    from .tool_runtime.files import walk_inventory

    def inventory(base: Path) -> list[str]:
        try:
            argv = [*git.base_argv(), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
            done = subprocess.run(argv, cwd=base, env=git.env(), capture_output=True, timeout=15, check=False)
        except Exception:
            return walk_inventory(base)
        if done.returncode != 0:
            return walk_inventory(base)
        names = done.stdout.decode("utf-8", "replace").split("\0")
        return sorted(name for name in names if name)

    return inventory


def build_tools(project_root: Path, config: Any, journal: Any = None) -> tuple[Any, Any]:
    """Return (CompositeToolRuntime, Executor) for ``project_root``; ``journal`` (a CheckpointStore) records undo data."""
    import os

    from .tool_runtime.composite import CompositeToolRuntime
    from .tool_runtime.exec import Executor
    from .tool_runtime.files import FileTools
    from .tool_runtime.git import SafeGit
    from .tool_runtime.search import SearchTools

    executor = Executor(project_root, config)
    git = SafeGit(project_root, executor)
    inventory = git_inventory(git)

    def fingerprint() -> dict[str, tuple[int, int]]:
        """(size, mtime_ns) of the current inventory, so a command that rewrites sources is detected."""
        result: dict[str, tuple[int, int]] = {}
        for rel in inventory(project_root)[:FINGERPRINT_MAX_FILES]:
            try:
                st = os.stat(project_root / rel)
            except OSError:
                continue
            result[rel] = (st.st_size, st.st_mtime_ns)
        return result

    runtime = CompositeToolRuntime(
        sync_tools=[FileTools(project_root, inventory=inventory, journal=journal), SearchTools(project_root, inventory=inventory)],
        job_tools=[executor, git],
        policy=config.policy,
        resolve_argv=executor.resolve_argv,
        journal=journal,
        fingerprint=fingerprint,
    )
    return runtime, executor


class NativeSession:
    """One project session on the native engine. Not thread-safe; one run at a time."""

    def __init__(
        self,
        config: Any,
        *,
        emit: Callable[[EngineEvent], None] | None = None,
        model: str | None = None,
        session_id: str | None = None,
        resume_latest: bool = False,
        title: str = "",
        lazy: bool = False,
        role: str | None = None,
    ) -> None:
        from .engine.loop import Engine
        from .session_store.checkpoints import CheckpointStore
        from .session_store.sqlite_store import SqliteSessionStore
        from .transport.ollama import OllamaTransport

        self.config = config
        self.project_root = Path(config.runtime.project_root).resolve()
        state_dir = Path(config.runtime.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        db_path = state_dir / "sessions.sqlite3"
        self.store = SqliteSessionStore(db_path)
        self.checkpoints = CheckpointStore(db_path)
        self.recovery_notes = self.checkpoints.recover()
        root = str(self.project_root)
        name = normalize_title(title)
        self._pending_title = ""  # stored when a lazily-created session is finally written
        self.started_new = False
        if session_id is not None:
            # A full id, a unique prefix or an exact title, scoped to this project (F01/F02/F07).
            session_id = str(self.store.resolve_session(root, session_id)["id"])
            self.session_id = session_id
            if name:  # an explicit --name renames the session it resumes
                self.store.set_title(self.session_id, name)
        elif resume_latest:
            latest = self.store.latest_non_empty(root)  # never continue an empty launch artifact (F06)
            if latest is None:
                self.started_new = True
                self.session_id = None if lazy else self.store.create(root, name)
                self._pending_title = name
            else:
                self.session_id = latest
                if name:
                    self.store.set_title(self.session_id, name)
        elif lazy:
            # No session row until the first submitted prompt (F06): opening and closing the UI
            # leaves nothing behind.
            self.started_new = True
            self.session_id = None
            self._pending_title = name
        else:
            self.started_new = True
            self.session_id = self.store.create(root, name)
        self.model = worker_model(config, model)
        self.tools, self.executor = build_tools(self.project_root, config, self.checkpoints)
        self.checkpoints.on_undone.append(self.tools.invalidate_after_undo)
        self.transport = OllamaTransport(config.models.ollama_url)
        soul_mode = str(getattr(config.runtime, "soul", "off") or "off")
        role_name = role or self._saved_role() or str(getattr(config.runtime, "role", "code") or "code")
        self.engine = Engine(
            self.transport,
            self.tools,
            model=self.model,
            store=self.store,
            budget=engine_budget(config, self.model),
            emit=emit,
            project_root=self.project_root,
            soul=soul_mode,
            role=role_name,
        )
        from .engine.prompt import session_instructions, session_soul

        # A lazily-created session has no row yet: preview without storing; the engine snapshots both
        # for the real id at the first turn.
        snapshot_store = self.store if self.session_id else None
        self.instructions = session_instructions(snapshot_store, self.session_id, self.project_root)
        self.soul = session_soul(snapshot_store, self.session_id, soul_mode)

    def info(self) -> dict[str, Any]:
        status = self.executor.backend_status()
        return {
            "model": self.model,
            "context_tokens": self.engine.budget.context_tokens,
            "session_id": self.session_id,
            "project_root": str(self.project_root),
            "sandbox": status.name if status else "none",
            "isolation": status.isolation if status else "none",
            "instructions": (
                {
                    "name": self.instructions.name,
                    "size": self.instructions.size,
                    "truncated": self.instructions.truncated,
                }
                if self.instructions.present
                else None
            ),
            "soul": self.soul.mode,
            "role": self.engine.role_name,
            "role_label": self.engine.role_label,
        }

    def _saved_role(self) -> str | None:
        """The role a resumed session last ran under (an explicit ``role=`` still wins).

        Sessions saved before roles were persisted have no entry and keep the configured role.
        """
        if self.started_new or not self.session_id:
            return None
        from .engine.loop import ROLE_STATE
        from .engine.roles import role_names

        saved = self.store.get_state(self.session_id, ROLE_STATE)
        text = str(saved).strip().lower() if saved else ""
        return text if text in role_names() else None

    def set_role(self, name: str) -> None:
        """Switch the active role (see ``engine.roles``); the tool gate and session prefix are updated in place.

        Refused by the engine while a turn runs; the accepted role is recorded with the session.
        """
        self.engine.set_role(name)
        if self.session_id:
            from .engine.loop import ROLE_STATE

            self.store.set_state(self.session_id, ROLE_STATE, self.engine.role_name)

    def note_brief(self, path: str, digest: str) -> None:
        """Record a ``--brief`` file's provenance in the session transcript (display history, not model history)."""
        append = getattr(self.store, "append_transcript", None)
        if callable(append):
            append(self.ensure_session_id(), "user", f"brief: {path} (sha256 {digest[:12]})")

    async def run(self, text: str, *, intent: Any = None) -> AcceptanceResult:
        """``intent``: an ``engine.intent.Intent`` that overrides the classifier (``nedster run --expect``)."""
        self.ensure_session_id()
        self._auto_title(text)
        self.checkpoints.begin_turn(self.session_id, self.project_root)
        try:
            return await self.engine.run(self.session_id, text, intent=intent)
        finally:
            self.checkpoints.end_turn()  # also on cancel/error: stores post-images of partial writes

    def ensure_session_id(self) -> str:
        """Persist the session row if lazy creation still owes one; always return the id.

        Called at the first submitted prompt (and by the TUI, so it can take the activity lock
        *before* the turn runs). Idempotent.
        """
        if not self.session_id:
            self.session_id = self.store.create(str(self.project_root), self._pending_title)
            self._pending_title = ""
        return str(self.session_id)

    def _auto_title(self, text: str) -> None:
        """Name a still-unnamed session from its first user turn (one place for CLI and TUI).

        Never overwrites a name; for an old blank-title session the first stored message wins.
        """
        session = self.store.get_session(self.session_id)
        if session is None or str(session.get("title") or "").strip():
            return
        name = normalize_title(self.store.first_user_text(self.session_id) or text)
        if name:
            self.store.set_title(self.session_id, name)

    def undo_last(self) -> Any:
        """Undo the last turn's file-tool changes (UndoReport; ``str()`` is the user message). Command side
        effects are not covered and the message says so."""
        return self.checkpoints.undo_last(self.session_id)

    def cancel(self) -> None:
        """Request cancellation; ``run`` returns CANCELLED once running jobs have really stopped."""
        self.engine.cancel()

    async def aclose(self) -> None:
        await self.transport.aclose()
        self.executor.close()
        self.checkpoints.close()
        self.store.close()
