"""Assemble the v4 native engine from configuration: transport, tools, session store and policy."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .contracts import AcceptanceResult, EngineEvent

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
        if session_id is None and resume_latest:
            session_id = self.store.latest_session(root)
        self.session_id = session_id or self.store.create(root, title)
        self.model = worker_model(config, model)
        self.tools, self.executor = build_tools(self.project_root, config, self.checkpoints)
        self.checkpoints.on_undone.append(self.tools.invalidate_after_undo)
        self.transport = OllamaTransport(config.models.ollama_url)
        soul_mode = str(getattr(config.runtime, "soul", "off") or "off")
        self.engine = Engine(
            self.transport,
            self.tools,
            model=self.model,
            store=self.store,
            budget=engine_budget(config, self.model),
            emit=emit,
            project_root=self.project_root,
            soul=soul_mode,
        )
        from .engine.prompt import session_instructions, session_soul

        self.instructions = session_instructions(self.store, self.session_id, self.project_root)
        self.soul = session_soul(self.store, self.session_id, soul_mode)

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
        }

    async def run(self, text: str, *, intent: Any = None) -> AcceptanceResult:
        """``intent``: an ``engine.intent.Intent`` that overrides the classifier (``nedster run --expect``)."""
        self.checkpoints.begin_turn(self.session_id, self.project_root)
        try:
            return await self.engine.run(self.session_id, text, intent=intent)
        finally:
            self.checkpoints.end_turn()  # also on cancel/error: stores post-images of partial writes

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
