"""Assemble the v4 native engine from configuration: transport, tools, session store and policy."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .contracts import AcceptanceResult, EngineEvent

DEFAULT_WORKER = "nedster-worker"


def worker_model(config: Any, override: str | None = None) -> str:
    if override:
        return override
    coder = str(getattr(config.models, "coder", "") or "")
    return coder if coder and coder != "auto" else DEFAULT_WORKER


def build_tools(project_root: Path, config: Any) -> tuple[Any, Any]:
    """Return (CompositeToolRuntime, Executor). File/search tools are optional until package B1 lands."""
    from .tool_runtime.composite import CompositeToolRuntime
    from .tool_runtime.exec import Executor
    from .tool_runtime.git import SafeGit

    executor = Executor(project_root, config)
    sync_tools: list[Any] = []
    try:
        from .tool_runtime.files import FileTools

        sync_tools.append(FileTools(project_root))
    except ImportError:
        pass
    try:
        from .tool_runtime import search

        provider = getattr(search, "SearchTools", None)
        if provider is not None:
            sync_tools.append(provider(project_root))
    except ImportError:
        pass
    runtime = CompositeToolRuntime(
        sync_tools=sync_tools,
        job_tools=[executor, SafeGit(project_root, executor)],
        policy=config.policy,
        resolve_argv=executor.resolve_argv,
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
        from .engine.state import Budget
        from .session_store.sqlite_store import SqliteSessionStore
        from .transport.ollama import OllamaTransport

        self.config = config
        self.project_root = Path(config.runtime.project_root).resolve()
        state_dir = Path(config.runtime.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        self.store = SqliteSessionStore(state_dir / "sessions.sqlite3")
        root = str(self.project_root)
        if session_id is None and resume_latest:
            session_id = self.store.latest_session(root)
        self.session_id = session_id or self.store.create(root, title)
        self.model = worker_model(config, model)
        self.tools, self.executor = build_tools(self.project_root, config)
        self.transport = OllamaTransport(config.models.ollama_url)
        self.engine = Engine(
            self.transport,
            self.tools,
            model=self.model,
            store=self.store,
            budget=Budget(context_tokens=int(getattr(config.runtime, "context_tokens", 32_768) or 32_768)),
            emit=emit,
            project_root=self.project_root,
        )

    def info(self) -> dict[str, Any]:
        status = self.executor.backend_status()
        return {
            "model": self.model,
            "session_id": self.session_id,
            "project_root": str(self.project_root),
            "sandbox": status.name if status else "none",
            "isolation": status.isolation if status else "none",
        }

    async def run(self, text: str) -> AcceptanceResult:
        return await self.engine.run(self.session_id, text)

    def cancel(self) -> None:
        """Request cancellation; ``run`` returns CANCELLED once running jobs have really stopped."""
        self.engine.cancel()

    async def aclose(self) -> None:
        await self.transport.aclose()
        self.executor.close()
        self.store.close()
