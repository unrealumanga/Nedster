from __future__ import annotations

import asyncio
import difflib
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from rich.syntax import Syntax
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.events import Resize
from textual.widgets import Footer, OptionList, RichLog, Static

from ironclad.config import NedsterConfig
from ironclad.contracts import EngineEvent
from ironclad.session_store.sqlite_store import SHORT_ID, make_owner
from ironclad.session_store.titles import display_title, normalize_title, relative_age

from .notify import notification
from .widgets.composer import Composer, History, project_files
from .widgets.delete_confirm import DeleteConfirm
from .widgets.glyphs import Glyphs
from .widgets.output_viewer import OutputViewer
from .widgets.session import (
    EngineSession,
    EventCallback,
    NativeSession,
    SessionEvent,
    SessionNote,
    failure_reason,
)
from .widgets.session_picker import SessionOption, SessionPicker
from .widgets.status import StatusBar, isolation_label
from .widgets.transcript import (
    RESUME_BANNER,
    AssistantMessage,
    Banner,
    Line,
    OutcomeBadge,
    ToolRow,
    Transcript,
    bounded,
    replay_widgets,
)

COMMANDS: dict[str, str] = {
    "/help": "show commands and keys",
    "/new": "start a new conversation",
    "/clear": "clear the display (conversation is kept)",
    "/sessions": "browse saved sessions (Ctrl+S)",
    "/resume": "resume a session: /resume [id] (no id opens the picker)",
    "/model": "show or switch the model: /model [name]",
    "/status": "model, context, sandbox, session",
    "/rename": "rename this session: /rename <name>",
    "/delete": "delete a conversation and its checkpoints: /delete [id]",
    "/diff": "diff of files changed in this session",
    "/undo": "undo the last edit (when the engine supports it)",
    "/doctor": "readiness check (inference / read-only / execution)",
    "/models": "browse and switch installed models",
    "/permissions": "show the autonomy policy",
    "/quit": "exit Nedster",
}
# Commands that change engine/session state wait for the running turn (the text is kept).
IDLE_ONLY = {"/new", "/resume", "/model", "/models", "/undo", "/sessions", "/rename", "/delete"}
# Commands that take no argument: extra text is rejected with a usage hint instead of being ignored.
NO_ARG_COMMANDS = frozenset({
    "/help", "/new", "/clear", "/sessions", "/models", "/status",
    "/diff", "/undo", "/doctor", "/permissions", "/quit", "/exit",
})

#: How much stored history /resume replays into the view (display only, never re-executed).
RESUME_REPLAY_LIMIT = 200
#: Activity-lock heartbeat period; well below the store's 60 s lock TTL.
LOCK_REFRESH_S = 5.0
#: Longest the @file inventory may stay stale when no turn changed files.
FILE_REFRESH_S = 30.0
#: Acceptance outcomes that count as a failed turn for the completion notice.
FAILURE_OUTCOMES = frozenset({"failed", "incomplete", "unvalidated"})

DoctorProvider = Callable[[NedsterConfig], Mapping[str, Mapping[str, Any]]]
SessionFactoryFor = Callable[[NedsterConfig, EventCallback], EngineSession]


class _WiredNativeSession(NativeSession):
    """TUI adapter over ``ironclad.native.NativeSession``: shares its store/session and closes its resources."""

    core: Any = None

    def ensure_session(self) -> str:
        """Persist the (lazy) session row so the activity lock can be taken before the turn runs."""
        if self.core is None:
            return str(self.session_id)
        self.session_id = self.core.ensure_session_id()
        return str(self.session_id)

    async def submit(self, text: str) -> None:
        # Through the core session, not the bare engine: it opens the checkpoint turn that file mutations need
        # (and that /undo restores). /resume may have switched session_id, so the core follows it.
        if self.core is None:
            await super().submit(text)
            return
        self.ensure_session()
        self.core.session_id = self.session_id
        await self.core.run(text)

    def undo_last(self) -> Any:
        self.core.session_id = self.session_id
        return self.core.undo_last()

    def set_model(self, name: str) -> None:
        super().set_model(name)
        if self.core is None:
            return
        from ironclad.native import context_budget

        self.core.model = name
        budget = getattr(self.core.engine, "budget", None)
        if budget is not None:  # a switch must rebuild the compaction budget for the new window
            budget.context_tokens = context_budget(self.core.config, name)

    async def close(self) -> None:
        if self.core is not None:
            await self.core.aclose()


def native_session(
    config: NedsterConfig,
    emit: EventCallback,
    *,
    session_id: str | None = None,
    resume_latest: bool = False,
    lazy: bool = True,
) -> EngineSession:
    from ironclad.native import NativeSession as CoreSession

    holder: dict[str, CoreSession] = {}

    def build(engine_emit: Callable[[Any], None]) -> Any:
        holder["core"] = CoreSession(
            config,
            emit=engine_emit,
            session_id=session_id,
            resume_latest=resume_latest,
            lazy=lazy,
        )
        return holder["core"].engine

    session = _WiredNativeSession(build, emit, endpoint=config.models.ollama_url)
    core = holder["core"]
    info = core.info()
    session.core = core
    session.store = core.store
    session.session_id = core.session_id
    session.started_new = core.started_new
    session.project_root = info["project_root"]
    session.sandbox, session.isolation = info["sandbox"], info["isolation"]
    return session


def _default_doctor(config: NedsterConfig) -> Mapping[str, Mapping[str, Any]]:
    from .model_manager import readiness

    return readiness(config)


def session_diff(root: Path, paths: Sequence[str]) -> str:
    """Unified diff of ``paths`` against HEAD; untracked/new files are shown as additions."""
    paths = list(dict.fromkeys(paths))
    if not paths:
        return ""
    chunks: list[str] = []
    covered: set[str] = set()
    for base in (["HEAD"], []):
        try:
            result = subprocess.run(
                ["git", "diff", "--no-color", "--no-ext-diff", *base, "--", *paths],
                cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            break
        if result.returncode == 0:
            chunks.append(result.stdout)
            covered = {p for p in paths if f"+++ b/{p}" in result.stdout or f"--- a/{p}" in result.stdout}
            break
    for path in paths:
        if path in covered:
            continue
        file = root / path
        if not file.is_file():
            continue
        try:
            lines = file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        except OSError:
            continue
        diff = difflib.unified_diff([], lines, "/dev/null", f"b/{path}")
        chunks.append("".join(diff))
    return "".join(chunks).strip("\n")


class NedsterApp(App[None]):
    TITLE = "Nedster"
    SUB_TITLE = "local AI agent"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen { background: #0b0d10; color: #e6edf3; layout: vertical; }
    #identity { height: 1; padding: 0 2; background: #0b0d10; color: #f0f6fc; text-style: bold; }
    #work { height: 1fr; padding: 0 2; background: #0b0d10; color: #e6edf3; scrollbar-color: #394150; scrollbar-color-hover: #596579; scrollbar-size-vertical: 1; }
    #details { height: 1fr; padding: 0 2; display: none; background: #0b0d10; color: #e6edf3; }
    #completions { height: auto; max-height: 8; margin: 0 2; display: none; background: #11151b; border: round #394150; }
    #search-hint { height: 1; padding: 0 2; display: none; background: #0b0d10; color: #d29922; }
    #follow-hint { height: 1; padding: 0 2; display: none; background: #0b0d10; color: #58a6ff; }
    #prompt { height: auto; min-height: 3; max-height: 8; margin: 0 2; background: #11151b; color: #f0f6fc; border: round #394150; }
    #prompt:focus { border: round #7aa2f7; }
    #status { height: 1; padding: 0 2; background: #0b0d10; color: #c9d1d9; }
    #message { height: 1; padding: 0 2; background: #0b0d10; color: #8b949e; }
    Footer { background: #0b0d10; color: #8b949e; }
    #too-small { display: none; height: 1fr; content-align: center middle; text-align: center; }
    .queued { color: #d29922; }
    .no-color, .no-color #identity, .no-color #work, .no-color #details, .no-color #message, .no-color #status,
    .no-color Footer, .no-color #prompt, .no-color #completions, .no-color #search-hint, .no-color #follow-hint { color: ansi_default; background: ansi_default; }
    .no-color #prompt, .no-color #completions { border: round ansi_default; }
    .no-color #prompt:focus { border: round ansi_default; }
    .no-color .queued { color: ansi_default; text-style: italic; }
    .ascii #prompt, .ascii #prompt:focus, .ascii #completions { border: ascii; }
    .ascii #work { scrollbar-size-vertical: 0; }
    """

    BINDINGS = [
        Binding("f1", "help", "help"),
        Binding("f4", "models", "models"),
        Binding("ctrl+s", "session_picker", "sessions"),
        Binding("ctrl+r", "history", "history"),
        Binding("escape", "escape", "stop", priority=True),
        Binding("f2", "details", "events", show=False),
        Binding("ctrl+c", "interrupt", "quit x2", priority=True, show=False),
        Binding("ctrl+l", "clear", "clear", show=False),
        Binding("ctrl+o", "verbose", "verbose", show=False),
    ]

    def __init__(
        self,
        config: NedsterConfig,
        *,
        ascii_only: bool = False,
        no_color: bool = False,
        session_factory: SessionFactoryFor | None = None,
        doctor: DoctorProvider | None = None,
        files: Callable[[], Sequence[str]] | None = None,
        session_ref: str | None = None,
        resume_latest: bool = False,
        open_picker_on_start: bool = False,
        model_source: Callable[[str], Sequence[Any]] | None = None,
    ):
        super().__init__()
        self.config = config
        self.ascii_only = ascii_only
        self.no_color_mode = no_color
        self.glyphs = Glyphs(ascii_only=ascii_only, no_color=no_color)
        self.session_factory = session_factory
        self.doctor = doctor or _default_doctor
        self.session_ref = session_ref
        self.resume_latest = resume_latest
        self.open_picker_on_start = open_picker_on_start
        root = Path(config.runtime.project_root)
        self._files_fetch = files or (lambda: project_files(root))
        self._files_cache: list[str] = []  # refreshed by a worker; read from the UI thread without blocking
        self._files_at = 0.0
        self._files_refreshing = False
        self.model_source = model_source or self._default_model_source
        self.history = History.for_project(Path(config.runtime.state_dir), root)
        self.session: EngineSession | None = None
        self.busy = False
        self.cancelling = False
        self.queue: list[str] = []
        self.queued_lines: list[Line] = []
        self.details_visible = False
        self.event_buffer: list[str] = []
        self.tool_rows: dict[str, ToolRow] = {}
        self.changed_paths: list[str] = []
        self._turn_changed = False  # did the current turn touch any file (drives @file inventory refresh)?
        self.current_reply: AssistantMessage | None = None
        self.turn_replies: list[AssistantMessage] = []
        self.pending_badge: OutcomeBadge | None = None
        self.turn_closed = True
        self.last_interrupt = 0.0
        self.verbose = False  # Ctrl+O: auto-expand every tool row (in-memory, per session)
        self._output_handles: dict[str, str] = {}  # call_id -> stored-output handle (for the full-output modal)
        self._turn_started = 0.0  # wall-clock start of the running turn (drives the completion notice)
        self._turn_outcome = ""  # acceptance outcome reported by the engine for this turn
        self._notify_writer: Callable[[str], None] = self._default_notify_writer
        self._model_started: float | None = None  # wall-clock start of the model call in flight (speed fallback)
        self._loop_thread = threading.get_ident()
        self._lock_owner = make_owner("tui")
        self._locked: tuple[Any, str] | None = None  # (store, session_id) held by this window
        self._last_lock_refresh = 0.0

    # ------------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        g = self.glyphs
        yield Static("terminal too small - need at least 80x24", id="too-small")
        yield Static(g.text(f"Nedster / {self._project_name()}{g.sep}starting"), id="identity")
        yield Transcript(id="work")
        yield RichLog(id="details", markup=False, wrap=False, auto_scroll=True)
        yield OptionList(id="completions")
        yield Static("", id="search-hint")
        yield Static("", id="follow-hint")
        yield Composer(
            id="prompt",
            history=self.history,
            commands=list(COMMANDS),
            command_descriptions=COMMANDS,
            files=lambda: self._files_cache,
            menu=lambda: self.query_one("#completions", OptionList),
            hint=self._search_hint,
        )
        yield StatusBar(g, id="status")
        yield Static("", id="message")
        yield Footer(show_command_palette=False)

    def _project_name(self) -> str:
        return Path(self.config.runtime.project_root).name or str(self.config.runtime.project_root)

    def _default_model_source(self, current: str) -> Sequence[Any]:
        """Installed Ollama models; runs in a worker thread so the UI never blocks on the network."""
        from .model_manager import OllamaManager, list_installed_models

        manager = OllamaManager(self.config.models.ollama_url)
        return list_installed_models(manager, current)

    async def on_mount(self) -> None:
        self._loop_thread = threading.get_ident()
        if self.no_color_mode:
            self.screen.add_class("no-color")
        if self.ascii_only:
            self.screen.add_class("ascii")
        self._apply_size(self.size.width, self.size.height)
        self.transcript.on_follow_change = self._show_follow_hint
        prompt = self.query_one("#prompt", Composer)
        prompt.cursor_blink = False
        prompt.focus()
        g = self.glyphs
        self.transcript.add(Line(g.text("Nedster  local coding agent"), g.style("strong")))
        self.transcript.add(Line(g.text(str(self.config.runtime.project_root)), g.style("muted")))
        self.transcript.add(Line(g.text(
            "Ask a question or describe the outcome you want. Enter sends, Shift+Enter or Ctrl+J adds a line, "
            "@ mentions a file, / lists commands, Esc stops a running turn."
        ), g.style("muted")))
        self.set_interval(0.5, self._tick)
        self.run_worker(self._refresh_files(), group="files", exclusive=True)
        self.run_worker(self._initialize(), group="startup", exclusive=True)

    @property
    def transcript(self) -> Transcript:
        return self.query_one("#work", Transcript)

    @property
    def status_bar(self) -> StatusBar:
        return self.query_one("#status", StatusBar)

    def _apply_size(self, width: int, height: int) -> None:
        too_small = width < 80 or height < 24
        self.query_one("#too-small", Static).display = too_small
        self.query_one("#identity").display = not too_small
        self.query_one("#work").display = not too_small and not self.details_visible
        self.query_one("#details").display = not too_small and self.details_visible
        if too_small:
            self.query_one("#completions").display = False
            self.query_one("#search-hint").display = False
            self.query_one("#follow-hint").display = False
        for selector in ("#prompt", "#status", "#message", "Footer"):
            self.query_one(selector).display = not too_small

    def on_resize(self, event: Resize) -> None:
        self._apply_size(event.size.width, event.size.height)

    def _tick(self) -> None:
        self._heartbeat_lock()
        if self.busy:
            try:
                status = self.status_bar
            except NoMatches:  # the timer can fire while the app is shutting down and widgets are gone
                return
            status.refresh()
            for row in self.tool_rows.values():
                if row.finished is None:
                    row.refresh()
        elif (
            not self._files_refreshing
            and self._files_at
            and time.monotonic() - self._files_at >= FILE_REFRESH_S
        ):
            # keep the @file inventory from going stale even when no turn has run for a while
            self.run_worker(self._refresh_files(), group="files", exclusive=True)

    # ------------------------------------------------------------------ startup

    def _make_session(self) -> EngineSession:
        if self.session_factory is not None:
            return self.session_factory(self.config, self.handle_event)
        return native_session(
            self.config,
            self.handle_event,
            session_id=self.session_ref,
            resume_latest=self.resume_latest,
            lazy=self.session_ref is None and not self.resume_latest,
        )

    async def _initialize(self) -> None:
        g = self.glyphs
        try:
            self.session = self._make_session()
            start = getattr(self.session, "start", None)
            if start is not None:
                await start()
        except Exception as exc:  # noqa: BLE001 - shown as a setup remedy, never a crash
            self.session = None
            self.query_one("#identity", Static).update(
                g.text(f"Nedster / {self._project_name()}{g.sep}SETUP NEEDED"))
            self._line(f"{g.icon('fail')} Could not start the runtime: {exc}", "fail")
            self._line("Run `nedster setup` to install Ollama and matched models, then reopen Nedster. /doctor shows details.")
            self.status_bar.set_state("setup needed")
            return
        self._refresh_info()
        session_id = str(getattr(self.session, "session_id", "") or "")
        if session_id:  # a lazily-created session has no id yet and needs no lock
            self._acquire_lock(session_id)
            if not bool(getattr(self.session, "started_new", True)):
                self._replay_session(session_id)  # opened straight on an existing conversation
        self._set_state("idle")
        self._message("Ready")
        if self.open_picker_on_start:
            self._open_session_picker()

    def _refresh_info(self) -> None:
        if self.session is None:
            return
        info = self.session.info()
        self.status_bar.set_info(info)
        g = self.glyphs
        provider = str(info.get("provider") or "LOCAL")
        where = provider
        title = self._current_title()
        self.query_one("#identity", Static).update(
            g.text(f"Nedster / {self._project_name()}{g.sep}{title}{g.sep}{where}{g.sep}{info.get('model') or 'no model'}")
        )

    # ------------------------------------------------------------------ helpers

    def _text(self, value: str) -> str:
        return self.glyphs.text(value)

    def _line(self, text: str, style: str = "muted") -> Line:
        line = Line(self._text(text), self.glyphs.style(style))
        self.transcript.add(line)
        return line

    def _message(self, text: str) -> None:
        self.query_one("#message", Static).update(Text(self._text(text)))

    def _search_hint(self, text: str | None) -> None:
        """Composer callback: show/hide the one-line reverse-search hint above the prompt."""
        hint = self.query_one("#search-hint", Static)
        hint.display = text is not None
        hint.update(Text(self._text(text or "")))

    async def _refresh_files(self, *, force: bool = False) -> None:
        """Refresh the @file inventory in a worker thread; a changed turn forces it, otherwise 30 s TTL."""
        if not force and self._files_cache and time.monotonic() - self._files_at < FILE_REFRESH_S:
            return
        self._files_refreshing = True
        try:
            files = await asyncio.to_thread(self._files_fetch)
        except Exception:  # noqa: BLE001 - a failed scan keeps the last inventory rather than crashing
            return
        finally:
            self._files_refreshing = False
        self._files_cache = list(files)
        self._files_at = time.monotonic()

    def _set_state(self, state: str, detail: str = "", *, force: bool = False) -> None:
        if self.cancelling and not force:
            return  # "cancelling…" persists until the engine confirms
        self.status_bar.set_state(state, detail)

    # ------------------------------------------------------------------ session locks + resume

    def _session_store(self) -> Any:
        return getattr(self.session, "store", None) if self.session is not None else None

    def _title_of(self, session_id: str, title: str = "") -> str:
        """Display name for a session: stored title, else its first message, else "Untitled conversation"."""
        store = self._session_store()
        if store is None:
            return display_title(title)
        if not str(title or "").strip():
            getter = getattr(store, "get_session", None)
            record = getter(session_id) if callable(getter) else None
            title = str(record.get("title") or "") if record else title
        if str(title or "").strip():
            return display_title(title)
        first = getattr(store, "first_user_text", None)
        fallback = first(session_id) if callable(first) else ""
        return display_title("", fallback)

    def _current_title(self) -> str:
        if self.session is None:
            return ""
        return self._title_of(str(getattr(self.session, "session_id", "")))

    def _acquire_lock(self, session_id: str, *, warn: bool = True) -> bool:
        """Mark this window as the holder of ``session_id``; False when another live Nedster has it."""
        store = self._session_store()
        acquire = getattr(store, "acquire_lock", None)
        if not callable(acquire):
            return True
        try:
            acquired = bool(acquire(session_id, self._lock_owner))
        except Exception as exc:  # noqa: BLE001 - shown, never fatal
            if warn:
                self._line(f"Could not lock session: {exc}", "warn")
            return False
        if acquired:
            self._locked = (store, session_id)
            self._last_lock_refresh = time.monotonic()
        elif warn:
            self._line("That session is active in another Nedster window; close it there first.", "warn")
        return acquired

    def _release_lock(self) -> None:
        if self._locked is None:
            return
        store, session_id = self._locked
        self._locked = None
        release = getattr(store, "release_lock", None)
        if callable(release):
            with suppress(Exception):
                release(session_id, self._lock_owner)

    def _heartbeat_lock(self) -> None:
        """Keep the lock of the session this window is holding alive (see LOCK_REFRESH_S)."""
        if self._locked is None:
            return
        now = time.monotonic()
        if now - self._last_lock_refresh < LOCK_REFRESH_S:
            return
        self._last_lock_refresh = now
        store, session_id = self._locked
        refresh = getattr(store, "refresh_lock", None)
        if not callable(refresh):
            return
        try:
            renewed = bool(refresh(session_id, self._lock_owner))
        except Exception:  # noqa: BLE001 - a failed heartbeat must not crash the UI
            return
        if renewed:
            return
        acquire = getattr(store, "acquire_lock", None)
        try:
            recovered = bool(acquire(session_id, self._lock_owner)) if callable(acquire) else False
        except Exception:  # noqa: BLE001
            recovered = False
        self._message("" if recovered else "Session lock lost - another Nedster may have taken over")

    def _activate_session(self) -> None:
        """Create the lazily-held session row and take its activity lock before the first turn runs."""
        if self.session is None:
            return
        ensure = getattr(self.session, "ensure_session", None)
        if not callable(ensure):
            return
        try:
            session_id = str(ensure() or "")
        except Exception as exc:  # noqa: BLE001 - the turn itself will surface the real failure
            self._line(f"Could not prepare the session: {exc}", "warn")
            return
        if session_id and self._locked is None:
            self._acquire_lock(session_id)

    def _resolve_session_id(self, wanted: str | None) -> str:
        """Resolve ``/resume`` so the lock is taken *before* switching.

        An explicit ref goes through the store's one resolver (full id, unique prefix or exact
        title, over every session); a bare ``/resume`` keeps the newest-session behaviour.
        """
        store = self._session_store()
        resolver = getattr(store, "resolve_session", None)
        if wanted is not None and callable(resolver):
            return str(resolver(str(self.config.runtime.project_root), wanted)["id"])
        lister = getattr(self.session, "list_sessions", None)
        if lister is not None:
            sessions = list(lister())
        else:
            store_lister = getattr(store, "list_sessions", None)
            sessions = list(store_lister(str(self.config.runtime.project_root))) if callable(store_lister) else []
        if not sessions:
            raise LookupError("no saved sessions for this project")
        return str(sessions[0]["id"])

    def _replay_session(self, session_id: str) -> None:
        """Rebuild the view from the stored display transcript, then warn that it is history."""
        store = self._session_store()
        self.action_clear()
        self.changed_paths.clear()
        entries: list[Mapping[str, Any]] = []
        if store is not None:
            repair = getattr(store, "repair_interrupted", None)
            if callable(repair):
                try:
                    added = int(repair(session_id))
                except Exception as exc:  # noqa: BLE001 - replay must survive a damaged history
                    self._line(f"Could not repair interrupted calls: {exc}", "warn")
                else:
                    if added:
                        self._line(f"Repaired {added} interrupted tool call(s) from the last run.", "warn")
            read = getattr(store, "transcript", None)
            if callable(read):
                try:
                    entries = list(read(session_id, RESUME_REPLAY_LIMIT))
                except Exception as exc:  # noqa: BLE001
                    self._line(f"Could not read the transcript: {exc}", "warn")
        self._line(f"Resumed session {session_id[:SHORT_ID]}", "accent")
        for widget in replay_widgets(self.glyphs, entries, output_loader=self._output_loader_for):
            self.transcript.add(widget)
        if not entries:
            self._line("No saved transcript for this session.", "muted")
        self.transcript.add(Banner(self.glyphs, RESUME_BANNER))
        self._refresh_info()
        self.status_bar.reset_usage()  # the resumed session's context usage is unknown until the next call

    # ------------------------------------------------------------------ picker

    def _picker_options(self, store: Any) -> list[SessionOption]:
        """Every non-empty conversation of the project (never a capped page), annotated for the picker."""
        root = str(self.config.runtime.project_root)
        searcher = getattr(store, "search_sessions", None)
        if callable(searcher):
            rows = list(searcher(root, exclude_empty=True))
        else:  # a store double without search_sessions: fall back to the list page
            rows = [row for row in store.list_sessions(root) if int(row.get("message_count", 1)) > 0]
        current = str(getattr(self.session, "session_id", "") or "")
        held = self._locked[1] if self._locked is not None else ""
        is_active = getattr(store, "is_active", None)
        options: list[SessionOption] = []
        for row in rows:
            session_id = str(row["id"])
            active = bool(is_active(session_id)) if callable(is_active) else False
            options.append(
                SessionOption(
                    id=session_id,
                    title=str(row.get("title") or ""),
                    message_count=int(row.get("message_count") or 0),
                    updated_ns=int(row.get("updated_ns") or 0),
                    first_user_text=str(row.get("first_user_text") or ""),
                    last_user_text=str(row.get("last_user_text") or ""),
                    current=session_id == current,
                    locked=active and session_id != held,
                )
            )
        return options

    def _open_session_picker(self) -> None:
        """Open the conversation picker (``/sessions``, bare ``/resume``, Ctrl+S and ``--continue``)."""
        if self.session is None:
            self._line("Saved sessions are not available with this session.", "warn")
            return
        store = self._session_store()
        if store is None or not callable(getattr(store, "list_sessions", None)):
            self._line("Saved sessions are not available with this session.", "warn")
            return
        options = self._picker_options(store)
        if not options:
            self._line("No saved conversations for this project.")
            return
        self.push_screen(
            SessionPicker(
                options, busy=self.busy, ascii_only=self.ascii_only, on_delete=self._delete_from_picker
            ),
            self._picker_chosen,
        )

    def _picker_chosen(self, session_id: str | None) -> None:
        if session_id is None:  # Esc: the composer draft is untouched
            return
        self._cmd_resume([session_id])

    # ------------------------------------------------------------------ delete

    def _checkpoints(self) -> Any:
        return getattr(getattr(self.session, "core", None), "checkpoints", None)

    def _delete_preview(self, store: Any, session_id: str) -> str:
        record = store.get_session(session_id) or {}
        counter = getattr(self._checkpoints(), "checkpoint_count", None)
        count = counter(session_id) if callable(counter) else 0
        title = self._title_of(session_id, str(record.get("title") or ""))
        updated = relative_age(int(record.get("updated_ns") or 0)) or "unknown"
        messages = len(store.messages(session_id))
        return (
            f"{session_id[:SHORT_ID]}  {title}\n"
            f"updated {updated}{self.glyphs.sep}{messages} message(s){self.glyphs.sep}{count} checkpoint(s)"
        )

    def _confirm_delete(self, session_id: str, *, after: Any = None) -> None:
        current = str(getattr(self.session, "session_id", "") or "") if self.session is not None else ""
        if session_id == current:
            self._line("Can't delete the current session; switch to another session first.", "warn")
            return
        store = self._session_store()
        if store is None or not callable(getattr(store, "delete_session", None)):
            self._line("Deleting is not available with this session.", "warn")
            return
        preview = self._delete_preview(store, session_id)

        def chosen(confirmed: bool | None) -> None:
            if confirmed:
                self._perform_delete(store, session_id, after)

        self.push_screen(DeleteConfirm(preview), chosen)

    def _delete_from_picker(self, session_id: str) -> None:
        self._confirm_delete(session_id, after=self._close_picker)

    def _close_picker(self) -> None:
        if isinstance(self.screen, SessionPicker):
            self.pop_screen()

    def _perform_delete(self, store: Any, session_id: str, after: Any = None) -> None:
        from ironclad.session_store.checkpoints import delete_session_cascade

        checkpoints = self._checkpoints()
        if checkpoints is None:
            deleted = bool(store.delete_session(session_id))
        else:
            deleted = delete_session_cascade(store, checkpoints, session_id)
        if after is not None:  # pop the picker before writing to the transcript behind it
            after()
        if not deleted:
            self._line("That session is active in another Nedster window; close it there first.", "warn")
            return
        self._line(f"Deleted session {session_id[:SHORT_ID]}.", "accent")

    # ------------------------------------------------------------------ events

    def handle_event(self, event: SessionEvent) -> None:
        """Session callback; safe to call from any thread."""
        if threading.get_ident() != self._loop_thread:
            self.call_from_thread(self._dispatch, event)
        else:
            self._dispatch(event)

    def _dispatch(self, event: SessionEvent) -> None:
        if isinstance(event, SessionNote):
            if event.level == "raw":
                self._record(event.text)
                return
            self._record(f"note {event.text}")
            self._line(event.text, event.level)
            return
        self._record(f"{event.kind} {dict(event.data)}")
        handler = getattr(self, "_on_" + event.kind.replace(".", "_"), None)
        if handler is not None:
            handler(event)

    def _record(self, line: str) -> None:
        line = self._text(line)
        self.event_buffer.append(line)
        if len(self.event_buffer) > 500:
            del self.event_buffer[:100]
        if self.details_visible:
            self.query_one("#details", RichLog).write(line)

    def _on_turn_started(self, event: EngineEvent) -> None:
        self.turn_closed = False
        self._model_started = None
        self._set_state("thinking")

    def _on_model_delta(self, event: EngineEvent) -> None:
        if self._model_started is None:
            self._model_started = time.monotonic()
        if self.current_reply is None:
            self.current_reply = AssistantMessage(self.glyphs)
            self.turn_replies.append(self.current_reply)
            self.transcript.add(self.current_reply)
        self.current_reply.append(str(event.data.get("text") or ""), str(event.data.get("kind") or "content"))
        self.transcript.follow_output()
        self._set_state("thinking")

    def _on_model_usage(self, event: EngineEvent) -> None:
        data = event.data
        wall = time.monotonic() - self._model_started if self._model_started is not None else None
        self._model_started = None
        self.status_bar.note_usage(
            int(data.get("prompt_tokens") or 0),
            int(data.get("completion_tokens") or 0),
            decode_ms=float(data.get("decode_ms") or 0.0),
            wall_s=wall,
        )
        self.status_bar.advance_step()

    def _on_tool_started(self, event: EngineEvent) -> None:
        self.current_reply = None
        call_id = str(event.data.get("call_id") or f"call-{len(self.tool_rows)}")
        tool = str(event.data.get("tool") or "tool")
        row = ToolRow(
            self.glyphs, call_id, tool, dict(event.data.get("arguments") or {}),
            diff_loader=self._diff_loader,
            output_loader=lambda cid=call_id: self._load_full_output(cid),
            expanded=self.verbose,
        )
        self.tool_rows[call_id] = row
        self.transcript.add(row)
        self._set_state("running tool", tool)

    def _on_tool_finished(self, event: EngineEvent) -> None:
        data = dict(event.data)
        call_id = str(data.get("call_id") or "")
        handle = str(data.get("output_handle") or "")
        if handle:
            self._output_handles[call_id] = handle
        row = self.tool_rows.get(call_id)
        if row is None:
            row = ToolRow(
                self.glyphs, call_id, str(data.get("tool") or "tool"), {},
                diff_loader=self._diff_loader,
                output_loader=lambda cid=call_id: self._load_full_output(cid),
                expanded=self.verbose,
            )
            self.tool_rows[call_id or f"call-{len(self.tool_rows)}"] = row
            self.transcript.add(row)
        row.finish(data)
        if failure_reason(str(data.get("status") or ""), data) is None:
            self._note_changes(data.get("mutated_paths") or [])
        running = [r for r in self.tool_rows.values() if r.finished is None]
        if running:
            self._set_state("running tool", running[-1].tool)
        else:
            self._set_state("thinking")

    def _on_acceptance(self, event: EngineEvent) -> None:
        data = event.data
        changed = [str(p) for p in data.get("changed_paths") or []]
        self._note_changes(changed)
        self.pending_badge = OutcomeBadge(
            self.glyphs, str(data.get("outcome") or ""), [str(r) for r in data.get("reasons") or []], changed
        )

    def _on_turn_finished(self, event: EngineEvent) -> None:
        answer = str(event.data.get("answer") or "").strip()
        self._turn_outcome = str(event.data.get("outcome") or "")
        self._show_answer(answer)
        if self.pending_badge is None and event.data.get("outcome"):
            self.pending_badge = OutcomeBadge(self.glyphs, str(event.data["outcome"]))
        self._close_turn()
        if self.cancelling:
            self.cancelling = False
            self._message("Cancelled")
        self._set_state("idle")

    def _on_error(self, event: EngineEvent) -> None:
        message = str(event.data.get("message") or "error")
        remedy = str(event.data.get("remedy") or "")
        g = self.glyphs
        self._line(f"{g.icon('fail')} {message}" + (f" {g.icon('dash')} {remedy}" if remedy else ""), "fail")

    def _show_answer(self, answer: str) -> None:
        streamed = "".join(r.content for r in self.turn_replies).strip()
        for reply in self.turn_replies:
            reply.finalize()
        if not answer:
            return
        last = self.turn_replies[-1] if self.turn_replies else None
        if last is not None and last.content.strip() and answer.startswith(last.content.strip()):
            last.finalize(answer)
        elif not streamed or answer != streamed:
            self.transcript.add(AssistantMessage(self.glyphs, answer, final=True))

    def _close_turn(self) -> None:
        if self.turn_closed:
            return
        self.turn_closed = True
        for row in self.tool_rows.values():
            if row.finished is None:
                row.finish({"status": "cancelled", "summary": "not completed"})
        if self.pending_badge is not None:
            self.transcript.add(self.pending_badge)
            self.pending_badge = None
        self.current_reply = None
        self.turn_replies = []

    def _note_changes(self, paths: Sequence[Any]) -> None:
        for path in paths:
            if str(path) not in self.changed_paths:
                self.changed_paths.append(str(path))
                self._turn_changed = True

    def _diff_loader(self, paths: Sequence[str]) -> str:
        return session_diff(Path(self.config.runtime.project_root), paths)

    # ------------------------------------------------------------------ completion notice

    def _turn_outcome_word(self) -> str:
        """One word for the completion notice: cancelled / failed / completed."""
        if self.cancelling:
            return "cancelled"
        return "failed" if self._turn_outcome in FAILURE_OUTCOMES else "completed"

    def _notify_turn(self, outcome: str, seconds: float) -> None:
        """Emit the configured completion notice; silent in machine modes (the TUI never runs there)."""
        mode = str(getattr(self.config.runtime, "notify", "bell") or "off")
        threshold = float(getattr(self.config.runtime, "notify_after_s", 15.0) or 0.0)
        data = notification(mode, outcome, seconds, threshold=threshold, interactive=True)
        if data is not None:
            self._notify_writer(data)

    def _default_notify_writer(self, data: str) -> None:
        """Write an escape sequence straight to the terminal (bypasses the screen buffer)."""
        file = getattr(self.console, "file", None)
        if file is None:
            return
        with suppress(Exception):
            file.write(data)
            file.flush()

    # ------------------------------------------------------------------ full output + follow hint

    def _load_full_output(self, call_id: str) -> str:
        """Read a tool call's complete stored output (blocking; call from a worker)."""
        store = self._session_store()
        session_id = str(getattr(self.session, "session_id", "")) if self.session is not None else ""
        reader = getattr(store, "output_for_call", None)
        if callable(reader) and session_id and call_id:
            try:
                return str(reader(session_id, call_id) or "")
            except Exception:  # noqa: BLE001 - fall back to whatever the row already has
                return ""
        return ""

    @on(ToolRow.FullOutput)
    def _open_full_output(self, message: ToolRow.FullOutput) -> None:
        row = message.row
        loader = row.output_loader
        if loader is None and not row.event_output():
            self._message("No stored output to show for this tool call")
            return
        name = self.glyphs.text(row.tool)
        self.push_screen(
            OutputViewer(
                self.glyphs,
                title=name,
                loader=loader,
                fallback=row.event_output(),
                historical=row.historical,
            )
        )

    def _output_loader_for(self, call_id: str) -> Callable[[], str]:
        return lambda: self._load_full_output(call_id)

    def _show_follow_hint(self, pending: bool) -> None:
        hint = self.query_one("#follow-hint", Static)
        if pending and self.size.width >= 80 and self.size.height >= 24:
            hint.update(Text(self._text("New output - End to follow")))
            hint.display = True
        else:
            hint.display = False

    # ------------------------------------------------------------------ input

    @on(Composer.Submitted, "#prompt")
    def on_submit(self, event: Composer.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        composer = event.composer
        if value.startswith("/"):
            command = value.split()[0].lower()
            if self.busy and command in IDLE_ONLY:
                self._message(f"{command} waits for the running turn {self.glyphs.sep} Esc stops it")
                return  # text stays in the composer
            composer.prompt_history.add(value)
            composer.set_text("")
            self._command(value)
            return
        composer.prompt_history.add(value)
        composer.set_text("")
        self.submit_text(value)

    def submit_text(self, value: str) -> None:
        if self.busy:
            self.queue.append(value)
            g = self.glyphs
            self.queued_lines.append(self._line(f"{g.icon('queued')} queued: {value}", "warn"))
            self.queued_lines[-1].add_class("queued")
            self.status_bar.queued = len(self.queue)
            self.status_bar.refresh()
            self._message(f"Queued {g.sep} sent after the current turn")
            return
        if self.session is None:
            self.query_one("#prompt", Composer).set_text(value)
            self._message("Runtime unavailable · run nedster setup (your text was kept)")
            return
        self._start_turn(value)

    def _start_turn(self, value: str) -> None:
        self._activate_session()
        self.busy = True
        self.turn_closed = False
        self._turn_started = time.monotonic()
        self._turn_outcome = ""
        self.status_bar.start_turn()
        self._set_state("thinking")
        self.run_worker(self._run_turn(value), group="turn")

    async def _run_turn(self, value: str) -> None:
        assert self.session is not None
        g = self.glyphs
        self._line(f"{g.icon('user')} {value}", "user")
        try:
            await self.session.submit(value)
        except asyncio.CancelledError:
            self.busy = False  # app shutdown: widgets are going away, nothing to render
            raise
        except Exception as exc:  # noqa: BLE001 - shown in the transcript
            self._line(f"{g.icon('fail')} Turn failed: {exc}", "fail")
        self._close_turn()
        self._notify_turn(self._turn_outcome_word(), time.monotonic() - self._turn_started)
        self.busy = False
        if self.cancelling:
            self.cancelling = False
            self._message("Cancelled")
        self.status_bar.end_turn()
        self._set_state("idle")
        self._refresh_info()
        self.run_worker(self._refresh_files(force=self._turn_changed), group="files", exclusive=True)
        self._turn_changed = False
        self._dequeue()

    def _dequeue(self) -> None:
        if self.busy or not self.queue or self.session is None:
            return
        value = self.queue.pop(0)
        line = self.queued_lines.pop(0)
        line.remove()
        self.status_bar.queued = len(self.queue)
        self._message("")
        self._start_turn(value)

    # ------------------------------------------------------------------ actions

    def action_escape(self) -> None:
        from textual.screen import ModalScreen

        if isinstance(self.screen, ModalScreen):  # app-level Esc has priority; let the modal handle it
            cancel = getattr(self.screen, "action_cancel", None)
            if cancel is not None:
                cancel()
            return
        composer = self.query_one("#prompt", Composer)
        if composer.search_active:
            composer.cancel_search()  # Esc restores the draft that was in the composer
            return
        if composer.completion is not None:
            composer.close_completion()
            return
        if self.busy:
            self.cancel_turn()
        # idle: Esc does nothing (Ctrl+C twice or /quit exits)

    def cancel_turn(self) -> None:
        if not self.busy or self.cancelling or self.session is None:
            return
        self.cancelling = True
        self._set_state("cancelling…", force=True)
        self._message("Cancelling… waiting for the engine to stop")
        if self.queue:
            composer = self.query_one("#prompt", Composer)
            restored = "\n".join([*self.queue, composer.text] if composer.text.strip() else self.queue)
            composer.set_text(restored)
            for line in self.queued_lines:
                line.remove()
            self.queue.clear()
            self.queued_lines.clear()
            self.status_bar.queued = 0
            self._line("Queued input was returned to the composer.")
        self.run_worker(self.session.cancel(), group="cancel")

    def action_interrupt(self) -> None:
        now = time.monotonic()
        if now - self.last_interrupt < 2.0:
            self.exit()
            return
        self.last_interrupt = now
        if self.busy:
            self.cancel_turn()
        self._message("Press Ctrl+C again to exit")

    def action_help(self) -> None:
        self._command("/help")

    def action_history(self) -> None:
        self.query_one("#prompt", Composer).cycle_search()

    def action_details(self) -> None:
        self.details_visible = not self.details_visible
        self.query_one("#work").display = not self.details_visible
        details = self.query_one("#details", RichLog)
        details.display = self.details_visible
        if self.details_visible:
            details.clear()
            for line in self.event_buffer:
                details.write(line)
        self._message("Raw events · F2 returns to conversation" if self.details_visible else "")

    def action_clear(self) -> None:
        self.transcript.remove_children()
        self.tool_rows = {k: v for k, v in self.tool_rows.items() if v.finished is None}
        self.transcript.following = True
        self.transcript.new_output = False
        self._show_follow_hint(False)

    def action_verbose(self) -> None:
        """Ctrl+O: expand every tool row in this session (in-memory; not persisted)."""
        self.verbose = not self.verbose
        for row in self.tool_rows.values():
            row.expanded = self.verbose
            row.refresh(layout=True)
        self._message("Verbose transcript · tool rows expanded" if self.verbose else "")

    def action_models(self) -> None:
        self._open_model_picker()

    def action_session_picker(self) -> None:
        self._open_session_picker()

    # ------------------------------------------------------------------ commands

    def _command(self, value: str) -> None:
        head = value.split(None, 1)
        command = head[0].lower()
        args = head[1].split() if len(head) > 1 else []
        rest = head[1] if len(head) > 1 else ""  # spaces kept for /rename
        handlers: dict[str, Callable[[list[str]], Any]] = {
            "/quit": lambda _a: self.exit(),
            "/exit": lambda _a: self.exit(),
            "/help": self._cmd_help,
            "/clear": lambda _a: self.action_clear(),
            "/new": self._cmd_new,
            "/sessions": self._cmd_sessions,
            "/resume": self._cmd_resume,
            "/rename": lambda _a: self._cmd_rename(rest),
            "/delete": self._cmd_delete,
            "/model": self._cmd_model,
            "/models": self._cmd_models,
            "/status": self._cmd_status,
            "/diff": lambda _a: self.run_worker(self._cmd_diff(), group="command"),
            "/undo": lambda _a: self.run_worker(self._cmd_undo(), group="command"),
            "/doctor": lambda _a: self.run_worker(self._cmd_doctor(), group="command"),
            "/permissions": self._cmd_permissions,
        }
        handler = handlers.get(command)
        if handler is None:
            self._line(f"Unknown command {command}. /help lists commands.", "warn")
            return
        if command in NO_ARG_COMMANDS and args:
            self._line(f"{command} takes no arguments · usage: {command}", "warn")
            return
        handler(args)

    def _cmd_help(self, _args: list[str]) -> None:
        self._line("Commands", "accent")
        for name, text in COMMANDS.items():
            self._line(f"  {name:<13}{text}")
        self._line("Keys", "accent")
        self._line("  Enter send · Shift+Enter/Ctrl+J newline · Up/Down history · Ctrl+R search history")
        self._line("  @ file · / command · Esc stop the running turn · Ctrl+C twice exit")
        self._line("  F1 help · F4 models · Ctrl+S switch conversation · F2 raw events · Ctrl+L clear · Ctrl+O verbose")
        self._line("  Tool rows: focus with Tab/click, Enter expands argv, output and diff")
        self._line("  On a tool row, press o for the full stored output (Esc closes)")

    def _cmd_new(self, _args: list[str]) -> None:
        new = getattr(self.session, "new", None)
        if new is None:
            self._line("This engine cannot start a new conversation; restart Nedster.", "warn")
            return
        session_id = new()
        self._release_lock()
        self._acquire_lock(str(session_id))
        self.action_clear()
        self.changed_paths.clear()
        self._line(f"New conversation {str(session_id)[:SHORT_ID]}", "accent")
        self._refresh_info()
        self.status_bar.reset_usage()

    def _cmd_sessions(self, _args: list[str]) -> None:
        self._open_session_picker()

    def _cmd_rename(self, name: str) -> None:
        store = self._session_store()
        setter = getattr(store, "set_title", None)
        if self.session is None or not callable(setter):
            self._line("Renaming is not available with this session.", "warn")
            return
        clean = normalize_title(name)
        if not clean:
            self._line("Usage: /rename <name>", "warn")
            return
        setter(str(self.session.session_id), clean)
        self._line(f"Renamed session {str(self.session.session_id)[:SHORT_ID]} to {clean}", "accent")
        self._refresh_info()

    def _cmd_delete(self, args: list[str]) -> None:
        """``/delete [ref]``: confirm, then remove the conversation and its checkpoints.

        Without a reference it opens the picker, where Del (or ``/delete`` on a row) runs the
        same flow against the highlighted conversation.
        """
        if not args:
            self._open_session_picker()
            return
        store = self._session_store()
        wanted = " ".join(args)
        if store is None or not callable(getattr(store, "resolve_session", None)):
            self._line("Deleting is not available with this session.", "warn")
            return
        from ironclad.session_store.sqlite_store import SessionResolutionError

        try:
            session_id = str(store.resolve_session(str(self.config.runtime.project_root), wanted)["id"])
        except SessionResolutionError as exc:
            self._line(str(exc), "warn")
            return
        self._confirm_delete(session_id)

    def _cmd_resume(self, args: list[str]) -> None:
        if not args:  # bare /resume: choose from the picker
            self._open_session_picker()
            return
        resume = getattr(self.session, "resume", None)
        if resume is None:
            self._line("Resuming a saved session is not available with this session.", "warn")
            return
        wanted = " ".join(args) if args else None  # join so an exact title may contain spaces
        store = self._session_store()
        target: str | None = None
        if callable(getattr(store, "transcript", None)):
            try:
                target = self._resolve_session_id(wanted)
            except LookupError as exc:
                self._line(str(exc), "warn")
                return
            if not self._acquire_lock(target):  # active elsewhere: keep the current conversation
                return
        try:
            session_id = str(resume(wanted if target is None else target))
        except LookupError as exc:
            self._line(str(exc), "warn")
            return
        except Exception as exc:  # noqa: BLE001 - shown in the transcript, never fatal
            self._line(f"Resume failed: {exc}", "fail")
            return
        if target is not None and session_id != target:
            self._release_lock()
            self._acquire_lock(session_id, warn=False)
        self._replay_session(session_id)

    def _cmd_models(self, _args: list[str]) -> None:
        self._open_model_picker()

    def _cmd_model(self, args: list[str]) -> None:
        if not args:
            self._open_model_picker()
            return
        self.run_worker(self._switch_model(args[0]), group="command")

    def _current_model(self) -> str:
        return str(self.session.info().get("model") or "") if self.session is not None else ""

    def _open_model_picker(self) -> None:
        if self.session is None:
            self._line("Runtime unavailable · run nedster setup", "warn")
            return
        if self.busy:
            self._message(f"Models waits for the running turn {self.glyphs.sep} Esc stops it")
            return
        self.run_worker(self._load_models_and_pick(), group="command")

    async def _load_models_and_pick(self) -> None:
        from .widgets.model_picker import ModelPicker

        current = self._current_model()
        try:
            models = await asyncio.to_thread(self.model_source, current)
        except Exception:  # noqa: BLE001 - a dead endpoint is explained, never a crash
            self._line(self._models_unreachable(), "warn")
            return
        if not models:
            self._line("No models are installed. Add one with: ollama pull <model>", "warn")
            return
        self.push_screen(ModelPicker(models, current=current, glyphs=self.glyphs), self._on_model_picked)

    def _on_model_picked(self, name: str | None) -> None:
        self._message("")
        if not name or self.session is None:
            return
        if name == self._current_model():
            self._message(f"{name} is already the active model")
            return
        self._apply_model(name)

    async def _switch_model(self, name: str) -> None:
        if self.session is None:
            self._line("Runtime unavailable · run nedster setup", "warn")
            return
        from .model_manager import match_installed_model, model_not_installed_message

        current = self._current_model()
        try:
            models = await asyncio.to_thread(self.model_source, current)
        except Exception:  # noqa: BLE001 - cannot validate offline; keep the previous model
            self._line(self._models_unreachable(), "warn")
            return
        installed = [model.name for model in models]
        canonical = match_installed_model(name, installed)
        if canonical is None:
            self._line(model_not_installed_message(name, installed), "warn")
            return
        self._apply_model(canonical)

    def _apply_model(self, name: str) -> None:
        setter = getattr(self.session, "set_model", None)
        if setter is None:
            self._line("This engine picks models from the fleet; change models.coder in config.toml.", "warn")
            return
        try:
            setter(name)
        except Exception as exc:  # noqa: BLE001 - shown, never fatal
            self._line(f"Could not switch model: {exc}", "fail")
            return
        self._line(f"Model set to {name}", "accent")
        self._refresh_info()
        self.status_bar.reset_usage()  # a different model's measured context/speed does not apply

    def _models_unreachable(self) -> str:
        g = self.glyphs
        return (
            f"{g.icon('fail')} Ollama is not reachable at {self.config.models.ollama_url}. "
            "Start it (`ollama serve`), then retry."
        )

    def _cmd_status(self, _args: list[str]) -> None:
        if self.session is None:
            self._line("Runtime not started. Run /doctor.", "warn")
            return
        info = self.session.info()

        label, danger = isolation_label(str(info.get("sandbox") or ""), str(info.get("isolation") or ""))
        rows = [
            ("engine", info.get("engine", "")),
            ("model", f"{info.get('model', '')} ({info.get('provider', 'LOCAL')})"),
            ("context", f"{info.get('context_used', 0)}/{info.get('context_max', 0) or 'n/a'} tokens"),
            ("sandbox", label),
            ("session", str(info.get("session_id", ""))[:SHORT_ID]),
            ("title", self._current_title()),
            ("changed", f"{len(self.changed_paths)} file(s)"),
            ("queued", str(len(self.queue))),
            ("state", self.status_bar.state),
        ]
        for name, value in rows:
            self._line(f"  {name:<9}{value}", "fail" if name == "sandbox" and danger else "muted")

    async def _cmd_diff(self) -> None:
        if not self.changed_paths:
            self._line("No files changed in this session.")
            return
        root = Path(self.config.runtime.project_root)
        diff = await asyncio.to_thread(session_diff, root, list(self.changed_paths))
        self._line(f"Changed in this session: {', '.join(self.changed_paths)}", "accent")
        if not diff:
            self._line("(no textual diff available)")
            return
        text = self._text(bounded(diff, 400, 80_000))
        renderable = Text(text) if self.no_color_mode else Syntax(text, "diff", theme="monokai", background_color="default")
        self.transcript.add(Static(renderable))

    async def _cmd_undo(self) -> None:
        undo = getattr(self.session, "undo_last", None)
        if undo is None:
            self._line(
                "Undo is not available with this engine: it keeps no edit checkpoints. "
                "Review with /diff and revert with git (git checkout -- <file>).",
                "warn",
            )
            return
        try:
            result = undo()
            if asyncio.iscoroutine(result):
                result = await result
        except NotImplementedError as exc:
            self._line(f"Undo unavailable: {exc}", "warn")
            return
        except Exception as exc:  # noqa: BLE001
            self._line(f"Undo failed: {exc}", "fail")
            return
        self._line(f"Undo: {result}", "accent")

    async def _cmd_doctor(self) -> None:
        self._line("Doctor · checking readiness…", "accent")
        try:
            tiers = await asyncio.to_thread(self.doctor, self.config)
        except Exception as exc:  # noqa: BLE001
            self._line(f"Doctor failed: {exc}", "fail")
            return
        g = self.glyphs
        labels = {"inference": "inference-ready", "read_only": "read-only-ready", "execution": "execution-ready"}
        for key, label in labels.items():
            tier = tiers.get(key) or {}
            ok = bool(tier.get("ready"))
            mark = f"{g.icon('ok')} ready" if ok else f"{g.icon('fail')} NOT READY"
            self._line(f"  {label:<16}{mark} {g.sep} {tier.get('detail', '')}", "ok" if ok else "fail")
            if not ok and tier.get("remedy"):
                self._line(f"      {g.icon('arrow')} {tier['remedy']}", "warn")

    def _cmd_permissions(self, _args: list[str]) -> None:
        policy = self.config.policy
        self._line(f"Autonomous: {policy.autonomous} · automatic risk ceiling: {policy.max_auto_risk}")

    async def on_unmount(self) -> None:
        self._release_lock()
        close = getattr(self.session, "close", None)
        if close is not None:
            with suppress(Exception):
                result = close()
                if asyncio.iscoroutine(result):
                    await result


def run_tui(
    config: NedsterConfig,
    *,
    ascii_only: bool = False,
    no_color: bool = False,
    session_factory: SessionFactoryFor | None = None,
    session_ref: str | None = None,
    resume_latest: bool = False,
    open_picker_on_start: bool = False,
) -> None:
    NedsterApp(
        config,
        ascii_only=ascii_only,
        no_color=no_color,
        session_factory=session_factory,
        session_ref=session_ref,
        resume_latest=resume_latest,
        open_picker_on_start=open_picker_on_start,
    ).run()
