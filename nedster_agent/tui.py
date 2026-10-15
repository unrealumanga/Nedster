from __future__ import annotations

import asyncio
import difflib
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import datetime
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
from ironclad.session_store.sqlite_store import make_owner

from .personality import pick as personality_line
from .widgets.composer import Composer, History, project_files
from .widgets.glyphs import Glyphs
from .widgets.session import (
    EngineSession,
    EventCallback,
    NativeSession,
    SessionEvent,
    SessionNote,
    failure_reason,
)
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
    "/init": "write a starter AGENTS.md from repository facts (/init --print to preview)",
    "/sessions": "list saved sessions for this project",
    "/resume": "resume a session: /resume [id]",
    "/model": "show or switch the model: /model [name]",
    "/status": "model, context, sandbox, session",
    "/diff": "diff of files changed in this session",
    "/undo": "undo the last edit (when the engine supports it)",
    "/doctor": "readiness check (inference / read-only / execution)",
    "/models": "list the active model fleet",
    "/permissions": "show the autonomy policy",
    "/soul": "show or edit the opt-in personality (soul.md)",
    "/quit": "exit Nedster",
}
# Commands that change engine/session state wait for the running turn (the text is kept).
IDLE_ONLY = {"/new", "/resume", "/model", "/undo", "/sessions", "/soul"}

#: How much stored history /resume replays into the view (display only, never re-executed).
RESUME_REPLAY_LIMIT = 200
#: Activity-lock heartbeat period; well below the store's 60 s lock TTL.
LOCK_REFRESH_S = 5.0

DoctorProvider = Callable[[NedsterConfig], Mapping[str, Mapping[str, Any]]]
SessionFactoryFor = Callable[[NedsterConfig, EventCallback], EngineSession]


class _WiredNativeSession(NativeSession):
    """TUI adapter over ``ironclad.native.NativeSession``: shares its store/session and closes its resources."""

    core: Any = None

    async def submit(self, text: str) -> None:
        # Through the core session, not the bare engine: it opens the checkpoint turn that file mutations need
        # (and that /undo restores). /resume may have switched session_id, so the core follows it.
        if self.core is None:
            await super().submit(text)
            return
        self.core.session_id = self.session_id
        await self.core.run(text)

    def undo_last(self) -> Any:
        self.core.session_id = self.session_id
        return self.core.undo_last()

    async def close(self) -> None:
        if self.core is not None:
            await self.core.aclose()


def native_session(config: NedsterConfig, emit: EventCallback) -> EngineSession:
    from ironclad.native import NativeSession as CoreSession

    holder: dict[str, CoreSession] = {}

    def build(engine_emit: Callable[[Any], None]) -> Any:
        holder["core"] = CoreSession(config, emit=engine_emit, resume_latest=False)
        return holder["core"].engine

    session = _WiredNativeSession(build, emit, endpoint=config.models.ollama_url)
    core = holder["core"]
    info = core.info()
    session.core = core
    session.store = core.store
    session.session_id = core.session_id
    session.project_root = info["project_root"]
    session.sandbox, session.isolation = info["sandbox"], info["isolation"]
    session.instructions = info.get("instructions")
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
    #prompt { height: auto; min-height: 3; max-height: 8; margin: 0 2; background: #11151b; color: #f0f6fc; border: round #394150; }
    #prompt:focus { border: round #7aa2f7; }
    #status { height: 1; padding: 0 2; background: #0b0d10; color: #c9d1d9; }
    #message { height: 1; padding: 0 2; background: #0b0d10; color: #8b949e; }
    Footer { background: #0b0d10; color: #8b949e; }
    #too-small { display: none; height: 1fr; content-align: center middle; text-align: center; }
    .queued { color: #d29922; }
    .no-color, .no-color #identity, .no-color #work, .no-color #details, .no-color #message, .no-color #status,
    .no-color Footer, .no-color #prompt, .no-color #completions { color: ansi_default; background: ansi_default; }
    .no-color #prompt, .no-color #completions { border: round ansi_default; }
    .no-color #prompt:focus { border: round ansi_default; }
    .no-color .queued { color: ansi_default; text-style: italic; }
    .ascii #prompt, .ascii #prompt:focus, .ascii #completions { border: ascii; }
    .ascii #work { scrollbar-size-vertical: 0; }
    """

    BINDINGS = [
        Binding("escape", "escape", "Stop", priority=True),
        Binding("ctrl+c", "interrupt", "Quit (x2)", priority=True),
        Binding("f2", "details", "Events"),
        Binding("ctrl+l", "clear", "Clear"),
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
    ):
        super().__init__()
        self.config = config
        self.ascii_only = ascii_only
        self.no_color_mode = no_color
        self.glyphs = Glyphs(ascii_only=ascii_only, no_color=no_color)
        self.session_factory = session_factory
        self.doctor = doctor or _default_doctor
        root = Path(config.runtime.project_root)
        self.files_source = files or (lambda: project_files(root))
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
        self.current_reply: AssistantMessage | None = None
        self.turn_replies: list[AssistantMessage] = []
        self.pending_badge: OutcomeBadge | None = None
        self.turn_closed = True
        self.last_interrupt = 0.0
        self._loop_thread = threading.get_ident()
        self._lock_owner = make_owner("tui")
        self._locked: tuple[Any, str] | None = None  # (store, session_id) held by this window
        self._last_lock_refresh = 0.0
        self.soul_mode = str(getattr(config.runtime, "soul", "off") or "off")
        self.turn_number = 0
        self._pending_soul: tuple[str, Any] | None = None  # ("remember"|"reset", SoulEdit|None)

    # ------------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        g = self.glyphs
        yield Static("terminal too small - need at least 80x24", id="too-small")
        yield Static(g.text(f"Nedster / {self._project_name()}{g.sep}starting"), id="identity")
        yield Transcript(id="work")
        yield RichLog(id="details", markup=False, wrap=False, auto_scroll=True)
        yield OptionList(id="completions")
        yield Composer(
            id="prompt",
            history=self.history,
            commands=list(COMMANDS),
            files=self.files_source,
            menu=lambda: self.query_one("#completions", OptionList),
        )
        yield StatusBar(g, id="status")
        yield Static("", id="message")
        yield Footer(show_command_palette=False)

    def _project_name(self) -> str:
        return Path(self.config.runtime.project_root).name or str(self.config.runtime.project_root)

    async def on_mount(self) -> None:
        self._loop_thread = threading.get_ident()
        if self.no_color_mode:
            self.screen.add_class("no-color")
        if self.ascii_only:
            self.screen.add_class("ascii")
        self._apply_size(self.size.width, self.size.height)
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

    # ------------------------------------------------------------------ startup

    def _make_session(self) -> EngineSession:
        if self.session_factory is not None:
            return self.session_factory(self.config, self.handle_event)
        return native_session(self.config, self.handle_event)

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
        self._acquire_lock(str(getattr(self.session, "session_id", "")))
        self._set_state("idle")
        self._message("Ready")
        self._say("idle")
        self._announce_instructions()

    def _announce_instructions(self) -> None:
        """One-line notice about the project instruction file that was loaded for this session."""
        info = self.session.info() if self.session is not None else {}
        loaded = info.get("instructions")
        if not isinstance(loaded, Mapping) or not loaded.get("name"):
            return
        size = int(loaded.get("size") or 0)
        note = " [truncated]" if loaded.get("truncated") else ""
        self._line(f"Loaded {loaded['name']} ({size / 1024:.1f} KB){note}", "muted")

    def _refresh_info(self) -> None:
        if self.session is None:
            return
        info = self.session.info()
        self.status_bar.set_info(info)
        g = self.glyphs
        provider = str(info.get("provider") or "LOCAL")
        where = provider
        self.query_one("#identity", Static).update(
            g.text(f"Nedster / {self._project_name()}{g.sep}{where}{g.sep}{info.get('model') or 'no model'}")
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

    def _say(self, moment: str) -> None:
        """Opt-in personality line in the status message area; never enters the answer or events."""
        if self.soul_mode not in ("ui", "full"):
            return
        line = personality_line(moment, self.turn_number)
        if line:
            self._message(line)

    def _say_done(self, outcome: str) -> None:
        moment = {
            "completed": "done_ok",
            "answered": "done_ok",
            "no_change_needed": "done_ok",
            "unvalidated": "done_unvalidated",
            "incomplete": "done_failed",
            "failed": "done_failed",
            "cancelled": "done_failed",
        }.get(outcome)
        if moment:
            self._say(moment)

    def _set_state(self, state: str, detail: str = "", *, force: bool = False) -> None:
        if self.cancelling and not force:
            return  # "cancelling…" persists until the engine confirms
        self.status_bar.set_state(state, detail)

    # ------------------------------------------------------------------ session locks + resume

    def _session_store(self) -> Any:
        return getattr(self.session, "store", None) if self.session is not None else None

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

    def _resolve_session_id(self, wanted: str | None) -> str:
        """Mirror of ``NativeSession.resume`` resolution, so the lock is taken *before* switching."""
        lister = getattr(self.session, "list_sessions", None)
        if lister is not None:
            sessions = list(lister())
        else:
            store = self._session_store()
            store_lister = getattr(store, "list_sessions", None)
            sessions = list(store_lister(str(self.config.runtime.project_root))) if callable(store_lister) else []
        if not sessions:
            raise LookupError("no saved sessions for this project")
        if wanted is None:
            return str(sessions[0]["id"])
        matches = [item for item in sessions if str(item.get("id", "")).startswith(wanted)]
        if len(matches) != 1:
            raise LookupError(f"no unique session matches {wanted!r}")
        return str(matches[0]["id"])

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
        self._line(f"Resumed session {session_id[:12]}", "accent")
        for widget in replay_widgets(self.glyphs, entries):
            self.transcript.add(widget)
        if not entries:
            self._line("No saved transcript for this session.", "muted")
        self.transcript.add(Banner(self.glyphs, RESUME_BANNER))
        self._refresh_info()

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
        self._set_state("thinking")

    def _on_model_delta(self, event: EngineEvent) -> None:
        if self.current_reply is None:
            self.current_reply = AssistantMessage(self.glyphs)
            self.turn_replies.append(self.current_reply)
            self.transcript.add(self.current_reply)
        self.current_reply.append(str(event.data.get("text") or ""), str(event.data.get("kind") or "content"))
        self.transcript.call_after_refresh(self.transcript.scroll_end, animate=False)
        self._set_state("thinking")

    def _on_model_usage(self, event: EngineEvent) -> None:
        used = int(event.data.get("prompt_tokens") or 0) + int(event.data.get("completion_tokens") or 0)
        info = dict(self.status_bar.info)
        info["context_used"] = used
        self.status_bar.set_info(info)

    def _on_tool_started(self, event: EngineEvent) -> None:
        self.current_reply = None
        call_id = str(event.data.get("call_id") or f"call-{len(self.tool_rows)}")
        tool = str(event.data.get("tool") or "tool")
        row = ToolRow(self.glyphs, call_id, tool, dict(event.data.get("arguments") or {}), diff_loader=self._diff_loader)
        self.tool_rows[call_id] = row
        self.transcript.add(row)
        self._set_state("running tool", tool)

    def _on_tool_finished(self, event: EngineEvent) -> None:
        data = dict(event.data)
        call_id = str(data.get("call_id") or "")
        row = self.tool_rows.get(call_id)
        if row is None:
            row = ToolRow(self.glyphs, call_id, str(data.get("tool") or "tool"), {}, diff_loader=self._diff_loader)
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
        self._show_answer(answer)
        if self.pending_badge is None and event.data.get("outcome"):
            self.pending_badge = OutcomeBadge(self.glyphs, str(event.data["outcome"]))
        self._close_turn()
        if self.cancelling:
            self.cancelling = False
            self._message("Cancelled")
        else:
            self._say_done(str(event.data.get("outcome") or ""))
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

    def _diff_loader(self, paths: Sequence[str]) -> str:
        return session_diff(Path(self.config.runtime.project_root), paths)

    # ------------------------------------------------------------------ input

    @on(Composer.Submitted, "#prompt")
    def on_submit(self, event: Composer.Submitted) -> None:
        value = event.value.strip()
        composer = event.composer
        if self._pending_soul is not None:  # y/N answer to /soul remember|reset (Enter = no)
            composer.set_text("")
            self._confirm_soul(value)
            return
        if not value:
            return
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
        if self._pending_soul is not None:  # treat a programmatic submit as the y/N answer
            self._confirm_soul(value)
            return
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
        self.busy = True
        self.turn_closed = False
        self.turn_number += 1
        self._say("thinking")
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
        self.busy = False
        if self.cancelling:
            self.cancelling = False
            self._message("Cancelled")
        self.status_bar.end_turn()
        self._set_state("idle")
        self._refresh_info()
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
        composer = self.query_one("#prompt", Composer)
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

    # ------------------------------------------------------------------ commands

    def _command(self, value: str) -> None:
        parts = value.split()
        command, args = parts[0].lower(), parts[1:]
        handlers: dict[str, Callable[[list[str]], Any]] = {
            "/quit": lambda _a: self.exit(),
            "/exit": lambda _a: self.exit(),
            "/help": self._cmd_help,
            "/clear": lambda _a: self.action_clear(),
            "/init": self._cmd_init,
            "/new": self._cmd_new,
            "/sessions": self._cmd_sessions,
            "/resume": self._cmd_resume,
            "/model": self._cmd_model,
            "/models": self._cmd_models,
            "/status": self._cmd_status,
            "/diff": lambda _a: self.run_worker(self._cmd_diff(), group="command"),
            "/undo": lambda _a: self.run_worker(self._cmd_undo(), group="command"),
            "/doctor": lambda _a: self.run_worker(self._cmd_doctor(), group="command"),
            "/permissions": self._cmd_permissions,
            "/soul": self._cmd_soul,
        }
        handler = handlers.get(command)
        if handler is None:
            self._line(f"Unknown command {command}. /help lists commands.", "warn")
            return
        handler(args)

    def _cmd_help(self, _args: list[str]) -> None:
        self._line("Commands", "accent")
        for name, text in COMMANDS.items():
            self._line(f"  {name:<13}{text}")
        self._line("Keys", "accent")
        self._line("  Enter send · Shift+Enter/Ctrl+J newline · Up/Down history · @ file · / command")
        self._line("  Esc stop the running turn · Ctrl+C twice exit · F2 raw events · Ctrl+L clear")
        self._line("  Tool rows: focus with Tab/click, Enter expands argv, output and diff")

    def _cmd_init(self, args: list[str]) -> None:
        from ironclad.scaffold import AGENTS_FILENAME, line_count, starter_agents_md, write_agents_md

        root = Path(self.config.runtime.project_root)
        preview = bool(args) and args[0] in ("--print", "-p", "print")
        try:
            content = starter_agents_md(root, self.config)
        except Exception as exc:  # noqa: BLE001 - shown in the transcript, never fatal
            self._line(f"init failed: {exc}", "fail")
            return
        if preview:
            self.transcript.add(Static(self._text(content)))
            self._line(f"{AGENTS_FILENAME} preview ({line_count(content)} lines, nothing written)", "accent")
            return
        try:
            path = write_agents_md(root, content)
        except FileExistsError:
            self._line(f"{AGENTS_FILENAME} already exists in {root}; not overwritten.", "warn")
            return
        except Exception as exc:  # noqa: BLE001
            self._line(f"init failed: {exc}", "fail")
            return
        self._line(f"Wrote {path.name} ({line_count(content)} lines). Edit it to describe this project.", "accent")

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
        self._line(f"New conversation {str(session_id)[:12]}", "accent")
        self._refresh_info()

    def _cmd_sessions(self, _args: list[str]) -> None:
        lister = getattr(self.session, "list_sessions", None)
        if lister is None:
            self._line("Saved sessions are not available with this session.", "warn")
            return
        sessions = list(lister())
        if not sessions:
            self._line("No saved sessions for this project.")
            return
        current = self.status_bar.info.get("session_id")
        self._line("Sessions (newest first) · /resume <id>", "accent")
        for item in sessions:
            sid = str(item.get("id", ""))
            updated = item.get("updated_ns")
            when = datetime.fromtimestamp(int(updated) / 1e9).strftime("%Y-%m-%d %H:%M") if updated else ""
            marker = "*" if sid == current else " "
            self._line(f" {marker} {sid[:12]}  {when}  {item.get('title') or ''}")

    def _cmd_resume(self, args: list[str]) -> None:
        resume = getattr(self.session, "resume", None)
        if resume is None:
            self._line("Resuming a saved session is not available with this session.", "warn")
            return
        wanted = args[0] if args else None
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

    def _cmd_model(self, args: list[str]) -> None:
        info = self.session.info() if self.session else {}
        if not args:
            self._line(f"Model: {info.get('model') or 'none'} ({info.get('provider') or 'LOCAL'})")
            return
        setter = getattr(self.session, "set_model", None)
        if setter is None:
            self._line("This engine picks models from the fleet; change models.coder in config.toml.", "warn")
            return
        setter(args[0])
        self._line(f"Model set to {args[0]}", "accent")
        self._refresh_info()

    def _cmd_models(self, _args: list[str]) -> None:
        rows = getattr(self.session, "models", lambda: [])()
        if not rows:
            self._line("No model fleet is active. Run `nedster setup`.")
        for role, model, backend in rows:
            self._line(f"{role}: {model} ({backend})")

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
            ("session", info.get("session_id", "")),
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

    def _read_soul_text(self, path: Path) -> str:
        from ironclad.engine.prompt import DEFAULT_SOUL_TEMPLATE

        try:
            return path.read_text(encoding="utf-8") if path.is_file() else DEFAULT_SOUL_TEMPLATE
        except OSError:
            return DEFAULT_SOUL_TEMPLATE

    def _cmd_soul(self, args: list[str]) -> None:
        from ironclad.engine.prompt import soul_path

        path = soul_path()
        if not args:
            try:
                size = path.stat().st_size if path.is_file() else 0
            except OSError:
                size = 0
            self._line(f"Soul: {self.soul_mode} · {path} · {size} bytes")
            if self.soul_mode != "full":
                self._line('soul.md reaches the prompt only in "full" mode (runtime.soul = "full").')
            self._line("Usage: /soul remember <preference> | /soul reset")
            return
        action = args[0].lower()
        if action == "remember":
            from ironclad.engine.prompt import with_preference

            preference = " ".join(args[1:]).strip()
            if not preference:
                self._line("Usage: /soul remember <preference>", "warn")
                return
            edit = with_preference(self._read_soul_text(path), preference)
            if not edit.ok:
                self._line("That preference is already saved." if edit.action == "duplicate" else "Empty preference.", "warn")
                return
            notice = f" (drops oldest: {edit.dropped})" if edit.dropped else ""
            self._line(f"Will add: {edit.bullet}{notice}", "accent")
            if self.soul_mode != "full":
                self._line(f'Note: soul.md reaches the prompt only in "full" mode (runtime.soul = "{self.soul_mode}").', "warn")
            self._pending_soul = ("remember", edit)
            self._message("Append this preference? y/N · applies from your next session")
            return
        if action == "reset":
            self._pending_soul = ("reset", None)
            self._message("Restore the default soul.md template? y/N · your edits are lost")
            return
        self._line("Usage: /soul remember <preference> | /soul reset", "warn")

    def _confirm_soul(self, answer: str) -> None:
        from ironclad.engine.prompt import DEFAULT_SOUL_TEMPLATE, soul_path, write_soul

        pending, self._pending_soul = self._pending_soul, None
        if pending is None:
            return
        kind, edit = pending
        if answer.strip().lower() not in ("y", "yes"):
            self._message("")
            self._line("Cancelled; soul.md unchanged.")
            return
        path = soul_path()
        if kind == "remember":
            text, notice = edit.text, (f" (dropped oldest: {edit.dropped})" if edit.dropped else "")
            message = f"Saved preference{notice}. Applies from your next session."
        else:
            text, message = DEFAULT_SOUL_TEMPLATE, "Restored the default soul.md template. Applies from your next session."
        try:
            write_soul(path, text)
        except OSError as exc:
            self._line(f"Could not write {path}: {exc}", "fail")
            return
        self._message("")
        self._line(message, "accent")

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
) -> None:
    NedsterApp(config, ascii_only=ascii_only, no_color=no_color, session_factory=session_factory).run()
