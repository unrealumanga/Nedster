"""Conversation widgets: user/assistant messages, expandable tool rows, outcome badges."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widget import Widget

from .glyphs import Glyphs
from .session import failure_reason

MAX_OUTPUT_LINES = 40
MAX_OUTPUT_CHARS = 4_000
MAX_DIFF_LINES = 400
#: The full-output modal renders at most this much of a stored output (keeps the UI responsive).
FULL_OUTPUT_MAX_CHARS = 200_000
EDIT_TOOLS = {"write_file", "edit_file", "apply_patch", "replace_in_file", "write", "edit", "patch"}

BADGES: dict[str, tuple[str, str]] = {
    "completed": ("COMPLETED", "ok"),
    "answered": ("ANSWERED", "ok"),
    "no_change_needed": ("NO CHANGE NEEDED", "ok"),
    "unvalidated": ("UNVALIDATED", "warn"),
    "incomplete": ("INCOMPLETE", "fail"),
    "failed": ("FAILED", "fail"),
    "cancelled": ("CANCELLED", "muted"),
}


def bounded(text: str, max_lines: int = MAX_OUTPUT_LINES, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    lines = text[:max_chars].splitlines()
    clipped = len(text) > max_chars or len(lines) > max_lines
    body = "\n".join(lines[:max_lines])
    return f"{body}\n[… {len(text) - len(body)} more characters]" if clipped else body


def render_full_output(text: str, max_chars: int = FULL_OUTPUT_MAX_CHARS) -> str:
    """The complete stored output, capped at ``max_chars`` with a byte-count truncation note."""
    if len(text) <= max_chars:
        return text
    total = len(text.encode("utf-8", "replace"))
    return f"{text[:max_chars]}\n\n[truncated, {total} bytes total]"


def short_args(arguments: Mapping[str, Any], limit: int = 48) -> str:
    for key in ("path", "file", "pattern", "query", "command", "argv", "cmd"):
        value = arguments.get(key)
        if value:
            text = " ".join(map(str, value)) if isinstance(value, list | tuple) else str(value)
            break
    else:
        text = " ".join(f"{k}={v}" for k, v in arguments.items() if not isinstance(v, dict | list))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _elapsed(ms: float) -> str:
    return f"{ms:.0f}ms" if ms < 1_000 else f"{ms / 1_000:.1f}s"


class Line(Widget):
    """One plain line (notice, user message, queued input) rendered with a named style."""

    DEFAULT_CSS = "Line { height: auto; }"

    def __init__(self, text: str, style: str = "", *, classes: str | None = None) -> None:
        super().__init__(classes=classes)
        self.text = text
        self.line_style = style

    def set(self, text: str, style: str | None = None) -> None:
        self.text = text
        if style is not None:
            self.line_style = style
        self.refresh(layout=True)

    def render(self) -> RenderableType:
        return Text(self.text, style=self.line_style)


class AssistantMessage(Widget):
    """Streams model deltas live; thinking stays collapsed and dim; Markdown once final."""

    DEFAULT_CSS = "AssistantMessage { height: auto; margin-top: 1; }"
    can_focus = True
    BINDINGS = [Binding("enter", "toggle", "Expand", show=False)]

    def __init__(self, glyphs: Glyphs, content: str = "", *, final: bool = False) -> None:
        super().__init__()
        self.glyphs = glyphs
        self.content = content
        self.thinking = ""
        self.final = final
        self.show_thinking = False

    def append(self, text: str, kind: str = "content") -> None:
        if kind == "thinking":
            self.thinking += text
        else:
            self.content += text
        self.refresh(layout=True)

    def finalize(self, content: str | None = None) -> None:
        if content is not None:
            self.content = content
        self.final = True
        self.refresh(layout=True)

    def action_toggle(self) -> None:
        self.show_thinking = not self.show_thinking
        self.refresh(layout=True)

    def on_click(self, event: events.Click) -> None:
        self.action_toggle()

    def render(self) -> RenderableType:
        g = self.glyphs
        parts: list[RenderableType] = []
        if self.thinking.strip():
            if self.show_thinking:
                parts.append(Text(g.text(self.thinking.strip()), style="dim italic"))
            else:
                tail = " ".join(self.thinking.split())[-60:]
                label = f"{g.icon('thinking')} thinking ({len(self.thinking)} chars) {g.icon('expand')} {tail}"
                parts.append(Text(g.text(label), style="dim", no_wrap=True, overflow="ellipsis"))
        if self.content.strip():
            content = g.text(self.content.strip())
            if self.final and not g.ascii_only:
                parts.append(Markdown(content, code_theme="monokai"))
            else:
                parts.append(Text(content))
        elif not self.final and not self.thinking:
            parts.append(Text(g.icon("ellipsis"), style="dim"))
        return Group(*parts)


class ToolRow(Widget):
    """One compact line per tool call; Enter/click expands argv, path, bounded output and diff.

    Press ``o`` for the full stored output (a scrollable modal), loaded lazily via ``output_loader``.
    """

    DEFAULT_CSS = """
    ToolRow { height: auto; padding: 0 0 0 2; }
    ToolRow:focus { background: $boost; }
    """
    can_focus = True
    BINDINGS = [
        Binding("enter", "toggle", "Expand", show=False),
        Binding("o", "full_output", "Full output", show=False),
    ]

    class FullOutput(Message):
        """Ask the app to open the full stored output for this row."""

        def __init__(self, row: ToolRow) -> None:
            super().__init__()
            self.row = row

    def __init__(
        self,
        glyphs: Glyphs,
        call_id: str,
        tool: str,
        arguments: Mapping[str, Any],
        *,
        diff_loader: Callable[[Sequence[str]], str] | None = None,
        output_loader: Callable[[], str] | None = None,
        historical: bool = False,
        expanded: bool = False,
    ) -> None:
        super().__init__()
        self.glyphs = glyphs
        self.call_id = call_id
        self.tool = tool
        self.arguments = dict(arguments)
        self.started = time.monotonic()
        self.finished: dict[str, Any] | None = None
        self.expanded = expanded
        self.diff_loader = diff_loader
        self.output_loader = output_loader
        self.historical = historical
        self._diff: str | None = None

    @property
    def failure(self) -> str | None:
        if self.finished is None:
            return None
        return failure_reason(str(self.finished.get("status") or ""), self.finished)

    @property
    def status(self) -> str:
        if self.finished is None:
            return "running"
        return "ok" if self.failure is None else "failed"

    def finish(self, data: Mapping[str, Any]) -> None:
        self.finished = dict(data)
        self.refresh(layout=True)

    def action_toggle(self) -> None:
        self.expanded = not self.expanded
        self.refresh(layout=True)

    def action_full_output(self) -> None:
        self.post_message(self.FullOutput(self))

    def on_click(self, event: events.Click) -> None:
        self.focus()
        self.action_toggle()

    def event_output(self) -> str:
        """The bounded output carried on the finish event (empty until the call finishes)."""
        return str((self.finished or {}).get("output") or "")

    @property
    def mutated_paths(self) -> list[str]:
        return [str(p) for p in (self.finished or {}).get("mutated_paths") or []]

    def diff(self) -> str:
        if self.finished is None:
            return ""
        if self._diff is None:
            text = str(self.finished.get("diff") or "")
            if not text and self.diff_loader and self.mutated_paths:
                try:
                    text = self.diff_loader(self.mutated_paths)
                except Exception as exc:  # noqa: BLE001 - shown inline, never fatal
                    text = f"(diff unavailable: {exc})"
            self._diff = text
        return self._diff

    def summary_line(self) -> Text:
        g = self.glyphs
        data = self.finished or {}
        if self.finished is None:
            icon, style, state = g.icon("running"), g.style("info"), "running"
            elapsed = (time.monotonic() - self.started) * 1_000
        elif self.failure is not None:
            icon, style, state = g.icon("fail"), g.style("fail"), self.failure
            elapsed = float(data.get("elapsed_ms") or 0.0)
        else:
            icon, style, state = g.icon("ok"), g.style("ok"), str(data.get("summary") or "ok")
            elapsed = float(data.get("elapsed_ms") or 0.0)
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append(g.icon("collapse" if self.expanded else "expand") + " ", style="dim")
        line.append(f"{icon} ", style=style)
        line.append(self.tool, style="bold")
        args = short_args(self.arguments)
        if args:
            line.append(f"  {g.text(args)}", style=g.style("muted"))
        if elapsed:
            line.append(f"  {_elapsed(elapsed)}", style=g.style("faint"))
        line.append(f"  {g.text(state)}", style=style if self.failure is not None else g.style("muted"))
        return line

    def details(self) -> list[RenderableType]:
        g = self.glyphs
        data = self.finished or {}
        parts: list[RenderableType] = []
        argv = self.arguments.get("argv") or self.arguments.get("command")
        if isinstance(argv, list | tuple):
            parts.append(Text(g.text("argv: " + json.dumps([str(a) for a in argv])), style=g.style("muted")))
        elif argv:
            parts.append(Text(g.text(f"command: {argv}"), style=g.style("muted")))
        path = self.arguments.get("path") or data.get("path")
        if path:
            parts.append(Text(g.text(f"path: {path}"), style=g.style("muted")))
        if not argv and not path and self.arguments:
            parts.append(Text(g.text("args: " + json.dumps(self.arguments, default=str)), style=g.style("muted")))
        if data:
            fields = [f"status {data.get('status', '?')}"]
            fields += [f"{k} {data[k]}" for k in ("exit_code", "returncode", "tests_run") if k in data]
            fields.append(f"elapsed {_elapsed(float(data.get('elapsed_ms') or 0.0))}")
            parts.append(Text(g.text(g.sep.join(fields)), style=g.style("faint")))
        output = str(data.get("output") or data.get("summary") or "")
        if output:
            parts.append(Text(g.text(bounded(output))))
        if self.tool in EDIT_TOOLS or data.get("diff") or self.mutated_paths:
            diff = self.diff()
            if diff:
                parts.append(self._render_diff(diff))
        return parts

    def _render_diff(self, diff: str) -> RenderableType:
        g = self.glyphs
        text = g.text(bounded(diff, MAX_DIFF_LINES, MAX_DIFF_LINES * 200))
        if g.no_color:
            return Text(text)
        return Syntax(text, "diff", theme="monokai", background_color="default", word_wrap=True)

    def render(self) -> RenderableType:
        if not self.expanded:
            return self.summary_line()
        return Group(self.summary_line(), *(_indent(part) for part in self.details()))


def _indent(renderable: RenderableType) -> RenderableType:
    from rich.padding import Padding

    return Padding(renderable, (0, 0, 0, 4))


class OutcomeBadge(Widget):
    DEFAULT_CSS = "OutcomeBadge { height: auto; margin-bottom: 1; }"

    def __init__(self, glyphs: Glyphs, outcome: str, reasons: Sequence[str] = (), changed: Sequence[str] = ()) -> None:
        super().__init__()
        self.glyphs = glyphs
        self.outcome = outcome
        self.reasons = list(reasons)
        self.changed = list(changed)

    @property
    def label(self) -> str:
        return BADGES.get(self.outcome, (self.outcome.upper() or "UNKNOWN", "warn"))[0]

    def render(self) -> RenderableType:
        g = self.glyphs
        label, style_name = BADGES.get(self.outcome, (self.label, "warn"))
        style = g.style(style_name)
        line = Text()
        line.append(f"[{label}]", style=f"{style} reverse" if style and not g.no_color else style or "bold")
        if self.changed:
            line.append(f"  {len(self.changed)} file(s) changed", style=g.style("muted"))
        rows: list[RenderableType] = [line]
        for reason in self.reasons:
            rows.append(Text(g.text(f"  {g.icon('dash')} {reason}"), style=g.style("muted")))
        return Group(*rows)


class Banner(Widget):
    """One high-visibility notice line (the resume banner, retention warnings, …)."""

    DEFAULT_CSS = """
    Banner { height: auto; margin: 1 0; padding: 0 1; background: #3b2f00; color: #e3b341; text-style: bold; }
    .no-color Banner { background: ansi_default; color: ansi_default; }
    """

    def __init__(self, glyphs: Glyphs, text: str) -> None:
        super().__init__()
        self.glyphs = glyphs
        self.text = text

    def render(self) -> RenderableType:
        return Text(self.glyphs.text(f"{self.glyphs.icon('warn')} {self.text}"))


#: Shown after every /resume: replayed history is context, not fresh evidence.
RESUME_BANNER = "resumed - earlier validation results are not current evidence"
REPLAY_MAX_LINES = 400
REPLAY_MAX_CHARS = 40_000
TOOL_REPLAY_NOTE = "earlier tool result - not replayed"


def replay_widgets(
    glyphs: Glyphs,
    entries: Sequence[Mapping[str, Any]],
    *,
    output_loader: Callable[[str], Callable[[], str] | None] | None = None,
) -> list[Widget]:
    """Display-only reconstruction of a stored transcript for ``/resume``.

    Nothing is ever re-executed: tool entries become a marker row and their stored output text is
    dropped on purpose, so an old tool result can never be mistaken for current evidence. The full
    stored output is still reachable on demand (press ``o``) and is labelled "historical output".
    """
    widgets: list[Widget] = []
    for entry in entries:
        role = str(entry.get("role") or "")
        meta = entry.get("meta") if isinstance(entry.get("meta"), Mapping) else {}
        text = bounded(str(entry.get("text") or ""), REPLAY_MAX_LINES, REPLAY_MAX_CHARS)
        if role == "tool":
            tool = str(meta.get("tool") or meta.get("name") or "tool")
            call_id = str(meta.get("call_id") or "")
            loader = output_loader(call_id) if output_loader is not None and call_id else None
            row = ToolRow(
                glyphs, call_id, tool, {}, historical=True, output_loader=loader,
            )
            row.finish({"status": "ok", "summary": TOOL_REPLAY_NOTE})
            widgets.append(row)
            continue
        if not text.strip():
            calls = [call for call in (meta.get("tool_calls") or []) if isinstance(call, Mapping)]
            for call in calls:
                arguments = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
                label = str(call.get("name") or "tool")
                args = short_args(arguments)
                widgets.append(
                    Line(
                        glyphs.text(f"{glyphs.icon('dash')} {label} {args}".rstrip()),
                        glyphs.style("muted"),
                    )
                )
            continue
        if role == "user":
            widgets.append(Line(glyphs.text(f"{glyphs.icon('user')} {text}"), glyphs.style("user")))
        elif role == "assistant":
            message = AssistantMessage(glyphs, text, final=True)
            message.thinking = bounded(str(meta.get("thinking") or ""), REPLAY_MAX_LINES, REPLAY_MAX_CHARS)
            widgets.append(message)
        else:  # system notices and unknown roles: context, never evidence
            widgets.append(Line(glyphs.text(text), glyphs.style("muted")))
    return widgets


class Transcript(VerticalScroll):
    """The central conversation; focusable rows inside.

    New output auto-scrolls only while the user is following the tail; after scrolling up the view
    stays put and ``new_output`` flags the pending text so the app can show an "End to follow" hint.
    """

    DEFAULT_CSS = "Transcript { height: 1fr; padding: 0 2; }"
    can_focus = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.following = True  # user intent: are we pinned to the newest text?
        self.new_output = False
        self.on_follow_change: Callable[[bool], None] | None = None

    def add(self, widget: Widget) -> Widget:
        self.mount(widget)
        self.follow_output()
        return widget

    def follow_output(self) -> None:
        """Called whenever content grows: pin the tail only when the user is following it."""
        if not self.following:
            self.new_output = True
            self._signal_follow()
            return
        self.new_output = False
        self.call_after_refresh(self._pin_to_end)

    def _pin_to_end(self) -> None:
        self.scroll_end(animate=False)

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        at_end = self.is_vertical_scroll_end
        if at_end and not self.following:
            self.following = True
            self.new_output = False
            self._signal_follow()
        elif not at_end and self.following:
            self.following = False

    def _signal_follow(self) -> None:
        if self.on_follow_change is not None:
            self.on_follow_change(self.new_output)

    def plain_text(self) -> str:
        """Everything currently shown, as plain text (used by tests and /export-like features)."""
        lines: list[str] = []
        for child in self.children:
            rendered = child.render()
            lines.append(_plain(rendered))
        return "\n".join(lines)


def _plain(renderable: Any) -> str:
    if isinstance(renderable, Text):
        return renderable.plain
    if isinstance(renderable, Group):
        return "\n".join(_plain(r) for r in renderable.renderables)
    if isinstance(renderable, Markdown):
        return renderable.markup
    if isinstance(renderable, Syntax):
        return renderable.code
    inner = getattr(renderable, "renderable", None)
    if inner is not None:
        return _plain(inner)
    return str(renderable)
