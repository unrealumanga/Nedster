"""Icons and styles that honour --ascii and --no-color everywhere in the TUI."""

from __future__ import annotations

from dataclasses import dataclass

UNICODE_ICONS: dict[str, str] = {
    "ok": "✓",
    "fail": "✗",
    "running": "…",
    "queued": "⧗",
    "user": "›",
    "thinking": "◇",
    "step": "◆",
    "arrow": "→",
    "retry": "↻",
    "dot": "●",
    "sep": "·",
    "ellipsis": "…",
    "dash": "—",
    "expand": "▸",
    "collapse": "▾",
    "bar_full": "█",
    "bar_empty": "░",
    "warn": "!",
}

ASCII_ICONS: dict[str, str] = {
    "ok": "OK",
    "fail": "X",
    "running": "...",
    "queued": "~",
    "user": ">",
    "thinking": ">",
    "step": ">",
    "arrow": "->",
    "retry": "~",
    "dot": "*",
    "sep": "|",
    "ellipsis": "...",
    "dash": "-",
    "expand": "+",
    "collapse": "-",
    "bar_full": "#",
    "bar_empty": ".",
    "warn": "!",
}

# Fallbacks for free text (legacy event lines, engine summaries) shown in ASCII mode.
ASCII_REPLACEMENTS: dict[str, str] = {
    "…": "...",
    "·": "|",
    "—": "-",
    "–": "-",
    "●": "*",
    "◇": ">",
    "◆": ">",
    "→": "->",
    "✓": "OK",
    "✕": "X",
    "✗": "X",
    "↻": "~",
    "›": ">",
    "⧗": "~",
    "▸": "+",
    "▾": "-",
    "█": "#",
    "░": ".",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "•": "*",
}

COLORS: dict[str, str] = {
    "ok": "#3fb950",
    "fail": "bold #f85149",
    "warn": "#d29922",
    "info": "#7aa2f7",
    "accent": "#d2a8ff",
    "muted": "#8b949e",
    "faint": "#6e7681",
    "strong": "bold #f0f6fc",
    "user": "bold #f0f6fc",
    "danger": "bold #f85149",
}

# Monochrome keeps emphasis (bold/dim/reverse) but never sets a color.
MONOCHROME: dict[str, str] = {
    "ok": "bold",
    "fail": "bold reverse",
    "warn": "bold",
    "info": "",
    "accent": "bold",
    "muted": "dim",
    "faint": "dim",
    "strong": "bold",
    "user": "bold",
    "danger": "bold reverse",
}


@dataclass(slots=True, frozen=True)
class Glyphs:
    ascii_only: bool = False
    no_color: bool = False

    def icon(self, name: str) -> str:
        return (ASCII_ICONS if self.ascii_only else UNICODE_ICONS)[name]

    def style(self, name: str) -> str:
        return (MONOCHROME if self.no_color else COLORS).get(name, "")

    def text(self, value: str) -> str:
        """Make arbitrary display text safe for the active mode."""
        if not self.ascii_only:
            return value
        for symbol, fallback in ASCII_REPLACEMENTS.items():
            value = value.replace(symbol, fallback)
        return value.encode("ascii", "replace").decode("ascii")

    @property
    def sep(self) -> str:
        return f" {self.icon('sep')} "

    def bar(self, used: int, maximum: int, width: int = 10) -> str:
        if maximum <= 0:
            return ""
        filled = max(0, min(width, round(width * used / maximum)))
        return self.icon("bar_full") * filled + self.icon("bar_empty") * (width - filled)
