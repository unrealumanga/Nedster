"""Nedster TUI - Rich terminal UI for code agent"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from difflib import unified_diff


class NedsterTUI:
    COLORS = {
        "prompt": "bold white",
        "user": "bold white",
        "agent": "color(253)",  # ~90% white
        "tool": "color(245)",  # ~50% white
        "edit": "color(245)",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "thinking": "color(245) italic",  # ~50% white italic
        "footer": "dim",
    }

    def __init__(self):
        self.console = Console()

    def print_diff(self, path: str, original: str, new_content: str) -> None:
        """Print colored unified diff with + green / - red lines."""
        diff_lines = list(
            unified_diff(
                original.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=3,
            )
        )
        if not diff_lines:
            return

        diff_text = Text()
        for line in diff_lines:
            if line.startswith("+") and not line.startswith("+++"):
                diff_text.append(line, style="green")
            elif line.startswith("-") and not line.startswith("---"):
                diff_text.append(line, style="red")
            elif line.startswith("@"):
                diff_text.append(line, style="bold cyan")
            else:
                diff_text.append(line, style="dim")

        self.console.print(
            Panel(
                diff_text,
                title=f"[bold]Diff: {path}[/]",
                border_style=self.COLORS["edit"],
            )
        )

    def print_tool_call(self, name, args, result=None, valid=True):
        """Print tool call in one line, dim."""
        path = args.get("path", args.get("cmd", ""))[:40]
        self.console.print(f"  [→ {name}] {path}", style=self.COLORS["tool"])
        if result:
            # Show first line of result immediately
            first_line = str(result).split("\n")[0][:60]
            status = "✓" if "ERROR" not in str(result) else "✗"
            self.console.print(f"  [{status}] {first_line}", style=self.COLORS["tool"])

    def print_tool_result(
        self, tool_name: str, result: str, verbose: bool = False
    ) -> None:
        pass

    def print_edit_preview(self, edit: dict) -> bool:
        """Show diff and prompt Y/n. Return True if approved."""
        path = edit.get("path", "unknown")
        old = edit.get("old", "")
        new = edit.get("new", "")

        self.print_diff(path, old, new)

        try:
            response = input("Apply? [Y/n] ").strip().lower()
            return response in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            self.console.print("\n[Cancelled]", style=self.COLORS["warning"])
            return False

    def print_response(self, text: str):
        """Print agent response, stripping raw tool XML."""
        import re

        # Remove tool call blocks — activity feed handles display
        clean = re.sub(r'<tool\s+name="[^"]*">.*?</tool>\s*', "", text, flags=re.DOTALL)
        # Remove any trailing whitespace artifacts
        clean = clean.strip()
        if clean:
            self.console.print(clean)
        # If nothing left to print, print nothing

    def print_response_stream(self, chunk: str) -> None:
        """Stream text to console in green."""
        self.console.print(chunk, style=self.COLORS["agent"], end="")

    def print_status(self, msg: str, style: str = "color(245)") -> None:
        """Print status message."""
        if style == "color(245)":
            self.console.print(f"[{style}][Nedster] {msg}[/]")
        elif style == "":
            self.console.print(msg)
        else:
            self.console.print(f"[{style}][Nedster] {msg}[/]")

    def print_boot_logo(self):
        """Prints the ASCII logo."""
        self.console.print(
            "[bold cyan] ██   ██  ███████  ██████   ███████  ███████  ███████  ██████  [/]"
        )
        self.console.print(
            "[bold cyan] ███  ██  ██       ██   ██  ██         ███    ██       ██   ██ [/]"
        )
        self.console.print(
            "[bold cyan] ██ █ ██  █████    ██   ██  ███████    ███    █████    ██████  [/]"
        )
        self.console.print(
            "[bold cyan] ██  ███  ██       ██   ██       ██    ███    ██       ██  ██  [/]"
        )
        self.console.print(
            "[bold cyan] ██   ██  ███████  ██████   ███████    ███    ███████  ██   ██ [/]"
        )
        self.console.print("                      [dim]Unchained Local AI[/]\n")

    def print_status_bar(self, text: str):
        """Prints a simple, single-line status bar."""
        self.console.print(Panel(text, style="dim", border_style="dim", padding=(0, 1)))

    def print_boot(
        self,
        project: str,
        files: int,
        vectors: int,
        sessions: int,
        model: str,
        model_size: str,
        vram_free: str,
        vram_total: str,
        tools_ok_str: str,
        tools_warn_str: str,
        think: bool,
        auto: bool,
    ) -> None:
        """Print detailed boot screen."""
        self.console.print(
            "[bold cyan] ██   ██  ███████  ██████   ███████  ███████  ███████  ██████  [/]"
        )
        self.console.print(
            "[bold cyan] ███  ██  ██       ██   ██  ██         ███    ██       ██   ██ [/]"
        )
        self.console.print(
            "[bold cyan] ██ █ ██  █████    ██   ██  ███████    ███    █████    ██████  [/]"
        )
        self.console.print(
            "[bold cyan] ██  ███  ██       ██   ██       ██    ███    ██       ██  ██  [/]"
        )
        self.console.print(
            "[bold cyan] ██   ██  ███████  ██████   ███████    ███    ███████  ██   ██ [/]"
        )
        self.console.print("                      [dim]Unchained Local AI[/]\n")

        think_str = "ON" if think else "OFF"
        auto_str = "ON" if auto else "OFF"

        try:
            free_f = float(vram_free.replace(" GB", ""))
            tot_f = float(vram_total.replace(" GB", ""))
            pct = int((free_f / tot_f) * 100) if tot_f > 0 else 0
            vram_line = f"VRAM:   {vram_free} free / {vram_total} ({pct}% available)"
        except:
            vram_line = f"VRAM:   {vram_free} free / {vram_total}"

        content = (
            f"[bold]Nedster v2[/]  ·  {project}\n"
            f"[dim]{files} files · {vectors} vectors · {sessions} sessions[/]\n\n"
            f"Model:  {model} ({model_size})\n"
            f"{vram_line}\n"
            f"Tools:  {tools_ok_str}  │  {tools_warn_str}\n"
            f"Think:  {think_str}  │  Auto: {auto_str}"
        )

        panel = Panel(content, style="white", padding=(1, 2))
        self.console.print(panel)

    def print_error(self, msg: str) -> None:
        """Print error message."""
        self.console.print(f"[{self.COLORS['error']}][ERROR] {msg}[/]")

    def print_success(self, msg: str) -> None:
        """Print success message."""
        self.console.print(f"[{self.COLORS['success']}][OK] {msg}[/]")

    def print_warning(self, msg: str) -> None:
        """Print warning message."""
        self.console.print(f"[{self.COLORS['warning']}][WARN] {msg}[/]")

    def print_thinking(self, content: str) -> None:
        """Print thinking content in dim italic."""
        self.console.print(content, style=self.COLORS["thinking"])

    def print_status_bar(self, text: str):
        """Prints a single-line status bar with the provided text."""
        width = 85
        panel_text = text.ljust(width)
        self.console.print(f"┌{'─' * width}┐", style="color(244)")
        self.console.print(f"│{panel_text}│", style="color(244)")
        self.console.print(f"└{'─' * width}┘", style="color(244)")


# ── nedster_fixer: output strip ───────────────────────────────────────────────


def _strip_model_artifacts(text: str) -> str:
    """Remove hallucinated tool XML and identity anchors from model output."""
    import re as _re

    # Raw tool call XML echoed by weak models
    text = _re.sub(r'<tool\s+name="[^"]*">.*?</tool>', "", text, flags=_re.DOTALL)
    # [YOU ARE NEDSTER. ...] echoed from system prompt
    text = _re.sub(r"\[YOU ARE NEDSTER\..*?\]", "", text, flags=_re.DOTALL)
    text = _re.sub(r"YOU ARE NEDSTER[.,][^\n]*", "", text)
    # === FILE: ... === echoed format markers
    text = _re.sub(r"={3,}\s*FILE:.*?={3,}", "", text, flags=_re.DOTALL)
    # **Final response:** / **Final reply:**
    text = _re.sub(r"\*\*Final (?:response|reply):\*\*\s*", "", text)
    # Trailing open code fence
    text = _re.sub(r"```\s*$", "", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
