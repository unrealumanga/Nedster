"""search_text: ripgrep (``rg --json``) when a *trusted* one is available, pure-Python fallback otherwise.

``rg`` is resolved exactly like every other trusted program (``exec.find_program`` over the
allowlisted PATH): absolute, never from a relative PATH entry and never from inside the project
(a planted ``./rg`` is ignored and the pure-Python search runs instead). The child gets the same
allowlisted environment sandboxed commands use (``exec.build_env``), so secrets never pass.

Literal by default; regex mode compiles the pattern up front so a bad expression is
INVALID before anything runs. Output lines are ``relpath:line: text`` (``line-`` for
context lines when ``context`` is requested).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import weakref
from pathlib import Path

from ..contracts import ToolCall, ToolResult, ToolSpec, ToolStatus
from ..types import RiskLevel
from .exec import build_env, find_program, host_search_path
from .files import Inventory, ToolError, locate, walk_inventory
from .schema import validate

SEARCH_TEXT = "search_text"
TOOLS = (SEARCH_TEXT,)
MAX_PATTERN_LENGTH = 500
DEFAULT_MAX_RESULTS = 100
MAX_RESULTS_LIMIT = 1000
MAX_CONTEXT = 20
MAX_SEARCH_FILE_BYTES = 2_000_000
RG_TIMEOUT_S = 30

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Literal text by default; a regex when regex=true."},
        "path": {"type": "string", "description": "File or directory to search, relative to the project root."},
        "regex": {"type": "boolean"},
        "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS_LIMIT},
        "context": {"type": "integer", "minimum": 0, "maximum": MAX_CONTEXT},
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


def find_ripgrep(root: Path) -> Path | None:
    """``rg`` resolved the same way ``Executor`` resolves trusted programs, or ``None``.

    Reuses ``exec.find_program`` over ``exec.host_search_path`` (the user's PATH with relative
    entries and every directory inside the project removed), so an ``rg`` planted in the project
    or reached through a project PATH entry is never returned. ``None`` means "use the Python
    fallback".
    """
    search_path = host_search_path(root, os.environ.get("PATH", ""))
    return find_program("rg", search_path, root)


class SearchTools:
    """Sync ToolRuntime exposing one read-only tool: ``search_text``."""

    def __init__(self, project_root: str | Path, *, inventory: Inventory | None = None):
        self.project_root = Path(project_root).resolve()
        self._inventory = inventory or walk_inventory
        self._environment: dict[str, str] | None = None

    def _child_env(self) -> dict[str, str]:
        """Allowlisted environment for the rg child, built by the same ``exec.build_env`` used
        for sandboxed commands (secrets dropped, PATH sanitized). Created once per instance."""
        if self._environment is None:
            temp = Path(tempfile.mkdtemp(prefix="nedster-search-"))
            weakref.finalize(self, shutil.rmtree, str(temp), True)
            self._environment = build_env(self.project_root, temp)
        return dict(self._environment)

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                SEARCH_TEXT,
                "Search project files for a literal string (default) or regular expression. "
                "Returns 'relpath:line: text' lines.",
                SEARCH_SCHEMA,
                int(RiskLevel.READ),
            )
        ]

    def run(self, call: ToolCall) -> ToolResult:
        if call.name != SEARCH_TEXT:
            return ToolResult(call.id, call.name, ToolStatus.INVALID, f"unknown tool {call.name!r}")
        errors = validate(SEARCH_SCHEMA, dict(call.arguments or {}))
        if errors:
            return ToolResult(call.id, call.name, ToolStatus.INVALID, "\n".join(errors))
        try:
            return self._search(call)
        except ToolError as exc:
            return ToolResult(call.id, call.name, ToolStatus.INVALID, str(exc))
        except OSError as exc:
            return ToolResult(call.id, call.name, ToolStatus.FAILED, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # a tool bug must not escape as an exception
            return ToolResult(call.id, call.name, ToolStatus.FAILED, f"tool error: {exc}")

    # -- internals ---------------------------------------------------------------------

    def _prepare(self, call: ToolCall) -> tuple[str, bool, Path, str, Path, int, int]:
        args = call.arguments
        pattern = str(args["pattern"])
        if not pattern:
            raise ToolError("pattern must not be empty")
        if len(pattern) > MAX_PATTERN_LENGTH:
            raise ToolError(f"pattern is too long ({len(pattern)} > {MAX_PATTERN_LENGTH} characters)")
        regex = bool(args.get("regex", False))
        if regex:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ToolError(f"invalid regular expression: {exc}") from exc
        located = locate(self.project_root, str(args.get("path") or "."), for_write=False)
        if not located.path.exists():
            raise ToolError(f"no such path: {located.rel}")
        max_results = int(args.get("max_results", DEFAULT_MAX_RESULTS))
        context = int(args.get("context", 0))
        return pattern, regex, located.path, located.rel, located.root, max_results, context

    def _search(self, call: ToolCall) -> ToolResult:
        pattern, regex, path, rel, base, max_results, context = self._prepare(call)
        rg = find_ripgrep(self.project_root)
        if rg is not None:
            result = self._search_ripgrep(call, str(rg), pattern, rel, base, regex, max_results, context)
            if result is not None:
                return result
        return self._search_python(call, pattern, path, rel, base, regex, max_results, context)

    @staticmethod
    def _finish(call: ToolCall, lines: list[str], count: int, truncated: bool, backend: str) -> ToolResult:
        output = "\n".join(lines)
        if truncated:
            output += f"\n[truncated: showing first {count} matches]"
        data = {"count": count, "truncated": truncated, "backend": backend}
        return ToolResult(call.id, call.name, ToolStatus.OK, output, data)

    def _search_ripgrep(
        self,
        call: ToolCall,
        rg: str,
        pattern: str,
        rel: str,
        base: Path,
        regex: bool,
        max_results: int,
        context: int,
    ) -> ToolResult | None:
        argv = [rg, "--json", "--no-config", "--no-messages", "--line-number"]
        if not regex:
            argv.append("--fixed-strings")
        if context:
            argv.append(f"--context={context}")
        argv.extend(["--", pattern, rel if rel != "." else "."])
        env = self._child_env()
        env["GIT_OPTIONAL_LOCKS"] = "0"
        try:
            proc = subprocess.run(
                argv,
                cwd=base,
                env=env,
                capture_output=True,
                text=True,
                timeout=RG_TIMEOUT_S,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode not in (0, 1):  # 1 = no matches; anything else: let Python handle it
            return None
        lines: list[str] = []
        count = 0
        truncated = False
        for raw_line in proc.stdout.splitlines():
            try:
                entry = json.loads(raw_line)
            except ValueError:
                continue
            kind = entry.get("type")
            if kind not in ("match", "context"):
                continue
            data = entry.get("data") or {}
            path_text = (((data.get("path") or {}).get("text")) or "").replace("\\", "/")
            if path_text.startswith("./"):
                path_text = path_text[2:]
            line_number = data.get("line_number") or 0
            text = ((data.get("lines") or {}).get("text") or "").rstrip("\r\n")
            if kind == "match":
                if count >= max_results:
                    truncated = True
                    break
                count += 1
            lines.append(f"{path_text}:{line_number}{':' if kind == 'match' else '-'} {text}")
        return self._finish(call, lines, count, truncated, "ripgrep")

    def _search_python(
        self,
        call: ToolCall,
        pattern: str,
        path: Path,
        rel: str,
        base: Path,
        regex: bool,
        max_results: int,
        context: int,
    ) -> ToolResult:
        matcher = re.compile(pattern) if regex else None
        entries = sorted(set(self._inventory(base)))
        if path.is_file():
            entries = [rel] if rel in entries else []
        elif rel != ".":
            prefix = rel + "/"
            entries = [entry for entry in entries if entry.startswith(prefix)]
        lines: list[str] = []
        count = 0
        truncated = False
        for entry in entries:
            full = base / entry
            try:
                if full.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                data = full.read_bytes()
            except OSError:
                continue
            if b"\x00" in data:
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            file_lines = text.splitlines()
            hits = [
                index
                for index, line in enumerate(file_lines)
                if (matcher.search(line) if matcher is not None else pattern in line)
            ]
            if not hits:
                continue
            match_set = set(hits)
            emitted_up_to = -1
            stop = False
            for index in hits:
                if count >= max_results:
                    truncated = True
                    stop = True
                    break
                start = max(0, index - context, emitted_up_to + 1)
                end = min(len(file_lines), index + context + 1)
                for position in range(start, end):
                    is_match = position in match_set
                    if is_match:
                        if count >= max_results:
                            truncated = True
                            stop = True
                            break
                        count += 1
                    marker = ":" if is_match else "-"
                    lines.append(f"{entry}:{position + 1}{marker} {file_lines[position]}")
                emitted_up_to = max(emitted_up_to, end - 1)
                if stop:
                    break
            if stop:
                break
        return self._finish(call, lines, count, truncated, "python")


__all__ = ["TOOLS", "SearchTools", "find_ripgrep"]
