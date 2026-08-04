from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .sandbox import SandboxExecutor
from .security import CapabilityAuthority, PolicyEngine
from .types import RiskLevel, ToolAction, ToolRequest

ToolFunction = Callable[[dict[str, Any]], dict[str, Any]]


class ToolRegistry:
    def __init__(
        self,
        project_root: str | Path,
        authority: CapabilityAuthority,
        policy: PolicyEngine,
        sandbox: SandboxExecutor,
    ):
        self.project_root = Path(project_root).resolve()
        self.authority = authority
        self.policy = policy
        self.sandbox = sandbox
        self._tools: dict[str, tuple[RiskLevel, ToolFunction, str]] = {}
        self._register_builtins()

    def register(self, name: str, risk: RiskLevel, function: ToolFunction, description: str) -> None:
        self._tools[name] = (risk, function, description)

    def descriptions(self) -> list[dict[str, str]]:
        return [
            {"name": name, "risk": risk.name.lower(), "description": description}
            for name, (risk, _, description) in sorted(self._tools.items())
        ]

    def _path(self, raw: str) -> Path:
        candidate = (self.project_root / raw).resolve()
        try:
            candidate.relative_to(self.project_root)
        except ValueError as exc:
            raise PermissionError(f"path escapes project root: {raw}") from exc
        return candidate

    def prepare(self, action: ToolAction) -> ToolRequest:
        if action.tool not in self._tools:
            raise KeyError(f"unknown tool: {action.tool}")
        if action.tool in {"read_file", "write_file", "replace_text"}:
            self._path(str(action.arguments["path"]))
        elif action.tool == "run_command":
            self._path(str(action.arguments.get("cwd", ".")))
        risk = self._tools[action.tool][0]
        return ToolRequest(action.tool, action.arguments, risk, str(self.project_root))

    def authorize(self, request: ToolRequest) -> tuple[str | None, str]:
        allowed, reason = self.policy.evaluate(request)
        return (self.authority.mint(request), reason) if allowed else (None, reason)

    def execute(self, request: ToolRequest, token: str) -> dict[str, Any]:
        self.authority.verify(token, request, consume=True)
        _, function, _ = self._tools[request.tool]
        return function(request.arguments)

    def _register_builtins(self) -> None:
        self.register("read_file", RiskLevel.READ, self._read_file, "Read a UTF-8 project file.")
        self.register("list_files", RiskLevel.READ, self._list_files, "List project files matching a glob.")
        self.register("search_text", RiskLevel.READ, self._search_text, "Search project text with a regular expression.")
        self.register("git_diff", RiskLevel.READ, self._git_diff, "Read the current git diff.")
        self.register("write_file", RiskLevel.WRITE, self._write_file, "Atomically create or replace a project file.")
        self.register("replace_text", RiskLevel.WRITE, self._replace_text, "Replace one exact text occurrence in a file.")
        self.register("run_command", RiskLevel.EXECUTE, self._run_command, "Run an argv command inside the configured sandbox.")

    def _read_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(str(arguments["path"]))
        max_chars = min(int(arguments.get("max_chars", 80_000)), 250_000)
        content = path.read_text(encoding="utf-8", errors="replace")
        return {"path": str(path.relative_to(self.project_root)), "content": content[:max_chars], "truncated": len(content) > max_chars}

    def _list_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        pattern = str(arguments.get("glob", "**/*"))
        limit = min(int(arguments.get("limit", 500)), 2_000)
        paths = []
        for path in self.project_root.glob(pattern):
            if path.is_symlink():
                continue
            try:
                path.resolve().relative_to(self.project_root)
            except (OSError, ValueError):
                continue
            if path.is_file() and ".git" not in path.parts:
                paths.append(path.relative_to(self.project_root).as_posix())
                if len(paths) >= limit:
                    break
        return {"files": sorted(paths), "truncated": len(paths) >= limit}

    def _search_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expression = re.compile(str(arguments["pattern"]))
        file_glob = str(arguments.get("glob", "**/*"))
        limit = min(int(arguments.get("limit", 200)), 1_000)
        matches: list[dict[str, Any]] = []
        for path in self.project_root.glob(file_glob):
            if path.is_symlink():
                continue
            try:
                path.resolve().relative_to(self.project_root)
            except (OSError, ValueError):
                continue
            if not path.is_file() or ".git" in path.parts or path.stat().st_size > 2_000_000:
                continue
            try:
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if expression.search(line):
                        matches.append(
                            {
                                "path": path.relative_to(self.project_root).as_posix(),
                                "line": line_number,
                                "text": line[:500],
                            }
                        )
                        if len(matches) >= limit:
                            return {"matches": matches, "truncated": True}
            except (UnicodeDecodeError, OSError):
                continue
        return {"matches": matches, "truncated": False}

    def _git_diff(self, arguments: dict[str, Any]) -> dict[str, Any]:
        paths = [str(item) for item in arguments.get("paths", [])]
        for path in paths:
            self._path(path)
        command = ["git", "diff", "--", *paths]
        result = subprocess.run(
            command,
            cwd=self.project_root,
            text=True,
            capture_output=True,
            timeout=20,
            shell=False,
            check=False,
        )
        return {"returncode": result.returncode, "diff": result.stdout[-120_000:], "stderr": result.stderr[-10_000:]}

    def _write_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(str(arguments["path"]))
        content = str(arguments.get("content", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = path.read_text(encoding="utf-8", errors="replace") if path.exists() else None
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {
            "path": path.relative_to(self.project_root).as_posix(),
            "bytes": len(content.encode("utf-8")),
            "created": previous is None,
            "changed": previous != content,
        }

    def _replace_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(str(arguments["path"]))
        old, new = str(arguments["old"]), str(arguments["new"])
        content = path.read_text(encoding="utf-8")
        count = content.count(old)
        if count != 1:
            raise ValueError(f"expected exactly one match, found {count}")
        return self._write_file({"path": str(path.relative_to(self.project_root)), "content": content.replace(old, new, 1)})

    def _run_command(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = arguments.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
            raise ValueError("command must be a non-empty argv string array")
        cwd = self._path(str(arguments.get("cwd", ".")))
        result = self.sandbox.execute(command, cwd)
        return {
            "backend": result.backend,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
        }
