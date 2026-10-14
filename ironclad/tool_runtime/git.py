"""Safe, read-only git: diff / status / log / show.

The argv is always constructed here (never taken from the model) and runs on the host
through ``Executor.start_host``. Every repository-controlled way of executing code during
these commands is switched off: fsmonitor, external diff drivers, textconv, hooks, pagers,
signature verification (gpg.program), and filter drivers defined outside the user's global
config. Submodules are ignored so their (unscanned) configs never run.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from ironclad.contracts import ToolCall, ToolResult, ToolSpec, ToolStatus
from ironclad.types import RiskLevel

from .exec import ArgumentError, Executor, Job

GIT_DIFF, GIT_STATUS, GIT_LOG, GIT_SHOW = "git_diff", "git_status", "git_log", "git_show"
TOOLS = (GIT_DIFF, GIT_STATUS, GIT_LOG, GIT_SHOW)
NO_EXTERNAL = ("--no-ext-diff", "--no-textconv")
REV = re.compile(r"^(?![-.])[A-Za-z0-9._/~^@{}:-]{1,200}$")
DEFAULT_TIMEOUT_S = 30.0
MAX_LOG = 200
TRUSTED_SCOPES = {"system", "global"}  # system is disabled anyway (GIT_CONFIG_NOSYSTEM=1)


class SafeGit:
    def __init__(
        self, project_root: str | Path, executor: Executor | None = None, timeout_s: float = DEFAULT_TIMEOUT_S
    ):
        self.project_root = Path(project_root).resolve()
        self.executor = executor or Executor(self.project_root)
        self.timeout_s = timeout_s
        self.hooks_dir = self.executor.temp_dir / "empty-hooks"
        self.hooks_dir.mkdir(parents=True, exist_ok=True)

    # -- environment and base argv -------------------------------------------------------

    def env(self) -> dict[str, str]:
        env = self.executor.env()
        for key in [key for key in env if key.upper().startswith("GIT_")]:
            del env[key]
        env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat", "PAGER": "cat",
                    "GIT_OPTIONAL_LOCKS": "0"})  # fmt: skip
        return env

    def _hardening(self) -> list[str]:
        settings = {
            "core.fsmonitor": "false",
            "diff.external": "",
            "core.pager": "cat",
            "core.hooksPath": str(self.hooks_dir),
            "log.showSignature": "false",
            "color.ui": "false",
        }
        args: list[str] = []
        for key, value in settings.items():
            args.extend(["-c", f"{key}={value}"])
        return args

    def base_argv(self) -> list[str]:
        return ["git", "--no-pager", *self._hardening(), "--no-optional-locks"]

    def untrusted_filters(self) -> list[str]:
        """Filter drivers configured by repository-controlled config (local/worktree/includes)."""
        argv = [*self.base_argv(), "config", "--show-scope", "--includes", "--get-regexp", r"^filter\."]
        try:
            result = subprocess.run(
                argv, cwd=self.project_root, env=self.env(), capture_output=True, timeout=10, check=False,
                stdin=subprocess.DEVNULL,
            )  # fmt: skip
        except (OSError, subprocess.TimeoutExpired):
            return []
        drivers: set[str] = set()
        for line in result.stdout.decode("utf-8", errors="replace").splitlines():
            scope, _, rest = line.partition("\t")
            key = rest.split(" ", 1)[0]
            if scope in TRUSTED_SCOPES or not key.startswith("filter.") or key.count(".") < 2:
                continue
            drivers.add(key[len("filter.") : key.rindex(".")])
        return sorted(drivers)

    def _filter_overrides(self) -> list[str]:
        args: list[str] = []
        for driver in self.untrusted_filters():
            for key, value in (("clean", ""), ("smudge", ""), ("process", ""), ("required", "false")):
                args.extend(["-c", f"filter.{driver}.{key}={value}"])
        return args

    # -- argument validation ---------------------------------------------------------------

    def _paths(self, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ArgumentError("paths must be an array of strings")
        paths: list[str] = []
        for raw in value:
            if not raw or raw.startswith(("-", ":")) or "\x00" in raw:
                raise ArgumentError(f"invalid path {raw!r}")
            target = (self.project_root / raw).resolve()
            try:
                rel = target.relative_to(self.project_root).as_posix()
            except ValueError as exc:
                raise ArgumentError(f"path {raw!r} is outside the project") from exc
            paths.append(rel)
        return paths

    @staticmethod
    def _rev(value: Any) -> str:
        if not isinstance(value, str) or not REV.match(value):
            raise ArgumentError(f"invalid revision {value!r}")
        return value

    @staticmethod
    def _flag(args: dict[str, Any], name: str) -> bool:
        value = args.get(name, False)
        if not isinstance(value, bool):
            raise ArgumentError(f"{name} must be a boolean")
        return value

    def argv_for(self, call: ToolCall) -> list[str]:
        args = dict(call.arguments or {})
        paths = self._paths(args.get("paths"))
        if call.name == GIT_DIFF:
            sub = ["diff", "--no-color", *NO_EXTERNAL, "--ignore-submodules=all"]
            if self._flag(args, "staged"):
                sub.append("--cached")
            if self._flag(args, "stat"):
                sub.append("--stat")
        elif call.name == GIT_STATUS:
            sub = ["status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=all"]
        elif call.name == GIT_LOG:
            count = args.get("max_count", 20)
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_LOG:
                raise ArgumentError(f"max_count must be an integer between 1 and {MAX_LOG}")
            sub = ["log", "--no-color", *NO_EXTERNAL, f"--max-count={count}", "--date=short",
                   "--format=%h %ad %an %s"]  # fmt: skip
            if self._flag(args, "patch"):
                sub.append("-p")
        elif call.name == GIT_SHOW:
            sub = ["show", "--no-color", *NO_EXTERNAL, "--stat", "-p", self._rev(args.get("rev", "HEAD"))]
        else:
            raise ArgumentError(f"unknown git tool {call.name!r}")
        return [*self.base_argv(), *self._filter_overrides(), *sub, "--", *paths]

    # -- tools -------------------------------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        paths = {"type": "array", "items": {"type": "string"}, "description": "Project-relative paths."}

        def spec(name: str, description: str, properties: dict[str, Any]) -> ToolSpec:
            schema = {"type": "object", "properties": properties, "additionalProperties": False}
            return ToolSpec(name, description, schema, int(RiskLevel.READ), mutates=False)

        return [
            spec(GIT_DIFF, "Show uncommitted changes (git diff).",
                 {"paths": paths, "staged": {"type": "boolean"}, "stat": {"type": "boolean"}}),
            spec(GIT_STATUS, "List changed and untracked files (git status --porcelain).", {"paths": paths}),
            spec(GIT_LOG, "Show recent commits (git log).",
                 {"paths": paths, "max_count": {"type": "integer", "minimum": 1, "maximum": MAX_LOG},
                  "patch": {"type": "boolean"}}),
            spec(GIT_SHOW, "Show one commit with its diff (git show).",
                 {"rev": {"type": "string"}, "paths": paths}),
        ]  # fmt: skip

    def start(self, call: ToolCall) -> Job:
        try:
            argv = self.argv_for(call)
        except ArgumentError as exc:
            return Job.completed(ToolResult(call.id, call.name, ToolStatus.INVALID, str(exc)))
        return self.executor.start_host(call.id, call.name, argv, self.timeout_s, env=self.env())

    async def run(self, call: ToolCall) -> ToolResult:
        return await self.start(call).wait()


__all__ = ["TOOLS", "SafeGit"]
