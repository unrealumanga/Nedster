from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import httpx

from .sandbox import SandboxExecutor
from .security import CapabilityAuthority, PolicyEngine
from .types import RiskLevel, ToolAction, ToolRequest

ToolFunction = Callable[[dict[str, Any]], dict[str, Any]]
IGNORED_INDEX_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
        "venv",
    }
)
PROJECT_SEARCH_STOP_WORDS = frozenset(
    {
        "about",
        "active",
        "and",
        "are",
        "find",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "locate",
        "of",
        "project",
        "repo",
        "repository",
        "says",
        "search",
        "show",
        "statement",
        "summarize",
        "that",
        "the",
        "this",
        "to",
        "where",
        "with",
    }
)
CODE_SEARCH_INTENT = re.compile(
    r"\b(?:class|code|defined|function|implement(?:ed|ation)?|method|rejects?|verif(?:y|ied))\b",
    re.IGNORECASE,
)


def _search_term(token: str) -> str:
    value = token.casefold()
    if len(value) > 5 and value.endswith("ied"):
        return value[:-3] + "y"
    if len(value) > 5 and value.endswith("ing"):
        return value[:-3]
    if len(value) > 4 and value.endswith("ed"):
        return value[:-2]
    if len(value) > 4 and value.endswith("s"):
        return value[:-1]
    return value


class _SearchResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._field = ""
        self._buffer: list[str] = []

    @staticmethod
    def _classes(attributes: list[tuple[str, str | None]]) -> set[str]:
        value = next((item or "" for key, item in attributes if key == "class"), "")
        return set(value.split())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = self._classes(attrs)
        if tag == "a" and "result__a" in classes:
            href = next((value or "" for key, value in attrs if key == "href"), "")
            parsed = urlparse(href)
            redirected = parse_qs(parsed.query).get("uddg", [href])[0]
            self.results.append({"title": "", "url": unquote(redirected), "snippet": ""})
            self._field = "title"
            self._buffer = []
        elif self.results and "result__snippet" in classes:
            self._field = "snippet"
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._field:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._field and tag in {"a", "div"}:
            value = " ".join("".join(self._buffer).split())
            if value and self.results:
                self.results[-1][self._field] = value
            self._field = ""
            self._buffer = []


class _ReadableHTMLParser(HTMLParser):
    BLOCK_TAGS = frozenset({"article", "br", "div", "h1", "h2", "h3", "h4", "li", "main", "p", "section", "title"})
    SKIP_TAGS = frozenset({"script", "style", "noscript", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.fragments: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif not self._skip_depth and tag in self.BLOCK_TAGS:
            self.fragments.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif not self._skip_depth and tag in self.BLOCK_TAGS:
            self.fragments.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.fragments.append(data)

    def text(self) -> str:
        lines = [" ".join(line.split()) for line in "".join(self.fragments).splitlines()]
        return "\n".join(line for line in lines if line)


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

    def descriptions(self) -> list[dict[str, Any]]:
        parameters: dict[str, dict[str, Any]] = {
            "read_file": {
                "required": {"path": "project-relative string"},
                "optional": {"max_chars": "integer"},
            },
            "list_files": {"optional": {"glob": "glob string", "limit": "integer"}},
            "search_text": {
                "required": {"pattern": "regular-expression string"},
                "optional": {"glob": "glob string", "limit": "integer"},
            },
            "project_search": {
                "required": {"query": "natural-language search query"},
                "optional": {"glob": "glob string", "limit": "integer"},
            },
            "git_diff": {"optional": {"paths": "array of project-relative strings"}},
            "write_file": {
                "required": {"path": "project-relative string", "content": "string"}
            },
            "replace_text": {
                "required": {
                    "path": "project-relative string",
                    "old": "exact string occurring once",
                    "new": "replacement string",
                }
            },
            "run_command": {
                "required": {"command": "argv string array"},
                "optional": {
                    "cwd": "project-relative string",
                    "expect_exit": "integer exit code (default 0)",
                    "expect_stdout_contains": "required stdout substring",
                    "expect_stderr_contains": "required stderr substring",
                },
            },
            "run_tests": {"optional": {"cwd": "project-relative string"}},
            "web_search": {
                "required": {"query": "search query string"},
                "optional": {"limit": "integer from 1 to 10"},
            },
            "web_fetch": {
                "required": {"url": "public HTTPS URL"},
                "optional": {"max_chars": "integer"},
            },
            "market_price": {"required": {"symbol": "market pair such as BTCUSDT"}},
        }
        return [
            {
                "name": name,
                "risk": risk.name.lower(),
                "description": description,
                "parameters": parameters.get(name, {}),
            }
            for name, (risk, _, description) in sorted(self._tools.items())
        ]

    def _path(self, raw: str) -> Path:
        # Resolve the root on every boundary check. Windows may expose the same
        # temporary directory through both an 8.3 alias and its long path; a
        # lexical comparison between those spellings incorrectly looks like an
        # escape even though both paths identify the same directory.
        root = self.project_root.resolve()
        self.project_root = root
        candidate = (root / raw).resolve()
        try:
            candidate.relative_to(root)
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
        self.register(
            "project_search",
            RiskLevel.READ,
            self._project_search,
            "Rank relevant project passages for a natural-language query.",
        )
        self.register("git_diff", RiskLevel.READ, self._git_diff, "Read the current git diff.")
        self.register("write_file", RiskLevel.WRITE, self._write_file, "Atomically create or replace a project file.")
        self.register("replace_text", RiskLevel.WRITE, self._replace_text, "Replace one exact text occurrence in a file.")
        self.register("run_command", RiskLevel.EXECUTE, self._run_command, "Run an argv command inside the configured sandbox.")
        self.register(
            "run_tests",
            RiskLevel.EXECUTE,
            self._run_tests,
            "Detect the project test runner and run its tests inside the sandbox.",
        )
        self.register(
            "web_search",
            RiskLevel.READ,
            self._web_search,
            "Search the public web through DuckDuckGo without an API key.",
        )
        self.register(
            "web_fetch",
            RiskLevel.READ,
            self._web_fetch,
            "Fetch readable text or JSON from a public HTTPS URL; private hosts are blocked.",
        )
        self.register(
            "market_price",
            RiskLevel.READ,
            self._market_price,
            "Get and cross-check a current public market quote from Binance and Coinbase.",
        )

    @staticmethod
    def _glob_variants(pattern: str) -> list[str]:
        match = re.search(r"\{([^{}]+)}", pattern)
        variants = [pattern]
        if match:
            variants = [
                pattern[: match.start()] + choice.strip() + pattern[match.end() :]
                for choice in match.group(1).split(",")
                if choice.strip()
            ]
        return [value if "/" in value else f"**/{value}" for value in variants]

    def _matching_files(self, pattern: str) -> list[Path]:
        self.project_root = self.project_root.resolve()
        paths: set[Path] = set()
        for variant in self._glob_variants(pattern):
            for path in self.project_root.glob(variant):
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    relative = path.resolve().relative_to(self.project_root)
                except (OSError, ValueError):
                    continue
                if any(part in IGNORED_INDEX_DIRS for part in relative.parts):
                    continue
                paths.add(path)
        return sorted(paths, key=lambda path: path.relative_to(self.project_root).as_posix())

    def _read_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(str(arguments["path"]))
        max_chars = min(int(arguments.get("max_chars", 80_000)), 250_000)
        content = path.read_text(encoding="utf-8", errors="replace")
        return {"path": str(path.relative_to(self.project_root)), "content": content[:max_chars], "truncated": len(content) > max_chars}

    def _list_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        pattern = str(arguments.get("glob", "**/*"))
        limit = min(int(arguments.get("limit", 500)), 2_000)
        matched = self._matching_files(pattern)
        paths = [path.relative_to(self.project_root).as_posix() for path in matched[:limit]]
        return {
            "files": paths,
            "count": len(paths),
            "total_matches": len(matched),
            "truncated": len(matched) > limit,
        }

    def _search_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expression = re.compile(str(arguments["pattern"]))
        file_glob = str(arguments.get("glob", "**/*"))
        limit = min(int(arguments.get("limit", 200)), 1_000)
        matches: list[dict[str, Any]] = []
        for path in self._matching_files(file_glob):
            if path.stat().st_size > 2_000_000:
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
                            return {"matches": matches, "count": len(matches), "truncated": True}
            except (UnicodeDecodeError, OSError):
                continue
        return {"matches": matches, "count": len(matches), "truncated": False}

    def _project_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = " ".join(str(arguments.get("query") or "").split())
        terms = {
            _search_term(term)
            for term in re.findall(r"[A-Za-z0-9]+", query)
            if len(term) > 1 and _search_term(term) not in PROJECT_SEARCH_STOP_WORDS
        }
        if not terms:
            raise ValueError("query must contain at least one meaningful search term")
        file_glob = str(arguments.get("glob", "**/*"))
        limit = min(max(int(arguments.get("limit", 20)), 1), 100)
        code_intent = bool(CODE_SEARCH_INTENT.search(query))
        ranked: list[dict[str, Any]] = []
        for path in self._matching_files(file_glob):
            if path.stat().st_size > 2_000_000:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for index, line in enumerate(lines):
                line_terms = {
                    _search_term(term) for term in re.findall(r"[A-Za-z0-9]+", line)
                }
                overlap = terms & line_terms
                if not overlap:
                    continue
                start, end = max(0, index - 2), min(len(lines), index + 3)
                passage = "\n".join(
                    f"{line_number + 1}: {lines[line_number][:500]}"
                    for line_number in range(start, end)
                )
                passage_terms = {
                    _search_term(term)
                    for term in re.findall(r"[A-Za-z0-9]+", passage)
                }
                passage_overlap = terms & passage_terms
                path_terms = {
                    _search_term(term)
                    for term in re.findall(
                        r"[A-Za-z0-9_-]+", f"{path.as_posix()} {path.stem}"
                    )
                }
                relative_parts = path.relative_to(self.project_root).parts
                score = len(passage_overlap) * 3 + len(terms & path_terms)
                if any(part == "tests" or part.startswith("test_") for part in relative_parts):
                    score -= 6
                if code_intent and path.suffix.casefold() in {".py", ".js", ".ts", ".rs", ".go"}:
                    score += 5
                if code_intent and "docs" in relative_parts:
                    score -= 3
                elif "docs" in relative_parts or path.name.casefold().startswith("readme"):
                    score += 4
                ranked.append(
                    {
                        "path": path.relative_to(self.project_root).as_posix(),
                        "line": index + 1,
                        "score": score,
                        "matched_terms": sorted(passage_overlap),
                        "text": passage,
                    }
                )
        ranked.sort(key=lambda item: (-int(item["score"]), str(item["path"]), int(item["line"])))
        matches = ranked[:limit]
        return {
            "query": query,
            "terms": sorted(terms),
            "matches": matches,
            "count": len(matches),
            "truncated": len(ranked) > limit,
        }

    def _git_diff(self, arguments: dict[str, Any]) -> dict[str, Any]:
        paths = [str(item) for item in arguments.get("paths", [])]
        for path in paths:
            self._path(path)
        diff_result = subprocess.run(
            ["git", "diff", "--", *paths],
            cwd=self.project_root,
            text=True,
            capture_output=True,
            timeout=20,
            shell=False,
            check=False,
        )
        stat_result = subprocess.run(
            ["git", "diff", "--stat", "--", *paths],
            cwd=self.project_root,
            text=True,
            capture_output=True,
            timeout=20,
            shell=False,
            check=False,
        )
        status_result = subprocess.run(
            ["git", "status", "--short", "--", *paths],
            cwd=self.project_root,
            text=True,
            capture_output=True,
            timeout=20,
            shell=False,
            check=False,
        )
        status_entries = []
        for line in status_result.stdout.splitlines():
            if len(line) < 3:
                continue
            status_entries.append({"status": line[:2], "path": line[3:]})
        tracked_files = [
            item["path"] for item in status_entries if item["status"] != "??"
        ]
        untracked_files = [
            item["path"] for item in status_entries if item["status"] == "??"
        ]
        max_diff_chars = 120_000
        return {
            "returncode": diff_result.returncode,
            "status": status_result.stdout[-40_000:],
            "tracked_files": tracked_files,
            "tracked_count": len(tracked_files),
            "untracked_files": untracked_files,
            "untracked_count": len(untracked_files),
            "stat": stat_result.stdout[-40_000:],
            "diff": diff_result.stdout[:max_diff_chars],
            "truncated": len(diff_result.stdout) > max_diff_chars,
            "stderr": diff_result.stderr[-10_000:],
        }

    def _write_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(str(arguments["path"]))
        content = self._text_argument(arguments, "content", default="")
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
        old = self._text_argument(arguments, "old")
        new = self._text_argument(arguments, "new")
        content = path.read_text(encoding="utf-8")
        count = content.count(old)
        if count != 1:
            raise ValueError(f"expected exactly one match, found {count}")
        return self._write_file({"path": str(path.relative_to(self.project_root)), "content": content.replace(old, new, 1)})

    @staticmethod
    def _text_argument(
        arguments: dict[str, Any], key: str, *, default: str | None = None
    ) -> str:
        if key not in arguments:
            if default is not None:
                return default
            raise ValueError(f"{key} is required")
        value = arguments[key]
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string or one-item string array")
        return value

    def _run_command(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = arguments.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
            raise ValueError("command must be a non-empty argv string array")
        command = self._normalize_cargo_run(command)
        expectations: dict[str, str] = {}
        for key in ("expect_stdout_contains", "expect_stderr_contains"):
            if key not in arguments:
                continue
            raw_expected = arguments[key]
            if isinstance(raw_expected, list) and len(raw_expected) == 1:
                raw_expected = raw_expected[0]
            if not isinstance(raw_expected, str):
                raise ValueError(f"{key} must be a string or one-item string array")
            expectations[key] = raw_expected
        cwd = self._path(str(arguments.get("cwd", ".")))
        result = self.sandbox.execute(command, cwd)
        assertions: list[dict[str, Any]] = []
        has_expectation = any(
            key in arguments
            for key in ("expect_exit", "expect_stdout_contains", "expect_stderr_contains")
        )
        if has_expectation:
            expected_exit = int(arguments.get("expect_exit", 0))
            assertions.append(
                {
                    "check": "exit_code",
                    "expected": expected_exit,
                    "actual": result.returncode,
                    "passed": result.returncode == expected_exit,
                }
            )
            if "expect_stdout_contains" in expectations:
                expected = expectations["expect_stdout_contains"]
                assertions.append(
                    {
                        "check": "stdout_contains",
                        "expected": expected,
                        "passed": expected in result.stdout,
                    }
                )
            if "expect_stderr_contains" in expectations:
                expected = expectations["expect_stderr_contains"]
                assertions.append(
                    {
                        "check": "stderr_contains",
                        "expected": expected,
                        "passed": expected in result.stderr,
                    }
                )
        combined = f"{result.stdout}\n{result.stderr}"
        return {
            "command": command,
            "backend": result.backend,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "assertions": assertions,
            "assertions_passed": (
                all(bool(assertion["passed"]) for assertion in assertions)
                if has_expectation
                else None
            ),
            "tests_run": self._count_tests(command, combined),
        }

    @staticmethod
    def _normalize_cargo_run(command: list[str]) -> list[str]:
        """Insert Cargo's argument separator when the model supplied clear app arguments."""
        if command[:2] != ["cargo", "run"] or "--" in command[2:]:
            return command
        no_value = {
            "--quiet",
            "-q",
            "--release",
            "--frozen",
            "--locked",
            "--offline",
            "--ignore-rust-version",
        }
        takes_value = {
            "--package",
            "-p",
            "--bin",
            "--example",
            "--features",
            "--profile",
            "--target",
            "--target-dir",
            "--manifest-path",
            "--jobs",
            "-j",
        }
        index = 2
        while index < len(command):
            token = command[index]
            if token in no_value:
                index += 1
                continue
            if token in takes_value and index + 1 < len(command):
                index += 2
                continue
            return [*command[:index], "--", *command[index:]]
        return command

    @staticmethod
    def _count_tests(command: list[str], combined: str) -> int | None:
        if command[:2] == ["cargo", "test"]:
            return sum(map(int, re.findall(r"\brunning (\d+) tests?\b", combined)))
        if "unittest" in command:
            match = re.search(r"\bRan (\d+) tests?\b", combined)
            return int(match.group(1)) if match else None
        if "pytest" in command:
            match = re.search(r"\b(\d+) passed\b", combined)
            return int(match.group(1)) if match else None
        if command[:2] == ["go", "test"]:
            return len(re.findall(r"(?m)^--- PASS:", combined)) or None
        if command[:2] == ["npm", "test"]:
            match = re.search(r"\bTests:\s+(\d+) passed\b", combined)
            return int(match.group(1)) if match else None
        return None

    def _run_tests(self, arguments: dict[str, Any]) -> dict[str, Any]:
        cwd = self._path(str(arguments.get("cwd", ".")))
        if (cwd / "Cargo.toml").exists():
            command = ["cargo", "test"]
        elif (cwd / "go.mod").exists():
            command = ["go", "test", "./..."]
        elif (cwd / "package.json").exists():
            command = ["npm", "test", "--", "--runInBand"]
        elif (cwd / "pyproject.toml").exists() or any(cwd.glob("**/test*.py")):
            python_tests = list(cwd.glob("**/test*.py"))
            uses_unittest = any(
                "unittest" in path.read_text(encoding="utf-8", errors="ignore")
                for path in python_tests[:100]
            )
            if uses_unittest and shutil.which("python3"):
                command = ["python3", "-m", "unittest", "discover", "-v"]
            elif shutil.which("uv"):
                command = ["uv", "run", "--with", "pytest", "pytest", "-q"]
            elif shutil.which("python3"):
                command = ["python3", "-m", "pytest", "-q"]
            else:
                raise RuntimeError("no Python test runner is available")
        else:
            raise RuntimeError("could not detect a supported test suite")
        result = self.sandbox.execute(command, cwd)
        combined = f"{result.stdout}\n{result.stderr}"
        tests_run = self._count_tests(command, combined)
        return {
            "command": command,
            "backend": result.backend,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "tests_run": tests_run,
        }

    @staticmethod
    def _public_https_url(raw: str) -> str:
        parsed = urlparse(raw.strip())
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            raise ValueError("web_fetch requires a public HTTPS URL")
        if parsed.username or parsed.password or parsed.port not in {None, 443}:
            raise ValueError("URL credentials and non-standard ports are not allowed")
        hostname = parsed.hostname.rstrip(".").casefold()
        if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
            raise PermissionError("local and private hosts are blocked")
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
            }
        except socket.gaierror as exc:
            raise ValueError(f"could not resolve public host: {hostname}") from exc
        if not addresses:
            raise ValueError(f"could not resolve public host: {hostname}")
        for raw_address in addresses:
            address = ipaddress.ip_address(raw_address)
            if not address.is_global:
                raise PermissionError("local, private, reserved, and link-local hosts are blocked")
        return parsed.geturl()

    @classmethod
    def _get_public(cls, url: str, *, max_bytes: int = 1_000_000) -> tuple[str, str, str]:
        current = cls._public_https_url(url)
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Nedster/3.1; +https://github.com)",
            "Accept": "text/html,application/json,text/plain,application/xml;q=0.9,*/*;q=0.1",
        }
        with httpx.Client(timeout=20, follow_redirects=False, headers=headers) as client:
            for _ in range(4):
                with client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise RuntimeError("redirect response omitted a destination")
                        current = cls._public_https_url(urljoin(current, location))
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    allowed = content_type.startswith("text/") or content_type in {
                        "application/json",
                        "application/ld+json",
                        "application/xml",
                        "application/xhtml+xml",
                    }
                    if not allowed:
                        raise ValueError(f"unsupported web content type: {content_type or 'unknown'}")
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            chunks.append(chunk[: max_bytes - (size - len(chunk))])
                            break
                        chunks.append(chunk)
                    encoding = response.encoding or "utf-8"
                    return current, content_type, b"".join(chunks).decode(encoding, errors="replace")
        raise RuntimeError("too many redirects")

    def _web_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = " ".join(str(arguments.get("query") or "").split())
        if not query:
            raise ValueError("query must not be empty")
        limit = min(max(int(arguments.get("limit", 5)), 1), 10)
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        failures: list[str] = []
        try:
            final_url, _, content = self._get_public(search_url)
            parser = _SearchResultParser()
            parser.feed(content)
            results = [
                item for item in parser.results if item.get("title") and item.get("url")
            ][:limit]
            if results:
                return {
                    "query": query,
                    "provider": "DuckDuckGo",
                    "url": final_url,
                    "results": results,
                    "retrieved_at": datetime.now(UTC).isoformat(),
                }
            reason = "blocked by a bot challenge" if "anomaly-modal" in content else "no results"
            failures.append(f"DuckDuckGo: {reason}")
        except Exception as exc:
            failures.append(f"DuckDuckGo: {type(exc).__name__}: {exc}")

        news_query = bool(re.search(r"\b(?:breaking|headline|news)\b", query, re.IGNORECASE))
        search_kind = "news/search" if news_query else "search"
        bing_url = (
            f"https://www.bing.com/{search_kind}?format=rss&q={quote_plus(query)}"
            "&mkt=en-US&setlang=en-US&cc=US"
        )
        try:
            final_url, _, content = self._get_public(bing_url)
            results = self._bing_rss_results(content, limit)
            if results:
                return {
                    "query": query,
                    "provider": "Bing News" if news_query else "Bing",
                    "url": final_url,
                    "results": results,
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "provider_warnings": failures,
                }
            failures.append(f"{'Bing News' if news_query else 'Bing'}: no results")
        except Exception as exc:
            failures.append(f"Bing: {type(exc).__name__}: {exc}")
        raise RuntimeError("public search failed: " + "; ".join(failures))

    @staticmethod
    def _bing_rss_results(content: str, limit: int) -> list[dict[str, str]]:
        root = ET.fromstring(content)
        results: list[dict[str, str]] = []
        for item in root.findall(".//item"):
            title = " ".join((item.findtext("title") or "").split())
            raw_url = (item.findtext("link") or "").strip()
            parsed = urlparse(raw_url)
            target = parse_qs(parsed.query).get("url", [""])[0]
            url = unquote(target) if target else raw_url
            if not title or not url.startswith("https://"):
                continue
            description = re.sub(r"<[^>]+>", " ", item.findtext("description") or "")
            description = " ".join(description.split())
            published = " ".join((item.findtext("pubDate") or "").split())
            snippet = description
            if published:
                snippet = f"{description} Published: {published}".strip()
            results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= limit:
                break
        return results

    def _web_fetch(self, arguments: dict[str, Any]) -> dict[str, Any]:
        max_chars = min(max(int(arguments.get("max_chars", 30_000)), 1_000), 100_000)
        final_url, content_type, content = self._get_public(str(arguments.get("url") or ""))
        if "html" in content_type:
            parser = _ReadableHTMLParser()
            parser.feed(content)
            content = parser.text()
        return {
            "url": final_url,
            "content_type": content_type,
            "content": content[:max_chars],
            "truncated": len(content) > max_chars,
            "retrieved_at": datetime.now(UTC).isoformat(),
        }

    def _market_price(self, arguments: dict[str, Any]) -> dict[str, Any]:
        symbol = re.sub(r"[^A-Za-z0-9]", "", str(arguments.get("symbol") or "")).upper()
        match = re.fullmatch(r"([A-Z0-9]{2,12})(USDT|USDC|USD|EUR)", symbol)
        if not match:
            raise ValueError("symbol must be a pair such as BTCUSDT, ETHUSD, or SOLUSDC")
        base, quote = match.groups()
        observed_at = datetime.now(UTC).isoformat()
        quotes: list[dict[str, str]] = []
        errors: list[str] = []
        endpoints = [
            ("Binance", f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"),
            ("Coinbase", f"https://api.exchange.coinbase.com/products/{base}-{quote}/ticker"),
        ]
        for provider, url in endpoints:
            try:
                final_url, _, content = self._get_public(url, max_bytes=100_000)
                value = json.loads(content)
                price = str(value.get("price") or "").strip()
                if not price or Decimal(price) <= 0:
                    raise ValueError("response did not contain a positive price")
                quotes.append(
                    {
                        "provider": provider,
                        "symbol": symbol,
                        "price": price,
                        "quote_currency": quote,
                        "source_url": final_url,
                        "source_time": str(value.get("time") or observed_at),
                    }
                )
            except Exception as exc:
                errors.append(f"{provider}: {type(exc).__name__}: {exc}")
        if not quotes:
            raise RuntimeError("all public quote providers failed: " + "; ".join(errors))
        prices = [Decimal(item["price"]) for item in quotes]
        spread_pct = Decimal("0")
        if len(prices) > 1:
            midpoint = (max(prices) + min(prices)) / 2
            spread_pct = (max(prices) - min(prices)) / midpoint * 100
        return {
            "symbol": symbol,
            "quotes": quotes,
            "provider_errors": errors,
            "spread_percent": format(spread_pct, ".6f"),
            "observed_at": observed_at,
        }
