"""Starter project-instruction file (AGENTS.md) generated from repository facts, never from a model call.

Everything here is deterministic local detection: file extensions for languages, marker files for package
managers, and the engine's own test-suite detection for the test command. ``nedster init`` and the TUI
``/init`` command share these helpers so both produce the same file.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

AGENTS_FILENAME = "AGENTS.md"
MAX_AGENTS_LINES = 40

#: file extension -> language, for the "Languages" section of the starter file
LANGUAGES: dict[str, str] = {
    ".py": "Python", ".pyi": "Python",
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".rs": "Rust", ".go": "Go", ".java": "Java", ".rb": "Ruby", ".php": "PHP",
    ".c": "C", ".h": "C", ".cc": "C++", ".cpp": "C++", ".hpp": "C++",
    ".cs": "C#", ".swift": "Swift", ".kt": "Kotlin", ".scala": "Scala",
    ".sh": "Shell", ".ps1": "PowerShell", ".lua": "Lua", ".dart": "Dart",
    ".html": "HTML", ".css": "CSS", ".scss": "SCSS", ".sql": "SQL",
    ".r": "R", ".jl": "Julia", ".ex": "Elixir", ".exs": "Elixir", ".clj": "Clojure",
}

#: directories never walked when counting languages
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".nedster", "node_modules", "__pycache__", "target",
    "dist", "build", "vendor", "site-packages", ".venv", "venv", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".idea", ".vscode", ".next", ".cache",
})

#: marker file -> package manager, first match wins per ecosystem
PACKAGE_MANAGERS: tuple[tuple[str, str], ...] = (
    ("uv.lock", "uv"),
    ("poetry.lock", "poetry"),
    ("Pipfile", "pipenv"),
    ("requirements.txt", "pip"),
    ("pyproject.toml", "pip"),
    ("pnpm-lock.yaml", "pnpm"),
    ("yarn.lock", "yarn"),
    ("package-lock.json", "npm"),
    ("package.json", "npm"),
    ("Cargo.toml", "cargo"),
    ("go.mod", "go"),
    ("Gemfile", "bundler"),
    ("composer.json", "composer"),
)

MAX_LANGUAGES = 5


def detect_languages(root: str | Path, limit: int = MAX_LANGUAGES) -> list[str]:
    """Most frequent source languages under ``root``, by file extension (build/VCS dirs skipped)."""
    base = Path(root)
    counts: Counter[str] = Counter()
    for _dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")]
        for name in filenames:
            language = LANGUAGES.get(Path(name).suffix.lower())
            if language:
                counts[language] += 1
    return [language for language, _ in counts.most_common(limit)]


def detect_package_managers(root: str | Path) -> list[str]:
    """Package managers evidenced by a lock or manifest file directly in ``root``."""
    base = Path(root)
    return list(dict.fromkeys(name for marker, name in PACKAGE_MANAGERS if (base / marker).is_file()))


def detect_test_command(root: str | Path, config: Any = None) -> str | None:
    """The engine's own ``run_tests`` detection as a printable argv, or None when no suite is found.

    A host-only sandbox config keeps detection pure: no sandbox backend is probed.
    """
    from ironclad.config import SandboxConfig
    from ironclad.tool_runtime.exec import Executor

    executor = Executor(root, config or SandboxConfig(backend="host"))
    try:
        argv = executor.detect_test_argv()
    except Exception:  # noqa: BLE001 - no detectable suite; the scaffold says so instead
        return None
    finally:
        executor.close()
    return " ".join(argv)


def starter_agents_md(root: str | Path, config: Any = None) -> str:
    """Starter AGENTS.md content, under ``MAX_AGENTS_LINES``, built only from repository facts."""
    languages = detect_languages(root)
    managers = detect_package_managers(root)
    test_command = detect_test_command(root, config)
    lines = [
        "# Project instructions",
        "",
        "Guidance for AI coding agents working in this repository. Edit this file to match the project.",
        "These instructions are prompt text only; they never change Nedster's tool permissions.",
        "",
    ]
    if languages:
        lines += ["## Languages", f"- {', '.join(languages)}", ""]
    if managers:
        lines += ["## Package managers", f"- {', '.join(managers)}", ""]
    lines += ["## Commands"]
    lines.append(f"- Test: `{test_command}`" if test_command else "- Test: (add the command that runs the test suite)")
    lines += ["- Lint / type-check: (add here)", "", "## Conventions"]
    lines += [
        "- Match the existing code style; keep changes small and focused.",
        "- After changing code, run the test command above and report what you ran.",
        "",
        "## Notes",
        "- Keep secrets and credentials out of the repository.",
    ]
    return "\n".join(lines[:MAX_AGENTS_LINES]) + "\n"


def write_agents_md(root: str | Path, content: str) -> Path:
    """Create ``AGENTS.md`` in ``root``; refuse to overwrite an existing file (``FileExistsError``)."""
    path = Path(root) / AGENTS_FILENAME
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return path


def line_count(content: str) -> int:
    return len(content.splitlines())
