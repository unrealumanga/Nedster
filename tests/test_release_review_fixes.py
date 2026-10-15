"""Final-review S5 (cross-kind session refs), S6 (evidence-neutral moments), S7 (one TTY admission rule)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from ironclad.session_store.sqlite_store import SessionAmbiguous, SqliteSessionStore
from nedster_agent import personality
from nedster_agent.cli import EXIT_OK, EXIT_USAGE, main


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("NEDSTER_") or name == "OLLAMA_HOST":
            monkeypatch.delenv(name)
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("NEDSTER_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def _store(root: Path) -> SqliteSessionStore:
    return SqliteSessionStore(root / "data" / "sessions.sqlite3")


# --- S5 ------------------------------------------------------------------------------------------


def test_title_equal_to_another_sessions_prefix_is_ambiguous_and_delete_refuses(clean_env, capsys):
    root = str(Path(clean_env).resolve())
    store = _store(clean_env)
    first = store.create(root, "first")
    second = store.create(root, first[:8])
    with pytest.raises(SessionAmbiguous) as info:
        store.resolve_session(root, first[:8])
    assert "matches 2 sessions" in str(info.value)
    assert store.resolve_session(root, first)["id"] == first  # full id stays authoritative
    store.close()

    code = main(["sessions", "delete", first[:8], "--yes", "--project", str(clean_env)])
    assert code == EXIT_USAGE
    assert "matches 2 sessions" in capsys.readouterr().err
    after = _store(clean_env)
    assert after.get_session(first) is not None and after.get_session(second) is not None
    after.close()


# --- S6 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcome", "moment"),
    [
        ("answered", "done_answered"),
        ("completed", "done_ok"),
        ("unvalidated", "done_unvalidated"),
        ("failed", "done_failed"),
        ("no_change_needed", "done_answered"),
    ],
)
def test_outcome_to_moment_mapping(outcome, moment):
    assert personality.moment_for_outcome(outcome) == moment


def test_only_completed_lines_mention_tests_or_checks():
    claims = ("test", "pass", "verified", "green", "check", "suite", "works", "citation", "source", "proof", "evidence")
    for outcome in ("answered", "no_change_needed", "unvalidated", "failed", "incomplete", "cancelled"):
        moment = personality.moment_for_outcome(outcome)
        for line in personality.lines(moment):
            lowered = line.lower().replace("not verified", "").replace("verified nothing", "")
            assert not any(word in lowered for word in claims), (outcome, line)
    assert "Answered. Read it with the usual suspicion." in personality.lines("done_answered")


# --- S7 ------------------------------------------------------------------------------------------


def _ttys(monkeypatch, *, stdin: bool, stdout: bool) -> list:
    import nedster_agent.tui as tui

    monkeypatch.setattr(sys.stdin, "isatty", lambda: stdin)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: stdout)
    calls: list = []
    monkeypatch.setattr(tui, "run_tui", lambda config, **kwargs: calls.append(kwargs))
    return calls


@pytest.mark.parametrize("command", [["tui"], ["resume"]])
@pytest.mark.parametrize(("stdin", "stdout"), [(False, True), (True, False), (False, False)])
def test_ui_entry_requires_both_streams_to_be_terminals(clean_env, monkeypatch, capsys, command, stdin, stdout):
    calls = _ttys(monkeypatch, stdin=stdin, stdout=stdout)
    assert main([*command, "--project", str(clean_env)]) == EXIT_USAGE
    assert calls == []
    assert "interactive terminal" in capsys.readouterr().err


@pytest.mark.parametrize("command", [["tui"], ["resume"]])
@pytest.mark.parametrize("flag", ["--json", "--plain", "--no-tui"])
def test_ui_entry_refuses_machine_and_no_tui_flags(clean_env, monkeypatch, capsys, command, flag):
    calls = _ttys(monkeypatch, stdin=True, stdout=True)
    assert main([*command, flag, "--project", str(clean_env)]) == EXIT_USAGE
    assert calls == []
    assert "interactive terminal" in capsys.readouterr().err


def test_bare_launch_without_stdin_tty_prints_help_and_never_opens_the_ui(clean_env, monkeypatch, capsys):
    calls = _ttys(monkeypatch, stdin=False, stdout=True)
    assert main(["--project", str(clean_env)]) == EXIT_OK
    assert calls == []
    assert "usage: nedster" in capsys.readouterr().out


def test_ui_entry_opens_when_both_streams_are_terminals(clean_env, monkeypatch):
    calls = _ttys(monkeypatch, stdin=True, stdout=True)
    assert main(["tui", "--project", str(clean_env)]) == EXIT_OK
    assert len(calls) == 1
