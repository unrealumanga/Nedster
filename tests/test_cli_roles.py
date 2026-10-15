"""Tests for --role and --brief flags on the run subcommand (no Ollama needed)."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from nedster_agent.cli import build_parser, main


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("NEDSTER_") or name == "OLLAMA_HOST":
            monkeypatch.delenv(name)
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("NEDSTER_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


# --- --role validation (exit 2 for invalid role) ----------------------------


def test_run_role_invalid_exits_2(clean_env, capsys):
    # argparse exits with status 2 for invalid choices — we assert via SystemExit
    fake_stdin = io.StringIO("hello\n")
    with patch.object(__import__("sys"), "stdin", fake_stdin), pytest.raises(SystemExit) as exc:
        main(["run", "--project", str(clean_env), "--role", "invalid"])
    assert exc.value.code == 2


def test_run_role_valid_choices_work(clean_env):
    parser = build_parser()
    for role in ("code", "plan", "review", "explain"):
        args = parser.parse_args(["run", "--project", ".", "--role", role])
        assert args.role == role
    # default is None
    args2 = parser.parse_args(["run", "--project", "."])
    assert args2.role is None


# --- --brief validation (exit 2 for bad brief) ------------------------------


def test_run_brief_missing_file_exits_2(clean_env, capsys):
    with patch("sys.stdin", io.StringIO("hello\n")):
        code = main(["run", "--project", str(clean_env), "--brief", "/nonexistent/path"])
    assert code == 2
    err = capsys.readouterr().err
    assert "not found" in err or "Error" in err


def test_run_brief_empty_file_exits_2(clean_env, capsys):
    # Create an empty file
    tmpfile = clean_env / "brief.txt"
    tmpfile.write_text("", encoding="utf-8")
    with patch("sys.stdin", io.StringIO("hello\n")):
        code = main(["run", "--project", str(clean_env), "--brief", str(tmpfile)])
    assert code == 2


def test_run_brief_whitespace_only_exits_2(clean_env, capsys):
    tmpfile = clean_env / "brief.txt"
    tmpfile.write_text("   \n\n  ", encoding="utf-8")
    with patch("sys.stdin", io.StringIO("hello\n")):
        code = main(["run", "--project", str(clean_env), "--brief", str(tmpfile)])
    assert code == 2


def test_run_brief_too_large_exits_2(clean_env, capsys):
    tmpfile = clean_env / "brief.txt"
    # Write a file larger than 32768 bytes
    tmpfile.write_bytes(b"x" * 32769)
    with patch("sys.stdin", io.StringIO("hello\n")):
        code = main(["run", "--project", str(clean_env), "--brief", str(tmpfile)])
    assert code == 2
    err = capsys.readouterr().err
    assert "too large" in err


def test_run_brief_together_with_objective_exits_2(clean_env, capsys):
    tmpfile = clean_env / "brief.txt"
    tmpfile.write_text("objective from file", encoding="utf-8")
    with patch("sys.stdin", io.StringIO("hello\n")):
        code = main(["run", "--project", str(clean_env), "--brief", str(tmpfile), "some objective"])
    assert code == 2
    err = capsys.readouterr().err
    assert "either an objective or --brief" in err


def test_run_brief_with_valid_content_used_as_objective(clean_env, capsys):
    """Brief file content is used as objective when valid (no model actually called)."""
    tmpfile = clean_env / "brief.txt"
    tmpfile.write_text("from brief file", encoding="utf-8")

    class FakeSession:
        def __init__(self, config, *, emit=None, model=None, session_id=None, resume_latest=False, title="", role=None):
            self.session_id = "abc123"
        async def run(self, text, *, intent=None):
            from ironclad.contracts import AcceptanceResult, Outcome
            return AcceptanceResult(outcome=Outcome.ANSWERED, answer=text)
        async def aclose(self):
            pass
        def note_brief(self, path: str, sha: str) -> None:
            pass

    with patch("ironclad.native.NativeSession", FakeSession), \
         patch("sys.stdin.isatty", return_value=True):
        code = main(["run", "--project", str(clean_env), "--brief", str(tmpfile)])
    assert code == 0
