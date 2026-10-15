"""Tests for conflicting CLI flags — each conflict returns exit 2 with the message on stderr."""

import pytest

from nedster_agent.cli import build_parser, main

# --- _check_flags conflicts ---------------------------------------------------


def test_json_and_plain_conflict(capsys):
    code = main(["config", "show", "--json", "--plain"])
    assert code == 2
    err = capsys.readouterr().err
    assert "Choose one output mode: --plain, --json, or --jsonl." in err


def test_json_and_jsonl_conflict(capsys):
    code = main(["run", "--project", ".", "--json", "--jsonl"])
    assert code == 2
    err = capsys.readouterr().err
    assert "Choose one output mode: --plain, --json, or --jsonl." in err


def test_plain_and_jsonl_conflict(capsys):
    code = main(["run", "--project", ".", "--plain", "--jsonl"])
    assert code == 2
    err = capsys.readouterr().err
    assert "Choose one output mode: --plain, --json, or --jsonl." in err


def test_color_and_no_color_conflict(capsys):
    code = main(["config", "show", "--color", "--no-color"])
    assert code == 2
    err = capsys.readouterr().err
    assert "Choose either --color or --no-color, not both." in err


def test_session_and_continue_conflict(capsys):
    with pytest.raises(SystemExit) as exc:  # argparse mutually exclusive group
        main(["run", "--project", ".", "--session", "abc", "--continue"])
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


# --- _tui_or_help when TUI cannot open (non-TTY) -----------------------------


def test_tui_without_interactive_terminal_returns_2(capsys):
    """When args.command == 'tui' and stdout is not a TTY, return 2 with the message."""
    code = main(["tui"])
    assert code == 2
    err = capsys.readouterr().err
    assert "The TUI needs an interactive terminal. For piped output use: nedster run \"...\"" in err


def test_bare_command_still_returns_0_when_stdout_not_tty(capsys):
    """When args.command is None (bare command), print help and return 0."""
    code = main([])
    assert code == 0
    out = capsys.readouterr().out
    assert "usage: nedster" in out


# --- help text ----------------------------------------------------------------


def test_run_help_contains_machine_readable(capsys):
    """`build_parser().parse_args(["run", "--help"])` must contain 'machine-readable'."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--help"])
    out = capsys.readouterr().out
    assert "machine-readable" in out
