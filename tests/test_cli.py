from unittest.mock import patch

from nedster_agent.cli import build_parser, main


def test_machine_output_flags_work_after_subcommand():
    args = build_parser().parse_args(["models", "recommend", "--plain", "--ascii"])
    assert args.plain
    assert args.ascii


def test_bare_command_does_not_open_tui_when_stdout_is_not_a_terminal(capsys):
    with patch("sys.stdout.isatty", return_value=False), patch("sys.stdin.isatty", return_value=True):
        assert main([]) == 0
    assert "usage: nedster" in capsys.readouterr().out
