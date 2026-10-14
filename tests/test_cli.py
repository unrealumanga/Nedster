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


# --- output contracts / lazy imports (package F) ---------------------------------------------

import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("NEDSTER_") or name == "OLLAMA_HOST":
            monkeypatch.delenv(name)
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("NEDSTER_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def test_config_show_json_has_provenance(clean_env, capsys):
    assert main(["config", "show", "--json", "--project", str(clean_env)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["provenance"]["runtime.state_dir"] == "env"
    assert doc["config"]["engine"] == "legacy"


def test_config_show_human_lists_sources(clean_env, capsys):
    assert main(["config", "show", "--project", str(clean_env)]) == 0
    assert "engine = legacy  [default]" in capsys.readouterr().out


def test_missing_explicit_config_exits_2(clean_env, capsys):
    assert main(["--config", str(clean_env / "nope.toml"), "config", "show"]) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "nope.toml" in err


def test_verify_ledger_json_is_valid_json(clean_env, capsys):
    assert main(["verify-ledger", "--json", "--project", str(clean_env)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_run_without_prompt_and_tty_stdin_is_usage_error(clean_env, capsys):
    with patch("sys.stdin.isatty", return_value=True):
        assert main(["run", "--project", str(clean_env)]) == 2
    assert "No objective" in capsys.readouterr().err


class _FakeRuntime:
    def __init__(self, config):
        self.config = config

    async def initialize(self): ...

    async def respond(self, objective):
        return {"answer": f"echo:{objective}", "passed": True, "mode": "ask", "completed_steps": 1, "total_steps": 1}

    def close(self): ...


@pytest.mark.parametrize("flag", [None, "--json", "--jsonl"])
def test_run_output_contracts_and_piped_prompt(clean_env, capsys, flag):
    argv = ["run", "--project", str(clean_env)] + ([flag] if flag else [])
    with patch("ironclad.orchestrator.NedsterRuntime", _FakeRuntime), patch("sys.stdin", io.StringIO("hello\n")):
        assert main(argv) == 0
    captured = capsys.readouterr()
    if flag is None:
        assert captured.out == "echo:hello\n"
        assert "passed" in captured.err
    elif flag == "--json":
        assert json.loads(captured.out)["answer"] == "echo:hello"
    else:
        events = [json.loads(line) for line in captured.out.splitlines()]
        assert [e["type"] for e in events] == ["start", "result", "end"]


def test_doctor_exit_3_when_inference_not_ready(clean_env, capsys):
    tiers = {
        "inference": {"ready": False, "remedy": "Run: nedster setup"},
        "read_only": {"ready": False, "remedy": "Run: nedster setup"},
        "execution": {"ready": True, "detail": "x"},
    }
    with patch("nedster_agent.model_manager.readiness", return_value=tiers):
        assert main(["doctor", "--json", "--project", str(clean_env)]) == 3
    assert json.loads(capsys.readouterr().out)["tiers"]["inference"]["ready"] is False


def test_models_list_uses_resolved_endpoint(clean_env, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://example:9")
    seen = []

    class FakeManager:
        def __init__(self, url, *_):
            seen.append(url)
            self.url = url

        def api_available(self):
            return False

    with patch("nedster_agent.model_manager.OllamaManager", FakeManager):
        assert main(["models", "--project", str(clean_env)]) == 3
    assert seen == ["http://example:9"]


def test_cli_module_import_is_lazy():
    # nedster_agent/__init__ eagerly imports the orchestrator (not owned by this package), so stub the
    # package to test cli's own imports.
    code = (
        "import sys, types\n"
        "pkg = types.ModuleType('nedster_agent'); pkg.__path__ = ['nedster_agent']; pkg.__version__ = 'x'\n"
        "sys.modules['nedster_agent'] = pkg\n"
        "import nedster_agent.cli as cli\n"
        "cli.build_parser()\n"
        "print(','.join(m for m in ('textual', 'httpx', 'rich', 'ironclad.orchestrator') if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True, text=True, check=True, cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert out.stdout.strip() == ""
