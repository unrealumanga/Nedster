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


# --- setup execution consent (SETUP-CONSENT) --------------------------------------------------

import tomllib  # noqa: E402

ISOLATED = {"ready": True, "backend": "docker", "isolation": "full", "detail": "docker (full isolation)"}
PROCESS_ONLY = {"ready": True, "backend": "winjob", "isolation": "process-only", "detail": "winjob"}
NO_BACKEND = {"ready": False, "backend": "auto", "isolation": "unknown", "detail": "no backend"}


class _SetupManager:
    url = "http://fake"

    def __init__(self, *_a, **_k): ...
    def install(self, **_): ...
    def ensure_service(self, **_): ...
    def list_models(self): return []
    def pull(self, *_a, **_k): ...
    def create_alias(self, *_a, **_k): ...
    def copy_model(self, *_a): ...
    def remove_model(self, *_a): ...

    def verify_capabilities(self, router="", embedding=None):
        result = {"generation": True, "structured_output": True, "tool_calling": True}
        if embedding:
            result["embeddings"] = True
        return result


def _run_setup(clean_env, monkeypatch, capsys, probe, flags=(), answers=(), tty=False, prior=None):
    cfg = clean_env / "cfg"
    (cfg / "config.toml").unlink(missing_ok=True)  # each call starts from the given prior state
    if prior is not None:
        cfg.mkdir(parents=True, exist_ok=True)
        (cfg / "config.toml").write_text(prior, encoding="utf-8")
    asked: list[str] = []
    replies = iter(answers)

    def fake_input(prompt):
        asked.append(prompt)
        return next(replies)

    monkeypatch.setattr("nedster_agent.cli._read_line", fake_input)
    monkeypatch.setattr("nedster_agent.model_manager.probe_execution", lambda _c: probe)
    monkeypatch.setattr("nedster_agent.model_manager.OllamaManager", _SetupManager)
    monkeypatch.setattr("sys.stdin.isatty", lambda: tty)
    argv = ["setup", "--project", str(clean_env), *flags]
    if tty:
        monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *_a, **_k: "balanced")
        monkeypatch.setattr("rich.prompt.Confirm.ask", lambda *_a, **_k: True)
    else:
        argv.append("--plain")
    code = main(argv)
    out = capsys.readouterr()
    path = cfg / "config.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return code, data, asked, out


def test_yes_enables_execute_only_with_isolated_backend(clean_env, monkeypatch, capsys):
    code, data, asked, out = _run_setup(clean_env, monkeypatch, capsys, ISOLATED, ["--yes"])
    assert code == 0 and data["policy"]["max_auto_risk"] == "execute"
    assert data["sandbox"]["allow_host_execution"] is False and not asked
    assert "isolated sandbox" in out.out


def test_yes_without_isolated_backend_stays_write(clean_env, monkeypatch, capsys):
    for probe in (NO_BACKEND, PROCESS_ONLY):
        code, data, _, out = _run_setup(clean_env, monkeypatch, capsys, probe, ["--yes"])
        assert code == 0 and data["policy"]["max_auto_risk"] == "write"
        assert data["sandbox"]["allow_host_execution"] is False
        assert "read and write" in out.out


def test_allow_execution_needs_isolated_backend(clean_env, monkeypatch, capsys):
    _, data, _, _ = _run_setup(clean_env, monkeypatch, capsys, ISOLATED, ["--allow-execution"])
    assert data["policy"]["max_auto_risk"] == "execute"
    _, data, _, out = _run_setup(clean_env, monkeypatch, capsys, NO_BACKEND, ["--allow-execution"])
    assert data["policy"]["max_auto_risk"] == "write" and "NOT enabled" in out.err


def test_allow_host_execution_implies_execute_and_warns(clean_env, monkeypatch, capsys):
    _, data, _, out = _run_setup(clean_env, monkeypatch, capsys, NO_BACKEND, ["--allow-host-execution"])
    assert data["policy"]["max_auto_risk"] == "execute" and data["sandbox"]["allow_host_execution"] is True
    assert "WARNING" in out.err and "NO isolation" in out.out


def test_no_flags_non_interactive_keeps_previous_answers(clean_env, monkeypatch, capsys):
    prior = '[policy]\nmax_auto_risk = "execute"\n[sandbox]\nallow_host_execution = true\n'
    _, data, asked, _ = _run_setup(clean_env, monkeypatch, capsys, ISOLATED, [], prior=prior)
    assert data["policy"]["max_auto_risk"] == "execute" and data["sandbox"]["allow_host_execution"] is True
    assert not asked


def test_interactive_isolated_default_yes(clean_env, monkeypatch, capsys):
    _, data, asked, _ = _run_setup(clean_env, monkeypatch, capsys, ISOLATED, answers=[""], tty=True)
    assert len(asked) == 1 and "inside docker (full isolation)" in asked[0] and "[Y/n]" in asked[0]
    assert data["policy"]["max_auto_risk"] == "execute" and data["sandbox"]["allow_host_execution"] is False


def test_interactive_isolated_decline_asks_no_host_question(clean_env, monkeypatch, capsys):
    _, data, asked, _ = _run_setup(clean_env, monkeypatch, capsys, ISOLATED, answers=["n"], tty=True)
    assert len(asked) == 1
    assert data["policy"]["max_auto_risk"] == "write" and data["sandbox"]["allow_host_execution"] is False


def test_interactive_no_backend_host_default_no(clean_env, monkeypatch, capsys):
    _, data, asked, _ = _run_setup(clean_env, monkeypatch, capsys, NO_BACKEND, answers=[""], tty=True)
    assert len(asked) == 1 and "[y/N]" in asked[0] and "no isolation" in asked[0]
    assert data["policy"]["max_auto_risk"] == "write" and data["sandbox"]["allow_host_execution"] is False


def test_interactive_no_backend_host_yes(clean_env, monkeypatch, capsys):
    _, data, _, _ = _run_setup(clean_env, monkeypatch, capsys, NO_BACKEND, answers=["y"], tty=True)
    assert data["policy"]["max_auto_risk"] == "execute" and data["sandbox"]["allow_host_execution"] is True


def test_interactive_process_only_default_no_then_host_question(clean_env, monkeypatch, capsys):
    _, data, asked, _ = _run_setup(clean_env, monkeypatch, capsys, PROCESS_ONLY, answers=["", "y"], tty=True)
    assert "[y/N]" in asked[0] and len(asked) == 2
    assert data["policy"]["max_auto_risk"] == "execute" and data["sandbox"]["allow_host_execution"] is True


def test_interactive_decline_overrides_previous_execute(clean_env, monkeypatch, capsys):
    prior = '[policy]\nmax_auto_risk = "execute"\n'
    _, data, _, _ = _run_setup(clean_env, monkeypatch, capsys, NO_BACKEND, answers=["n"], tty=True, prior=prior)
    assert data["policy"]["max_auto_risk"] == "write"
