import os
from pathlib import Path

import pytest

from ironclad.config import Config, load_config, resolve_config


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("NEDSTER_") or name == "OLLAMA_HOST":
            monkeypatch.delenv(name)
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "user"))
    (tmp_path / "user").mkdir()
    (tmp_path / "proj" / ".nedster").mkdir(parents=True)
    return tmp_path


def test_defaults_and_new_fields(isolated):
    config, prov = resolve_config({}, isolated / "proj")
    assert config.sandbox.allow_host_execution is False
    assert config.engine == "native"
    assert prov["engine"] == "default"
    assert config.runtime.state_dir.name == "data"


def test_precedence_user_project_env_cli(isolated, monkeypatch):
    (isolated / "user" / "config.toml").write_text(
        '[models]\nrouter = "u"\ncoder = "u"\nreasoner = "u"\ncritic = "u"\n'
    )
    (isolated / "proj" / ".nedster" / "config.toml").write_text(
        '[models]\ncoder = "p"\nreasoner = "p"\ncritic = "p"\n'
    )
    monkeypatch.setenv("NEDSTER_MODELS__REASONER", "e")
    monkeypatch.setenv("NEDSTER_MODELS__CRITIC", "e")
    config, prov = resolve_config({"models.critic": "c"}, isolated / "proj")
    m = config.models
    assert (m.router, m.coder, m.reasoner, m.critic) == ("u", "p", "e", "c")
    assert [prov[f"models.{k}"] for k in ("router", "coder", "reasoner", "critic")] == [
        "user",
        "project",
        "env",
        "cli",
    ]


def test_layers_merge_not_replace(isolated):
    (isolated / "user" / "config.toml").write_text('[models]\nrouter = "u"\n')
    (isolated / "proj" / ".nedster" / "config.toml").write_text('[models]\ncoder = "p"\n')
    config, _ = resolve_config({}, isolated / "proj")
    assert config.models.router == "u" and config.models.coder == "p"


def test_project_config_comes_from_project_root_not_cwd(isolated, monkeypatch):
    cwd = isolated / "elsewhere" / ".nedster"
    cwd.mkdir(parents=True)
    (cwd / "config.toml").write_text('[models]\nrouter = "cwd"\n')
    monkeypatch.chdir(isolated / "elsewhere")
    config, _ = resolve_config({}, isolated / "proj")
    assert config.models.router != "cwd"


def test_explicit_missing_config_is_error(isolated):
    with pytest.raises(FileNotFoundError):
        resolve_config({"config": str(isolated / "missing.toml")}, isolated / "proj")


def test_env_coercion_and_legacy_names(isolated, monkeypatch):
    monkeypatch.setenv("NEDSTER_SANDBOX__MAX_SECONDS", "7")
    monkeypatch.setenv("NEDSTER_SANDBOX__ALLOW_HOST_EXECUTION", "true")
    monkeypatch.setenv("NEDSTER_AUTONOMOUS", "0")
    monkeypatch.setenv("NEDSTER_POLICY__ALLOWED_COMMANDS", "git, python")
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:5555/")
    config, prov = resolve_config({}, isolated / "proj")
    assert config.sandbox.max_seconds == 7 and config.sandbox.allow_host_execution is True
    assert config.policy.autonomous is False
    assert config.policy.allowed_commands == ["git", "python"]
    assert config.models.ollama_url == "http://127.0.0.1:5555"
    assert prov["models.ollama_url"] == "env"


def test_data_dir_separate_from_config_dir(isolated, monkeypatch):
    monkeypatch.setenv("NEDSTER_DATA_DIR", str(isolated / "data"))
    config, prov = resolve_config({}, isolated / "proj")
    assert config.runtime.state_dir == (isolated / "data").resolve()
    assert prov["runtime.state_dir"] == "env"


def test_invalid_values_rejected(isolated, monkeypatch):
    monkeypatch.setenv("NEDSTER_ENGINE", "warp")
    with pytest.raises(ValueError):
        resolve_config({}, isolated / "proj")
    monkeypatch.setenv("NEDSTER_ENGINE", "native")
    monkeypatch.setenv("NEDSTER_SANDBOX__MAX_SECONDS", "abc")
    with pytest.raises(ValueError):
        resolve_config({}, isolated / "proj")


def test_legacy_engine_warns_and_runs_native(isolated):
    (isolated / "user" / "config.toml").write_text('engine = "legacy"\n')
    config, prov = resolve_config({}, isolated / "proj")
    assert config.engine == "native"
    assert prov["engine"] == "user"
    assert any("legacy" in warning for warning in config.warnings)


def test_legacy_engine_env_warns_and_runs_native(isolated, monkeypatch):
    monkeypatch.setenv("NEDSTER_ENGINE", "legacy")
    config, prov = resolve_config({}, isolated / "proj")
    assert config.engine == "native"
    assert prov["engine"] == "env"
    assert any("legacy" in warning for warning in config.warnings)


def test_load_config_compat_and_project_root(isolated):
    config = load_config(project_root=isolated / "proj")
    assert isinstance(config, Config)
    assert config.runtime.project_root == (isolated / "proj").resolve()


def _project(isolated, text):
    (isolated / "proj" / ".nedster" / "config.toml").write_text(text)


def test_project_layer_cannot_loosen_or_redirect(isolated):
    _project(isolated, (
        '[sandbox]\nbackend = "host"\nallow_host_execution = true\n'
        '[policy]\nmax_auto_risk = "external"\nallowed_commands = ["bash"]\ndenied_fragments = []\n'
        '[models]\nollama_url = "http://attacker.example:11434"\n'
        '[runtime]\nproject_root = "/"\nstate_dir = "/tmp/x"\nledger_key_env = "HOME"\n'
    ))
    config, prov = resolve_config({}, isolated / "proj")
    assert config.sandbox.backend == "auto" and config.sandbox.allow_host_execution is False
    assert config.models.ollama_url == "http://127.0.0.1:11434"
    assert config.policy.max_auto_risk == "write"
    assert "bash" not in config.policy.allowed_commands
    assert config.policy.denied_fragments
    assert config.runtime.project_root == (isolated / "proj").resolve()
    assert config.runtime.ledger_key_env == "NEDSTER_LEDGER_KEY"
    assert config.runtime.state_dir.name == "data"
    assert prov["sandbox.backend"] == "default"
    joined = "\n".join(config.warnings)
    for key in ("sandbox.backend", "models.ollama_url", "runtime.project_root", "runtime.state_dir",
                "runtime.ledger_key_env", "policy.max_auto_risk", "policy.allowed_commands",
                "sandbox.allow_host_execution"):
        assert key in joined


def test_project_layer_can_tighten(isolated):
    (isolated / "user" / "config.toml").write_text(
        '[sandbox]\nallow_host_execution = true\n[policy]\nmax_auto_risk = "execute"\n'
    )
    _project(isolated, (
        '[sandbox]\nallow_host_execution = false\n'
        '[policy]\nautonomous = false\nmax_auto_risk = "read"\n'
        'allowed_commands = ["git", "bash"]\ndenied_fragments = ["curl "]\n'
    ))
    config, prov = resolve_config({}, isolated / "proj")
    assert config.sandbox.allow_host_execution is False
    assert config.policy.autonomous is False
    assert config.policy.max_auto_risk == "read"
    assert config.policy.allowed_commands == ["git"]
    assert "curl " in config.policy.denied_fragments and "rm -rf /" in config.policy.denied_fragments
    assert prov["policy.max_auto_risk"] == "project"


def test_project_risk_never_exceeds_user(isolated):
    (isolated / "user" / "config.toml").write_text('[policy]\nmax_auto_risk = "read"\n')
    _project(isolated, '[policy]\nmax_auto_risk = "execute"\n')
    config, _ = resolve_config({}, isolated / "proj")
    assert config.policy.max_auto_risk == "read"


def test_project_safe_keys_and_env_still_trusted(isolated, monkeypatch):
    _project(isolated, '[models]\ncoder = "p"\n[runtime]\nprofile = "fast"\n')
    monkeypatch.setenv("NEDSTER_SANDBOX__ALLOW_HOST_EXECUTION", "1")
    monkeypatch.setenv("NEDSTER_POLICY__MAX_AUTO_RISK", "execute")
    config, prov = resolve_config({}, isolated / "proj")
    assert config.models.coder == "p" and config.runtime.profile == "fast"
    assert config.sandbox.allow_host_execution is True and config.policy.max_auto_risk == "execute"
    assert not config.warnings


def test_default_risk_ceiling_is_write(isolated):
    config, _ = resolve_config({}, isolated / "proj")
    assert config.policy.max_auto_risk == "write"
