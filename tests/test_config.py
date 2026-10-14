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
    assert config.engine == "legacy"
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


def test_load_config_compat_and_project_root(isolated):
    config = load_config(project_root=isolated / "proj")
    assert isinstance(config, Config)
    assert config.runtime.project_root == (isolated / "proj").resolve()
