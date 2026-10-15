"""Tests for nedster config get / path subcommands."""

import json
import os

import pytest

from nedster_agent.cli import main


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("NEDSTER_") or name == "OLLAMA_HOST":
            monkeypatch.delenv(name)
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("NEDSTER_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def test_config_get_models_coder(clean_env, capsys):
    result = main(["config", "get", "models.coder", "--project", str(clean_env)])
    assert result == 0
    out = capsys.readouterr().out.strip()
    assert out  # non-empty


def test_config_get_json_has_key_value_source(clean_env, capsys):
    result = main(["config", "get", "models.coder", "--json", "--project", str(clean_env)])
    assert result == 0
    doc = json.loads(capsys.readouterr().out)
    assert "key" in doc
    assert "value" in doc
    assert "source" in doc


def test_config_get_unknown_key_exits_2(clean_env, capsys):
    result = main(["config", "get", "models.nonexistent", "--project", str(clean_env)])
    assert result == 2
    err = capsys.readouterr().err
    assert "Unknown config key" in err


def test_config_get_no_key_exits_2(capsys):
    result = main(["config", "get"])
    assert result == 2
    err = capsys.readouterr().err
    assert "Usage: nedster config get" in err


def test_config_path_prints_lines(clean_env, capsys):
    result = main(["config", "path", "--project", str(clean_env)])
    assert result == 0
    out = capsys.readouterr().out
    assert "user config:" in out
    assert "project config:" in out
    assert "data dir:" in out
    # neither should exist in tmp_env
    assert "(not created yet)" in out


def test_config_path_json_has_keys(clean_env, capsys):
    result = main(["config", "path", "--json", "--project", str(clean_env)])
    assert result == 0
    doc = json.loads(capsys.readouterr().out)
    assert "user_config" in doc
    assert "project_config" in doc
    assert "data_dir" in doc


def test_config_alone_still_shows_full_listing(clean_env, capsys):
    result = main(["config", "show", "--project", str(clean_env)])
    assert result == 0
    out = capsys.readouterr().out
    assert "engine" in out
