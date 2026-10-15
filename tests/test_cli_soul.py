from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nedster_agent.cli import main


@pytest.fixture
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in list(os.environ):
        if name.startswith("NEDSTER_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("NEDSTER_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def test_soul_path_json_points_at_the_config_dir(clean_env: Path, capsys):
    assert main(["soul", "path", "--json", "--project", str(clean_env)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["path"] == str(clean_env / "cfg" / "soul.md")


def test_soul_show_reports_mode_off_without_a_file(clean_env: Path, capsys):
    assert main(["soul", "show", "--json", "--project", str(clean_env)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["mode"] == "off" and doc["exists"] is False and doc["size"] == 0


def test_soul_remember_yes_writes_a_bullet_applied_next_session(clean_env: Path, capsys):
    assert main(["soul", "remember", "Use metric units", "--yes", "--json", "--project", str(clean_env)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["added"] == "- Use metric units" and doc["applies"] == "next session"
    text = (clean_env / "cfg" / "soul.md").read_text(encoding="utf-8")
    assert "- Use metric units" in text


def test_soul_remember_dedupes(clean_env: Path, capsys):
    argv = ["soul", "remember", "Be blunt", "--yes", "--project", str(clean_env)]
    assert main(argv) == 0
    capsys.readouterr()
    assert main(argv) == 2
    assert "already in soul.md" in capsys.readouterr().err


def test_soul_remember_refuses_without_confirmation_on_non_tty(clean_env: Path, capsys):
    with patch("sys.stdin.isatty", return_value=False):
        assert main(["soul", "remember", "Be blunt", "--project", str(clean_env)]) == 2
    assert "pass --yes" in capsys.readouterr().err
    assert not (clean_env / "cfg" / "soul.md").exists()


def test_soul_remember_prompts_and_respects_no(clean_env: Path, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    with patch("sys.stdin.isatty", return_value=True):
        assert main(["soul", "remember", "Be blunt", "--project", str(clean_env)]) == 0
    assert "Cancelled" in capsys.readouterr().out
    assert not (clean_env / "cfg" / "soul.md").exists()


def test_soul_remember_keeps_at_most_five_bullets(clean_env: Path, capsys):
    for index in range(5):
        assert main(["soul", "remember", f"pref {index}", "--yes", "--project", str(clean_env)]) == 0
    capsys.readouterr()
    assert main(["soul", "remember", "pref late", "--yes", "--json", "--project", str(clean_env)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["dropped"] == "- pref 0"
    text = (clean_env / "cfg" / "soul.md").read_text(encoding="utf-8")
    assert "- pref 0" not in text and "- pref late" in text
