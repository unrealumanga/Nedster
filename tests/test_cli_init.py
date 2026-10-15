from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from nedster_agent.cli import main


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in list(os.environ):
        if name.startswith("NEDSTER_") or name == "OLLAMA_HOST":
            monkeypatch.delenv(name)
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("NEDSTER_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / "proj"
    (root / "tests").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "demo"\ndependencies = ["pytest"]\n', encoding="utf-8")
    (root / "tests" / "test_demo.py").write_text("def test_demo():\n    assert True\n", encoding="utf-8")
    return root


def test_init_writes_starter_and_refuses_to_overwrite(project: Path, capsys) -> None:
    assert main(["init", "--project", str(project)]) == 0
    assert "AGENTS.md" in capsys.readouterr().out
    path = project / "AGENTS.md"
    content = path.read_text(encoding="utf-8")
    assert len(content.splitlines()) < 40
    assert "Project instructions" in content
    # the interpreter name follows the platform (python on Windows, python3 on POSIX)
    assert re.search(r"^- Test: `\S*python[\d.]*(\.exe)? -m pytest -q`$", content, re.M), content

    assert main(["init", "--project", str(project)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert path.read_text(encoding="utf-8") == content  # untouched


def test_init_print_writes_nothing(project: Path, capsys) -> None:
    assert main(["init", "--print", "--project", str(project)]) == 0
    assert "# Project instructions" in capsys.readouterr().out
    assert not (project / "AGENTS.md").exists()


def test_init_json_reports_the_written_file(project: Path, capsys) -> None:
    assert main(["init", "--json", "--project", str(project)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["path"].endswith("AGENTS.md")
    assert 0 < doc["lines"] < 40


def test_config_show_lists_loaded_project_instructions(project: Path, capsys) -> None:
    (project / "AGENTS.md").write_text("Run the tests before finishing.", encoding="utf-8")
    assert main(["config", "show", "--project", str(project)]) == 0
    assert "project instructions: AGENTS.md" in capsys.readouterr().out

    assert main(["config", "show", "--json", "--project", str(project)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["project_instructions"]["name"] == "AGENTS.md"
