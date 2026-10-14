from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ironclad.contracts import ToolCall, ToolStatus
from ironclad.tool_runtime.search import SearchTools

needs_rg = pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")


def call(name: str = "search_text", call_id: str = "s1", **arguments) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "a.txt").write_text("alpha\na(b here\nomega\n", encoding="utf-8", newline="\n")
    (project / "sub").mkdir()
    (project / "sub" / "b.py").write_text("def a(b):\n    return 'needle'\n", encoding="utf-8", newline="\n")
    (project / "blob.bin").write_bytes(b"needle\x00\x01")
    return project


@pytest.fixture
def tools(root: Path) -> SearchTools:
    return SearchTools(root)


def force_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ironclad.tool_runtime.search.shutil.which", lambda program: None)


def test_specs(tools: SearchTools):
    specs = tools.specs()
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "search_text" and not spec.mutates and spec.risk == 0
    assert spec.parameters["required"] == ["pattern"]


def test_literal_parenthesis_search_python(tools: SearchTools, root: Path, monkeypatch: pytest.MonkeyPatch):
    force_python(monkeypatch)
    result = tools.run(call(pattern="a(b"))
    assert result.status is ToolStatus.OK, result.output
    assert result.data["backend"] == "python" and result.data["count"] == 2
    assert "a.txt:2: a(b here" in result.output
    assert "sub/b.py:1: def a(b):" in result.output
    assert "blob.bin" not in result.output


@needs_rg
def test_literal_parenthesis_search_ripgrep(tools: SearchTools):
    result = tools.run(call(pattern="a(b"))
    assert result.status is ToolStatus.OK, result.output
    assert result.data["backend"] == "ripgrep" and result.data["count"] == 2
    assert "a.txt:2: a(b here" in result.output
    assert "sub/b.py:1: def a(b):" in result.output


def test_regex_mode_and_invalid_regex(tools: SearchTools, root: Path, monkeypatch: pytest.MonkeyPatch):
    force_python(monkeypatch)
    result = tools.run(call(pattern=r"need\w+", regex=True))
    assert result.status is ToolStatus.OK
    assert "sub/b.py:2:     return 'needle'" in result.output
    assert "a.txt" not in result.output
    bad = tools.run(call(pattern=r"a(", regex=True))
    assert bad.status is ToolStatus.INVALID and "invalid regular expression" in bad.output


@needs_rg
def test_regex_mode_ripgrep(tools: SearchTools):
    result = tools.run(call(pattern=r"need\w+", regex=True))
    assert result.status is ToolStatus.OK
    assert "needle" in result.output and result.data["backend"] == "ripgrep"


def test_path_argument_and_escape(tools: SearchTools, root: Path, monkeypatch: pytest.MonkeyPatch):
    force_python(monkeypatch)
    only_sub = tools.run(call(pattern="a(b", path="sub"))
    assert only_sub.status is ToolStatus.OK and "a.txt" not in only_sub.output and "sub/b.py" in only_sub.output
    single_file = tools.run(call(pattern="a(b", path="a.txt"))
    assert single_file.status is ToolStatus.OK and single_file.data["count"] == 1
    escape = tools.run(call(pattern="a(b", path=".."))
    assert escape.status is ToolStatus.INVALID
    missing = tools.run(call(pattern="a(b", path="void"))
    assert missing.status is ToolStatus.INVALID


def test_max_results_cap(tools: SearchTools, root: Path, monkeypatch: pytest.MonkeyPatch):
    force_python(monkeypatch)
    (root / "many.txt").write_text("\n".join(f"hit {index}" for index in range(1, 21)), encoding="utf-8", newline="\n")
    result = tools.run(call(pattern="hit", max_results=5))
    assert result.status is ToolStatus.OK
    assert result.data["count"] == 5 and result.data["truncated"]
    assert "[truncated: showing first 5 matches]" in result.output


def test_context_lines(tools: SearchTools, root: Path, monkeypatch: pytest.MonkeyPatch):
    force_python(monkeypatch)
    (root / "ctx.txt").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8", newline="\n")
    result = tools.run(call(pattern="three", context=1))
    assert result.status is ToolStatus.OK
    assert "ctx.txt:3: three" in result.output
    assert "ctx.txt:2- two" in result.output and "ctx.txt:4- four" in result.output
    assert "ctx.txt:1" not in result.output


@needs_rg
def test_context_lines_ripgrep(tools: SearchTools):
    (Path(tools.project_root) / "ctx.txt").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8", newline="\n")
    result = tools.run(call(pattern="three", context=1))
    assert result.status is ToolStatus.OK
    assert "ctx.txt:3: three" in result.output
    assert "ctx.txt:2- two" in result.output


def test_invalid_arguments(tools: SearchTools, monkeypatch: pytest.MonkeyPatch):
    force_python(monkeypatch)
    assert tools.run(call(name="grep_everything")).status is ToolStatus.INVALID
    assert tools.run(call()).status is ToolStatus.INVALID
    empty = tools.run(call(pattern=""))
    assert empty.status is ToolStatus.INVALID
    long_pattern = tools.run(call(pattern="x" * 501))
    assert long_pattern.status is ToolStatus.INVALID and "too long" in long_pattern.output
    bad_types = tools.run(call(pattern="x", context=-1))
    assert bad_types.status is ToolStatus.INVALID


def test_no_matches_is_ok(tools: SearchTools, monkeypatch: pytest.MonkeyPatch):
    force_python(monkeypatch)
    result = tools.run(call(pattern="zzz-absent"))
    assert result.status is ToolStatus.OK and result.data["count"] == 0 and result.output == ""


@needs_rg
def test_ripgrep_no_matches_is_ok(tools: SearchTools):
    result = tools.run(call(pattern="zzz-absent"))
    assert result.status is ToolStatus.OK and result.data["count"] == 0


def test_python_search_uses_injected_inventory(root: Path, monkeypatch: pytest.MonkeyPatch):
    force_python(monkeypatch)
    (root / "hidden.txt").write_text("needle\n", encoding="utf-8")
    (root / "shown.txt").write_text("needle\n", encoding="utf-8")
    seen: list[Path] = []

    def fake(base: Path) -> list[str]:
        seen.append(base)
        return ["shown.txt"]

    result = SearchTools(root, inventory=fake).run(call(pattern="needle"))
    assert result.status is ToolStatus.OK and seen
    assert "shown.txt:1" in result.output and "hidden.txt" not in result.output
