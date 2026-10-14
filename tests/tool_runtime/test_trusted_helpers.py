"""Exec trust gaps: rg is resolved like every other trusted program, and the docker
backend also protects every case spelling of ``.git``/``.nedster`` on case-folding hosts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ironclad.contracts import ToolCall, ToolStatus
from ironclad.tool_runtime.backends.docker import (
    DockerBackend,
    case_variants,
    mount_alias_relpaths,
)
from ironclad.tool_runtime.search import SearchTools, find_ripgrep

RG_NAMES = ("rg.exe", "rg") if os.name == "nt" else ("rg", "rg.exe")


def _plant(root: Path, name: str) -> Path:
    path = root / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o755)
    return path


@pytest.mark.parametrize("name", RG_NAMES)
def test_planted_rg_in_project_is_never_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str):
    """A relative ``./rg`` (or ``./rg.exe``) must never be selected, even via the project path."""
    planted = _plant(tmp_path, name)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_ripgrep(tmp_path) is None
    assert find_ripgrep(tmp_path) != planted


def test_path_entry_inside_project_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A PATH entry pointing into the project is dropped before lookup, so its rg is invisible."""
    _plant(tmp_path, RG_NAMES[0])
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path), str(outside)]))
    assert find_ripgrep(tmp_path) is None


@pytest.mark.parametrize("name", RG_NAMES)
def test_planted_rg_is_never_executed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str):
    """End to end: with only a project rg on PATH the search falls back to Python, running nothing."""
    _plant(tmp_path, name)
    (tmp_path / "a.txt").write_text("needle\n", encoding="utf-8", newline="\n")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(
        "ironclad.tool_runtime.search.subprocess.run",
        lambda *a, **k: pytest.fail("a project-local rg was executed"),
    )
    result = SearchTools(tmp_path).run(ToolCall(id="s1", name="search_text", arguments={"pattern": "needle"}))
    assert result.status is ToolStatus.OK and result.data["backend"] == "python"


def test_case_variants_cover_every_spelling():
    variants = case_variants(".git")
    assert variants[0] == ".git" and ".GIT" in variants and len(variants) == 8
    assert len(case_variants(".nedster")) == 128


def test_mount_alias_relpaths_only_expands_protected_names():
    assert len(mount_alias_relpaths(".git", True)) == 8
    assert mount_alias_relpaths("vendor/.git", True)[0] == "vendor/.git"
    assert mount_alias_relpaths("some-gitdir", True) == ("some-gitdir",)
    assert mount_alias_relpaths(".git", False) == (".git",)


def _mounts(argv: list[str]) -> list[str]:
    return [argv[index + 1] for index, item in enumerate(argv[:-1]) if item == "--mount"]


def _docker(root: Path, case_insensitive: bool) -> list[str]:
    backend = DockerBackend(root, "img:1")
    backend.binary = "docker"
    backend.case_insensitive = case_insensitive
    return backend.wrap(["pytest"], backend.root, {})


def test_docker_protects_case_aliases_on_case_folding_hosts(tmp_path: Path):
    """Writing /workspace/.GIT must hit the read-only mount, not the host's real .git."""
    mounts = _mounts(_docker(tmp_path, case_insensitive=True))
    assert any(m.endswith("dst=/workspace/.git,readonly") for m in mounts)
    assert any(m.endswith("dst=/workspace/.GIT,readonly") for m in mounts)
    assert any(m.endswith("dst=/workspace/.NEDSTER,readonly") for m in mounts)


def test_docker_skips_case_aliases_on_case_sensitive_hosts(tmp_path: Path):
    mounts = _mounts(_docker(tmp_path, case_insensitive=False))
    assert any(m.endswith("dst=/workspace/.git,readonly") for m in mounts)
    assert not any(".GIT" in m or ".NEDSTER" in m for m in mounts)


def test_docker_comma_path_is_refused_not_a_broken_mount(tmp_path: Path):
    """A comma-bearing project path is rejected before docker sees a malformed --mount."""
    root = tmp_path / "a,b"
    root.mkdir()
    backend = DockerBackend(root, "img:1")
    backend.binary = "docker"
    with pytest.raises(ValueError, match="--mount"):
        backend.wrap(["pytest"], backend.root, {})
