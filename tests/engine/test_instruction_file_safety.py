"""Final-review B4/S1: project instructions never follow links out of the project; capped reads are bounded."""

from __future__ import annotations

import hashlib
import os
import tracemalloc
from pathlib import Path

import pytest

from ironclad.engine.prompt import (
    MAX_INSTRUCTIONS_BYTES,
    MAX_SOUL_BYTES,
    load_project_instructions,
    load_soul,
    snapshot_digest,
)

MARKER = "PRIVATE-OUTSIDE-MARKER"


def _symlink(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:  # Windows without developer mode / privilege
        pytest.skip(f"cannot create symlinks here: {exc}")


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text(MARKER, encoding="utf-8")
    return project, outside


def test_symlinked_agents_md_pointing_outside_is_skipped_with_a_warning(tmp_path: Path) -> None:
    project, outside = _project(tmp_path)
    _symlink(project / "AGENTS.md", outside)
    loaded = load_project_instructions(project)
    assert loaded is not None
    assert not loaded.present
    assert MARKER not in loaded.text
    assert "AGENTS.md" in loaded.warning and "link" in loaded.warning


def test_symlinked_nedster_md_is_skipped_too(tmp_path: Path) -> None:
    project, outside = _project(tmp_path)
    _symlink(project / "NEDSTER.md", outside)
    loaded = load_project_instructions(project)
    assert loaded is not None and not loaded.present
    assert "NEDSTER.md" in loaded.warning


def test_symlink_even_inside_the_project_is_refused(tmp_path: Path) -> None:
    project, _ = _project(tmp_path)
    (project / "real.md").write_text("inside", encoding="utf-8")
    _symlink(project / "AGENTS.md", project / "real.md")
    loaded = load_project_instructions(project)
    assert loaded is not None and not loaded.present and loaded.warning


def test_rejected_agents_md_falls_back_to_a_regular_nedster_md_and_keeps_the_notice(tmp_path: Path) -> None:
    project, outside = _project(tmp_path)
    _symlink(project / "AGENTS.md", outside)
    (project / "NEDSTER.md").write_text("use tabs", encoding="utf-8")
    loaded = load_project_instructions(project)
    assert loaded is not None
    assert loaded.name == "NEDSTER.md" and loaded.text == "use tabs"
    assert "AGENTS.md" in loaded.warning
    assert MARKER not in loaded.text


def test_rejected_file_never_aborts_session_startup(tmp_path: Path) -> None:
    from ironclad.engine.prompt import session_instructions

    project, outside = _project(tmp_path)
    _symlink(project / "AGENTS.md", outside)
    snapshot = session_instructions(None, "s1", project)
    assert not snapshot.present and snapshot.warning


def test_regular_file_is_still_loaded_without_a_warning(tmp_path: Path) -> None:
    project, _ = _project(tmp_path)
    (project / "AGENTS.md").write_bytes(b"be terse")
    loaded = load_project_instructions(project)
    assert loaded is not None and loaded.text == "be terse" and loaded.warning == ""
    assert loaded.sha256 == hashlib.sha256(b"be terse").hexdigest()  # untruncated: plain sha256


# --- S1: bounded reads ---------------------------------------------------------------------------


def _big(path: Path, size: int) -> None:
    with path.open("wb") as handle:
        handle.truncate(size)  # sparse where supported; never allocated in this process


def _peak(fn) -> int:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_large_instruction_file_is_read_with_bounded_memory(tmp_path: Path) -> None:
    size = 16 * 1024 * 1024
    _big(tmp_path / "AGENTS.md", size)
    result: dict = {}
    peak = _peak(lambda: result.update(loaded=load_project_instructions(tmp_path)))
    loaded = result["loaded"]
    assert peak < 1024 * 1024
    assert loaded.truncated and loaded.text.endswith("[truncated]")
    assert loaded.size == size
    assert loaded.sha256 == snapshot_digest(b"\0" * MAX_INSTRUCTIONS_BYTES, size)


def test_large_soul_file_is_read_with_bounded_memory(tmp_path: Path) -> None:
    size = 16 * 1024 * 1024
    soul = tmp_path / "soul.md"
    _big(soul, size)
    result: dict = {}
    peak = _peak(lambda: result.update(soul=load_soul(soul)))
    loaded = result["soul"]
    assert peak < 1024 * 1024
    assert loaded.truncated and loaded.text.endswith("[truncated]")
    assert loaded.size == size
    assert loaded.sha256 == snapshot_digest(b"\0" * MAX_SOUL_BYTES, size)


def test_snapshot_digest_mixes_in_the_source_size_only_when_truncated() -> None:
    assert snapshot_digest(b"abc", 3) == hashlib.sha256(b"abc").hexdigest()
    assert snapshot_digest(b"abc", 10) != snapshot_digest(b"abc", 11)
