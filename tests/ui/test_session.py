from __future__ import annotations

import asyncio
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from ironclad.contracts import EngineEvent
from nedster_agent.tui import session_diff
from nedster_agent.widgets.composer import HISTORY_LIMIT, History, match_files
from nedster_agent.widgets.session import NativeSession, failure_reason


class FakeEngine:
    def __init__(self, emit) -> None:
        self.emit = emit
        self.model = "qwen3:8b"
        self.budget = SimpleNamespace(context_tokens=32768)
        self.cancelled = 0
        self.runs: list[tuple[str, str]] = []

    async def run(self, session_id: str, text: str) -> None:
        self.runs.append((session_id, text))
        self.emit(EngineEvent("model.usage", session_id, "r1", {"prompt_tokens": 900, "completion_tokens": 100}))

    def cancel(self) -> None:
        self.cancelled += 1


class FakeStore:
    def __init__(self) -> None:
        self.created = 0

    def create(self, project_root: str, title: str = "") -> str:
        self.created += 1
        return f"s{self.created}"

    def list_sessions(self, project_root: str, limit: int = 20):
        return [{"id": "abc123", "title": "latest", "updated_ns": 2}, {"id": "abd999", "title": "", "updated_ns": 1}]


def test_native_session_adapts_engine_events_info_and_sessions():
    seen: list = []
    store = FakeStore()
    session = NativeSession(FakeEngine, seen.append, store=store, project_root="/p",
                            sandbox="bwrap", isolation="full", endpoint="http://127.0.0.1:11434")
    asyncio.run(session.submit("hi"))
    assert session.engine.runs == [("s1", "hi")]
    assert [e.kind for e in seen] == ["model.usage"]
    info = session.info()
    assert info["context_used"] == 1000 and info["context_max"] == 32768
    assert info["provider"] == "LOCAL" and info["sandbox"] == "bwrap" and info["session_id"] == "s1"
    asyncio.run(session.cancel())
    assert session.engine.cancelled == 1
    assert session.new() == "s2"
    assert session.resume() == "abc123"
    assert session.resume("abd") == "abd999"
    with pytest.raises(LookupError):
        session.resume("ab")  # ambiguous
    session.set_model("other")
    assert session.info()["model"] == "other"
    with pytest.raises(NotImplementedError):
        asyncio.run(session.undo_last())


@pytest.mark.parametrize(
    "status, data, expected",
    [
        ("ok", {}, None),
        ("ok", {"exit_code": 0, "tests_run": 3}, None),
        ("ok", {"exit_code": 1}, "exit 1"),
        ("ok", {"returncode": 2}, "exit 2"),
        ("ok", {"tests_run": 0}, "no tests ran"),
        ("ok", {"tests_failed": 2, "tests_run": 5}, "2 tests failed"),
        ("failed", {"summary": "boom"}, "failed: boom"),
        ("timeout", {}, "timeout"),
        ("invalid", {}, "invalid"),
        ("denied", {}, "denied"),
        ("ok", {"exit_code": True}, None),  # bools are not exit codes
    ],
)
def test_failure_reason(status, data, expected):
    assert failure_reason(status, data) == expected


def test_history_persists_multiline_bounded_and_survives_corruption(tmp_path):
    history = History.for_project(tmp_path / "data", tmp_path / "proj")
    history.add("first\nsecond line")
    history.add("first\nsecond line")  # consecutive duplicate ignored
    history.add("")
    reloaded = History.for_project(tmp_path / "data", tmp_path / "proj")
    assert reloaded.entries == ["first\nsecond line"]
    other = History.for_project(tmp_path / "data", tmp_path / "other")
    assert other.entries == []  # per project
    for index in range(HISTORY_LIMIT + 5):
        reloaded.add(f"e{index}")
    assert len(History.for_project(tmp_path / "data", tmp_path / "proj").entries) == HISTORY_LIMIT
    assert reloaded.path is not None
    reloaded.path.write_text("{not json\n", encoding="utf-8")
    assert History.for_project(tmp_path / "data", tmp_path / "proj").entries == []


def test_history_navigation_restores_draft():
    history = History(None)
    for text in ("a", "b"):
        history.add(text)
    assert history.previous("draft") == "b"
    assert history.previous("b") == "a"
    assert history.previous("a") is None
    assert history.next() == "b"
    assert history.next() == "draft"
    assert history.next() is None


def test_match_files_prefers_path_then_name_then_substring():
    files = ["docs/main.md", "src/main.py", "src/util.py", "tests/test_main.py"]
    assert match_files(files, "src/") == ["src/main.py", "src/util.py"]
    assert match_files(files, "main")[:2] == ["docs/main.md", "src/main.py"]
    assert "tests/test_main.py" in match_files(files, "main")


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_session_diff_uses_git_for_tracked_and_shows_new_files(tmp_path):
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    git("add", "a.py")
    git("commit", "-q", "-m", "init")
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 1\n", encoding="utf-8")
    diff = session_diff(tmp_path, ["a.py", "b.py"])
    assert "-x = 1" in diff and "+x = 2" in diff
    assert "+++ b/b.py" in diff and "+y = 1" in diff
    assert diff.count("+++ b/a.py") == 1
