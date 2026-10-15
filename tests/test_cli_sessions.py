"""W2-RESUME: `nedster sessions` (list/show/delete/prune) and its project binding."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from ironclad.contracts import ChatMessage
from ironclad.session_store.sqlite_store import MIN_PREFIX_CHARS, SqliteSessionStore, make_owner
from nedster_agent.cli import EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, main


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("NEDSTER_") or name == "OLLAMA_HOST":
            monkeypatch.delenv(name)
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("NEDSTER_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def _root(clean_env) -> str:
    return str(Path(clean_env).resolve())


def _store(clean_env) -> SqliteSessionStore:
    return SqliteSessionStore(Path(clean_env) / "data" / "sessions.sqlite3")


def _seed(clean_env, project_root: str, *, title: str = "") -> tuple[SqliteSessionStore, str]:
    store = _store(clean_env)
    return store, store.create(project_root, title)


def _backdate(clean_env, handles: list[tuple[str, int]], *, days: float = 10.0) -> None:
    """Make stored outputs look old enough to be prune candidates (creation is always 'now')."""
    cutoff = time.time_ns() - int(days * 86_400 * 1e9)
    connection = sqlite3.connect(Path(clean_env) / "data" / "sessions.sqlite3")
    try:
        for session_id, handle in handles:
            connection.execute(
                "UPDATE outputs SET created_ns = ? WHERE session_id = ? AND handle = ?",
                (cutoff, session_id, handle),
            )
        connection.commit()
    finally:
        connection.close()


def test_sessions_list_human_is_empty_and_successful(clean_env, capsys):
    assert main(["sessions", "list", "--project", str(clean_env)]) == EXIT_OK
    assert "No saved sessions for this project." in capsys.readouterr().out


def test_sessions_defaults_to_list_and_accepts_flags_after_the_action(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="write the report")
    store.close()
    assert main(["sessions", "list", "--project", str(clean_env), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert [entry["id"] for entry in payload["sessions"]] == [session]
    assert payload["sessions"][0]["title"] == "write the report"
    assert payload["sessions"][0]["active"] is False
    # `--json` also works on the parent parser, before the action
    assert main(["sessions", "--json", "--project", str(clean_env), "list"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["sessions"][0]["id"] == session


def test_sessions_list_json_only_lists_this_projects_sessions(clean_env, capsys):
    other = clean_env / "other-project"
    other.mkdir()
    mine, mine_id = _seed(clean_env, _root(clean_env))
    _, foreign_id = _seed(clean_env, str(other.resolve()))
    mine.close()

    assert main(["sessions", "list", "--project", str(clean_env), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["project_root"] == _root(clean_env)
    assert [entry["id"] for entry in payload["sessions"]] == [mine_id]
    assert foreign_id not in {entry["id"] for entry in payload["sessions"]}


def test_sessions_show_json_reports_transcript_metadata_and_outputs(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="inspect me")
    store.append_message(session, ChatMessage(role="user", content="do the thing"))
    store.append_message(session, ChatMessage(role="assistant", content="done"))
    store.append_transcript(session, "user", "do the thing")
    store.append_transcript(session, "assistant", "done", {"source": "cli"})
    store.append_transcript(session, "tool", "read_file", {"tool": "read_file", "status": "ok"})
    handle = store.put_output(session, "read_file", "c1", "file.py contents")
    store.set_state(session, "task_state", {"step": 1})
    store.close()

    assert main(["sessions", "show", session[:8], "--project", str(clean_env), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["session"]["id"] == session
    assert payload["messages"] == 2
    assert payload["transcript_total"] == 3
    assert [entry["role"] for entry in payload["entries"]] == ["user", "assistant", "tool"]
    assert payload["entries"][1]["meta"] == {"source": "cli"}
    assert payload["outputs"]["total"] == 1
    assert payload["state_keys"] == ["task_state"]
    assert payload["lock"] is None
    assert handle > 0


def test_sessions_show_human_prints_roles_and_the_stored_text(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="inspect me")
    store.append_transcript(session, "user", "do the thing")
    store.append_transcript(session, "assistant", "done")
    store.append_transcript(session, "tool", "read_file")
    store.close()

    assert main(["sessions", "show", session, "--project", str(clean_env)]) == EXIT_OK
    out = capsys.readouterr().out
    assert f"session     {session}" in out
    # the stored project_root is the canonical, case-normalised key (invariant 3)
    assert f"project     {os.path.normcase(_root(clean_env))}".lower() in out.lower()
    assert "--- transcript ---" in out
    assert "[user] do the thing" in out
    assert "[assistant] done" in out
    assert "[tool] read_file" in out


def test_sessions_delete_unknown_id_is_a_usage_error(clean_env, capsys):
    assert main(["sessions", "delete", "deadbeef", "--project", str(clean_env)]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "No session matches 'deadbeef' in this project. Run: nedster sessions list" in err


def test_sessions_delete_prefix_must_be_unique(clean_env, capsys):
    store = _store(clean_env)
    first = store.create(_root(clean_env))
    second = store.create(_root(clean_env))
    store.close()

    # an empty prefix matches every session of this project -> ambiguous
    assert main(["sessions", "delete", "", "--project", str(clean_env)]) == EXIT_USAGE
    assert "matches 2 sessions" in capsys.readouterr().err
    after = _store(clean_env)
    assert after.get_session(first) is not None and after.get_session(second) is not None
    after.close()

    # a full id is unique, and a 7-char prefix of it resolves when nothing else matches
    assert main(["sessions", "delete", first[:7], "--project", str(clean_env)]) == EXIT_OK
    assert main(["sessions", "delete", second, "--project", str(clean_env)]) == EXIT_OK


def test_sessions_delete_refuses_a_live_lock_and_keeps_the_session(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env))
    owner = make_owner("tui")
    assert store.acquire_lock(session, owner)
    store.close()

    assert main(["sessions", "delete", session, "--project", str(clean_env)]) == EXIT_RUNTIME
    assert "active" in capsys.readouterr().err
    after = _store(clean_env)
    assert after.get_session(session) is not None
    assert after.lock_info(session)["active"] is True
    after.release_lock(session, owner)
    assert after.delete_session(session) is True  # released: now deletable
    after.close()


def test_sessions_delete_removes_only_the_named_session(clean_env, capsys):
    store = _store(clean_env)
    keep = store.create(_root(clean_env), "keep")
    gone = store.create(_root(clean_env), "gone")
    store.append_transcript(gone, "user", "bye")
    store.close()

    assert main(["sessions", "delete", gone, "--project", str(clean_env)]) == EXIT_OK
    assert f"Deleted session {gone}." in capsys.readouterr().out
    after = _store(clean_env)
    assert after.get_session(gone) is None
    assert after.get_session(keep) is not None
    assert after.transcript(keep) == []
    after.close()


def test_sessions_prune_requires_an_explicit_older_than(clean_env, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["sessions", "prune", "--project", str(clean_env)])
    assert exit_info.value.code == EXIT_USAGE
    assert "--older-than" in capsys.readouterr().err


@pytest.mark.parametrize("days", ["0", "-3", "inf", "nan"])
def test_sessions_prune_rejects_non_positive_or_unbounded_days(clean_env, capsys, days):
    assert main(["sessions", "prune", "--older-than", days, "--project", str(clean_env)]) == EXIT_USAGE
    assert "positive number of days" in capsys.readouterr().err


def test_sessions_prune_json_deletes_only_old_unreferenced_outputs(clean_env, capsys):
    other = clean_env / "other-project"
    other.mkdir()
    store = _store(clean_env)
    session = store.create(_root(clean_env))
    old_unreferenced = store.put_output(session, "bash", "c1", "stale log")
    old_referenced = store.put_output(session, "bash", "c2", "needed later")
    store.append_message(session, ChatMessage(role="assistant", content=f'see {{"handle": {old_referenced}}}'))
    store.put_output(session, "bash", "c3", "fresh log")
    foreign = store.create(str(other.resolve()))  # another project: out of scope for this prune
    foreign_output = store.put_output(foreign, "bash", "c4", "foreign")
    store.close()
    _backdate(clean_env, [(session, old_unreferenced), (session, old_referenced)])

    assert main(["sessions", "prune", "--older-than", "1", "--project", str(clean_env), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["deleted"] == 1
    assert payload["retained_referenced"] == 1
    assert payload["retained_recent"] == 1
    assert payload["project_root"] == _root(clean_env)

    after = _store(clean_env)
    assert after.outputs_report(session)["total"] == 2  # the referenced one plus the recent one
    assert after.get_output(session, old_referenced)
    assert after.get_output(foreign, foreign_output) == "foreign"
    after.close()


def test_sessions_prune_human_reports_what_it_kept(clean_env, capsys):
    store = _store(clean_env)
    session = store.create(_root(clean_env))
    handle = store.put_output(session, "bash", "c1", "log")
    store.append_message(
        session, ChatMessage(role="tool", content=f'{{"handle": {handle}}}', tool_call_id="c1", name="bash")
    )
    store.close()
    _backdate(clean_env, [(session, handle)], days=40)

    assert main(["sessions", "prune", "--older-than", "30", "--project", str(clean_env)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Deleted 0 unreferenced output(s)." in out
    assert "Kept 1 referenced by a message or task state." in out
    assert "Nothing was prunable." in out


def test_sessions_show_marks_a_session_another_window_is_using(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env))
    owner = make_owner("tui")
    assert store.acquire_lock(session, owner)
    store.close()

    assert main(["sessions", "list", "--project", str(clean_env)]) == EXIT_OK
    assert "active" in capsys.readouterr().out
    assert main(["sessions", "show", session, "--project", str(clean_env), "--json"]) == EXIT_OK
    lock = json.loads(capsys.readouterr().out)["lock"]
    assert lock["active"] is True and lock["pid"] == os.getpid()
    cleanup = _store(clean_env)
    cleanup.release_lock(session, owner)
    cleanup.close()


# --- BP1-S1: one project-bound session resolver ------------------------------------------------


def _counts(clean_env) -> tuple[int, int]:
    connection = sqlite3.connect(Path(clean_env) / "data" / "sessions.sqlite3")
    try:
        sessions = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        connection.close()
    return int(sessions), int(messages)


def test_resolve_session_accepts_prefix_and_exact_title_and_searches_all(clean_env):
    store = _store(clean_env)
    ids = [store.create(_root(clean_env), f"title {index}") for index in range(25)]
    oldest = ids[0]
    assert oldest not in {entry["id"] for entry in store.list_sessions(_root(clean_env))}
    assert store.resolve_session(_root(clean_env), oldest)["id"] == oldest
    assert store.resolve_session(_root(clean_env), oldest[:MIN_PREFIX_CHARS])["id"] == oldest
    assert store.resolve_session(_root(clean_env), "Title 7")["id"] == ids[7]
    store.close()


def test_resolve_session_reports_unknown_ambiguous_and_foreign(clean_env):
    from ironclad.session_store.sqlite_store import (
        SessionAmbiguous,
        SessionForeignProject,
        SessionNotFound,
    )

    other = clean_env / "other-project"
    other.mkdir()
    store = _store(clean_env)
    store.create(_root(clean_env), "duplicate")
    store.create(_root(clean_env), "duplicate")
    foreign = store.create(str(other.resolve()), "elsewhere")
    with pytest.raises(SessionNotFound):
        store.resolve_session(_root(clean_env), "deadbeef")
    with pytest.raises(SessionAmbiguous) as ambiguous:
        store.resolve_session(_root(clean_env), "duplicate")
    assert "matches 2 sessions" in str(ambiguous.value)
    with pytest.raises(SessionForeignProject):
        store.resolve_session(_root(clean_env), foreign)
    store.close()


def test_sessions_show_and_delete_accept_a_unique_prefix(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="inspect me")
    store.append_transcript(session, "user", "hello")
    store.close()

    assert main(["sessions", "show", session[:6], "--project", str(clean_env)]) == EXIT_OK
    assert f"session     {session}" in capsys.readouterr().out
    assert main(["sessions", "delete", session[:6], "--project", str(clean_env)]) == EXIT_OK
    after = _store(clean_env)
    assert after.get_session(session) is None
    after.close()


def test_sessions_show_and_delete_accept_an_exact_title(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="Fix the range bug")
    store.append_transcript(session, "user", "hello")
    store.close()

    assert main(["sessions", "show", "fix the range bug", "--project", str(clean_env)]) == EXIT_OK
    assert f"session     {session}" in capsys.readouterr().out
    assert main(["sessions", "delete", "Fix the range bug", "--project", str(clean_env)]) == EXIT_OK
    assert f"Deleted session {session}." in capsys.readouterr().out


def test_sessions_ambiguous_title_lists_candidates_and_writes_nothing(clean_env, capsys):
    store = _store(clean_env)
    first = store.create(_root(clean_env), "duplicate")
    second = store.create(_root(clean_env), "duplicate")
    store.close()
    before = _counts(clean_env)

    assert main(["sessions", "delete", "duplicate", "--project", str(clean_env)]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "matches 2 sessions" in err
    assert first[:8] in err and second[:8] in err
    assert _counts(clean_env) == before


def test_sessions_foreign_id_is_a_usage_error_and_writes_nothing(clean_env, capsys):
    other = clean_env / "other-project"
    other.mkdir()
    store = _store(clean_env)
    _seed(clean_env, _root(clean_env))
    foreign = store.create(str(other.resolve()), "elsewhere")
    store.close()
    before = _counts(clean_env)

    assert main(["sessions", "show", foreign, "--project", str(clean_env)]) == EXIT_USAGE
    assert "belongs to another project" in capsys.readouterr().err
    assert main(["sessions", "delete", foreign, "--project", str(clean_env)]) == EXIT_USAGE
    assert "belongs to another project" in capsys.readouterr().err
    assert _counts(clean_env) == before
    after = _store(clean_env)
    assert after.get_session(foreign) is not None
    after.close()


def test_run_session_and_continue_are_mutually_exclusive(clean_env, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["run", "--session", "deadbeef", "--continue", "--project", str(clean_env), "hi"])
    assert exit_info.value.code == EXIT_USAGE
    assert "not allowed with" in capsys.readouterr().err


def test_run_session_prefix_resolves_to_the_full_id(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="resume me")
    store.close()
    seen: dict[str, str | None] = {}

    class _FakeSession:
        def __init__(self, config, *, emit=None, model=None, session_id=None, resume_latest=False, **_):
            seen["session_id"] = session_id
            self.session_id = session_id

        async def run(self, text, *, intent=None):
            from ironclad.contracts import AcceptanceResult, Outcome

            return AcceptanceResult(outcome=Outcome.ANSWERED, answer="ok")

        async def aclose(self): ...

    with patch("ironclad.native.NativeSession", _FakeSession):
        assert main(["run", "--session", session[:8], "--project", str(clean_env), "hi"]) == EXIT_OK
    assert seen["session_id"] == session
    assert capsys.readouterr().out == "ok\n"


def test_run_unknown_session_exits_2_and_writes_nothing(clean_env, capsys):
    store, _ = _seed(clean_env, _root(clean_env), title="the only one")
    store.close()
    before = _counts(clean_env)

    assert main(["run", "--session", "deadbeef", "--project", str(clean_env), "hi"]) == EXIT_USAGE
    assert "No session matches 'deadbeef'" in capsys.readouterr().err
    assert _counts(clean_env) == before


def test_run_foreign_session_exits_2_and_writes_nothing(clean_env, capsys):
    other = clean_env / "other-project"
    other.mkdir()
    store, _ = _seed(clean_env, _root(clean_env))
    foreign = store.create(str(other.resolve()), "elsewhere")
    store.close()
    before = _counts(clean_env)

    assert main(["run", "--session", foreign, "--project", str(clean_env), "hi"]) == EXIT_USAGE
    assert "belongs to another project" in capsys.readouterr().err
    assert _counts(clean_env) == before


def test_run_session_locked_elsewhere_exits_1_and_writes_nothing(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="busy")
    store.append_message(session, ChatMessage(role="user", content="in flight"))
    owner = make_owner("tui")
    assert store.acquire_lock(session, owner)
    store.close()
    before = _counts(clean_env)

    assert main(["run", "--session", session[:8], "--project", str(clean_env), "hi"]) == EXIT_RUNTIME
    err = capsys.readouterr().err
    assert f"Session {session[:8]} is active in another Nedster. Close it there and retry." in err
    assert _counts(clean_env) == before
    after = _store(clean_env)
    assert len(after.messages(session)) == 1
    after.release_lock(session, owner)
    after.close()


def test_cli_sessions_parsing_stays_lazy():
    # parsing `sessions` must not import the UI, the model client or the engine
    code = (
        "import sys, types\n"
        "pkg = types.ModuleType('nedster_agent'); pkg.__path__ = ['nedster_agent']; pkg.__version__ = 'x'\n"
        "sys.modules['nedster_agent'] = pkg\n"
        "import nedster_agent.cli as cli\n"
        "args = cli.build_parser().parse_args(['sessions', 'list', '--json', '--plain'])\n"
        "assert args.command == 'sessions' and args.action == 'list' and args.json and args.plain\n"
        "print(','.join(m for m in ('textual', 'httpx', 'rich', 'ironclad.native') if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True, text=True, check=True, cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert out.stdout.strip() == ""
