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
    assert "No saved conversations here. Start one: nedster run" in capsys.readouterr().out


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
    assert main(["sessions", "delete", first[:7], "--yes", "--project", str(clean_env)]) == EXIT_OK
    assert main(["sessions", "delete", second, "--yes", "--project", str(clean_env)]) == EXIT_OK


def test_sessions_delete_refuses_a_live_lock_and_keeps_the_session(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env))
    owner = make_owner("tui")
    assert store.acquire_lock(session, owner)
    store.close()

    assert main(["sessions", "delete", session, "--yes", "--project", str(clean_env)]) == EXIT_RUNTIME
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

    assert main(["sessions", "delete", gone, "--yes", "--project", str(clean_env)]) == EXIT_OK
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
    assert "Nothing was prunable." in out
    assert "This prunes stored tool output only; conversations are kept." in out


def test_sessions_prune_dry_run_deletes_nothing(clean_env, capsys):
    """Dry run computes the same stats but removes nothing."""
    store = _store(clean_env)
    session = store.create(_root(clean_env))
    unreferenced_old = store.put_output(session, "bash", "c1", "stale")
    handle = store.put_output(session, "bash", "c2", "log")
    store.append_message(
        session, ChatMessage(role="tool", content=f'{{"handle": {handle}}}', tool_call_id="c2", name="bash")
    )
    foreign = store.create(str((clean_env / "other").resolve()))
    foreign_handle = store.put_output(foreign, "bash", "x", "foreign")
    store.close()
    _backdate(clean_env, [(session, unreferenced_old), (session, handle)])

    # dry-run: same stats as real run but nothing deleted
    assert main(["sessions", "prune", "--older-than", "1", "--project", str(clean_env), "--dry-run"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "1 removable" in out
    assert "Dry run: nothing was deleted. Run again without --dry-run to delete 1 output(s)." in out
    assert "This prunes stored tool output only; conversations are kept." in out

    # both outputs still exist
    after = _store(clean_env)
    assert after.get_output(session, unreferenced_old) == "stale"
    assert after.get_output(session, handle) == "log"
    assert after.get_output(foreign, foreign_handle) == "foreign"
    after.close()

    # real run: deletes the unreferenced one
    assert main(["sessions", "prune", "--older-than", "1", "--project", str(clean_env)]) == EXIT_OK
    after2 = _store(clean_env)
    with pytest.raises(KeyError):
        after2.get_output(session, unreferenced_old)  # deleted
    assert after2.get_output(session, handle) == "log"  # referenced → kept
    after2.close()


def test_sessions_prune_json_includes_dry_run_key(clean_env, capsys):
    """When --dry-run is passed, the JSON payload contains 'dry_run': true."""
    store = _store(clean_env)
    session = store.create(_root(clean_env))
    handle = store.put_output(session, "bash", "c1", "log")
    store.close()
    _backdate(clean_env, [(session, handle)])

    assert main(["sessions", "prune", "--older-than", "1", "--project", str(clean_env), "--dry-run", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["deleted"] == 1  # same count as a real run


def test_sessions_prune_dry_run_plain_has_dry_run_line(clean_env, capsys):
    """--plain output with --dry-run includes the dry_run\t1 line."""
    store = _store(clean_env)
    session = store.create(_root(clean_env))
    handle = store.put_output(session, "bash", "c1", "log")
    store.close()
    _backdate(clean_env, [(session, handle)])

    assert main(["sessions", "prune", "--older-than", "1", "--project", str(clean_env), "--plain", "--dry-run"]) == EXIT_OK
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    keys = [line.split("\t")[0] for line in lines]
    assert "dry_run" in keys
    dry_run_line = [ln for ln in lines if ln.startswith("dry_run\t")][0]
    assert dry_run_line == "dry_run\t1"


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
    assert main(["sessions", "delete", session[:6], "--yes", "--project", str(clean_env)]) == EXIT_OK
    after = _store(clean_env)
    assert after.get_session(session) is None
    after.close()


def test_sessions_show_and_delete_accept_an_exact_title(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="Fix the range bug")
    store.append_transcript(session, "user", "hello")
    store.close()

    assert main(["sessions", "show", "fix the range bug", "--project", str(clean_env)]) == EXIT_OK
    assert f"session     {session}" in capsys.readouterr().out
    assert main(["sessions", "delete", "Fix the range bug", "--yes", "--project", str(clean_env)]) == EXIT_OK
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


# --- BP1-S2: session names (auto-title, --name, rename, list display) -------------------------


def _set_updated(clean_env, session_id: str, *, seconds_ago: float) -> None:
    connection = sqlite3.connect(Path(clean_env) / "data" / "sessions.sqlite3")
    try:
        connection.execute(
            "UPDATE sessions SET updated_ns = ? WHERE id = ?",
            (time.time_ns() - int(seconds_ago * 1e9), session_id),
        )
        connection.commit()
    finally:
        connection.close()


def test_sessions_list_human_shows_relative_age_count_and_resume_hint(clean_env, capsys):
    store, recent = _seed(clean_env, _root(clean_env), title="named one")
    blank = store.create(_root(clean_env))
    store.append_message(blank, ChatMessage(role="user", content="  fix   the range\nboundary "))
    store.append_message(blank, ChatMessage(role="assistant", content="done"))
    store.close()
    _set_updated(clean_env, recent, seconds_ago=5 * 60)
    _set_updated(clean_env, blank, seconds_ago=3 * 86_400)

    assert main(["sessions", "list", "--project", str(clean_env)]) == EXIT_OK
    out = capsys.readouterr().out
    header = out.splitlines()[0]
    assert header.split()[0] == "id" and "updated" in header and "messages" in header and "title" in header
    assert f"{recent[:8]}" in out and "5m ago" in out
    assert "named one" in out
    # an empty title falls back to the first stored user message
    assert "fix the range" in out
    assert "3d ago" in out
    assert 'Resume: nedster resume <id or title>' in out and 'nedster run --session <id or title>' in out


def test_sessions_list_json_keeps_keys_and_adds_title_and_message_count(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="keep me")
    store.append_message(session, ChatMessage(role="user", content="hi"))
    store.close()

    assert main(["sessions", "list", "--project", str(clean_env), "--json"]) == EXIT_OK
    entry = json.loads(capsys.readouterr().out)["sessions"][0]
    for key in ("id", "project_root", "title", "created_ns", "updated_ns", "active"):
        assert key in entry
    assert entry["title"] == "keep me"
    assert entry["message_count"] == 1


def test_sessions_show_human_falls_back_to_the_first_message(clean_env, capsys):
    store = _store(clean_env)
    session = store.create(_root(clean_env))
    store.append_message(session, ChatMessage(role="user", content="explain the widget"))
    store.close()

    assert main(["sessions", "show", session, "--project", str(clean_env)]) == EXIT_OK
    assert "title       explain the widget" in capsys.readouterr().out


def test_sessions_rename_by_prefix_and_title(clean_env, capsys):
    store = _store(clean_env)
    first = store.create(_root(clean_env), "original name")
    second = store.create(_root(clean_env), "second")
    store.close()

    assert main(["sessions", "rename", first[:8], "Fix range boundary", "--project", str(clean_env)]) == EXIT_OK
    assert "Renamed session" in capsys.readouterr().out
    # the new name resolves as an exact title, the old one no longer does
    assert main(["sessions", "rename", "second", "Fix range boundary 2", "--project", str(clean_env)]) == EXIT_OK
    capsys.readouterr()
    after = _store(clean_env)
    assert after.get_session(first)["title"] == "Fix range boundary"
    assert after.get_session(second)["title"] == "Fix range boundary 2"
    after.close()


def test_sessions_rename_rejects_an_empty_name(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="keep")
    store.close()

    assert main(["sessions", "rename", session[:8], "   ", "--project", str(clean_env)]) == EXIT_USAGE
    assert "visible character" in capsys.readouterr().err
    after = _store(clean_env)
    assert after.get_session(session)["title"] == "keep"
    after.close()


def test_run_name_is_normalized_and_used_at_creation(clean_env, capsys):
    seen: dict[str, str | None] = {}

    class _FakeSession:
        def __init__(self, config, *, emit=None, model=None, session_id=None, resume_latest=False, title="", **_):
            seen["session_id"] = session_id
            seen["title"] = title
            self.session_id = session_id or "new"

        async def run(self, text, *, intent=None):
            from ironclad.contracts import AcceptanceResult, Outcome

            return AcceptanceResult(outcome=Outcome.ANSWERED, answer="ok")

        async def aclose(self): ...

    with patch("ironclad.native.NativeSession", _FakeSession):
        assert main(["run", "--name", "  Fix  the range ", "--project", str(clean_env), "hi"]) == EXIT_OK
    assert seen["session_id"] is None
    assert seen["title"] == "Fix the range"


def test_run_name_renames_the_resumed_session(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="old name")
    store.close()
    seen: dict[str, str | None] = {}

    class _FakeSession:
        def __init__(self, config, *, emit=None, model=None, session_id=None, resume_latest=False, title="", **_):
            seen["session_id"] = session_id
            seen["title"] = title
            self.session_id = session_id or "new"

        async def run(self, text, *, intent=None):
            from ironclad.contracts import AcceptanceResult, Outcome

            return AcceptanceResult(outcome=Outcome.ANSWERED, answer="ok")

        async def aclose(self): ...

    with patch("ironclad.native.NativeSession", _FakeSession):
        assert main(["run", "--session", session[:8], "--name", "New name", "--project", str(clean_env), "hi"]) == EXIT_OK
    assert seen["session_id"] == session
    assert seen["title"] == "New name"


def test_run_empty_name_is_a_usage_error(clean_env, capsys):
    assert main(["run", "--name", "   ", "--project", str(clean_env), "hi"]) == EXIT_USAGE
    assert "--name needs at least one visible character" in capsys.readouterr().err


# --- BP1-S3: `nedster resume`, lazy `run --continue`, tui session flags ---------------------------


def _accept_ok():
    from ironclad.contracts import AcceptanceResult, Outcome

    return AcceptanceResult(outcome=Outcome.ANSWERED, answer="ok")


def test_resume_non_interactive_exits_2_with_a_script_hint(clean_env, capsys):
    assert main(["resume", "--project", str(clean_env)]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert 'nedster resume needs an interactive terminal; for scripts use: nedster run --session <ref> "..."' in err


def _force_interactive(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)


def _capture_run_tui(monkeypatch) -> dict:
    import nedster_agent.tui as tui

    calls: dict = {}
    monkeypatch.setattr(tui, "run_tui", lambda config, **kwargs: calls.update(kwargs))
    return calls


def test_resume_with_a_ref_resolves_and_opens_that_session(clean_env, monkeypatch):
    store, session = _seed(clean_env, _root(clean_env), title="resume me")
    store.close()
    _force_interactive(monkeypatch)
    calls = _capture_run_tui(monkeypatch)

    assert main(["resume", session[:8], "--project", str(clean_env)]) == EXIT_OK
    assert calls["session_ref"] == session
    assert calls.get("open_picker_on_start", False) is False


def test_resume_without_a_ref_opens_the_picker_first(clean_env, monkeypatch):
    _force_interactive(monkeypatch)
    calls = _capture_run_tui(monkeypatch)

    assert main(["resume", "--project", str(clean_env)]) == EXIT_OK
    assert calls.get("open_picker_on_start") is True
    assert calls.get("session_ref") is None


def test_resume_last_picks_the_most_recent_non_empty_session(clean_env, monkeypatch):
    store = _store(clean_env)
    older = store.create(_root(clean_env), "older")
    store.append_message(older, ChatMessage(role="user", content="older work"))
    empty = store.create(_root(clean_env))  # newer but empty: must never be chosen
    store.close()
    _set_updated(clean_env, older, seconds_ago=3600)
    _set_updated(clean_env, empty, seconds_ago=1)
    _force_interactive(monkeypatch)
    calls = _capture_run_tui(monkeypatch)

    assert main(["resume", "--last", "--project", str(clean_env)]) == EXIT_OK
    assert calls["session_ref"] == older


def test_resume_ref_and_last_are_mutually_exclusive(clean_env, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["resume", "deadbeef", "--last", "--project", str(clean_env)])
    assert exit_info.value.code == EXIT_USAGE
    assert "not allowed with" in capsys.readouterr().err


def test_resume_unknown_ref_is_a_usage_error(clean_env, monkeypatch, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="the only one")
    store.close()
    _force_interactive(monkeypatch)

    assert main(["resume", "deadbeef", "--project", str(clean_env)]) == EXIT_USAGE
    assert "No session matches 'deadbeef'" in capsys.readouterr().err


def test_tui_session_and_continue_are_mutually_exclusive(clean_env, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["tui", "--session", "deadbeef", "--continue", "--project", str(clean_env)])
    assert exit_info.value.code == EXIT_USAGE
    assert "not allowed with" in capsys.readouterr().err


def test_tui_session_flag_opens_that_conversation(clean_env, monkeypatch):
    store, session = _seed(clean_env, _root(clean_env), title="open me")
    store.close()
    _force_interactive(monkeypatch)
    calls = _capture_run_tui(monkeypatch)

    assert main(["tui", "--session", session[:8], "--project", str(clean_env)]) == EXIT_OK
    assert calls["session_ref"] == session


def test_tui_continue_flag_resumes_latest(clean_env, monkeypatch):
    _force_interactive(monkeypatch)
    calls = _capture_run_tui(monkeypatch)

    assert main(["tui", "--continue", "--project", str(clean_env)]) == EXIT_OK
    assert calls.get("resume_latest") is True


def test_run_continue_without_history_prints_a_notice(clean_env, capsys):
    class _FakeSession:
        def __init__(self, config, *, emit=None, model=None, session_id=None, resume_latest=False, title="", **_):
            self.session_id = session_id or "new"
            self.started_new = True  # no non-empty session in this project

        async def run(self, text, *, intent=None):
            return _accept_ok()

        async def aclose(self): ...

    with patch("ironclad.native.NativeSession", _FakeSession):
        assert main(["run", "--continue", "--project", str(clean_env), "hi"]) == EXIT_OK
    assert "No previous conversation here; starting a new one." in capsys.readouterr().err


def test_run_continue_with_history_has_no_notice(clean_env, capsys):
    class _FakeSession:
        def __init__(self, config, *, emit=None, model=None, session_id=None, resume_latest=False, title="", **_):
            self.session_id = session_id or "existing"
            self.started_new = False  # resumed a real conversation

        async def run(self, text, *, intent=None):
            return _accept_ok()

        async def aclose(self): ...

    with patch("ironclad.native.NativeSession", _FakeSession):
        assert main(["run", "--continue", "--project", str(clean_env), "hi"]) == EXIT_OK
    assert "No previous conversation here" not in capsys.readouterr().err


# --- BP1-S5: reviewed, complete session deletion ------------------------------------------------


def _checkpoint(clean_env, session: str, *, rel: str = "file.txt") -> None:
    """Journal one pre-image for ``session`` so the checkpoint tables have rows to remove."""
    from ironclad.session_store.checkpoints import CheckpointStore

    root = Path(clean_env)
    target = root / rel
    target.write_text("before", encoding="utf-8")
    store = CheckpointStore(root / "data" / "sessions.sqlite3")
    store.begin_turn(session, root)
    store.record_pre_image(rel, target)
    target.write_text("after", encoding="utf-8")
    store.end_turn()
    store.close()


def _checkpoint_counts(clean_env) -> tuple[int, int]:
    connection = sqlite3.connect(Path(clean_env) / "data" / "sessions.sqlite3")
    try:
        turns = connection.execute("SELECT COUNT(*) FROM ckpt_turns").fetchone()[0]
        files = connection.execute("SELECT COUNT(*) FROM ckpt_files").fetchone()[0]
    finally:
        connection.close()
    return int(turns), int(files)


def test_checkpoint_store_delete_session_removes_rows_of_one_session_only(clean_env):
    from ironclad.session_store.checkpoints import CheckpointStore

    store = _store(clean_env)
    keep = store.create(_root(clean_env), "keep")
    gone = store.create(_root(clean_env), "gone")
    store.close()
    _checkpoint(clean_env, keep, rel="keep.txt")
    _checkpoint(clean_env, gone, rel="gone.txt")

    checkpoints = CheckpointStore(Path(clean_env) / "data" / "sessions.sqlite3")
    assert checkpoints.checkpoint_count(gone) == 1
    assert checkpoints.delete_session(gone) == 1
    assert checkpoints.checkpoint_count(gone) == 0
    assert checkpoints.checkpoint_count(keep) == 1  # untouched
    checkpoints.close()
    turns, files = _checkpoint_counts(clean_env)
    assert (turns, files) == (1, 1)


def test_sessions_delete_removes_the_sessions_checkpoints(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="checkpointed")
    store.append_transcript(session, "user", "hello")
    store.close()
    _checkpoint(clean_env, session)
    assert _checkpoint_counts(clean_env) == (1, 1)

    assert main(["sessions", "delete", session, "--yes", "--project", str(clean_env)]) == EXIT_OK
    assert _checkpoint_counts(clean_env) == (0, 0)
    after = _store(clean_env)
    assert after.get_session(session) is None
    after.close()


def test_sessions_delete_dry_run_prints_preview_and_keeps_everything(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="preview me")
    store.append_message(session, ChatMessage(role="user", content="one"))
    store.append_message(session, ChatMessage(role="assistant", content="two"))
    store.close()
    _checkpoint(clean_env, session)

    assert main(["sessions", "delete", session, "--dry-run", "--project", str(clean_env)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "preview me" in out and "messages" in out and "checkpoints" in out
    assert "Dry run: nothing deleted." in out
    before = _counts(clean_env)
    assert before[0] == 1  # the session is still there
    assert _checkpoint_counts(clean_env) == (1, 1)


def test_sessions_delete_non_tty_refuses_without_yes(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="safe")
    store.close()

    assert main(["sessions", "delete", session, "--project", str(clean_env)]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "Refusing to delete without confirmation in a non-interactive shell. Re-run with --yes." in err
    after = _store(clean_env)
    assert after.get_session(session) is not None
    after.close()


def test_sessions_delete_json_requires_yes(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="machine")
    store.close()

    assert main(["sessions", "delete", session, "--json", "--project", str(clean_env)]) == EXIT_USAGE
    assert "--json needs --yes" in capsys.readouterr().err
    after = _store(clean_env)
    assert after.get_session(session) is not None
    after.close()

    assert main(["sessions", "delete", session, "--json", "--yes", "--project", str(clean_env)]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"deleted": True, "id": session}


def test_sessions_delete_dry_run_json_deletes_nothing(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="preview")
    store.close()

    assert (
        main(["sessions", "delete", session, "--dry-run", "--json", "--project", str(clean_env)]) == EXIT_OK
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True and payload["deleted"] is False
    assert payload["session"]["title"] == "preview"
    after = _store(clean_env)
    assert after.get_session(session) is not None
    after.close()


def test_sessions_delete_prompts_on_a_tty_and_only_yes_deletes(clean_env, monkeypatch, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="ask me")
    store.close()
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    prompts: list[str] = []
    answers = iter(["n"])
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or next(answers))

    assert main(["sessions", "delete", session, "--project", str(clean_env)]) == EXIT_OK
    assert prompts[-1] == "Delete this conversation and its saved checkpoints? [y/N] "
    out = capsys.readouterr().out
    assert "ask me" in out and "messages" in out and "checkpoints  0" in out
    after = _store(clean_env)
    assert after.get_session(session) is not None  # 'n' cancels
    after.close()

    answers = iter(["yes"])
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or next(answers))
    assert main(["sessions", "delete", session, "--project", str(clean_env)]) == EXIT_OK
    after = _store(clean_env)
    assert after.get_session(session) is None
    after.close()


def test_sessions_delete_lock_refusal_keeps_checkpoints(clean_env, capsys):
    store, session = _seed(clean_env, _root(clean_env), title="busy")
    owner = make_owner("tui")
    assert store.acquire_lock(session, owner)
    store.close()
    _checkpoint(clean_env, session)

    assert main(["sessions", "delete", session, "--yes", "--project", str(clean_env)]) == EXIT_RUNTIME
    assert "active" in capsys.readouterr().err
    assert _checkpoint_counts(clean_env) == (1, 1)  # refusal touched nothing
    after = _store(clean_env)
    assert after.get_session(session) is not None
    after.release_lock(session, owner)
    after.close()



