"""BP1-S2: session title text rules and the store fields they rely on."""

from __future__ import annotations

from pathlib import Path

from ironclad.contracts import ChatMessage
from ironclad.session_store.sqlite_store import SqliteSessionStore
from ironclad.session_store.titles import TITLE_LIMIT, UNTITLED, display_title, normalize_title, relative_age


def test_normalize_collapses_whitespace_and_takes_the_first_non_empty_line():
    assert normalize_title("  Fix   the   range\nboundary ") == "Fix the range"
    assert normalize_title("\n\n  second line wins  \n") == "second line wins"
    assert normalize_title("   \n \t \n") == ""


def test_normalize_strips_control_characters():
    assert normalize_title("Fix\x00 the\x07 range") == "Fix the range"
    assert all(ord(ch) >= 0x20 for ch in normalize_title("a\x00b\x1b two"))


def test_normalize_truncates_at_a_word_boundary_within_the_limit():
    text = "word " * 40
    result = normalize_title(text)
    assert len(result) <= TITLE_LIMIT
    assert result.endswith("…")
    assert result[:-1].endswith("word")

    long_token = "x" * 200
    hard = normalize_title(long_token)
    assert len(hard) == TITLE_LIMIT and hard.endswith("…")


def test_display_title_falls_back_to_the_first_message_then_untitled():
    assert display_title("  Fix the range  ") == "Fix the range"
    assert display_title("", "do the thing\nand more") == "do the thing"
    assert display_title("", "") == UNTITLED
    assert display_title("", "\x00\x07") == UNTITLED


def test_relative_age_buckets_by_elapsed_time():
    now = 2_000_000_000.0
    ns = lambda seconds: int((now - seconds) * 1e9)  # noqa: E731 - tiny helper for readability
    assert relative_age(0) == ""
    assert relative_age(ns(10), now=now) == "just now"
    assert relative_age(ns(5 * 60), now=now) == "5m ago"
    assert relative_age(ns(3 * 3600), now=now) == "3h ago"
    assert relative_age(ns(86_400 + 60), now=now) == "yesterday"
    assert relative_age(ns(3 * 86_400), now=now) == "3d ago"
    assert relative_age(ns(30 * 86_400), now=now).count("-") == 2  # a date when older than a week


def test_store_set_title_first_user_text_and_message_count(tmp_path: Path):
    with SqliteSessionStore(tmp_path / "sessions.sqlite3") as store:
        session = store.create(str(tmp_path))
        store.append_message(session, ChatMessage(role="user", content="   "))
        store.append_message(session, ChatMessage(role="assistant", content="hello"))
        store.append_message(session, ChatMessage(role="user", content="the real first prompt"))
        assert store.first_user_text(session) == "the real first prompt"

        store.set_title(session, "Named session")
        assert store.get_session(session)["title"] == "Named session"
        listing = store.list_sessions(str(tmp_path))
        assert listing[0]["title"] == "Named session"
        assert listing[0]["message_count"] == 3
