from __future__ import annotations

import pytest

from ironclad.engine.intent import Intent, classify, parse_expect

C, Q, U = Intent.CHANGE, Intent.QUESTION, Intent.UNKNOWN

CASES = [
    # required by the brief
    ("how do I fix X?", Q),
    ("fix X?", C),
    ("can you add a test", C),
    ("do not change anything, just explain", Q),
    ("why does test_y fail? fix it", C),
    # imperative edit verbs
    ("Fix the off-by-one in pager.py", C),
    ("add a --verbose flag", C),
    ("Implement caching for the loader.", C),
    ("rename foo to bar", C),
    ("Refactor utils.py into smaller functions", C),
    ("remove the dead code in api.py", C),
    ("delete tmp.txt", C),
    ("update the README", C),
    ("change the default port to 8080", C),
    ("create a CLI entry point", C),
    ("write tests for parse()", C),
    ("make the tests pass", C),
    ("convert config.ini to TOML", C),
    ("port this module to asyncio", C),
    ("migrate the db layer to sqlite3", C),
    ("bump the version to 2.0", C),
    ("Please refactor foo.py.", C),
    ("could you please rename the class", C),
    ("I want you to add type hints", C),
    ("help me fix the failing test", C),
    ("go ahead and implement it", C),
    ("explain it and fix the bug", C),
    ("Is X broken? If so, fix it", C),
    ("Could you look into why login fails and repair it", C),
    ("Read main.py, then add logging", C),
    # questions
    ("what does main.py do?", Q),
    ("why is the build slow", Q),
    ("where is the config loaded?", Q),
    ("Explain the retry logic", Q),
    ("describe the architecture", Q),
    ("list the public functions", Q),
    ("show me the entry point", Q),
    ("Can I add a test?", Q),
    ("should we refactor this?", Q),
    ("explain the add and remove functions", Q),
    ("what is utils.add", Q),
    ("explain `remove`", Q),
    ("no changes please, just describe the API", Q),
    ("can you explain how the cache works", Q),
    ("summarize the changes in the last module", Q),
    ("don't modify any files", Q),
    # unknown
    ("hello", U),
    ("run the tests", U),
    ("", U),
    ("the build", U),
]


@pytest.mark.parametrize(("objective", "expected"), CASES)
def test_classify(objective: str, expected: Intent) -> None:
    assert classify(objective) is expected


def test_at_least_forty_cases() -> None:
    assert len(CASES) >= 40


@pytest.mark.parametrize("override", list(Intent))
def test_caller_intent_always_wins(override: Intent) -> None:
    for objective, _ in CASES:
        assert classify(objective, override) is override


def test_parse_expect() -> None:
    assert parse_expect(None) is None and parse_expect(" ") is None
    assert parse_expect("change") is C and parse_expect("ANSWER") is Q
    with pytest.raises(ValueError):
        parse_expect("maybe")


def test_deterministic() -> None:
    assert [classify(o) for o, _ in CASES] == [classify(o) for o, _ in CASES]
