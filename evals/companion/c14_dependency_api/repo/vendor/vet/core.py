"""Core of vet.

A Checker holds rules.  Each rule pairs a field name with an expectation made by
one of the `expect.*` factories.  `Checker.judge(record)` inspects a plain dict
and returns a Verdict.

    chk = Checker()
    chk.demand("id", expect.whole(at_least=1))        # field must be present
    chk.allow("nick", expect.words(longest=12))       # field may be absent / None
    verdict = chk.judge({"id": 0})
    if not verdict:                                   # a Verdict is truthy only when clean
        for p in verdict.problems:                    # one Problem per failing field
            print(p.where, p.why)

At most one Problem is reported per field: the first reason that applies.
Reason codes are plain strings, listed with each expectation below.
"""

from __future__ import annotations

import re


class Problem:
    """One failed field: `where` is the field name, `why` the reason code."""

    __slots__ = ("where", "why")

    def __init__(self, where: str, why: str):
        self.where = where
        self.why = why

    def __repr__(self) -> str:
        return f"Problem({self.where!r}, {self.why!r})"


class Verdict:
    """Result of Checker.judge.  Truthy when there are no problems."""

    def __init__(self, problems):
        self.problems = list(problems)

    def __bool__(self) -> bool:
        return not self.problems

    def __len__(self) -> int:
        return len(self.problems)


class _Whole:
    """Integer expectation.  Reasons, checked in this order:

    not_whole  - value is not an int (bool and float do not count)
    too_small  - value < at_least
    too_big    - value > at_most
    """

    def __init__(self, at_least, at_most):
        self.at_least, self.at_most = at_least, at_most

    def inspect(self, value):
        if isinstance(value, bool) or not isinstance(value, int):
            return "not_whole"
        if self.at_least is not None and value < self.at_least:
            return "too_small"
        if self.at_most is not None and value > self.at_most:
            return "too_big"
        return None


class _Words:
    """String expectation.  Reasons, checked in this order:

    not_text   - value is not a str
    too_short  - len(value) < shortest
    too_long   - len(value) > longest
    no_match   - `pattern` given and re.fullmatch(pattern, value) fails
    lacks      - `contains` given and it is not a substring of value
    """

    def __init__(self, shortest, longest, pattern, contains):
        self.shortest, self.longest = shortest, longest
        self.pattern = re.compile(pattern) if pattern is not None else None
        self.contains = contains

    def inspect(self, value):
        if not isinstance(value, str):
            return "not_text"
        if len(value) < self.shortest:
            return "too_short"
        if self.longest is not None and len(value) > self.longest:
            return "too_long"
        if self.pattern is not None and not self.pattern.fullmatch(value):
            return "no_match"
        if self.contains is not None and self.contains not in value:
            return "lacks"
        return None


class expect:
    """Factories for expectations (used as expect.whole(...), expect.words(...))."""

    @staticmethod
    def whole(at_least=None, at_most=None):
        return _Whole(at_least, at_most)

    @staticmethod
    def words(shortest=0, longest=None, pattern=None, contains=None):
        return _Words(shortest, longest, pattern, contains)


class Checker:
    def __init__(self):
        self._rules = []

    def demand(self, field, expectation):
        """Field is required: absent or None gives the reason 'missing'."""
        self._rules.append((field, expectation, True))
        return self

    def allow(self, field, expectation):
        """Field is optional: absent or None is fine; anything else is inspected."""
        self._rules.append((field, expectation, False))
        return self

    def judge(self, record):
        problems = []
        for field, expectation, required in self._rules:
            value = record.get(field)
            if value is None:
                if required:
                    problems.append(Problem(field, "missing"))
                continue
            why = expectation.inspect(value)
            if why is not None:
                problems.append(Problem(field, why))
        return Verdict(problems)
