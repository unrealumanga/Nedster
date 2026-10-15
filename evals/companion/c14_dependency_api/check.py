from __future__ import annotations

from pathlib import Path

from evals.companion import helpers

HIDDEN = r'''
import sys
import signup
import copy

def v(record):
    before = copy.deepcopy(record)
    out = signup.validate_signup(record)
    assert record == before, 'validation mutated the input'
    assert isinstance(out, list) and all(isinstance(s, str) for s in out), out
    return sorted(out)

good = {"username": "ada99", "age": 30, "email": "a@b.c"}
cases = [
    (good, []),
    ({"username": "ada", "age": 13}, []),
    ({"username": "A" * 20, "age": 120, "email": None}, []),
    ({}, ["age:missing", "username:missing"]),
    ({**good, "username": "ab"}, ["username:too_short"]),
    ({**good, "username": "a" * 21}, ["username:too_long"]),
    ({**good, "username": "bad name"}, ["username:no_match"]),
    ({**good, "username": 12345}, ["username:not_text"]),
    ({**good, "age": 12}, ["age:too_small"]),
    ({**good, "age": 121}, ["age:too_big"]),
    ({**good, "age": "30"}, ["age:not_whole"]),
    ({**good, "age": 30.0}, ["age:not_whole"]),
    ({**good, "age": True}, ["age:not_whole"]),
    ({**good, "email": "nope"}, ["email:lacks"]),
    ({**good, "email": ""}, ["email:lacks"]),
    ({**good, "email": 5}, ["email:not_text"]),
    ({"username": "x!", "age": 5, "email": "q"}, ["age:too_small", "email:lacks", "username:too_short"]),
    ({"age": None, "username": "ok_name"}, ["age:missing", "username:no_match"]),
    ({**good, "username": "ada\n"}, ["username:no_match"]),
    ({**good, "username": "\u00e9mile"}, ["username:no_match"]),
    ({**good, "email": "@"}, []),
]
for record, want in cases:
    got = v(record)
    assert got == want, f"validate_signup({record!r}) -> {got}, expected {want}"

assert signup.handle_signup(good) == {"ok": True, "user": "ada99"}
assert signup.handle_signup({"username": "ab", "age": 30})["ok"] is False
used = [m for m in sys.modules if m == "vendor.vet" or m.startswith("vendor.vet.")]
assert used, "the vendored vet library was not used"
# Verify that returned reasons actually come from vet, rather than an unused
# import followed by hand-written validation. Both Checker and direct public
# expectation.inspect usage are allowed.
from vendor.vet import expect
for factory, bad_record, reason_code, wanted in (
    (expect.words, {**good, 'username': 'ab'}, 'too_long', 'username:too_long'),
    (expect.whole, {**good, 'age': 12}, 'too_big', 'age:too_big'),
):
    cls = type(factory())
    inspect = cls.inspect
    def tagged(self, value, original=inspect, replacement=reason_code):
        reason = original(self, value)
        return replacement if reason is not None else None
    cls.inspect = tagged
    try:
        assert v(bad_record) == [wanted], 'validation did not consume vet results'
    finally:
        cls.inspect = inspect
print("ok")
'''


def check(root: Path, answer: str) -> tuple[bool, str]:
    for rel, content in helpers.tree(helpers.original(__file__, "vendor")).items():
        if (root / "vendor" / rel).read_bytes() != content:
            return False, f"vendor/{rel} was modified"
    ok, detail = helpers.run_py(root, HIDDEN)
    return (True, "validates through the vendored library") if ok else (False, detail)
