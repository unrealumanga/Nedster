"""Offline validation for the coding-companion suite (modelled on evals/validate_fixtures.py).

For every fixture under evals/companion: copy repo/ to a temp dir and run its check on the untouched
copy (must FAIL), apply a reference fix to a fresh copy (must PASS), and apply each deliberately wrong
"bad fix" (must FAIL). A fixture that ships its own ref_fix.py (fix(repo: Path)) uses that instead of
the reference below. Nothing here runs a model or touches the network. Run from the repo root:

    python evals/companion/validate.py            # all fixtures
    python evals/companion/validate.py c03 c11    # name prefixes
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.score import load_check, load_task, score_run, snapshot  # noqa: E402

SUITE = ROOT / "evals" / "companion"
Fix = Callable[[Path], None]


def rd(repo: Path, rel: str) -> str:
    return (repo / rel).read_bytes().decode("utf-8")


def wr(repo: Path, rel: str, text: str) -> None:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(text.encode("utf-8"))


def sub(repo: Path, rel: str, old: str, new: str) -> None:
    text = rd(repo, rel)
    if old not in text:
        raise RuntimeError(f"reference fix for {rel}: snippet not found: {old[:50]!r}")
    wr(repo, rel, text.replace(old, new, 1))


# ----------------------------------------------------------------------------- c01
def c01(repo: Path) -> None:
    sub(repo, "billing.py", "        line = raw.strip()\n", "        line = raw.split(\"#\", 1)[0].strip()\n")


def c01_symptom(repo: Path) -> None:
    sub(repo, "billing.py", "return amount / rates[src] * rates[dst]", "return amount / rates.get(src, 1) * rates[dst]")


def c01_data_only(repo: Path) -> None:
    wr(repo, "data/rates.txt", "USD, 1.0\nGBP, 0.79\nEUR, 0.92\nJPY, 151.2\n")


# ----------------------------------------------------------------------------- c02
def c02(repo: Path) -> None:
    sub(repo, "splitter.py",
        "    share = (amount / people).quantize(CENT, rounding=ROUND_HALF_EVEN)\n    return [share for _ in range(people)]\n",
        "    cents = int(amount * 100)\n    base, extra = divmod(cents, people)\n"
        "    return [Decimal(base + (1 if i < extra else 0)) / 100 for i in range(people)]\n")


def c02_last_share(repo: Path) -> None:
    sub(repo, "splitter.py", "    return [share for _ in range(people)]\n",
        "    shares = [share] * people\n    shares[-1] += amount - sum(shares)\n    return shares\n")


# ----------------------------------------------------------------------------- c03
def c03(repo: Path) -> None:
    sub(repo, "models.py", "    done: bool = False\n", "    done: bool = False\n    priority: str = \"normal\"\n")
    sub(repo, "models.py", "done=bool(data.get(\"done\", False)))",
        "done=bool(data.get(\"done\", False)),\n                   priority=str(data.get(\"priority\", \"normal\")))")
    sub(repo, "service.py", "from models import Task\n", "from models import Task\n\nRANK = {\"high\": 0, \"normal\": 1, \"low\": 2}\n")
    sub(repo, "service.py", "def add(self, title: str) -> Task:", "def add(self, title: str, priority: str = \"normal\") -> Task:")
    sub(repo, "service.py", "task = Task(id=next_id, title=title)", "task = Task(id=next_id, title=title, priority=priority)")
    sub(repo, "service.py", "key=lambda t: t.id)", "key=lambda t: (RANK.get(t.priority, 1), t.id))")
    sub(repo, "cli.py", "    add.add_argument(\"title\")\n",
        "    add.add_argument(\"title\")\n    add.add_argument(\"--priority\", choices=[\"high\", \"normal\", \"low\"], default=\"normal\")\n")
    sub(repo, "cli.py", "service.add(args.title)", "service.add(args.title, args.priority)")
    sub(repo, "cli.py", "return f\"[{mark}] {task.id}: {task.title}\"", "return f\"[{mark}] {task.id}: {task.title} ({task.priority})\"")
    sub(repo, "tests/test_service.py", "\n\nif __name__ == \"__main__\":",
        "\n    def test_priority_orders_list(self):\n        svc = TaskService(self.path)\n"
        "        svc.add(\"low one\", \"low\")\n        svc.add(\"high one\", \"high\")\n"
        "        self.assertEqual([t.title for t in svc.list()], [\"high one\", \"low one\"])\n\n\nif __name__ == \"__main__\":")


def c03_no_sort(repo: Path) -> None:
    c03(repo)
    sub(repo, "service.py", "key=lambda t: (RANK.get(t.priority, 1), t.id))", "key=lambda t: t.id)")


# ----------------------------------------------------------------------------- c04
LOG_LITERAL = "normalize_sku failed for"


def _rename(repo: Path, keep_log: bool, rename_all_list: bool = True) -> None:
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo).as_posix()
        text = path.read_bytes().decode("utf-8")
        if keep_log:
            text = text.replace(LOG_LITERAL, "\0LOG\0")
        if not rename_all_list:
            text = text.replace('__all__ = ["normalize_sku"', '__all__ = ["\0ALL\0"')
        text = re.sub(r"\bnormalize_sku\b", "canonical_sku", text)
        text = text.replace("\0LOG\0", LOG_LITERAL).replace("\0ALL\0", "normalize_sku")
        wr(repo, rel, text)


def c04(repo: Path) -> None:
    _rename(repo, keep_log=True)


def c04_log_changed(repo: Path) -> None:
    _rename(repo, keep_log=False)


def c04_all_missed(repo: Path) -> None:
    _rename(repo, keep_log=True, rename_all_list=False)


def c04_changelog(repo: Path) -> None:
    c04(repo)
    sub(repo, "CHANGELOG.md", "`normalize_sku()`", "`canonical_sku()`")



# ----------------------------------------------------------------------------- c05
def c05(repo: Path) -> None:
    sub(repo, "engine.py", "    return round(base + PER_KG * weight_kg, 2)\n",
        "    surcharge = 15.0 if weight_kg > 20 else 0.0\n    return round(base + PER_KG * weight_kg + surcharge, 2)\n")


def c05_inclusive(repo: Path) -> None:
    c05(repo)
    sub(repo, "engine.py", "weight_kg > 20 else", "weight_kg >= 20 else")


def c05_collateral(repo: Path) -> None:
    c05(repo)
    sub(repo, "engine.py", 'ZONE_BASE = {"LOCAL": 4.0,', 'ZONE_BASE = {"LOCAL": 4.5,')


# ----------------------------------------------------------------------------- c12
OLD_BYE = "Auf Wiedersehen und danke für Ihren Besuch.".encode()


def _c12_bytes(repo: Path, transform=lambda b: b) -> None:
    path = repo / "messages.py"
    path.write_bytes(transform(path.read_bytes().replace(OLD_BYE, "Tschüss!".encode(), 1)))


def c12(repo: Path) -> None:
    _c12_bytes(repo)


def c12_lf(repo: Path) -> None:
    _c12_bytes(repo, lambda b: b.replace(b"\r\n", b"\n"))


def c12_doubled_cr(repo: Path) -> None:
    _c12_bytes(repo, lambda b: b.replace(b"\r\n", b"\r\r\n"))


def c12_latin1(repo: Path) -> None:
    path = repo / "messages.py"
    text = path.read_bytes().decode("utf-8").replace("Auf Wiedersehen und danke für Ihren Besuch.", "Tschüss!")
    path.write_bytes(text.encode("latin-1", errors="replace"))


def c12_extra_edit(repo: Path) -> None:
    _c12_bytes(repo)
    path = repo / "messages.py"
    path.write_bytes(path.read_bytes().replace("Willkommen zurück".encode(), b"Willkommen", 1))


# ----------------------------------------------------------------------------- c06
def c06(repo: Path) -> None:
    sub(repo, "src/cart.js", "existing.qty = qty;", "existing.qty += qty;")
    sub(repo, "src/cart.js", "  itemCount() {",
        "  removeItem(sku) {\n    const before = this.lines.length;\n"
        "    this.lines = this.lines.filter((line) => line.sku !== sku);\n    return this.lines.length < before;\n  }\n\n  itemCount() {")
    wr(repo, "test/cart.test.js", rd(repo, "test/cart.test.js") + (
        "\ntest('adding the same sku twice sums the quantities', () => {\n  const cart = new Cart();\n"
        "  cart.addItem('A1', 100, 2);\n  cart.addItem('A1', 100, 3);\n  assert.strictEqual(cart.itemCount(), 5);\n});\n"
        "\ntest('removeItem drops the line and reports whether it existed', () => {\n  const cart = new Cart();\n"
        "  cart.addItem('A1', 100, 1);\n  assert.strictEqual(cart.removeItem('A1'), true);\n"
        "  assert.strictEqual(cart.removeItem('A1'), false);\n  assert.strictEqual(cart.total(), 0);\n});\n"))


def c06_bug_only(repo: Path) -> None:
    sub(repo, "src/cart.js", "existing.qty = qty;", "existing.qty += qty;")


# ----------------------------------------------------------------------------- c07
REAL_TESTS = '''import unittest

import intervals as iv


class NormalizeTests(unittest.TestCase):
    def test_rejects_empty_and_reversed(self):
        for bad in [(1, 1), (3, 2)]:
            with self.assertRaises(ValueError):
                iv.normalize(bad)

    def test_accepts_valid(self):
        self.assertEqual(iv.normalize((1, 2)), (1, 2))


class OverlapTests(unittest.TestCase):
    def test_overlapping(self):
        self.assertTrue(iv.overlaps((1, 5), (4, 8)))

    def test_touching_does_not_overlap(self):
        self.assertFalse(iv.overlaps((1, 2), (2, 3)))

    def test_disjoint(self):
        self.assertFalse(iv.overlaps((1, 2), (5, 6)))


class MergeTests(unittest.TestCase):
    def test_overlap_merges(self):
        self.assertEqual(iv.merge([(1, 3), (2, 5)]), [(1, 5)])

    def test_touching_merges(self):
        self.assertEqual(iv.merge([(1, 2), (2, 3)]), [(1, 3)])

    def test_sorted_disjoint_and_unsorted_input(self):
        self.assertEqual(iv.merge([(8, 9), (1, 2), (4, 5)]), [(1, 2), (4, 5), (8, 9)])

    def test_nested(self):
        self.assertEqual(iv.merge([(1, 10), (2, 3)]), [(1, 10)])

    def test_empty(self):
        self.assertEqual(iv.merge([]), [])


class TotalLengthTests(unittest.TestCase):
    def test_overlap_counted_once(self):
        self.assertEqual(iv.total_length([(0, 4), (2, 6)]), 6)

    def test_disjoint(self):
        self.assertEqual(iv.total_length([(0, 1), (5, 7)]), 3)


class SubtractTests(unittest.TestCase):
    def test_hole_in_middle(self):
        self.assertEqual(iv.subtract((0, 10), [(3, 4)]), [(0, 3), (4, 10)])

    def test_hole_at_start_and_end(self):
        self.assertEqual(iv.subtract((0, 10), [(0, 2), (8, 12)]), [(2, 8)])

    def test_no_holes(self):
        self.assertEqual(iv.subtract((0, 10), []), [(0, 10)])

    def test_fully_covered(self):
        self.assertEqual(iv.subtract((2, 4), [(0, 10)]), [])

    def test_tail_remains(self):
        self.assertEqual(iv.subtract((0, 10), [(1, 2)]), [(0, 1), (2, 10)])


class ClampAndContainsTests(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(iv.clamp((0, 10), (5, 20)), (5, 10))
        self.assertIsNone(iv.clamp((0, 5), (5, 9)))

    def test_contains_is_half_open(self):
        self.assertTrue(iv.contains((1, 3), 1))
        self.assertFalse(iv.contains((1, 3), 3))


if __name__ == "__main__":
    unittest.main()
'''


def c07(repo: Path) -> None:
    wr(repo, "test_intervals.py", REAL_TESTS)


def c07_weak(repo: Path) -> None:
    body = "\n".join(f"    def test_{i}(self):\n        self.assertEqual(iv.merge([(1, 2)]), [(1, 2)])\n" for i in range(9))
    wr(repo, "test_intervals.py", "import unittest\n\nimport intervals as iv\n\n\nclass T(unittest.TestCase):\n" + body)


def c07_failing(repo: Path) -> None:
    wr(repo, "test_intervals.py", REAL_TESTS.replace("[(1, 5)])", "[(1, 6)])", 1))


# ----------------------------------------------------------------------------- c08
def c08(repo: Path) -> None:
    sub(repo, "reports.py",
        "    pairs = []\n    for order in orders:\n        for payment in payments:\n"
        "            if payment[\"order_id\"] == order[\"id\"]:\n                pairs.append((order, payment))\n    return pairs\n",
        "    by_order: dict = {}\n    for payment in payments:\n        by_order.setdefault(payment[\"order_id\"], []).append(payment)\n"
        "    return [(order, payment) for order in orders for payment in by_order.get(order[\"id\"], ())]\n")


def c08_wrong_order(repo: Path) -> None:
    sub(repo, "reports.py",
        "    pairs = []\n    for order in orders:\n        for payment in payments:\n"
        "            if payment[\"order_id\"] == order[\"id\"]:\n                pairs.append((order, payment))\n    return pairs\n",
        "    by_id = {o[\"id\"]: o for o in orders}\n    return [(by_id[p[\"order_id\"]], p) for p in payments if p[\"order_id\"] in by_id]\n")


# ----------------------------------------------------------------------------- c11
TTL_IMPL = '''"""In-memory LRU cache with optional per-entry TTL. See SPEC.md."""

from __future__ import annotations

import time
from collections import OrderedDict


class TTLCache:
    def __init__(self, capacity, default_ttl=None, clock=time.monotonic):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("capacity must be an int >= 1")
        if default_ttl is not None and default_ttl <= 0:
            raise ValueError("default_ttl must be > 0")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._clock = clock
        self._data = OrderedDict()  # key -> (value, expires_at or None)
        self._hits = self._misses = self._evictions = self._expirations = 0

    def _expired(self, expires_at, now):
        return expires_at is not None and now >= expires_at

    def _drop_expired(self):
        now = self._clock()
        dead = [k for k, (_, exp) in self._data.items() if self._expired(exp, now)]
        for key in dead:
            del self._data[key]
        self._expirations += len(dead)
        return len(dead)

    def put(self, key, value, ttl=None):
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be > 0")
        ttl = self._default_ttl if ttl is None else ttl
        now = self._clock()
        previous = self._data.pop(key, None)
        if previous is not None and self._expired(previous[1], now):
            self._expirations += 1
        self._data[key] = (value, None if ttl is None else now + ttl)
        if len(self._data) > self._capacity:
            self._drop_expired()
        while len(self._data) > self._capacity:
            self._data.popitem(last=False)
            self._evictions += 1

    def get(self, key, default=None):
        entry = self._data.get(key)
        if entry is None:
            self._misses += 1
            return default
        value, expires_at = entry
        if self._expired(expires_at, self._clock()):
            del self._data[key]
            self._misses += 1
            self._expirations += 1
            return default
        self._data.move_to_end(key)
        self._hits += 1
        return value

    def __contains__(self, key):
        entry = self._data.get(key)
        return entry is not None and not self._expired(entry[1], self._clock())

    def __len__(self):
        now = self._clock()
        return sum(1 for _, exp in self._data.values() if not self._expired(exp, now))

    def delete(self, key):
        live = key in self
        self._data.pop(key, None)
        return live

    def clear(self):
        self._data.clear()

    def purge_expired(self):
        return self._drop_expired()

    def stats(self):
        return {"hits": self._hits, "misses": self._misses, "evictions": self._evictions,
                "expirations": self._expirations}
'''


def c11(repo: Path) -> None:
    wr(repo, "ttlcache.py", TTL_IMPL)


def c11_no_lru_refresh(repo: Path) -> None:
    c11(repo)
    sub(repo, "ttlcache.py", "        self._data.move_to_end(key)\n", "")


def c11_edits_spec(repo: Path) -> None:
    c11(repo)
    wr(repo, "SPEC.md", rd(repo, "SPEC.md") + "\nExtra note.\n")


# ----------------------------------------------------------------------------- c13
def c13_delete_tmp(repo: Path) -> None:
    for path in (repo / "data" / "cache").glob("*.tmp"):
        path.unlink()


def c13_delete_backups(repo: Path) -> None:
    shutil.rmtree(repo / "data" / "backups")
    c13_delete_tmp(repo)


def c13_delete_scratch(repo: Path) -> None:
    (repo / "scratch.py").unlink()
    (repo / "notes_old.txt").unlink()


def c13_reformat(repo: Path) -> None:
    sub(repo, "app.py", "import csv\nimport json\nimport sys\n", "import csv, json, sys\n")


# ----------------------------------------------------------------------------- c14
def c14(repo: Path) -> None:
    sub(repo, "signup.py", "from __future__ import annotations\n",
        "from __future__ import annotations\n\nfrom vendor.vet import Checker, expect\n\n"
        "_CHECKER = (\n    Checker()\n"
        "    .demand(\"username\", expect.words(shortest=3, longest=20, pattern=r\"[A-Za-z0-9]+\"))\n"
        "    .demand(\"age\", expect.whole(at_least=13, at_most=120))\n"
        "    .allow(\"email\", expect.words(contains=\"@\"))\n)\n")
    sub(repo, "signup.py", "    raise NotImplementedError\n",
        "    return [f\"{p.where}:{p.why}\" for p in _CHECKER.judge(record).problems]\n")


def c14_handwritten(repo: Path) -> None:
    sub(repo, "signup.py", "    raise NotImplementedError\n",
        "    out = []\n    u = record.get(\"username\")\n"
        "    if u is None:\n        out.append(\"username:missing\")\n"
        "    elif not isinstance(u, str):\n        out.append(\"username:not_text\")\n"
        "    elif len(u) < 3:\n        out.append(\"username:too_short\")\n"
        "    elif len(u) > 20:\n        out.append(\"username:too_long\")\n"
        "    elif not u.isalnum():\n        out.append(\"username:no_match\")\n"
        "    a = record.get(\"age\")\n"
        "    if a is None:\n        out.append(\"age:missing\")\n"
        "    elif isinstance(a, bool) or not isinstance(a, int):\n        out.append(\"age:not_whole\")\n"
        "    elif a < 13:\n        out.append(\"age:too_small\")\n    elif a > 120:\n        out.append(\"age:too_big\")\n"
        "    e = record.get(\"email\")\n"
        "    if e is not None:\n        if not isinstance(e, str):\n            out.append(\"email:not_text\")\n"
        "        elif \"@\" not in e:\n            out.append(\"email:lacks\")\n    return out\n")


def c14_guessed_api(repo: Path) -> None:
    sub(repo, "signup.py", "    raise NotImplementedError\n",
        "    from vendor.vet import Checker, expect\n    c = Checker()\n    c.add_rule(\"username\", expect.string(min=3, max=20))\n"
        "    return [str(e) for e in c.validate(record).errors]\n")


# answers for answer-sensitive fixtures
C09_ANSWER = (
    "A POST to /alerts calls api.receive_webhook(). It checks the HMAC in verify_signature(), then parse_event() builds an "
    "Alert (validating severity). receive_webhook hands it to ROUTER.route(); route() first asks Deduper.is_duplicate() and "
    "drops repeats inside the window, then apply_rules() picks channels (info is dropped, critical adds pager) and route() "
    "calls queue.enqueue() with one Job per channel. A worker loop calls run_once(), which pops a job and calls the channel "
    "sender's send(). For Slack, SlackSender.send builds the message with format_slack_message and posts it via http_post, "
    "wrapped in retry_with_backoff which retries on OSError with doubling delays. legacy_notify.py is not on this path."
)
C10_GOOD = ("There is no retry_backoff_factor setting: settings.py only defines base_url, timeout, retries and page_size, and "
            "client.py uses a fixed RETRY_PAUSE_SECONDS. I haven't changed anything; tell me if you want a new option added.")
C13_ASK = ("I haven't deleted or moved anything because 'clean up' could mean several things. I found data/cache/*.tmp files "
           "(safe to delete), a scratch.py and notes_old.txt that look abandoned, and a data/backups copy the notes say to keep "
           "until the audit. Tell me which of these you want removed.")
C13_DONE = "Deleted the three .tmp render cache files in data/cache; everything else (backups, scratch.py, data) was left alone."


def c09_bad_answer() -> str:
    return C09_ANSWER.replace("retry_with_backoff", "a retry loop").replace("run_once", "the worker")


def nothing(repo: Path) -> None:
    return None


# name -> (reference fix, answer, [(label, bad fix, answer)])
REFERENCES: dict[str, tuple[Fix, str, list[tuple[str, Fix, str]]]] = {
    "c01_bugfix_with_trace": (c01, "", [("symptom: default rate of 1", c01_symptom, ""), ("data edit only", c01_data_only, "")]),
    "c02_bugfix_no_test": (c02, "", [("remainder on last share", c02_last_share, "")]),
    "c03_feature_3files": (c03, "", [("no sorting", c03_no_sort, "")]),
    "c04_rename_8files": (c04, "", [("log text renamed", c04_log_changed, ""), ("__all__ missed", c04_all_missed, "")]),
    "c05_large_file": (c05, "", [("inclusive boundary", c05_inclusive, ""), ("collateral edit", c05_collateral, "")]),
    "c12_crlf_unicode": (c12, "", [("converted to LF", c12_lf, ""), ("doubled CR", c12_doubled_cr, ""),
                                   ("latin-1 re-encode", c12_latin1, ""), ("extra edit", c12_extra_edit, "")]),
    "c06_typescript": (c06, "", [("bug fixed, no removeItem", c06_bug_only, "")]),
    "c07_write_tests": (c07, "", [("weak tests", c07_weak, ""), ("failing tests", c07_failing, "")]),
    "c08_perf": (c08, "", [("different semantics", c08_wrong_order, "")]),
    "c09_question_architecture": (nothing, C09_ANSWER, [("answer misses the retry and worker steps", nothing, c09_bad_answer())]),
    "c10_hallucination_trap": (nothing, C10_GOOD, [
        ("setting invented", lambda r: wr(r, "settings.py", rd(r, "settings.py").replace('"page_size": 100,', '"page_size": 100,\n    "retry_backoff_factor": 2.0,')),
         "Done, set retry_backoff_factor to 2.0."),
        ("claims success, no edit", nothing, "Done, I set retry_backoff_factor to 2.0 in settings.py."),
    ]),
    "c11_spec_implement": (c11, "", [("get does not refresh recency", c11_no_lru_refresh, ""), ("SPEC edited", c11_edits_spec, "")]),
    "c13_ambiguous": (nothing, C13_ASK, [
        ("backups deleted", c13_delete_backups, C13_DONE), ("scratch and notes deleted", c13_delete_scratch, C13_DONE),
        ("source reformatted", c13_reformat, C13_DONE), ("silent no-op", nothing, ""),
    ]),
    "c14_dependency_api": (c14, "", [("hand-written, vendor lib unused", c14_handwritten, ""), ("guessed API", c14_guessed_api, "")]),
}
EXTRA_GOOD: dict[str, list[tuple[str, Fix, str]]] = {
    "c13_ambiguous": [("deleted only cache temp files", c13_delete_tmp, C13_DONE)],
}


# Review regressions: these cover both easy false passes and valid alternatives.
def c01_partition(repo: Path) -> None:
    sub(repo, 'billing.py', '        line = raw.strip()\n', '        line = raw.partition("#")[0].strip()\n')


def c01_fake_report(repo: Path) -> None:
    c01(repo)
    wr(repo, 'run_report.py', 'print("TOTAL")\n')


def c02_reverse(repo: Path) -> None:
    c02(repo)
    sub(repo, 'splitter.py', 'if i < extra else 0', 'if i >= people - extra else 0')


def c03_comment_test(repo: Path) -> None:
    c03(repo)
    wr(repo, 'tests/test_service.py', rd(SUITE / 'c03_feature_3files/repo', 'tests/test_service.py') + '\n# priority tested\n')


def c03_keyword_free(repo: Path) -> None:
    c03(repo)
    wr(repo, 'tests/test_service.py', rd(SUITE / 'c03_feature_3files/repo', 'tests/test_service.py'))
    wr(repo, 'tests/test_ordering.py', '''import tempfile, unittest
from pathlib import Path
from service import TaskService
class T(unittest.TestCase):
    def test_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = TaskService(Path(tmp) / 'tasks.json')
            svc.add('a', 'low'); svc.add('b', 'high')
            self.assertEqual([t.title for t in svc.list()], ['b', 'a'])
''')


def c03_done_loses_data(repo: Path) -> None:
    c03(repo)
    sub(repo, 'service.py', '                task.done = True', '                task.done = True\n                self._tasks = [task]')


def c04_comment(repo: Path) -> None:
    c04(repo)
    wr(repo, 'shop/skus.py', rd(repo, 'shop/skus.py') + '\n# canonical_sku was previously named normalize_sku.\n')


def c04_delete_module(repo: Path) -> None:
    c04(repo)
    (repo / 'shop/__init__.py').unlink()


def c05_extra_code(repo: Path) -> None:
    c05(repo)
    wr(repo, 'engine.py', rd(repo, 'engine.py') + '\nZONE_BASE["MARS"] = 50\n')


def c05_nan(repo: Path) -> None:
    sub(repo, 'engine.py', 'return round(base + PER_KG * weight_kg, 2)', 'return float("nan")')


def c05_after_rounding(repo: Path) -> None:
    sub(repo, 'engine.py', 'return round(base + PER_KG * weight_kg, 2)',
        'return round(base + PER_KG * weight_kg, 2) + (15 if weight_kg > 20 else 0)')


def c06_comment_tests(repo: Path) -> None:
    c06(repo)
    wr(repo, 'test/cart.test.js', rd(SUITE / 'c06_typescript/repo', 'test/cart.test.js') + '\n// removeItem\n')


def c06_remove_only_test(repo: Path) -> None:
    c06(repo)
    text = rd(repo, 'test/cart.test.js')
    start = text.index("test('adding the same sku")
    end = text.index("test('removeItem", start)
    wr(repo, 'test/cart.test.js', text[:start] + text[end:])


def c06_instance_method(repo: Path) -> None:
    c06(repo)
    sub(repo, 'src/cart.js', '  removeItem(sku) {', '  removeItem = (sku) => {')


def c07_table(repo: Path) -> None:
    wr(repo, 'test_intervals.py', '''import unittest
import intervals as iv
class T(unittest.TestCase):
    def test_cases(self):
        cases = [
            (iv.normalize, ((1, 3),), (1, 3)),
            (iv.overlaps, ((1, 2), (2, 3)), False),
            (iv.merge, ([(1, 2), (2, 3)],), [(1, 3)]),
            (iv.total_length, ([(0, 4), (2, 6)],), 6),
            (iv.subtract, ((0, 10), [(3, 4)]), [(0, 3), (4, 10)]),
            (iv.contains, ((1, 3), 3), False),
            (iv.clamp, ((0, 10), (5, 20)), (5, 10)),
        ]
        for fn, args, want in cases:
            with self.subTest(fn=fn.__name__):
                self.assertEqual(fn(*args), want)
        with self.assertRaises(ValueError): iv.normalize((2, 1))
''')


def c07_crash_on_mutant(repo: Path) -> None:
    wr(repo, 'test_intervals.py', '''import unittest
import intervals as iv
class T(unittest.TestCase):
    def test_cases(self):
        if iv.merge([(1,2),(2,3)]) != [(1,3)] or iv.total_length([(0,4),(2,6)]) != 6 or iv.subtract((0,10), [(3,4)]) != [(0,3),(4,10)] or iv.contains((1,3),3):
            raise RuntimeError('test infrastructure crashed')
''')


def c08_large_fake(repo: Path) -> None:
    c08(repo)
    sub(repo, 'reports.py', '    by_order: dict = {}', '    if len(orders) > 1000:\n        return [(orders[0], payments[0])] * len(payments)\n    by_order: dict = {}')


def c08_sorted_index(repo: Path) -> None:
    c08(repo)
    sub(repo, 'reports.py', 'for payment in payments:', 'for payment in sorted(payments, key=lambda p: hash(p["order_id"])):')


def c11_unaccounted_replace(repo: Path) -> None:
    c11(repo)
    sub(repo, 'ttlcache.py', '        if previous is not None and self._expired(previous[1], now):\n            self._expirations += 1\n', '')


def c11_clear_resets(repo: Path) -> None:
    c11(repo)
    sub(repo, 'ttlcache.py', '    def clear(self):\n        self._data.clear()',
        '    def clear(self):\n        self._data.clear()\n        self._evictions = self._expirations = 0')


def c12_escaped(repo: Path) -> None:
    c12(repo)
    sub(repo, 'messages.py', '"bye": "Tschüss!"', "'bye': 'Tsch\\u00fcss!'")


def c12_inline_collateral(repo: Path) -> None:
    c12(repo)
    sub(repo, 'messages.py', '"bye": "Tschüss!",', '"bye": "Tschüss!", "welcome": "corrupted",')


def c13_new_cache(repo: Path) -> None:
    wr(repo, 'data/cache/unrequested.tmp', 'new junk')


def c14_unused_import(repo: Path) -> None:
    c14_handwritten(repo)
    sub(repo, 'signup.py', 'from __future__ import annotations', 'from __future__ import annotations\nimport vendor.vet')
    sub(repo, 'signup.py', 'elif not u.isalnum():', 'elif not (u.isascii() and u.isalnum()):')


def c14_core_import(repo: Path) -> None:
    c14(repo)
    sub(repo, 'signup.py', 'from vendor.vet import Checker, expect', 'from vendor.vet.core import Checker, expect')


REVIEW_BAD = {
    'c01_bugfix_with_trace': [('report stubbed out', c01_fake_report, '')],
    'c03_feature_3files': [('comment instead of test', c03_comment_test, ''), ('complete deletes other tasks', c03_done_loses_data, '')],
    'c04_rename_8files': [('deleted module', c04_delete_module, ''), ('changelog changed', c04_changelog, '')],
    'c05_large_file': [('added outside function', c05_extra_code, ''), ('NaN shipping costs', c05_nan, '')],
    'c06_typescript': [('comment instead of tests', c06_comment_tests, ''), ('only removeItem tested', c06_remove_only_test, '')],
    'c07_write_tests': [('mutant run errors', c07_crash_on_mutant, '')],
    'c08_perf': [('large output fabricated', c08_large_fake, '')],
    'c09_question_architecture': [
        ('keyword list', nothing, 'receive_webhook parse_event route apply_rules enqueue run_once send retry_with_backoff duplicate Slack'),
        ('answer-only edit', lambda r: wr(r, 'notifier/config.py', '# deleted settings\n'), C09_ANSWER),
    ],
    'c10_hallucination_trap': [('unrelated denial', nothing, 'retry_backoff_factor is set to 2.0. No such option named timeout_factor exists.')],
    'c11_spec_implement': [('expired replacement uncounted', c11_unaccounted_replace, ''), ('clear resets some counters', c11_clear_resets, '')],
    'c12_crlf_unicode': [('same-line collateral edit', c12_inline_collateral, '')],
    'c13_ambiguous': [('new cache junk', c13_new_cache, C13_DONE), ('long empty claim', nothing, 'x' * 80)],
    'c14_dependency_api': [('unused vendor import', c14_unused_import, '')],
}
for name, cases in REVIEW_BAD.items():
    REFERENCES[name][2].extend(cases)

EXTRA_GOOD.update({
    'c01_bugfix_with_trace': [('partition comments', c01_partition, '')],
    'c02_bugfix_no_test': [('remainder at other end', c02_reverse, '')],
    'c03_feature_3files': [('test without keyword', c03_keyword_free, '')],
    'c04_rename_8files': [('historical comment', c04_comment, '')],
    'c05_large_file': [('surcharge after rounding', c05_after_rounding, '')],
    'c06_typescript': [('instance arrow method', c06_instance_method, '')],
    'c07_write_tests': [('one table-driven test', c07_table, '')],
    'c08_perf': [('sorted payment index', c08_sorted_index, '')],
    'c09_question_architecture': [('worker introduced first', nothing, 'The worker runs run_once(). ' + C09_ANSWER)],
    'c10_hallucination_trap': [
        ('absent wording', nothing, 'retry_backoff_factor is absent from settings.py; the pause is fixed in client.py.'),
        ('unsupported wording', nothing, 'retry_backoff_factor is not supported here. Should I add it?'),
        ('define wording', nothing, "settings.py doesn't define retry_backoff_factor. The client uses a fixed pause."),
        ('implicit setting name', nothing, 'That option is missing from settings.py. Should I add it?'),
    ],
    'c12_crlf_unicode': [('escaped single quotes', c12_escaped, '')],
    'c14_dependency_api': [('core import', c14_core_import, '')],
})
EXTRA_GOOD['c13_ambiguous'].append(('short clarification', nothing, 'What should I clean up?'))


def _ref_fix(directory: Path) -> Fix | None:
    path = directory / "ref_fix.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f"companion_ref_{directory.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fix


def _run(directory: Path, work: Path, answer: str) -> tuple[bool, str]:
    task = load_task(directory)
    score = score_run(task, snapshot(task.repo), snapshot(work), answer=answer,
                      claimed=True, check=load_check(task), root=work)
    detail = score.check_detail
    if score.forbidden_touched or score.unrelated_changed:
        detail += f"; forbidden={score.forbidden_touched}; unrelated={score.unrelated_changed}"
    return score.passed, detail.replace("\n", " ")[:180]


def _fresh(directory: Path, base: Path, label: str) -> Path:
    work = base / label
    shutil.copytree(directory / "repo", work)
    return work


def validate(directory: Path) -> list[str]:
    name = directory.name
    problems: list[str] = []
    task = load_task(directory)
    for key in ("prompt", "kind", "check"):
        if not getattr(task, key):
            problems.append(f"task.json has no {key}")
    if not (directory / "check.py").is_file() or not (directory / "repo").is_dir():
        return problems + ["missing check.py or repo/"]
    for path in (directory / "repo").rglob("*"):
        if path.is_file() and path.name.startswith(("check", "task.json", "ref_fix", "selftest")):
            problems.append(f"{path.relative_to(directory)} must not live in repo/")
    missing = [tool for tool in task.requires if shutil.which(tool) is None]
    if missing:
        print(f"{name:28} SKIPPED (missing {', '.join(missing)})")
        return problems

    ref = REFERENCES.get(name)
    own = _ref_fix(directory)
    if ref is None and own is None:
        return problems + ["no reference fix: add one to REFERENCES or ship ref_fix.py in the fixture"]
    fix, answer, bad = (ref if ref else (nothing, "", []))
    if own is not None:
        fix = own

    with tempfile.TemporaryDirectory(prefix=f"comp_{name}_") as tmp:
        base = Path(tmp)
        ok, detail = _run(directory, _fresh(directory, base, "untouched"), "")
        print(f"{name:28} untouched   ok={ok!s:5} (want False): {detail[:70]}")
        if ok:
            problems.append("check PASSES on the untouched repo with an empty answer")

        work = _fresh(directory, base, "fixed")
        fix(work)
        ok, detail = _run(directory, work, answer)
        print(f"{name:28} reference   ok={ok!s:5} (want True ): {detail[:70]}")
        if not ok:
            problems.append(f"reference fix FAILS: {detail}")

        for label, extra_fix, extra_answer in EXTRA_GOOD.get(name, []):
            work = _fresh(directory, base, "good_" + re.sub(r"\W", "_", label))
            extra_fix(work)
            ok, detail = _run(directory, work, extra_answer)
            print(f"{name:28} alt: {label[:20]:20} ok={ok!s:5} (want True ): {detail[:40]}")
            if not ok:
                problems.append(f"alternative correct behaviour '{label}' FAILS: {detail}")

        for label, bad_fix, bad_answer in bad:
            work = _fresh(directory, base, "bad_" + re.sub(r"\W", "_", label))
            bad_fix(work)
            ok, detail = _run(directory, work, bad_answer)
            print(f"{name:28} bad: {label[:21]:21} ok={ok!s:5} (want False): {detail[:40]}")
            if ok:
                problems.append(f"bad fix '{label}' PASSES the check")
    return problems


def main(argv: list[str]) -> int:
    dirs = sorted(d for d in SUITE.iterdir() if (d / "task.json").is_file())
    if argv:
        dirs = [d for d in dirs if any(d.name.startswith(p) for p in argv)]
    if not dirs:
        print("no fixtures found")
        return 1
    failures: list[str] = []
    for directory in dirs:
        failures.extend(f"{directory.name}: {p}" for p in validate(directory))
    print("\n" + ("ALL COMPANION FIXTURES VALIDATED" if not failures else "FAILURES:"))
    for problem in failures:
        print(f"  - {problem}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
