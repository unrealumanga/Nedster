from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from evals.companion import helpers

# (name, old, new): each mutant breaks one documented behaviour of the original module.
MUTANTS = [
    ("touching_not_merged", "if merged and start <= merged[-1][1]:", "if merged and start < merged[-1][1]:"),
    ("overlap_counted_twice", "return sum(end - start for start, end in merge(intervals))",
     "return sum(end - start for start, end in intervals)"),
    ("subtract_drops_tail", "    if cursor < end:\n        result.append((cursor, end))\n    return result",
     "    return result"),
    ("contains_right_closed", "return start <= point < end", "return start <= point <= end"),
]
MUST_KILL = len(MUTANTS)


def check(root: Path, answer: str) -> tuple[bool, str]:
    original = helpers.original(__file__, "intervals.py").read_text(encoding="utf-8")
    if (root / "intervals.py").read_text(encoding="utf-8") != original:
        return False, "intervals.py must remain unchanged"
    if not any(root.rglob("test_*.py")):
        return False, "no test_*.py file was written"
    result, detail = helpers.unittest_result(root)
    if not result or not result['ok']:
        return False, f"the new tests fail on the real module: {detail}"
    if result['ran'] <= result['skipped'] + result['expected']:
        return False, "no successful tests ran"

    killed, survived = [], []
    for name, old, new in MUTANTS:
        if original.count(old) != 1:
            raise RuntimeError(f"mutant {name} no longer applies to the original module")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "repo"
            shutil.copytree(root, work, ignore=shutil.ignore_patterns("__pycache__", ".nedster"))
            (work / "intervals.py").write_text(original.replace(old, new), encoding="utf-8")
            mutated, _ = helpers.unittest_result(work)
        kill = mutated and mutated['failures'] > 0 and mutated['errors'] == 0
        (killed if kill else survived).append(name)
    if len(killed) < MUST_KILL:
        return False, f"tests are too weak: killed {killed}, mutants survived {survived}"
    return True, f"{result['ran']} tests pass; killed mutants {killed}"
