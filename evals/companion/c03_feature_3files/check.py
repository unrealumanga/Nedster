from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from evals.companion import helpers


def _cli(root: Path, db: Path, *args: str) -> tuple[bool, str]:
    return helpers.run_argv(root, [sys.executable, "-B", "cli.py", "--db", str(db), *args])


def check(root: Path, answer: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.json"
        for title, pri in (("zeta-low", "low"), ("mid-default", None), ("alpha-high", "high"), ("beta-normal", "normal"),
                           ("omega-high", "high")):
            args = ["add", title] + (["--priority", pri] if pri else [])
            ok, out = _cli(root, db, *args)
            if not ok:
                return False, f"add {title} failed: {out}"
        ok, out = _cli(root, db, "list")
        if not ok:
            return False, f"list failed: {out}"
        order = [t for t in ("alpha-high", "omega-high", "mid-default", "beta-normal", "zeta-low") if t in out]
        pos = [out.index(t) for t in order]
        if len(order) != 5 or pos != sorted(pos):
            return False, f"list not ordered high -> normal -> low (ties by id): {out!r}"
        ok, out = _cli(root, db, "add", "bad", "--priority", "urgent")
        if ok:
            return False, "an invalid priority must be rejected with a non-zero exit"
        data = json.loads(db.read_text(encoding="utf-8"))
        expected = {"zeta-low": "low", "mid-default": "normal", "alpha-high": "high",
                    "beta-normal": "normal", "omega-high": "high"}
        if len(data) != 5 or {d['title']: d.get('priority', 'normal') for d in data} != expected:
            return False, f"task data or priorities were lost/corrupted: {data}"
        ids = {d['title']: d['id'] for d in data}
        ok, detail = _cli(root, db, "done", str(ids['alpha-high']))
        ok2, opened = _cli(root, db, "list", "--open")
        if not ok or not ok2 or 'alpha-high' in opened or any(t not in opened for t in expected if t != 'alpha-high'):
            return False, f"done/list --open regressed: {detail} {opened}"
        saved = json.loads(db.read_text(encoding='utf-8'))
        if {d['title']: d.get('priority', 'normal') for d in saved} != expected:
            return False, "completing a task discarded priorities"

        legacy = Path(tmp) / "old.json"
        legacy.write_text(json.dumps([{"id": 1, "title": "old-one", "done": False},
                                      {"id": 2, "title": "old-two", "done": True}]), encoding="utf-8")
        ok, out = _cli(root, legacy, "list")
        if not ok or "old-one" not in out or "old-two" not in out:
            return False, f"legacy task file (no priority) no longer lists: {out}"
        ok, out = _cli(root, legacy, "add", "new-high", "--priority", "high")
        ok2, out2 = _cli(root, legacy, "list")
        if not ok or not ok2 or not all(t in out2 for t in ('new-high', 'old-one', 'old-two')) or out2.index("new-high") > out2.index("old-one"):
            return False, f"new high task should list before legacy normal ones: {out2!r}"
        saved = json.loads(legacy.read_text(encoding='utf-8'))
        if len(saved) != 3 or not next(d for d in saved if d['title'] == 'old-two')['done']:
            return False, "legacy task data was lost when saving priorities"

    result, detail = helpers.unittest_result(root)
    if not result or not result['ok'] or result['ran'] <= result['skipped']:
        return False, f"tests must run and pass: {detail}"
    # The added tests must notice the missing feature, not merely mention it.
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / 'repo'
        shutil.copytree(root, work, ignore=shutil.ignore_patterns('__pycache__'))
        for rel in ('models.py', 'service.py', 'cli.py'):
            shutil.copyfile(helpers.original(__file__, rel), work / rel)
        old, detail = helpers.unittest_result(work)
    if not old or old['ok'] or old['ran'] == 0:
        return False, "tests did not detect the missing priority feature"
    return True, "priority persistence, ordering, legacy data and regression tests passed"
