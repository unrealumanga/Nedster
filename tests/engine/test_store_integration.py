from __future__ import annotations

from pathlib import Path

import pytest

from ironclad.contracts import Outcome
from ironclad.engine.loop import Engine
from ironclad.engine.state import Budget

from .fakes import FakeRuntime, ScriptedTransport, Turn, call, ok

sqlite_store = pytest.importorskip("ironclad.session_store.sqlite_store")


async def test_engine_resumes_from_sqlite_store_after_compaction(tmp_path: Path) -> None:
    store = sqlite_store.SqliteSessionStore(tmp_path / "s.db")
    try:
        sid = store.create(str(tmp_path), "t")
        runtime = FakeRuntime({"read_file": lambda c: ok(c, f"head {c.id}\n" + "x" * 2_000)})
        script = [Turn(calls=[call("read_file", f"r{i}", path="a")]) for i in range(12)] + [Turn("done")]
        budget = Budget(context_tokens=3_000)
        engine = Engine(ScriptedTransport(script), runtime, model="m", store=store, budget=budget)
        assert (await engine.run(sid, "survey")).outcome is Outcome.ANSWERED

        stub = next(m for m in store.messages(sid) if m.content.startswith("[output #"))
        handle = int(stub.content.split("#", 1)[1].split(" ", 1)[0])
        resumed = Engine(
            ScriptedTransport([Turn(calls=[call("show_output", "s", handle=handle)]), Turn("ok")]),
            runtime, model="m", store=store, budget=budget,
        )
        await resumed.run(sid, "show it")
        shown = next(m for m in resumed.messages(sid) if m.tool_call_id == "s")
        assert shown.content.startswith("[ok] head r")
        assert store.get_state(sid, "task_state")["objective"] == "survey"
    finally:
        store.close()
