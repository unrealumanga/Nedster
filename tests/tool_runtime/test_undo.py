from __future__ import annotations

import asyncio
from pathlib import Path

from ironclad.contracts import ToolCall, ToolResult, ToolSpec, ToolStatus
from ironclad.session_store.checkpoints import CheckpointStore
from ironclad.tool_runtime.composite import CompositeToolRuntime
from ironclad.tool_runtime.files import FileTools


def make(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.txt").write_text("A0", encoding="utf-8")
    store = CheckpointStore(tmp_path / "s.sqlite3")
    files = FileTools(root, journal=store)
    runtime = CompositeToolRuntime(sync_tools=[files], journal=store)
    store.on_undone.append(runtime.invalidate_after_undo)
    return root, store, files, runtime


def call(runtime, name, **args):
    return asyncio.run(runtime.start(ToolCall("c", name, args)).wait())


def test_undo_invalidates_read_hashes_and_bumps_revision(tmp_path):
    root, store, files, runtime = make(tmp_path)
    store.begin_turn("s", root)
    sha = call(runtime, "read_file", path="a.txt").data["sha256"]
    assert call(runtime, "write_file", path="a.txt", content="A1", expected_sha=sha).status is ToolStatus.OK
    store.end_turn()
    assert files.read_hashes and runtime.revision().counter == 1
    assert store.undo_last("s").ok
    assert files.read_hashes == {}  # a stale read can no longer authorize an edit
    assert runtime.revision().counter == 2  # validation evidence from before the undo is stale
    store.close()


class Exec:
    def specs(self):
        return [ToolSpec("run_command", "c", {"type": "object"}, 2, mutates=True)]

    def start(self, c):
        class Job:
            async def wait(self):
                return ToolResult(c.id, c.name, ToolStatus.OK, "ok")

            async def cancel(self):
                return None

        return Job()


def test_execution_tools_mark_turn_as_partially_covered(tmp_path):
    root, store, files, _ = make(tmp_path)
    runtime = CompositeToolRuntime(sync_tools=[files], job_tools=[Exec()], journal=store)
    store.begin_turn("s", root)
    sha = call(runtime, "read_file", path="a.txt").data["sha256"]
    call(runtime, "write_file", path="a.txt", content="A1", expected_sha=sha)
    call(runtime, "run_command", argv=["echo"])
    store.end_turn()
    report = store.undo_last("s")
    assert report.ok and report.commands_not_covered and "NOT covered" in report.message
    assert (root / "a.txt").read_text() == "A0"
    store.close()


def test_command_only_turn_is_not_silently_undoable_as_files(tmp_path):
    root, store, files, _ = make(tmp_path)
    runtime = CompositeToolRuntime(sync_tools=[files], job_tools=[Exec()], journal=store)
    store.begin_turn("s", root)
    call(runtime, "run_command", argv=["echo"])
    store.end_turn()
    report = store.undo_last("s")
    assert report.ok and not report.paths and "NOT covered" in report.message
    store.close()
