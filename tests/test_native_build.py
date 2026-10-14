from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ironclad.config import NedsterConfig
from ironclad.native import build_tools


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_native_tools_use_hardened_git_inventory_that_respects_gitignore(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "big.log").write_text("x", encoding="utf-8")
    (tmp_path / "app.py").write_text("print(1)\n", encoding="utf-8")
    marker = tmp_path.parent / f"{tmp_path.name}-fsmonitor-ran"
    hook = tmp_path / "hook.py"
    hook.write_text(f"open({str(marker)!r}, 'w').close()\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "config", "core.fsmonitor", f"python {hook}"], check=True)
    config = NedsterConfig()
    config.runtime.project_root = tmp_path
    runtime, executor = build_tools(tmp_path, config)
    try:
        import asyncio

        from ironclad.contracts import ToolCall

        result = asyncio.run(runtime.start(ToolCall("1", "list_files", {})).wait())
        assert "app.py" in result.output
        assert "big.log" not in result.output
        assert not marker.exists()
    finally:
        executor.close()
