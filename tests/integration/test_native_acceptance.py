"""Engine + the REAL composite runtime (ironclad.native.build_tools): outcomes the fakes used to hide.

Review findings #6 (run_tests invalidated itself), #7 (validation spoofing), #8 (run_command changes ignored) and
#9 (failed mutation reported as answered). Commands run on the host backend with an explicit opt-in so the test is
deterministic on every OS; it only ever runs the current Python interpreter inside a temp project.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from engine.fakes import ScriptedTransport, Turn, call

from ironclad.config import NedsterConfig
from ironclad.contracts import AcceptanceResult, Outcome
from ironclad.engine.loop import Engine
from ironclad.native import build_tools

PY = sys.executable
SHA = hashlib.sha256(b"x = 1\n").hexdigest()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")
    (tmp_path / "app.py").write_bytes(b"x = 1\n")
    return tmp_path


@pytest.fixture
def config(project: Path) -> NedsterConfig:
    cfg = NedsterConfig()
    cfg.sandbox.backend = "host"
    cfg.sandbox.allow_host_execution = True
    cfg.runtime.project_root = project
    cfg.policy.max_auto_risk = "execute"
    cfg.policy.allowed_commands = [*cfg.policy.allowed_commands, Path(PY).stem]
    return cfg


@pytest.fixture
def run(project: Path, config: NedsterConfig) -> Iterator[object]:
    executors: list[object] = []

    async def go(script: list[Turn], auto_verify: int = 0) -> AcceptanceResult:
        # auto_verify=0: these tests pin the acceptance rules for what the model itself ran.
        tools, executor = build_tools(project, config)
        executors.append(executor)
        engine = Engine(ScriptedTransport(script), tools, model="m", project_root=project, auto_verify=auto_verify)
        return await engine.run("s", "change x to 2 and make sure tests pass")

    yield go
    for executor in executors:
        executor.close()  # type: ignore[attr-defined]


def edit_turn() -> Turn:
    return Turn(calls=[call("edit_file", "c1", path="app.py", old="x = 1", new="x = 2", expected_sha=SHA)])


async def test_review6_real_edit_then_real_passing_run_tests_is_completed(run, project: Path) -> None:
    result = await run([edit_turn(), Turn(calls=[call("run_tests", "c2", argv=[PY, "-m", "pytest", "-q"])]), Turn("Done.")])
    assert (project / "app.py").read_text().strip() == "x = 2"
    assert result.outcome is Outcome.COMPLETED, result.reasons
    assert result.changed_paths == ("app.py",)


async def test_review7_real_flagged_non_runner_command_is_not_completed(run) -> None:
    script = [
        edit_turn(),
        Turn(calls=[call("run_command", "c2", argv=[PY, "-c", "pass"], validation=True)]),
        Turn("Done, validated."),
    ]
    result = await run(script)
    assert result.outcome is not Outcome.COMPLETED, result.reasons
    assert result.outcome is Outcome.UNVALIDATED


async def test_review8_real_mutation_via_run_command_is_not_answered(run, project: Path) -> None:
    rewrite = "open('app.py','w').write('x = 2\\n')"
    result = await run([Turn(calls=[call("run_command", "c1", argv=[PY, "-c", rewrite])]), Turn("Changed x to 2.")])
    assert (project / "app.py").read_text().strip() == "x = 2"
    assert result.outcome is Outcome.UNVALIDATED, result.reasons
    assert any("unreported changes by run_command" in r for r in result.reasons)


async def test_review8_real_run_command_change_then_run_tests_is_completed(run) -> None:
    rewrite = "open('app.py','w').write('x = 2\\n')"
    script = [
        Turn(calls=[call("run_command", "c1", argv=[PY, "-c", rewrite])]),
        Turn(calls=[call("run_tests", "c2", argv=[PY, "-m", "pytest", "-q"])]),
        Turn("Done."),
    ]
    result = await run(script)
    assert result.outcome is Outcome.COMPLETED, result.reasons


async def test_review9_real_failed_command_is_incomplete(run) -> None:
    script = [Turn(calls=[call("run_command", "c1", argv=[PY, "-c", "import sys; sys.exit(3)"])]), Turn("All done, x is 2.")]
    result = await run(script)
    assert result.outcome is Outcome.INCOMPLETE and not result.passed, result.reasons
    assert any("requested change failed" in r for r in result.reasons)


async def test_auto_verify_runs_the_real_suite_when_the_model_skips_it(run, project: Path) -> None:
    (project / "tests" / "__init__.py").write_text("")
    (project / "tests" / "test_a.py").write_text(
        "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertTrue(True)\n"
    )
    result = await run([edit_turn(), Turn("Done.")], auto_verify=1)
    assert result.outcome is Outcome.COMPLETED, result.reasons


async def test_auto_verify_collecting_no_tests_is_not_a_failure(run, project: Path, monkeypatch) -> None:
    # pytest-style tests but the runner chosen is unittest: zero tests collected is inconclusive, not "failing".
    monkeypatch.setattr("ironclad.tool_runtime.exec.Executor._uses_pytest", lambda self: False)
    result = await run([edit_turn(), Turn("Done.")], auto_verify=1)
    assert result.outcome is Outcome.UNVALIDATED, result.reasons
