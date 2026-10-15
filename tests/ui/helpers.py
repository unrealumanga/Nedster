"""Fake EngineSession and helpers for headless Textual pilot tests."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from ironclad.config import NedsterConfig
from ironclad.contracts import EngineEvent
from nedster_agent.tui import NedsterApp
from nedster_agent.widgets.session import EventCallback, SessionNote

DEFAULT_INFO: dict[str, Any] = {
    "model": "nedster-worker",
    "provider": "LOCAL",
    "sandbox": "docker",
    "isolation": "full",
    "context_used": 1200,
    "context_max": 32768,
    "session_id": "sess-1",
    "engine": "fake",
}


class FakeSession:
    """Turns block until the test calls ``finish``/``confirm_cancel``; cancel never self-confirms."""

    def __init__(self, emit: EventCallback, info: dict[str, Any] | None = None) -> None:
        self.emit = emit
        self._info = dict(DEFAULT_INFO, **(info or {}))
        self.submitted: list[str] = []
        self.cancel_requests = 0
        self.new_calls = 0
        self._gate: asyncio.Event | None = None
        self.run = 0

    async def submit(self, text: str) -> None:
        self.submitted.append(text)
        self.run += 1
        self._gate = asyncio.Event()
        self.event("turn.started", user_text=text)
        await self._gate.wait()

    async def cancel(self) -> None:
        self.cancel_requests += 1

    def info(self) -> dict[str, Any]:
        return dict(self._info)

    def new(self) -> str:
        self.new_calls += 1
        self._info["session_id"] = f"sess-{self.new_calls + 1}"
        return self._info["session_id"]

    def set_model(self, name: str) -> None:
        self._info["model"] = name

    # ------------------------------------------------------------------ test controls

    def event(self, kind: str, /, **data: Any) -> None:
        self.emit(EngineEvent(kind, self._info["session_id"], f"run-{self.run}", data, time.time_ns()))  # type: ignore[arg-type]

    def note(self, text: str, level: str = "muted") -> None:
        self.emit(SessionNote(text, level))

    def finish(self, answer: str = "done", outcome: str = "answered", reasons: tuple[str, ...] = ()) -> None:
        self.event("acceptance", outcome=outcome, reasons=list(reasons), changed_paths=[])
        self.event("turn.finished", answer=answer, outcome=outcome)
        if self._gate is not None:
            self._gate.set()

    def confirm_cancel(self) -> None:
        self.finish("", "cancelled", ("cancelled by user",))


class Harness:
    def __init__(self, tmp_path: Path, **app_kwargs: Any) -> None:
        self.project = tmp_path / "project"
        self.project.mkdir(exist_ok=True)
        config = NedsterConfig()
        config.runtime.project_root = self.project
        config.runtime.state_dir = tmp_path / "data"
        self.config = config
        self.session: FakeSession | None = None
        info = app_kwargs.pop("info", None)

        def factory(_config: NedsterConfig, emit: EventCallback) -> FakeSession:
            self.session = FakeSession(emit, info)
            return self.session

        app_kwargs.setdefault("files", lambda: ["README.md", "src/main.py", "src/util.py", "tests/test_main.py"])
        self.app = NedsterApp(config, session_factory=factory, **app_kwargs)

    @property
    def fake(self) -> FakeSession:
        assert self.session is not None
        return self.session


def transcript(app: NedsterApp) -> str:
    return app.transcript.plain_text()


def screen_text(app: NedsterApp) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


async def type_text(pilot: Any, text: str) -> None:
    keys = ["space" if ch == " " else ch for ch in text]
    await pilot.press(*keys)


async def settle(pilot: Any, times: int = 3) -> None:
    for _ in range(times):
        await pilot.pause()


async def wait_for(pilot: Any, predicate: Any, timeout: float = 5.0) -> None:
    """Pause until ``predicate()`` holds (threads/workers finish at machine speed, not in N ticks)."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await pilot.pause(0.01)
    await pilot.pause()


async def ready(pilot: Any, app: NedsterApp) -> None:
    await wait_for(pilot, lambda: app.session is not None and app.status_bar.state == "idle")
