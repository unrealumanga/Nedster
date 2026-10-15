"""BP2-VIEW: the TUI completion notice fires only for long, interactive turns."""

from __future__ import annotations

import time

from nedster_agent.notify import BELL

from .helpers import Harness, ready, settle


async def test_long_turn_rings_the_bell(tmp_path):
    h = Harness(tmp_path)
    captured: list[str] = []
    h.app._notify_writer = captured.append
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("work")
        await settle(pilot)
        h.app._turn_started = time.monotonic() - 100  # pretend the turn ran for 100 s
        h.fake.finish("done", "answered")
        await settle(pilot, 5)
        assert captured == [BELL]


async def test_short_turn_stays_silent(tmp_path):
    h = Harness(tmp_path)
    captured: list[str] = []
    h.app._notify_writer = captured.append
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("work")
        await settle(pilot)
        h.fake.finish("done", "answered")  # well under the 15 s threshold
        await settle(pilot, 5)
        assert captured == []


async def test_off_mode_disables_the_notice(tmp_path):
    h = Harness(tmp_path)
    h.config.runtime.notify = "off"
    captured: list[str] = []
    h.app._notify_writer = captured.append
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("work")
        await settle(pilot)
        h.app._turn_started = time.monotonic() - 100
        h.fake.finish("done", "answered")
        await settle(pilot, 5)
        assert captured == []


async def test_osc9_mode_embeds_the_outcome(tmp_path):
    h = Harness(tmp_path)
    h.config.runtime.notify = "osc9"
    captured: list[str] = []
    h.app._notify_writer = captured.append
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("work")
        await settle(pilot)
        h.app._turn_started = time.monotonic() - 42
        h.fake.finish("done", "answered")
        await settle(pilot, 5)
        assert len(captured) == 1
        assert captured[0].startswith("\x1b]9;Nedster: completed in 4")
        assert captured[0].endswith("\x07")


async def test_failed_turn_reports_failed(tmp_path):
    h = Harness(tmp_path)
    h.config.runtime.notify = "osc9"
    captured: list[str] = []
    h.app._notify_writer = captured.append
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app.submit_text("work")
        await settle(pilot)
        h.app._turn_started = time.monotonic() - 100
        h.fake.finish("", "failed", ("boom",))
        await settle(pilot, 5)
        assert captured and "Nedster: failed in" in captured[0]
