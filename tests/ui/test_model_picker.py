"""The installed-model picker: filtering, switching, and truthful failures (BP2)."""

from __future__ import annotations

import pytest

from nedster_agent.model_manager import InstalledModel
from nedster_agent.widgets.model_picker import ModelPicker

from .helpers import Harness, ready, settle, transcript, type_text, wait_for


def _rows(current: str = "nedster-worker") -> list[InstalledModel]:
    return [
        InstalledModel(
            "nedster-worker:latest", size_gb=9.0, parameter_size="9B",
            quantization="Q4_K_M", context_length=40960, current=current == "nedster-worker",
        ),
        InstalledModel(
            "qwen3.6:27b", size_gb=18.0, parameter_size="27B",
            quantization="Q4_K_M", context_length=65536, current=current == "qwen3.6:27b",
        ),
        InstalledModel(
            "qwen3.5:9b", size_gb=6.0, parameter_size="9B",
            quantization="Q4_K_M", context_length=8192,
        ),
    ]


def test_picker_never_shows_models_that_cannot_chat():
    rows = [*_rows(), InstalledModel("nomic-embed-text:latest", embedding=True)]
    picker = ModelPicker(rows)
    assert [m.name for m in picker.models] == [m.name for m in _rows()]
    assert "nomic-embed-text:latest" not in [m.name for m in picker.filtered]


@pytest.mark.asyncio
async def test_models_command_filters_and_switches_the_session_model(tmp_path):
    h = Harness(tmp_path, model_source=lambda current: _rows(current))
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app._command("/models")
        await wait_for(pilot, lambda: isinstance(h.app.screen, ModelPicker))

        await type_text(pilot, "qwen3.6")
        await settle(pilot)
        assert [m.name for m in h.app.screen.filtered] == ["qwen3.6:27b"]

        await pilot.press("enter")
        await settle(pilot)
        assert h.fake.info()["model"] == "qwen3.6:27b"
        assert "Model set to qwen3.6:27b" in transcript(h.app)
        assert not isinstance(h.app.screen, ModelPicker)


@pytest.mark.asyncio
async def test_f4_opens_the_picker_and_escape_cancels(tmp_path):
    h = Harness(tmp_path, model_source=lambda current: _rows(current))
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        await pilot.press("f4")
        await wait_for(pilot, lambda: isinstance(h.app.screen, ModelPicker))

        await pilot.press("escape")
        await settle(pilot)
        assert not isinstance(h.app.screen, ModelPicker)
        assert h.fake.info()["model"] == "nedster-worker"  # unchanged


@pytest.mark.asyncio
async def test_typo_model_keeps_the_previous_model_and_suggests_the_closest(tmp_path):
    h = Harness(tmp_path, model_source=lambda current: _rows(current))
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app._command("/model qwen3.6:2")
        await wait_for(pilot, lambda: "is not installed" in transcript(h.app))

        text = transcript(h.app)
        assert "Model 'qwen3.6:2' is not installed" in text
        assert "qwen3.6:27b" in text  # difflib suggestion
        assert h.fake.info()["model"] == "nedster-worker"


@pytest.mark.asyncio
async def test_installed_model_without_a_tag_is_resolved_and_switched(tmp_path):
    h = Harness(tmp_path, model_source=lambda current: _rows(current))
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app._command("/model nedster-worker")
        await wait_for(pilot, lambda: "Model set to nedster-worker:latest" in transcript(h.app))
        assert h.fake.info()["model"] == "nedster-worker:latest"


@pytest.mark.asyncio
async def test_unreachable_ollama_explains_itself_instead_of_crashing(tmp_path):
    def boom(_current: str):
        raise RuntimeError("connection refused")

    h = Harness(tmp_path, model_source=boom)
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app._command("/models")
        await wait_for(pilot, lambda: "not reachable" in transcript(h.app))

        assert not isinstance(h.app.screen, ModelPicker)
        assert "ollama serve" in transcript(h.app)


@pytest.mark.asyncio
async def test_no_installed_models_says_how_to_add_one(tmp_path):
    h = Harness(tmp_path, model_source=lambda current: [])
    async with h.app.run_test(size=(100, 30)) as pilot:
        await ready(pilot, h.app)
        h.app._command("/models")
        await wait_for(pilot, lambda: "No models are installed" in transcript(h.app))

        assert not isinstance(h.app.screen, ModelPicker)
