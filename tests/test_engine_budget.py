from __future__ import annotations

from pathlib import Path

import pytest

from ironclad.config import NedsterConfig, load_config
from ironclad.native import engine_budget


def test_defaults_give_max_steps_40_and_max_tool_calls_120() -> None:
    config = NedsterConfig()
    config.runtime.context_tokens = 8192
    budget = engine_budget(config, "test-model")
    assert budget.max_steps == 40
    assert budget.max_tool_calls == 120


def test_runtime_max_steps_80_gives_80() -> None:
    config = NedsterConfig()
    config.runtime.max_steps = 80
    # Keep context_tokens > 0 to avoid network call in context_budget
    config.runtime.context_tokens = 8192
    budget = engine_budget(config, "test-model")
    assert budget.max_steps == 80


def test_runtime_max_steps_0_gives_none() -> None:
    config = NedsterConfig()
    config.runtime.max_steps = 0
    config.runtime.context_tokens = 8192
    budget = engine_budget(config, "test-model")
    assert budget.max_steps is None


def test_load_config_with_env_override_max_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use a temporary config directory to avoid user config leaking in
    monkeypatch.setenv("NEDSTER_CONFIG_DIR", str(tmp_path))
    config = load_config(project_root=tmp_path)
    assert config.runtime.max_steps == 40  # default

    monkeypatch.setenv("NEDSTER_RUNTIME__MAX_STEPS", "75")
    config2 = load_config(project_root=tmp_path)
    assert config2.runtime.max_steps == 75


def test_runtime_max_tool_calls_0_gives_none() -> None:
    config = NedsterConfig()
    config.runtime.max_tool_calls = 0
    config.runtime.context_tokens = 8192
    budget = engine_budget(config, "test-model")
    assert budget.max_tool_calls is None


def test_runtime_max_tool_calls_gives_value() -> None:
    config = NedsterConfig()
    config.runtime.max_tool_calls = 200
    config.runtime.context_tokens = 8192
    budget = engine_budget(config, "test-model")
    assert budget.max_tool_calls == 200
