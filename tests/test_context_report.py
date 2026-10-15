from __future__ import annotations

import httpx  # noqa: F401 (used in test functions below via assert)

from ironclad.config import NedsterConfig
from nedster_agent.model_manager import _num_ctx, _trained_ctx, context_report, kv_cache_advice


class _Manager:
    def __init__(self, window: tuple[int | None, int | None] | Exception):
        self.window = window

    def context_window(self, model: str) -> tuple[int | None, int | None]:
        if isinstance(self.window, Exception):
            raise self.window
        return self.window


def test_num_ctx_parses_the_modelfile_parameter_block() -> None:
    assert _num_ctx("top_p 0.95\nnum_ctx                        65536\ntemperature 1") == 65536
    assert _num_ctx("temperature 1") is None
    assert _trained_ctx({"qwen35moe.context_length": 262144, "general.name": "x"}) == 262144
    assert _trained_ctx({"x.context_length": True}) is None


def test_auto_budget_follows_the_model_window() -> None:
    report = context_report(NedsterConfig(), _Manager((65536, 262144)), env={})
    assert report["engine_budget"] == 65536
    assert not any("runtime.context_tokens" in tip for tip in report["advice"])


def test_explicit_budget_mismatch_is_reported_both_ways() -> None:
    config = NedsterConfig()
    config.runtime.context_tokens = 32768
    low = context_report(config, _Manager((65536, None)), env={})
    assert any("set runtime.context_tokens = 65536" in tip for tip in low["advice"])
    config.runtime.context_tokens = 131072
    high = context_report(config, _Manager((65536, None)), env={})
    assert any("Ollama will truncate" in tip for tip in high["advice"])


def test_kv_cache_advice_disappears_once_quantized_with_flash_attention() -> None:
    env = {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}
    report = context_report(NedsterConfig(), _Manager((65536, None)), env=env)
    assert report["flash_attention"] is True and report["kv_cache_type"] == "q8_0"
    assert report["advice"] == []
    plain = context_report(NedsterConfig(), _Manager((65536, None)), env={})
    # The advice text now uses kv_cache_advice() helper.
    assert any("kv_cache_advice" in tip.lower() or "setx" in tip.lower() or "launchctl" in tip.lower() or "systemctl" in tip.lower() for tip in plain["advice"])


def test_kv_source_reported_as_client_shell() -> None:
    report = context_report(NedsterConfig(), _Manager((65536, None)), env={})
    assert "server settings unknown" in str(report.get("kv_source", ""))


def test_kv_cache_advice_windows_contains_setx() -> None:
    advice = kv_cache_advice("Windows")
    assert "setx" in advice
    assert "OLLAMA_FLASH_ATTENTION 1" in advice
    assert "OLLAMA_KV_CACHE_TYPE q8_0" in advice


def test_kv_cache_advice_darwin_contains_launchctl() -> None:
    advice = kv_cache_advice("Darwin")
    assert "launchctl" in advice
    assert "restart the ollama app" in advice.lower()


def test_kv_cache_advice_linux_contains_systemctl() -> None:
    advice = kv_cache_advice("Linux")
    assert "systemctl" in advice
    assert "sudo systemctl edit ollama" in advice
