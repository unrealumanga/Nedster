from core.model_registry import RESCUE_TAG, resolve_model, resolve_tier, supports_native_tools


def test_model_resolution_uses_canonical_or_installed_qwen(monkeypatch):
    monkeypatch.delenv("NEDSTER_MODEL", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    assert resolve_model([]) == "nedster"
    assert resolve_model(["qwen3.5:9b"]) == "qwen3.5:9b"
    assert resolve_model(["vision-only:7b"]) == RESCUE_TAG


def test_tiers_do_not_split_small_cards():
    assert resolve_tier(8).num_ctx <= 16384
    assert resolve_tier(12).split_router is False
    assert resolve_tier(32).split_router is True


def test_canonical_model_is_tool_capable():
    assert supports_native_tools("nedster:latest") is True
