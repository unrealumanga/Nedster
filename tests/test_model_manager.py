from pathlib import Path
from unittest.mock import Mock, patch

from ironclad.types import HardwareSnapshot
from nedster_agent.model_manager import (
    OllamaManager,
    recommend_plan,
    render_config,
    render_modelfile,
    state_dir,
)


def snapshot(vram: float, ram: float = 32.0) -> HardwareSnapshot:
    return HardwareSnapshot(
        platform="test",
        cpu_count=16,
        ram_total_gb=ram,
        ram_available_gb=ram * 0.8,
        accelerator="test gpu" if vram else "cpu",
        vram_total_gb=vram,
        vram_free_gb=vram * 0.8,
    )


def test_balanced_titan_plan_leaves_room_for_router_and_embedding():
    plan = recommend_plan(snapshot(32))
    assert plan.worker == "qwen3.6:27b"
    assert plan.router == "qwen3.5:4b-q4_K_M"
    assert plan.estimated_download_gb < 22


def test_edge_plan_uses_quantized_models():
    plan = recommend_plan(snapshot(8))
    assert "q4" in plan.worker.lower()
    assert "q4" in plan.router.lower()


def test_modelfile_preserves_native_template_and_sets_context():
    value = render_modelfile("qwen3.6:27b", 65_536, role="worker")
    assert value.startswith("FROM qwen3.6:27b\n")
    assert "PARAMETER num_ctx 65536" in value
    assert "TEMPLATE" not in value
    assert "SYSTEM" not in value
    assert "temperature" not in value
    assert "top_p" not in value


def test_generated_config_uses_stable_aliases():
    value = render_config(recommend_plan(snapshot(32)), "http://localhost:11434/")
    assert 'router = "nedster-router"' in value
    assert 'coder = "nedster-worker"' in value
    assert 'ollama_url = "http://localhost:11434"' in value
    assert "[policy]" in value
    assert "autonomous = true" in value
    assert 'max_auto_risk = "execute"' in value


def test_state_directory_can_be_overridden(tmp_path: Path):
    with patch.dict("os.environ", {"NEDSTER_STATE_DIR": str(tmp_path)}):
        assert state_dir() == tmp_path


def test_capability_verification_checks_all_local_agent_primitives():
    manager = OllamaManager()
    responses = [
        Mock(is_success=True, json=lambda: {"version": "test"}),
        Mock(raise_for_status=lambda: None, json=lambda: {"response": "ready"}),
        Mock(raise_for_status=lambda: None, json=lambda: {"response": '{"ready": true}'}),
        Mock(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"tool_calls": [{"function": {"name": "ping"}}]}},
        ),
        Mock(raise_for_status=lambda: None, json=lambda: {"embeddings": [[0.1, 0.2]]}),
    ]
    with (
        patch("nedster_agent.model_manager.httpx.get", return_value=responses[0]),
        patch("nedster_agent.model_manager.httpx.post", side_effect=responses[1:]) as post,
    ):
        result = manager.verify_capabilities()
    assert all(result.values())
    assert all(call.kwargs["json"].get("think") is False for call in post.call_args_list[:3])
