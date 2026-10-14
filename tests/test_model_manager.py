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


# --- transactional setup / readiness (package F) ---------------------------------------------

import tomllib  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from nedster_agent.model_manager import SetupError, install_plan, merge_config, readiness  # noqa: E402


class FakeManager:
    url = "http://fake"

    def __init__(self, ok=True, models=()):
        self.ok, self.models, self.calls = ok, list(models), []

    def install(self, **_): ...
    def ensure_service(self, **_): ...
    def list_models(self): return self.models
    def api_available(self): return True
    def pull(self, model, **_): self.calls.append(("pull", model))
    def create_alias(self, alias, path, **_): self.calls.append(("create", alias))
    def copy_model(self, src, dst): self.calls.append(("cp", dst))
    def remove_model(self, name): self.calls.append(("rm", name))

    def verify_capabilities(self, router="", embedding=None):
        self.calls.append(("verify", router))
        good = self.ok or "worker" not in router
        result = {"generation": good, "structured_output": True, "tool_calling": True}
        if embedding:
            result["embeddings"] = True
        return result


def test_failed_worker_verification_keeps_old_config_and_aliases(tmp_path: Path):
    old = '[models]\nrouter = "old"\n'
    (tmp_path / "config.toml").write_text(old)
    manager = FakeManager(ok=False)
    with pytest.raises(SetupError) as info:
        install_plan(recommend_plan(snapshot(32)), manager=manager, config_dir=tmp_path)
    assert "worker.generation" in str(info.value) and info.value.remedy
    assert (tmp_path / "config.toml").read_text() == old
    assert not list(tmp_path.glob("config.toml.bak-*"))
    assert ("cp", "nedster-worker") not in manager.calls
    assert ("rm", "nedster-worker-next") in manager.calls  # staging cleaned up


def test_successful_setup_backs_up_merges_and_skips_embeddings(tmp_path: Path):
    (tmp_path / "config.toml").write_text('[models]\nrouter = "old"\ncloud_model = "mine"\n[custom]\nx = 1\n')
    manager = FakeManager()
    plan = recommend_plan(snapshot(32))
    result = install_plan(plan, manager=manager, config_dir=tmp_path)
    data = tomllib.loads((tmp_path / "config.toml").read_text())
    assert data["models"]["router"] == "nedster-router"  # managed key updated
    assert data["models"]["cloud_model"] == "mine" and data["custom"] == {"x": 1}  # user keys kept
    assert Path(result["backup"]).read_text().startswith('[models]\nrouter = "old"')
    assert ("pull", plan.embedding) not in manager.calls
    assert not (tmp_path / "config.toml.new").exists()


def test_with_embeddings_pulls_and_verifies_embedding(tmp_path: Path):
    manager = FakeManager()
    plan = recommend_plan(snapshot(32))
    result = install_plan(plan, manager=manager, config_dir=tmp_path, with_embeddings=True)
    assert ("pull", plan.embedding) in manager.calls
    assert result["verification"]["embedding"] == {"embeddings": True}


def test_dry_run_writes_nothing(tmp_path: Path):
    install_plan(recommend_plan(snapshot(32)), manager=FakeManager(), config_dir=tmp_path, dry_run=True)
    assert list(tmp_path.iterdir()) == []


def test_merge_config_adds_defaults_without_overriding_user_values():
    plan = recommend_plan(snapshot(32))
    merged = merge_config({"policy": {"autonomous": False}}, plan, "http://h:1")
    assert merged["policy"]["autonomous"] is False
    assert merged["models"]["ollama_url"] == "http://h:1"


def _config(backend="auto", host=False):
    return SimpleNamespace(
        models=SimpleNamespace(ollama_url="http://fake", coder="nedster-worker"),
        sandbox=SimpleNamespace(backend=backend, allow_host_execution=host, docker_image="x"),
        runtime=SimpleNamespace(project_root=Path(".")),
    )


def test_readiness_inference_tiers_and_remedies():
    tiers = readiness(_config(), FakeManager(models=["nedster-worker:latest"]))
    assert tiers["inference"]["ready"] and tiers["read_only"]["ready"]
    missing = readiness(_config(), FakeManager(models=["other"]))
    assert not missing["inference"]["ready"] and "nedster setup" in missing["inference"]["remedy"]


def test_readiness_host_mode_is_labelled_and_unopted_host_is_not_ready():
    host = readiness(_config("host", True), FakeManager())["execution"]
    assert host["ready"] and host["detail"] == "host process — no isolation (opted in)"
    denied = readiness(_config("host", False), FakeManager())["execution"]
    assert not denied["ready"] and denied["remedy"]
