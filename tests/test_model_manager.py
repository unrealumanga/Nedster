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
    assert 'max_auto_risk = "write"' in value  # execution is never enabled without consent
    assert "allow_host_execution = false" in value


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


# --- explicit execution consent (SETUP-CONSENT) -----------------------------------------------

from nedster_agent.model_manager import ExecutionChoice, describe_permissions  # noqa: E402


def _merge(existing, execution):
    return merge_config(existing, recommend_plan(snapshot(32)), "http://h:1", execution)


def test_merge_without_choice_defaults_to_write_and_keeps_previous_answers():
    fresh = _merge({}, None)
    assert fresh["policy"]["max_auto_risk"] == "write" and fresh["sandbox"]["allow_host_execution"] is False
    prior = {"policy": {"max_auto_risk": "execute"}, "sandbox": {"allow_host_execution": True}}
    kept = _merge(prior, None)
    assert kept["policy"]["max_auto_risk"] == "execute" and kept["sandbox"]["allow_host_execution"] is True


def test_merge_with_choice_overrides_consent_keys():
    prior = {"policy": {"max_auto_risk": "write"}, "sandbox": {"allow_host_execution": True}}
    isolated = _merge(prior, ExecutionChoice("execute"))
    assert isolated["policy"]["max_auto_risk"] == "execute" and isolated["sandbox"]["allow_host_execution"] is True
    host = _merge({}, ExecutionChoice("execute", True))
    assert host["sandbox"]["allow_host_execution"] is True
    declined = _merge({"policy": {"max_auto_risk": "execute"}}, ExecutionChoice("write", False))
    assert declined["policy"]["max_auto_risk"] == "write" and declined["sandbox"]["allow_host_execution"] is False


def test_install_plan_writes_the_consent_choice(tmp_path: Path):
    install_plan(recommend_plan(snapshot(32)), manager=FakeManager(), config_dir=tmp_path,
                 execution=ExecutionChoice("execute", True))
    data = tomllib.loads((tmp_path / "config.toml").read_text())
    assert data["policy"]["max_auto_risk"] == "execute" and data["sandbox"]["allow_host_execution"] is True


def test_describe_permissions_states_exact_capability():
    assert "only read" in describe_permissions("read", False)
    assert "read and write" in describe_permissions("write", False)
    assert "isolated sandbox" in describe_permissions("execute", False)
    assert "NO isolation" in describe_permissions("execute", True)


# --- installed-model discovery + truthful switching (BP2) --------------------------------------

import httpx  # noqa: E402

from nedster_agent.model_manager import (  # noqa: E402
    is_model_missing_error,
    list_installed_models,
    match_installed_model,
    model_not_installed_message,
    set_default_worker,
)


class _ListingManager:
    """Minimal OllamaManager double: tags details plus an /api/show probe that can fail."""

    def __init__(self, entries, contexts=None):
        self.entries = list(entries)
        self.contexts = contexts or {}
        self.probed: list[str] = []

    def list_model_details(self):
        return list(self.entries)

    def context_window(self, model):
        self.probed.append(model)
        if model in self.contexts:
            return self.contexts[model]
        raise httpx.HTTPError("dead endpoint")


def _entry(name, size=9_500_000_000, params="9B", quant="Q4_K_M"):
    return {"name": name, "size": size, "details": {"parameter_size": params, "quantization_level": quant}}


def test_list_installed_models_reports_details_and_marks_the_current_one():
    manager = _ListingManager(
        [_entry("qwen3.6:27b", size=18_000_000_000, params="27B"), _entry("nedster-worker:latest")],
        contexts={"qwen3.6:27b": (40960, None)},
    )
    rows = list_installed_models(manager, "nedster-worker")

    assert [row.name for row in rows] == ["qwen3.6:27b", "nedster-worker:latest"]
    assert rows[0].size_gb == 18.0 and rows[0].parameter_size == "27B" and rows[0].quantization == "Q4_K_M"
    assert rows[0].context_length == 40960 and rows[0].current is False
    assert rows[1].current is True  # "nedster-worker" matches "nedster-worker:latest"
    assert rows[1].context_length is None and rows[1].context_label() == "unknown"
    assert manager.probed == ["qwen3.6:27b", "nedster-worker:latest"]


def test_list_installed_models_can_skip_context_probes():
    manager = _ListingManager([_entry("qwen3.6:27b")], contexts={"qwen3.6:27b": (40960, None)})
    rows = list_installed_models(manager, with_context=False)
    assert rows[0].context_length is None and manager.probed == []


def test_match_installed_model_accepts_an_implicit_latest_tag():
    installed = ["nedster-worker:latest", "qwen3.6:27b"]
    assert match_installed_model("nedster-worker", installed) == "nedster-worker:latest"
    assert match_installed_model("qwen3.6:27b:latest", installed) == "qwen3.6:27b"
    assert match_installed_model("qwen3.6:27b", installed) == "qwen3.6:27b"
    assert match_installed_model("", installed) is None
    assert match_installed_model("missing", installed) is None


def test_model_not_installed_message_names_the_remedy_and_closest_installed():
    message = model_not_installed_message("qwen3.6:27", ["qwen3.6:27b", "llama3:8b"])
    assert message.startswith("Model 'qwen3.6:27' is not installed.")
    assert "Choose one with /models" in message
    assert "ollama pull qwen3.6:27" in message
    assert "qwen3.6:27b" in message  # difflib suggestion
    assert "llama3:8b" not in message


def test_is_model_missing_error_recognizes_ollama_404s():
    assert is_model_missing_error("Client error '404 Not Found' for url 'http://127.0.0.1:11434/api/chat'")
    assert is_model_missing_error("Ollama error: model 'x' not found")
    assert not is_model_missing_error("connection refused")


def test_set_default_worker_preserves_other_keys_and_backs_up(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[models]\nrouter = "nedster-router"\ncoder = "old"\n[policy]\nmax_auto_risk = "write"\n',
        encoding="utf-8",
    )
    backup = set_default_worker(path, "qwen3.6:27b")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert data["models"]["coder"] == "qwen3.6:27b"
    assert data["models"]["router"] == "nedster-router"  # untouched
    assert data["policy"]["max_auto_risk"] == "write"  # untouched
    assert backup is not None and Path(backup).read_text(encoding="utf-8").startswith("[models]")


def test_set_default_worker_creates_a_config_when_missing(tmp_path: Path):
    path = tmp_path / "nested" / "config.toml"
    assert set_default_worker(path, "qwen3.6:27b") is None
    assert tomllib.loads(path.read_text(encoding="utf-8"))["models"]["coder"] == "qwen3.6:27b"
