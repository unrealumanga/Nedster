from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from ironclad.hardware import choose_profile, probe_hardware
from ironclad.types import HardwareSnapshot

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class ModelPlan:
    profile: str
    label: str
    worker: str
    router: str
    embedding: str
    context_tokens: int
    estimated_download_gb: float
    notes: str


MODEL_PLANS: dict[str, dict[str, ModelPlan]] = {
    "edge-8g": {
        "fast": ModelPlan("fast", "Fast", "qwen3.5:4b-q4_K_M", "qwen3.5:4b-q4_K_M", "embeddinggemma:300m-qat-q4_0", 16_384, 3.7, "One compact model keeps memory pressure low."),
        "balanced": ModelPlan("balanced", "Balanced", "qwen3.5:9b-q4_K_M", "qwen3.5:4b-q4_K_M", "embeddinggemma:300m-qat-q4_0", 16_384, 10.3, "A stronger worker with a responsive router."),
        "maximum": ModelPlan("maximum", "Maximum", "qwen3.5:9b-q4_K_M", "qwen3.5:4b-q4_K_M", "embeddinggemma:300m-qat-q4_0", 32_768, 10.3, "Uses more context and may spill into system RAM."),
    },
    "forge-16g": {
        "fast": ModelPlan("fast", "Fast", "qwen3.5:9b-q4_K_M", "qwen3.5:4b-q4_K_M", "embeddinggemma:300m-qat-q4_0", 32_768, 10.3, "Both models fit typical 16 GB accelerators."),
        "balanced": ModelPlan("balanced", "Balanced", "qwen3.5:27b-q4_K_M", "qwen3.5:4b-q4_K_M", "embeddinggemma:300m-qat-q4_0", 32_768, 20.7, "Pages the worker when needed while the router stays quick."),
        "maximum": ModelPlan("maximum", "Maximum", "qwen3-coder:30b", "qwen3.5:4b-q4_K_M", "embeddinggemma:300m-qat-q4_0", 65_536, 22.7, "Coding-specialized worker; partial CPU offload may occur."),
    },
    "titan-32g": {
        "fast": ModelPlan("fast", "Fast", "qwen3.5:9b-q4_K_M", "qwen3.5:4b-q4_K_M", "embeddinggemma:300m-qat-q4_0", 32_768, 10.3, "Low latency and generous VRAM headroom."),
        "balanced": ModelPlan("balanced", "Balanced", "qwen3.6:27b", "qwen3.5:4b-q4_K_M", "embeddinggemma:300m-qat-q4_0", 65_536, 20.7, "Recommended: strong tool use with room for router and context."),
        "maximum": ModelPlan("maximum", "Maximum", "qwen3.6:35b", "qwen3.5:4b-q4_K_M", "embeddinggemma:300m-qat-q4_0", 65_536, 27.7, "Highest local quality; models are paged instead of kept together."),
    },
}


def recommend_plan(snapshot: HardwareSnapshot | None = None, preference: str = "balanced") -> ModelPlan:
    snapshot = snapshot or probe_hardware()
    tier = choose_profile(snapshot).name
    return MODEL_PLANS[tier][preference]


def state_dir() -> Path:
    return Path(os.getenv("NEDSTER_STATE_DIR", Path.home() / ".nedster")).expanduser()


def render_modelfile(base: str, context_tokens: int, *, role: str) -> str:
    temperature = "0.15" if role == "worker" else "0.05"
    return (
        f"FROM {base}\n"
        f"PARAMETER num_ctx {context_tokens}\n"
        f"PARAMETER temperature {temperature}\n"
        "PARAMETER top_p 0.9\n"
    )


def render_config(plan: ModelPlan, ollama_url: str) -> str:
    return f'''# Managed by `nedster setup`. Edit any value and rerun `nedster doctor`.
[models]
ollama_url = "{ollama_url.rstrip('/')}"
router = "nedster-router"
coder = "nedster-worker"
reasoner = "nedster-worker"
critic = "nedster-worker"
embedding = "{plan.embedding}"
router_keep_alive = "5m"
worker_keep_alive = "90s"
local_thinking = true

[runtime]
state_dir = "~/.nedster/runtime"
max_worker_turns = 8
max_parallel_workers = 2
profile = "{plan.profile}"
'''


class OllamaManager:
    def __init__(self, url: str = "http://127.0.0.1:11434", progress: ProgressCallback | None = None):
        self.url = url.rstrip("/")
        self.progress = progress or (lambda _message: None)

    @property
    def executable(self) -> str | None:
        found = shutil.which("ollama")
        if found:
            return found
        if platform.system() == "Windows":
            candidate = Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
            if candidate.exists():
                return str(candidate)
        return None

    def api_available(self) -> bool:
        try:
            return httpx.get(f"{self.url}/api/version", timeout=2).is_success
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        response = httpx.get(f"{self.url}/api/tags", timeout=10)
        response.raise_for_status()
        return [str(item["name"]) for item in response.json().get("models", [])]

    def install(self, *, dry_run: bool = False) -> None:
        if self.executable:
            self.progress("Ollama is already installed; keeping the existing installation.")
            return
        system = platform.system()
        if dry_run:
            self.progress(f"Would install Ollama for {system}.")
            return
        if system == "Windows":
            installer = Path(tempfile.gettempdir()) / "Nedster-OllamaSetup.exe"
            self.progress("Downloading the official Ollama Windows installer…")
            with httpx.stream("GET", "https://ollama.com/download/OllamaSetup.exe", follow_redirects=True, timeout=120) as response:
                response.raise_for_status()
                with installer.open("wb") as output:
                    for chunk in response.iter_bytes():
                        output.write(chunk)
            self.progress("Opening the Ollama installer. Complete it to continue.")
            subprocess.run([str(installer)], check=True)
        elif system == "Linux":
            script = Path(tempfile.gettempdir()) / "nedster-install-ollama.sh"
            response = httpx.get("https://ollama.com/install.sh", follow_redirects=True, timeout=30)
            response.raise_for_status()
            script.write_bytes(response.content)
            self.progress("Running the official Ollama Linux installer…")
            subprocess.run(["sh", str(script)], check=True, stdout=sys.stderr)
        elif system == "Darwin":
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "--cask", "ollama-app"], check=True, stdout=sys.stderr)
            else:
                subprocess.run(["open", "https://ollama.com/download/mac"], check=True)
                raise RuntimeError("Complete the Ollama app installation, then rerun `nedster setup`.")
        else:
            raise RuntimeError(f"Automatic Ollama installation is not supported on {system}.")

    def ensure_service(self, *, dry_run: bool = False, wait_seconds: int = 30) -> None:
        if self.api_available():
            self.progress("Ollama is running.")
            return
        executable = self.executable
        if dry_run:
            self.progress("Would start the local Ollama service.")
            return
        if not executable:
            raise RuntimeError("Ollama was installed but its executable is not on PATH. Restart the terminal and rerun setup.")
        self.progress("Starting Ollama in the background…")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if platform.system() == "Windows" else 0
        subprocess.Popen(
            [executable, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            start_new_session=platform.system() != "Windows",
        )
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if self.api_available():
                return
            time.sleep(0.5)
        raise RuntimeError("Ollama did not become ready within 30 seconds. Run `ollama serve` and try again.")

    def pull(self, model: str, *, dry_run: bool = False) -> None:
        if dry_run:
            self.progress(f"Would download {model}.")
            return
        executable = self.executable
        if not executable:
            raise RuntimeError("The Ollama executable is unavailable.")
        self.progress(f"Downloading {model}…")
        subprocess.run([executable, "pull", model], check=True, stdout=sys.stderr)

    def create_alias(self, alias: str, modelfile: Path, *, dry_run: bool = False) -> None:
        if dry_run:
            self.progress(f"Would create managed alias {alias} from {modelfile.name}.")
            return
        executable = self.executable
        if not executable:
            raise RuntimeError("The Ollama executable is unavailable.")
        subprocess.run([executable, "create", alias, "-f", str(modelfile)], check=True, stdout=sys.stderr)

    def verify_generation(self, model: str = "nedster-router") -> str:
        response = httpx.post(
            f"{self.url}/api/generate",
            json={"model": model, "prompt": "Reply with only: ready", "stream": False, "options": {"num_predict": 8}},
            timeout=120,
        )
        response.raise_for_status()
        return str(response.json().get("response", "")).strip()

    def verify_capabilities(
        self,
        *,
        router: str = "nedster-router",
        embedding: str = "embeddinggemma:300m-qat-q4_0",
    ) -> dict[str, bool]:
        generation = self.verify_generation(router)
        schema = {
            "type": "object",
            "properties": {"ready": {"type": "boolean"}},
            "required": ["ready"],
            "additionalProperties": False,
        }
        structured = httpx.post(
            f"{self.url}/api/generate",
            json={
                "model": router,
                "prompt": "Return JSON with ready set to true.",
                "format": schema,
                "stream": False,
                "options": {"num_predict": 32},
            },
            timeout=180,
        )
        structured.raise_for_status()
        structured_value = json.loads(structured.json().get("response", "{}"))

        tool_response = httpx.post(
            f"{self.url}/api/chat",
            json={
                "model": router,
                "messages": [{"role": "user", "content": "Call the ping tool once."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "ping",
                            "description": "Check readiness.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "stream": False,
            },
            timeout=180,
        )
        tool_response.raise_for_status()
        tool_calls = tool_response.json().get("message", {}).get("tool_calls", [])

        embed_response = httpx.post(
            f"{self.url}/api/embed",
            json={"model": embedding, "input": "Nedster local agent readiness"},
            timeout=180,
        )
        embed_response.raise_for_status()
        embeddings = embed_response.json().get("embeddings", [])
        return {
            "generation": bool(generation),
            "structured_output": structured_value.get("ready") is True,
            "tool_calling": bool(tool_calls),
            "embeddings": bool(embeddings and embeddings[0]),
        }


def install_plan(
    plan: ModelPlan,
    *,
    ollama_url: str = "http://127.0.0.1:11434",
    install_ollama: bool = True,
    dry_run: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    emit = progress or (lambda _message: None)
    manager = OllamaManager(ollama_url, emit)
    if install_ollama:
        manager.install(dry_run=dry_run)
    manager.ensure_service(dry_run=dry_run)
    installed = set() if dry_run else set(manager.list_models())
    for model in dict.fromkeys((plan.worker, plan.router, plan.embedding)):
        if model in installed:
            emit(f"Keeping installed model {model}.")
        else:
            manager.pull(model, dry_run=dry_run)

    root = state_dir()
    modelfiles = root / "models" / "modelfiles"
    config_path = root / "config.toml"
    worker_path = modelfiles / "worker.Modelfile"
    router_path = modelfiles / "router.Modelfile"
    if not dry_run:
        modelfiles.mkdir(parents=True, exist_ok=True)
        worker_path.write_text(render_modelfile(plan.worker, plan.context_tokens, role="worker"), encoding="utf-8")
        router_path.write_text(render_modelfile(plan.router, min(plan.context_tokens, 32_768), role="router"), encoding="utf-8")
        config_path.write_text(render_config(plan, ollama_url), encoding="utf-8")
    manager.create_alias("nedster-worker", worker_path, dry_run=dry_run)
    manager.create_alias("nedster-router", router_path, dry_run=dry_run)
    verification: str | dict[str, bool] = "dry-run"
    if not dry_run:
        emit("Verifying generation, structured output, tool calling, and embeddings…")
        verification = manager.verify_capabilities(embedding=plan.embedding)
        if not all(verification.values()):
            failed = ", ".join(name for name, passed in verification.items() if not passed)
            raise RuntimeError(f"Model verification failed: {failed}")
    result = {
        "plan": asdict(plan),
        "config": str(config_path),
        "aliases": ["nedster-worker", "nedster-router"],
        "verification": verification,
    }
    return result
