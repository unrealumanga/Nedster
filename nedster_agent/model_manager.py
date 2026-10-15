from __future__ import annotations

import contextlib
import difflib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True, slots=True)
class InstalledModel:
    """One model present in the local Ollama store, as shown by the picker and `models list`."""

    name: str
    size_gb: float | None = None
    parameter_size: str = ""
    quantization: str = ""
    context_length: int | None = None
    current: bool = False
    embedding: bool = False  # cannot chat (embedding-only): hidden unless `models list --all`

    def size_label(self) -> str:
        return f"{self.size_gb:.1f} GB" if self.size_gb is not None else "?"

    def context_label(self) -> str:
        return "unknown" if self.context_length is None else str(self.context_length)

    def kind_label(self) -> str:
        return "(embedding)" if self.embedding else ""


MODEL_PLANS: dict[str, dict[str, ModelPlan]] = {
    "edge-8g": {
        "fast": ModelPlan(
            "fast",
            "Fast",
            "qwen3.5:4b-q4_K_M",
            "qwen3.5:4b-q4_K_M",
            "embeddinggemma:300m-qat-q4_0",
            16_384,
            3.7,
            "One compact model keeps memory pressure low.",
        ),
        "balanced": ModelPlan(
            "balanced",
            "Balanced",
            "qwen3.5:9b-q4_K_M",
            "qwen3.5:4b-q4_K_M",
            "embeddinggemma:300m-qat-q4_0",
            16_384,
            10.3,
            "A stronger worker with a responsive router.",
        ),
        "maximum": ModelPlan(
            "maximum",
            "Maximum",
            "qwen3.5:9b-q4_K_M",
            "qwen3.5:4b-q4_K_M",
            "embeddinggemma:300m-qat-q4_0",
            32_768,
            10.3,
            "Uses more context and may spill into system RAM.",
        ),
    },
    "forge-16g": {
        "fast": ModelPlan(
            "fast",
            "Fast",
            "qwen3.5:9b-q4_K_M",
            "qwen3.5:4b-q4_K_M",
            "embeddinggemma:300m-qat-q4_0",
            32_768,
            10.3,
            "Both models fit typical 16 GB accelerators.",
        ),
        "balanced": ModelPlan(
            "balanced",
            "Balanced",
            "qwen3.5:27b-q4_K_M",
            "qwen3.5:4b-q4_K_M",
            "embeddinggemma:300m-qat-q4_0",
            32_768,
            20.7,
            "Pages the worker when needed while the router stays quick.",
        ),
        "maximum": ModelPlan(
            "maximum",
            "Maximum",
            "qwen3-coder:30b",
            "qwen3.5:4b-q4_K_M",
            "embeddinggemma:300m-qat-q4_0",
            65_536,
            22.7,
            "Coding-specialized worker; partial CPU offload may occur.",
        ),
    },
    "titan-32g": {
        "fast": ModelPlan(
            "fast",
            "Fast",
            "qwen3.5:9b-q4_K_M",
            "qwen3.5:4b-q4_K_M",
            "embeddinggemma:300m-qat-q4_0",
            32_768,
            10.3,
            "Low latency and generous VRAM headroom.",
        ),
        "balanced": ModelPlan(
            "balanced",
            "Balanced",
            "qwen3.6:27b",
            "qwen3.5:4b-q4_K_M",
            "embeddinggemma:300m-qat-q4_0",
            65_536,
            20.7,
            "Recommended: strong tool use with room for router and context.",
        ),
        "maximum": ModelPlan(
            "maximum",
            "Maximum",
            "qwen3.6:35b",
            "qwen3.5:4b-q4_K_M",
            "embeddinggemma:300m-qat-q4_0",
            65_536,
            27.7,
            "Highest local quality; models are paged instead of kept together.",
        ),
    },
}


def recommend_plan(snapshot: HardwareSnapshot | None = None, preference: str = "balanced") -> ModelPlan:
    snapshot = snapshot or probe_hardware()
    tier = choose_profile(snapshot).name
    return MODEL_PLANS[tier][preference]


def state_dir() -> Path:
    """Config directory (legacy name). NEDSTER_CONFIG_DIR wins over NEDSTER_STATE_DIR."""
    raw = os.getenv("NEDSTER_CONFIG_DIR") or os.getenv("NEDSTER_STATE_DIR") or Path.home() / ".nedster"
    return Path(raw).expanduser()


def render_modelfile(base: str, context_tokens: int, *, role: str) -> str:
    # Keep the publisher's sampler calibration. Role-specific prompts and
    # constrained schemas provide control without forcing Qwen off-distribution.
    _ = role
    return f"FROM {base}\nPARAMETER num_ctx {context_tokens}\n"


def render_config(plan: ModelPlan, ollama_url: str) -> str:
    return f'''# Managed by `nedster setup`. Edit any value and rerun `nedster doctor`.
[models]
ollama_url = "{ollama_url.rstrip("/")}"
router = "nedster-router"
coder = "nedster-worker"
reasoner = "nedster-worker"
critic = "nedster-worker"
embedding = "{plan.embedding}"
router_keep_alive = "5m"
worker_keep_alive = "90s"
local_thinking = true

[policy]
autonomous = true
max_auto_risk = "write"

[sandbox]
allow_host_execution = false

[runtime]
max_worker_turns = 8
max_worker_restarts = 1
max_step_attempts = 2
max_parallel_workers = 2
profile = "{plan.profile}"
chat_max_tokens = 96
answer_max_tokens = 768
conversation_history_chars = 6000
max_grounding_sources = 5
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

    def list_model_details(self) -> list[dict[str, Any]]:
        response = httpx.get(f"{self.url}/api/tags", timeout=10)
        response.raise_for_status()
        return [dict(item) for item in response.json().get("models", [])]

    def list_models(self) -> list[str]:
        return [str(item["name"]) for item in self.list_model_details()]

    def show(self, model: str) -> dict[str, Any]:
        """The raw ``/api/show`` payload for one model."""
        response = httpx.post(f"{self.url}/api/show", json={"model": model}, timeout=10)
        response.raise_for_status()
        return dict(response.json())

    def capabilities(self, model: str) -> set[str]:
        """The capability tags Ollama reports for a model (empty when the field is missing)."""
        caps = self.show(model).get("capabilities")
        return {str(item) for item in caps} if isinstance(caps, list) else set()

    def context_window(self, model: str) -> tuple[int | None, int | None]:
        """(configured num_ctx, trained context length) from /api/show; None when unknown."""
        data = self.show(model)
        return _num_ctx(str(data.get("parameters") or "")), _trained_ctx(data.get("model_info") or {})

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
            with httpx.stream(
                "GET", "https://ollama.com/download/OllamaSetup.exe", follow_redirects=True, timeout=120
            ) as response:
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
            raise RuntimeError(
                "Ollama was installed but its executable is not on PATH. Restart the terminal and rerun setup."
            )
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

    def copy_model(self, source: str, target: str) -> None:
        executable = self.executable
        if not executable:
            raise RuntimeError("The Ollama executable is unavailable.")
        subprocess.run([executable, "cp", source, target], check=True, stdout=sys.stderr)

    def remove_model(self, name: str) -> None:
        executable = self.executable
        if executable:
            subprocess.run([executable, "rm", name], check=False, capture_output=True)

    def verify_generation(self, model: str = "nedster-router") -> str:
        response = httpx.post(
            f"{self.url}/api/generate",
            json={
                "model": model,
                "prompt": "Reply with only: ready",
                "think": False,
                "stream": False,
                "options": {"num_predict": 8},
            },
            timeout=120,
        )
        response.raise_for_status()
        return str(response.json().get("response", "")).strip()

    def verify_capabilities(
        self,
        *,
        router: str = "nedster-router",
        embedding: str | None = "embeddinggemma:300m-qat-q4_0",
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
                "think": False,
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
                "think": False,
                "stream": False,
            },
            timeout=180,
        )
        tool_response.raise_for_status()
        tool_calls = tool_response.json().get("message", {}).get("tool_calls", [])

        result = {
            "generation": bool(generation),
            "structured_output": structured_value.get("ready") is True,
            "tool_calling": bool(tool_calls),
        }
        if embedding is None:
            return result
        embed_response = httpx.post(
            f"{self.url}/api/embed",
            json={"model": embedding, "input": "Nedster local agent readiness"},
            timeout=180,
        )
        embed_response.raise_for_status()
        embeddings = embed_response.json().get("embeddings", [])
        result["embeddings"] = bool(embeddings and embeddings[0])
        return result


class SetupError(RuntimeError):
    """Setup failed; the previous configuration was left untouched."""

    def __init__(self, message: str, remedy: str = ""):
        super().__init__(message)
        self.remedy = remedy


MANAGED_KEYS = {
    "models": {"ollama_url", "router", "coder", "reasoner", "critic", "embedding"},
    "runtime": {"profile"},
}


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(str(value))


def _dump_toml(data: dict[str, object]) -> str:
    lines = ["# Managed by `nedster setup`. Edit any value and rerun `nedster doctor`."]
    lines += [f"{k} = {_toml_value(v)}" for k, v in data.items() if not isinstance(v, dict)]
    for name, table in data.items():
        if isinstance(table, dict):
            lines += ["", f"[{name}]"] + [f"{k} = {_toml_value(v)}" for k, v in table.items()]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class ExecutionChoice:
    """What the user explicitly agreed to; None fields leave the existing value untouched."""

    max_auto_risk: str
    allow_host_execution: bool | None = None


def describe_permissions(max_auto_risk: object, allow_host: object) -> str:
    """One-line statement of exactly what Nedster may do without asking."""
    if max_auto_risk == "execute":
        where = "directly on this machine (NO isolation)" if allow_host else "only inside an isolated sandbox"
        return f"Nedster may read files, write files and run commands automatically, {where}."
    if max_auto_risk == "write":
        return "Nedster may read and write files automatically; it will ask before running commands."
    return "Nedster may only read files automatically; it will ask before writing or running commands."


def merge_config(
    existing: dict[str, object],
    plan: ModelPlan,
    ollama_url: str,
    execution: ExecutionChoice | None = None,
) -> dict[str, object]:
    """Managed keys take the new plan; every other user-edited key is preserved.

    An explicit `execution` choice (the user's answer) overrides policy/sandbox consent keys.
    """
    generated = tomllib.loads(render_config(plan, ollama_url))
    merged: dict[str, object] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in existing.items()}
    for section, table in generated.items():
        target = merged.get(section)
        if not isinstance(target, dict):
            target = merged[section] = {}
        for key, value in table.items():
            if key in MANAGED_KEYS.get(section, set()) or key not in target:
                target[key] = value
    if execution is not None:
        merged["policy"]["max_auto_risk"] = execution.max_auto_risk  # type: ignore[index]
        if execution.allow_host_execution is not None:
            merged["sandbox"]["allow_host_execution"] = execution.allow_host_execution  # type: ignore[index]
    return merged


def activate_config(config_dir: Path, text: str, *, stamp: str | None = None) -> Path | None:
    """Atomically replace config.toml; returns the backup path (None if there was no old config)."""
    config_dir.mkdir(parents=True, exist_ok=True)
    target = config_dir / "config.toml"
    candidate = config_dir / "config.toml.new"
    candidate.write_text(text, encoding="utf-8")
    backup: Path | None = None
    try:
        if target.exists():
            backup = config_dir / f"config.toml.bak-{stamp or time.strftime('%Y%m%d-%H%M%S')}"
            shutil.copy2(target, backup)
        os.replace(candidate, target)
    except OSError:
        candidate.unlink(missing_ok=True)
        raise
    return backup


def set_default_worker(config_path: Path, name: str) -> Path | None:
    """Persist ``models.coder = name`` in ``config.toml``, returning the backup path (None when new).

    Only the worker key changes; every other user value and section is preserved.
    """
    existing: dict[str, Any] = {}
    if config_path.is_file():
        existing = tomllib.loads(config_path.read_text(encoding="utf-8"))
    models = existing.get("models")
    if not isinstance(models, dict):
        models = {}
        existing["models"] = models
    models["coder"] = name
    return activate_config(config_path.parent, _dump_toml(existing))


STAGED_ALIASES = {"nedster-worker": "nedster-worker-next", "nedster-router": "nedster-router-next"}


def _verify_staged(manager: OllamaManager, plan: ModelPlan, with_embeddings: bool) -> dict[str, dict[str, bool]]:
    verification: dict[str, dict[str, bool]] = {}
    for alias, staged in STAGED_ALIASES.items():
        verification[alias.removeprefix("nedster-")] = manager.verify_capabilities(router=staged, embedding=None)
    if with_embeddings:
        checks = manager.verify_capabilities(router=STAGED_ALIASES["nedster-router"], embedding=plan.embedding)
        verification["embedding"] = {"embeddings": checks["embeddings"]}
    return verification


def install_plan(
    plan: ModelPlan,
    *,
    ollama_url: str = "http://127.0.0.1:11434",
    install_ollama: bool = True,
    dry_run: bool = False,
    with_embeddings: bool = False,
    progress: ProgressCallback | None = None,
    manager: OllamaManager | None = None,
    config_dir: Path | None = None,
    execution: ExecutionChoice | None = None,
) -> dict[str, object]:
    """Stage, verify, back up, then atomically activate. Any failure leaves the old config in place."""
    emit = progress or (lambda _message: None)
    manager = manager or OllamaManager(ollama_url, emit)
    root = config_dir or state_dir()
    config_path = root / "config.toml"
    created: list[str] = []
    verification: dict[str, dict[str, bool]] | str = "dry-run"
    backup: Path | None = None
    try:
        if install_ollama:
            manager.install(dry_run=dry_run)
        manager.ensure_service(dry_run=dry_run)
        installed = set() if dry_run else set(manager.list_models())
        wanted = (plan.worker, plan.router, *((plan.embedding,) if with_embeddings else ()))
        for model in dict.fromkeys(wanted):
            if model in installed:
                emit(f"Keeping installed model {model}.")
            else:
                manager.pull(model, dry_run=dry_run)

        modelfiles = root / "models" / "modelfiles"
        if not dry_run:
            modelfiles.mkdir(parents=True, exist_ok=True)
        specs = {
            "nedster-worker": (plan.worker, plan.context_tokens, "worker"),
            "nedster-router": (plan.router, min(plan.context_tokens, 32_768), "router"),
        }
        for alias, (base, context, role) in specs.items():
            path = modelfiles / f"{role}.Modelfile"
            if not dry_run:
                path.write_text(render_modelfile(base, context, role=role), encoding="utf-8")
            manager.create_alias(STAGED_ALIASES[alias], path, dry_run=dry_run)
            created.append(STAGED_ALIASES[alias])

        if not dry_run:
            emit("Verifying worker and router: generation, structured output, tool calling…")
            verification = _verify_staged(manager, plan, with_embeddings)
            failed = [f"{role}.{name}" for role, checks in verification.items()
                      for name, ok in checks.items() if not ok]
            if failed:
                raise SetupError(
                    f"Model verification failed: {', '.join(failed)}",
                    "Check `ollama ps`, or choose a smaller profile: nedster setup --profile fast",
                )
            for alias, staged in STAGED_ALIASES.items():
                manager.copy_model(staged, alias)
            existing = tomllib.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
            backup = activate_config(root, _dump_toml(merge_config(existing, plan, ollama_url, execution)))
    except SetupError:
        raise
    except (RuntimeError, OSError, subprocess.CalledProcessError, httpx.HTTPError, ValueError) as exc:
        raise SetupError(
            str(exc), "Fix the problem above and rerun `nedster setup`; your previous config is unchanged."
        ) from exc
    finally:
        if not dry_run:
            for name in created:
                manager.remove_model(name)
    return {
        "plan": asdict(plan),
        "config": str(config_path),
        "backup": str(backup) if backup else None,
        "aliases": ["nedster-worker", "nedster-router"],
        "embeddings": with_embeddings,
        "verification": verification,
    }


HOST_DETAIL = "host process — no isolation (opted in)"


def probe_execution(config: Any) -> dict[str, object]:
    """Probe the selected execution backend -> {ready, backend, isolation, detail[, remedy]}."""
    requested = config.sandbox.backend
    try:
        from ironclad.tool_runtime.backends import select_backend

        selection = select_backend(Path(config.runtime.project_root), Path(tempfile.gettempdir()), config)
    except Exception as exc:  # noqa: BLE001 — a broken backend probe means "not ready", never a crash
        return {"ready": False, "backend": requested, "isolation": "unknown", "detail": f"backend probe failed: {exc}"}
    status = selection.status
    if selection.backend is None or status is None:
        detail = selection.remedy or "no execution backend is ready"
        return {"ready": False, "backend": requested, "isolation": "unknown", "detail": detail}
    detail = HOST_DETAIL if status.name == "host" else f"{status.name} ({status.isolation} isolation)"
    return {"ready": True, "backend": status.name, "isolation": status.isolation, "detail": detail}


def _num_ctx(parameters: str) -> int | None:
    for line in parameters.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "num_ctx" and parts[1].isdigit():
            return int(parts[1])
    return None


def _trained_ctx(model_info: dict[str, Any]) -> int | None:
    for key, value in model_info.items():
        if key.endswith(".context_length") and isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def match_installed_model(name: str, installed: Sequence[str]) -> str | None:
    """The installed name matching ``name`` (exact, or with an implicit ``:latest``), else None."""
    wanted = name.strip()
    if not wanted:
        return None
    for candidate in installed:
        if candidate == wanted or candidate == f"{wanted}:latest" or wanted == f"{candidate}:latest":
            return candidate
    return None


def model_not_installed_message(name: str, installed: Sequence[str], *, suggestions: int = 3) -> str:
    """The one remedy the TUI and CLI print for a model that is not installed."""
    message = f"Model '{name}' is not installed. Choose one with /models, or run: ollama pull {name}"
    closest = difflib.get_close_matches(name, list(installed), n=suggestions, cutoff=0.4)
    if closest:
        message += f". Closest installed: {', '.join(closest)}"
    return message


def is_model_missing_error(text: str) -> bool:
    """True when a transport error looks like Ollama rejecting an unknown model."""
    lowered = text.lower()
    return "404" in lowered or "not found" in lowered


EMBEDDING_NAME_MARKERS = ("embed",)


def is_embedding_model(name: str, capabilities: Collection[str] | None = None) -> bool:
    """True when a model cannot chat: an obvious name, or Ollama reporting no ``completion`` capability."""
    if any(marker in name.lower() for marker in EMBEDDING_NAME_MARKERS):
        return True
    caps = {str(item).lower() for item in (capabilities or ())}
    return bool(caps) and "completion" not in caps


def list_installed_models(
    manager: OllamaManager,
    current: str = "",
    *,
    with_context: bool = True,
    include_embedding: bool = False,
) -> list[InstalledModel]:
    """Installed models with size, parameters, quantization and (cheap) context.

    Context and capabilities come from ``/api/show``; a model whose probe fails is reported as
    ``unknown`` rather than guessed. Models that cannot chat (embeddings) are hidden unless
    ``include_embedding`` is set. A dead endpoint raises the underlying ``httpx`` error so the
    caller can explain it.
    """
    models: list[InstalledModel] = []
    probe = getattr(manager, "context_window", None)
    show = getattr(manager, "show", None)
    seen: set[str] | None = getattr(manager, "_nedster_show_seen", None)
    shown_info: dict[str, Any] | None = getattr(manager, "_nedster_shown_info", None)
    if seen is None:
        seen = set()
        manager._nedster_show_seen = seen
    if shown_info is None:
        shown_info = {}
        manager._nedster_shown_info = shown_info
    for entry in manager.list_model_details():
        name = str(entry.get("name") or "")
        if not name:
            continue
        details = entry.get("details")
        details = details if isinstance(details, Mapping) else {}
        size = entry.get("size")
        context: int | None = None
        capabilities: set[str] = set()
        if with_context and callable(show):  # one /api/show gives both context and capabilities
            if name in seen:
                info = shown_info[name]
            else:
                try:
                    info = show(name)
                except (httpx.HTTPError, ValueError):
                    info = {}
                seen.add(name)
                shown_info[name] = info
            context = _num_ctx(str(info.get("parameters") or "")) or _trained_ctx(info.get("model_info") or {})
            caps = info.get("capabilities")
            capabilities = {str(item) for item in caps} if isinstance(caps, list) else set()
        elif with_context and callable(probe):
            try:
                configured, trained = probe(name)
            except (httpx.HTTPError, ValueError):
                configured = trained = None
            context = configured or trained
        embedding = is_embedding_model(name, capabilities)
        if embedding and not include_embedding:
            continue
        models.append(
            InstalledModel(
                name=name,
                size_gb=round(float(size) / 1_000_000_000, 1) if isinstance(size, (int, float)) else None,
                parameter_size=str(details.get("parameter_size") or ""),
                quantization=str(details.get("quantization_level") or ""),
                context_length=context,
                current=match_installed_model(current, [name]) is not None,
                embedding=embedding,
            )
        )
    return models


KV_CACHE_TYPES = ("f16", "q8_0", "q4_0")


def context_report(
    config: Any, manager: OllamaManager, env: Mapping[str, str] | None = None
) -> dict[str, object]:
    """Context window and KV-cache settings that bound how long a task can run.

    Ollama reads OLLAMA_FLASH_ATTENTION / OLLAMA_KV_CACHE_TYPE from the *server's* environment; Nedster can only see its
    own, so the KV values are reported as "as seen by this shell" and are advice, not a readiness gate.
    """
    env = os.environ if env is None else env
    models_cfg = config.models
    worker = models_cfg.coder if models_cfg.coder != "auto" else "nedster-worker"
    budget = int(getattr(config.runtime, "context_tokens", 0) or 0)
    report: dict[str, object] = {"worker": worker, "engine_budget": budget, "num_ctx": None, "trained": None}
    with contextlib.suppress(httpx.HTTPError, ValueError):
        report["num_ctx"], report["trained"] = manager.context_window(worker)
    flash = env.get("OLLAMA_FLASH_ATTENTION", "").strip().lower() in {"1", "true", "yes", "on"}
    kv_type = env.get("OLLAMA_KV_CACHE_TYPE", "").strip().lower() or "f16"
    report.update(flash_attention=flash, kv_cache_type=kv_type)
    # Track where we got these values from.
    report["kv_source"] = "client shell (Ollama server settings unknown)"

    advice: list[str] = []
    num_ctx = report["num_ctx"]
    if not budget:
        report["engine_budget"] = num_ctx or 32_768
    elif isinstance(num_ctx, int) and budget < num_ctx:
        advice.append(f"engine budget {budget} < model num_ctx {num_ctx}: set runtime.context_tokens = {num_ctx}")
    if isinstance(num_ctx, int) and budget > num_ctx:
        advice.append(f"engine budget {budget} > model num_ctx {num_ctx}: Ollama will truncate; lower runtime.context_tokens")
    if not flash or kv_type == "f16":
        advice.append(kv_cache_advice())
    report["advice"] = advice
    return report


def kv_cache_advice(system: str | None = None) -> str:
    """Return a platform-specific, copy-pasteable command to set KV-cache env vars for the Ollama server."""
    import platform

    system = system or platform.system()
    base = (
        "To halve KV-cache memory (about 2x context in the same VRAM), "
        "set these for the Ollama server and restart it (quit Ollama from the tray first):"
    )
    if system == "Windows":
        return (
            base
            + ' setx OLLAMA_FLASH_ATTENTION 1 ; setx OLLAMA_KV_CACHE_TYPE q8_0'
        )
    elif system == "Darwin":
        return (
            base
            + ' launchctl setenv OLLAMA_FLASH_ATTENTION 1 && launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0 (then restart the Ollama app)'
        )
    else:  # Linux / other
        return (
            base
            + ' add Environment="OLLAMA_FLASH_ATTENTION=1" and Environment="OLLAMA_KV_CACHE_TYPE=q8_0" with `sudo systemctl edit ollama`, then `sudo systemctl restart ollama`'
        )


def readiness(config: Any, manager: OllamaManager | None = None) -> dict[str, dict[str, object]]:
    """inference / read_only / execution tiers, all against the resolved endpoint."""
    models_cfg = config.models
    manager = manager or OllamaManager(models_cfg.ollama_url)
    worker = models_cfg.coder if models_cfg.coder != "auto" else "nedster-worker"
    inference: dict[str, object] = {"ready": False, "endpoint": manager.url, "worker": worker}
    if not manager.api_available():
        inference["detail"] = f"Ollama not reachable at {manager.url}"
        inference["remedy"] = "Start Ollama (`ollama serve`) or run: nedster setup"
    else:
        try:
            installed = manager.list_models()
        except (httpx.HTTPError, ValueError) as exc:
            installed = []
            inference["detail"] = f"could not list models: {exc}"
        if worker in installed or f"{worker}:latest" in installed:
            inference.update(ready=True, detail=f"worker model {worker} present")
        else:
            inference.setdefault("detail", f"worker model {worker} not installed")
            inference["remedy"] = "Run: nedster setup"
    read_only: dict[str, object] = {"ready": inference["ready"], "detail": "read-only tools need only the worker model"}
    if not read_only["ready"]:
        read_only["remedy"] = inference["remedy"]
    execution = probe_execution(config)
    if not execution["ready"]:
        execution["remedy"] = (
            "Start Docker or install a sandbox backend; or opt in to unisolated execution with "
            'sandbox.backend = "host" and sandbox.allow_host_execution = true'
        )
    return {"inference": inference, "read_only": read_only, "execution": execution}
