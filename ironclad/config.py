from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class ModelConfig:
    ollama_url: str = "http://127.0.0.1:11434"
    router: str = "auto"
    coder: str = "auto"
    reasoner: str = "auto"
    critic: str = "auto"
    embedding: str = "embeddinggemma:300m-qat-q4_0"
    router_keep_alive: str = "-1"
    worker_keep_alive: str = "90s"
    local_thinking: bool | str = True
    cloud_oracle_enabled: bool = False
    cloud_model: str = "gpt-5.6-sol"
    cloud_reasoning_effort: str = "high"
    cloud_reasoning_mode: str = "standard"
    cloud_reasoning_context: str = "current_turn"


@dataclass(slots=True)
class SandboxConfig:
    backend: str = "auto"
    network: str = "deny"
    max_seconds: int = 120
    max_output_chars: int = 100_000
    allow_unsafe_fallback: bool = False
    docker_image: str = "python:3.13-slim"
    firecracker_wrapper: str = ""


@dataclass(slots=True)
class PolicyConfig:
    autonomous: bool = False
    max_auto_risk: str = "write"
    allowed_commands: list[str] = field(
        default_factory=lambda: [
            "python",
            "python3",
            "pytest",
            "git",
            "rg",
            "ruff",
            "uv",
            "npm",
            "node",
            "cargo",
            "go",
        ]
    )
    denied_fragments: list[str] = field(
        default_factory=lambda: [
            "rm -rf /",
            "Remove-Item -Recurse",
            "format ",
            "mkfs",
            "shutdown",
            "reboot",
            ":(){:|:&};:",
        ]
    )


@dataclass(slots=True)
class RuntimeConfig:
    project_root: Path = field(default_factory=Path.cwd)
    state_dir: Path = field(default_factory=lambda: Path.home() / ".nedster" / "runtime")
    max_worker_turns: int = 8
    max_parallel_workers: int = 2
    profile: str = "balanced"
    ledger_key_env: str = "NEDSTER_LEDGER_KEY"


@dataclass(slots=True)
class NedsterConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _apply(obj: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(obj, key):
            current = getattr(obj, key)
            if isinstance(current, Path):
                value = Path(value).expanduser()
            setattr(obj, key, value)


def load_config(path: str | Path | None = None, project_root: str | Path | None = None) -> NedsterConfig:
    config = NedsterConfig()
    if path:
        candidate = Path(path).expanduser()
    else:
        user_config = Path.home() / ".nedster" / "config.toml"
        project_config = Path("config/nedster.toml")
        legacy_config = Path("config/ironclad.toml")
        candidate = next((item for item in (user_config, project_config, legacy_config) if item.exists()), user_config)
    if candidate.exists():
        data = tomllib.loads(candidate.read_text(encoding="utf-8"))
        _apply(config.models, dict(data.get("models") or {}))
        _apply(config.sandbox, dict(data.get("sandbox") or {}))
        _apply(config.policy, dict(data.get("policy") or {}))
        _apply(config.runtime, dict(data.get("runtime") or {}))

    config.runtime.project_root = Path(
        project_root or os.getenv("NEDSTER_PROJECT") or config.runtime.project_root
    ).expanduser().resolve()
    config.runtime.state_dir = Path(
        os.getenv("NEDSTER_STATE_DIR") or config.runtime.state_dir
    ).expanduser().resolve()
    config.models.ollama_url = os.getenv("OLLAMA_HOST", config.models.ollama_url).rstrip("/")
    config.models.cloud_model = os.getenv("NEDSTER_CLOUD_MODEL", config.models.cloud_model)
    config.models.cloud_oracle_enabled = _bool_env(
        "NEDSTER_CLOUD_ORACLE", config.models.cloud_oracle_enabled
    )
    config.policy.autonomous = _bool_env("NEDSTER_AUTONOMOUS", config.policy.autonomous)
    return config


# Compatibility for v3.0 integrations. New code should import NedsterConfig.
IroncladConfig = NedsterConfig
