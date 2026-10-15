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


@dataclass(slots=True)
class SandboxConfig:
    backend: str = "auto"
    network: str = "deny"
    max_seconds: int = 120
    max_output_chars: int = 100_000
    allow_unsafe_fallback: bool = False
    allow_host_execution: bool = False
    docker_image: str = "python:3.13-slim"
    docker_memory: str = "4g"
    docker_cpus: str = "2"
    firecracker_wrapper: str = ""


@dataclass(slots=True)
class PolicyConfig:
    # Execute-level autonomy needs an explicit opt-in (policy.max_auto_risk = "execute").
    # denied_fragments is advisory substring matching, NOT a security boundary.
    autonomous: bool = True
    max_auto_risk: str = "write"
    allowed_commands: list[str] = field(default_factory=lambda: [
        "python", "python3", "pytest", "git", "rg", "ruff", "uv", "npm", "node", "cargo", "go"
    ])
    denied_fragments: list[str] = field(default_factory=lambda: [
        "rm -rf /", "Remove-Item -Recurse", "format ", "mkfs", "shutdown", "reboot", ":(){:|:&};:"
    ])


@dataclass(slots=True)
class RuntimeConfig:
    project_root: Path = field(default_factory=Path.cwd)
    state_dir: Path = field(default_factory=lambda: Path.home() / ".nedster" / "data")
    max_worker_turns: int = 8
    max_worker_restarts: int = 1
    max_step_attempts: int = 2
    max_parallel_workers: int = 2
    profile: str = "balanced"
    ledger_key_env: str = "NEDSTER_LEDGER_KEY"
    chat_max_tokens: int = 96
    answer_max_tokens: int = 768
    conversation_history_chars: int = 6_000
    max_grounding_sources: int = 5
    context_tokens: int = 0  # engine context budget; 0 = the worker model's num_ctx (from Ollama), else 32768
    max_steps: int = 40  # model calls per run; 0 = no limit
    max_tool_calls: int = 120  # tool calls per run; 0 = no limit
    soul: str = "off"  # opt-in personality: "off" (plain), "ui" (status lines only) or "full" (soul.md in the prompt)


@dataclass(slots=True)
class NedsterConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    engine: str = "native"
    warnings: list[str] = field(default_factory=list)


Config = NedsterConfig


_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off", ""}
_ENGINES = ("native", "legacy")  # "legacy" is accepted once more so old configs warn instead of failing
_SOUL_MODES = ("off", "ui", "full")
_RISK_ORDER = ("read", "write", "execute", "destructive", "external")
# A cloned repo's .nedster/config.toml is untrusted: it may only pick model names / tuning,
# and may only tighten (never loosen) the security settings below.
_PROJECT_SAFE_KEYS = frozenset({
    "models.router", "models.coder", "models.reasoner", "models.critic", "models.summarizer",
    "runtime.context_tokens", "runtime.profile", "engine",
})
_PROJECT_TIGHTEN_KEYS = frozenset({
    "sandbox.allow_host_execution", "policy.autonomous", "policy.max_auto_risk",
    "policy.allowed_commands", "policy.denied_fragments",
})
_SECTIONS = ("models", "sandbox", "policy", "runtime")
# Legacy / convenience variables -> dotted field. Applied before NEDSTER_<SECTION>__<FIELD>.
_ENV_ALIASES: tuple[tuple[str, str], ...] = (
    ("OLLAMA_HOST", "models.ollama_url"),
    ("NEDSTER_AUTONOMOUS", "policy.autonomous"),
    ("NEDSTER_PROJECT", "runtime.project_root"),
    ("NEDSTER_STATE_DIR", "runtime.state_dir"),
    ("NEDSTER_DATA_DIR", "runtime.state_dir"),
    ("NEDSTER_ENGINE", "engine"),
)


def config_dir() -> Path:
    """Directory holding the user's config.toml (not the data directory)."""
    return Path(os.getenv("NEDSTER_CONFIG_DIR") or Path.home() / ".nedster").expanduser()


def data_dir() -> Path:
    return Path(os.getenv("NEDSTER_DATA_DIR") or os.getenv("NEDSTER_STATE_DIR") or Path.home() / ".nedster" / "data").expanduser()


def normalize_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if value and "://" not in value:
        value = "http://" + value
    return value


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        else:
            result[name] = value
    return result


def _field_names(config: Config) -> list[str]:
    names = ["engine"]
    for section in _SECTIONS:
        obj = getattr(config, section)
        names.extend(f"{section}.{k}" for k in obj.__dataclass_fields__)
    return names


def _coerce(current: Any, value: Any, key: str) -> Any:
    if isinstance(current, Path):
        return Path(str(value)).expanduser()
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _BOOL_TRUE:
            return True
        if text in _BOOL_FALSE:
            return False
        raise ValueError(f"{key}: expected a boolean, got {value!r}")
    if isinstance(current, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key}: expected an integer, got {value!r}") from None
    if isinstance(current, list):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)
    if key == "models.local_thinking" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _BOOL_TRUE:
            return True
        if lowered in _BOOL_FALSE:
            return False
    return value


def _set(config: Config, key: str, value: Any) -> bool:
    parts = key.split(".")
    target: Any = config
    for part in parts[:-1]:
        if part not in _SECTIONS or not hasattr(target, part):
            return False
        target = getattr(target, part)
    name = parts[-1]
    if name not in getattr(target, "__dataclass_fields__", {}):
        return False
    coerced = _coerce(getattr(target, name), value, key)
    if key == "engine":
        if coerced not in _ENGINES:
            raise ValueError(f"engine must be one of {', '.join(_ENGINES)}, got {coerced!r}")
        if coerced == "legacy":  # the v3 engine was removed (tag v3-legacy); run native with a warning
            coerced = "native"
            message = 'engine = "legacy" was removed; the native engine runs instead (rollback: tag v3-legacy)'
            if message not in config.warnings:
                config.warnings.append(message)
    if key == "runtime.soul":
        text = str(coerced).strip().lower()
        if text not in _SOUL_MODES:
            message = f'unknown runtime.soul {coerced!r}; using "off"'
            if message not in config.warnings:
                config.warnings.append(message)
            text = "off"
        coerced = text
    setattr(target, name, coerced)
    return True


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _flatten(tomllib.loads(path.read_text(encoding="utf-8")))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {path}: {exc}") from exc


def _apply_project_layer(config: Config, values: dict[str, Any], provenance: dict[str, str]) -> None:
    """Apply an untrusted repo-supplied layer: allowlisted keys plus tighten-only security keys."""
    for key, value in values.items():
        if key in _PROJECT_SAFE_KEYS:
            if _set(config, key, value):
                provenance[key] = "project"
        elif key in _PROJECT_TIGHTEN_KEYS:
            _tighten(config, key, value, provenance)
        else:
            config.warnings.append(f"ignored project config key {key!r}: not allowed in .nedster/config.toml")


def _tighten(config: Config, key: str, value: Any, provenance: dict[str, str]) -> None:
    probe = Config()
    if not _set(probe, key, value):
        return
    wanted = _get(probe, key)
    current = _get(config, key)
    if key == "sandbox.allow_host_execution" or key == "policy.autonomous":
        result = current and wanted
    elif key == "policy.max_auto_risk":
        wanted_text = str(wanted).lower()
        if wanted_text not in _RISK_ORDER:
            config.warnings.append(f"ignored project config key {key!r}: unknown risk level {wanted!r}")
            return
        current_rank = _RISK_ORDER.index(str(current).lower()) if str(current).lower() in _RISK_ORDER else 1
        result = wanted_text if _RISK_ORDER.index(wanted_text) < current_rank else current
    elif key == "policy.allowed_commands":
        allowed = {str(item) for item in wanted}
        result = [item for item in current if item in allowed]
    else:  # policy.denied_fragments: union
        result = list(current) + [item for item in wanted if item not in current]
    if result != current:
        _set(config, key, result)
        provenance[key] = "project"
    if key != "policy.denied_fragments" and wanted != result:
        config.warnings.append(f"project config key {key!r}: loosening ignored, only tightening is accepted")


def _get(config: Config, key: str) -> Any:
    target: Any = config
    for part in key.split("."):
        target = getattr(target, part)
    return target


def resolve_config(
    cli_overrides: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> tuple[Config, dict[str, str]]:
    """defaults < user < project < env < CLI. Returns (config, {dotted field: source})."""
    overrides = dict(cli_overrides or {})
    config = Config()
    provenance = {key: "default" for key in _field_names(config)}

    explicit = overrides.pop("config", None)
    layers: list[tuple[str, dict[str, Any]]] = []
    if explicit:
        explicit_path = Path(str(explicit)).expanduser()
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {explicit_path}")
        layers.append(("user", _read(explicit_path)))
    else:
        layers.append(("user", _read(config_dir() / "config.toml")))
    root = Path(project_root).expanduser().resolve() if project_root is not None else None
    if root is not None:
        layers.append(("project", _read(root / ".nedster" / "config.toml")))
    for source, values in layers:
        if source == "project":
            _apply_project_layer(config, values, provenance)
            continue
        for key, value in values.items():
            if _set(config, key, value):
                provenance[key] = source

    env = os.environ
    for name, key in _ENV_ALIASES:
        if env.get(name):
            _set(config, key, env[name])
            provenance[key] = "env"
    for name, raw in env.items():
        if name.startswith("NEDSTER_") and "__" in name:
            key = name[len("NEDSTER_"):].lower().replace("__", ".")
            if _set(config, key, raw):
                provenance[key] = "env"

    for key, value in overrides.items():
        if value is not None and _set(config, key.replace("__", "."), value):
            provenance[key.replace("__", ".")] = "cli"

    if root is not None and provenance["runtime.project_root"] in {"default", "user"}:
        config.runtime.project_root = root
        provenance["runtime.project_root"] = "project"
    config.runtime.project_root = Path(config.runtime.project_root).expanduser().resolve()
    config.runtime.state_dir = Path(config.runtime.state_dir).expanduser().resolve()
    config.models.ollama_url = normalize_url(config.models.ollama_url)
    return config, provenance


def load_config(path: str | Path | None = None, project_root: str | Path | None = None) -> NedsterConfig:
    overrides = {"config": path} if path is not None else None
    return resolve_config(overrides, project_root)[0]


# Compatibility for v3.0 integrations. New code should import NedsterConfig.
IroncladConfig = NedsterConfig
