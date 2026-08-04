"""
core/model_registry.py — single source of truth for which local model Nedster talks to.

Replaces the scattered, mutually-inconsistent legacy model strings with one
canonical name and a deterministic fallback.

Resolution order (first hit wins):
    1. NEDSTER_MODEL env var          (canonical, new)
    2. MODEL env var                  (legacy, still honoured)
    3. first tag present in ollama    from PREFERRED_TAGS
    4. RESCUE_TAG                     (plain upstream tag, always pullable)

Nothing in this module imports torch, ollama, or anything heavy at module scope.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field

# Ordered by preference. "nedster" is the merged Modelfile build (see Modelfile.nedster).
PREFERRED_TAGS = [
    "nedster",
    "nedster:latest",
]

# Always resolvable via `ollama pull`. Matches Modelfile's FROM line.
RESCUE_TAG = "qwen3.5:9b"

# Model families known to support Ollama's native tools= parameter.
# Used instead of a tag-prefix heuristic, which silently downgrades renamed models.
TOOL_CAPABLE_FAMILIES = ("qwen", "llama3.1", "llama3.2", "llama3.3", "mistral", "command-r", "hermes")

# Tags we build ourselves from a known tool-capable base (Modelfile FROM line).
# Listed explicitly so `nedster` is not misjudged just because the tag name
# carries no family string.
LOCAL_BUILD_TAGS = ("nedster",)


@dataclass
class Tier:
    """A VRAM-derived execution profile. num_ctx is the number that actually
    decides whether you OOM, so it is the primary knob here."""

    name: str
    min_vram_gb: float
    num_ctx: int
    kv_cache_type: str          # exported as OLLAMA_KV_CACHE_TYPE
    max_loaded_models: int      # exported as OLLAMA_MAX_LOADED_MODELS
    split_router: bool          # run a separate small planner model?
    max_tool_iterations: int
    notes: str = ""


# Ordered high -> low; first tier whose min_vram_gb is satisfied wins.
#
# split_router is False below 18GB on purpose. With OLLAMA_MAX_LOADED_MODELS=1
# a planner/worker split forces a full unload+reload of weights on every
# handoff — 10-30s of disk I/O per step on a 3060 Ti. Role separation is done
# with two system prompts against one resident model instead. See PLAN §3.
TIERS = [
    Tier("workstation", 30.0, 65536, "f16", 2, True, 12,
         "Both planner and worker stay resident; true two-model routing pays off here."),
    Tier("enthusiast", 18.0, 32768, "q8_0", 2, True, 10,
         "Planner (3b) + worker (14b) fit together at q4_K_M."),
    Tier("mainstream", 10.0, 24576, "q8_0", 1, False, 8,
         "Single resident model, dual system prompts. RTX 3060 Ti / 4060 sits here."),
    Tier("compact", 0.0, 12288, "q4_0", 1, False, 6,
         "8GB and below. Keep num_ctx honest; TurboQuant claims of 256K do not survive contact."),
]


def probe_vram_gb() -> float:
    """Total VRAM in GB. Returns 0.0 when no discrete GPU can be detected,
    which correctly lands on the 'compact' tier rather than guessing high.

    Deliberately does NOT import torch: torch is a ~2GB dependency and this
    runs on every CLI start.
    """
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            mibs = [int(line.strip()) for line in out.decode().splitlines() if line.strip()]
            if mibs:
                return max(mibs) / 1024.0
        except Exception:
            pass

    # Apple Silicon: unified memory, GPU can address most of it.
    if os.uname().sysname == "Darwin" if hasattr(os, "uname") else False:
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=5)
            return (int(out.decode().strip()) / (1024 ** 3)) * 0.65
        except Exception:
            pass

    return 0.0


def resolve_tier(vram_gb: float | None = None) -> Tier:
    if vram_gb is None:
        vram_gb = probe_vram_gb()
    for tier in TIERS:
        if vram_gb >= tier.min_vram_gb:
            return tier
    return TIERS[-1]


def list_installed_tags() -> list[str]:
    """Tags currently present in the local Ollama library. Empty list if the
    daemon is unreachable — callers must treat that as 'unknown', not 'none'."""
    try:
        import ollama

        data = ollama.Client(host="127.0.0.1").list()
        models = data.get("models", []) if isinstance(data, dict) else getattr(data, "models", [])
        tags = []
        for m in models:
            name = m.get("model") or m.get("name") if isinstance(m, dict) else getattr(m, "model", None)
            if name:
                tags.append(name)
        return tags
    except Exception:
        pass

    # Fall back to the CLI so this still works when the python client is absent.
    if shutil.which("ollama"):
        try:
            out = subprocess.check_output(["ollama", "list"], stderr=subprocess.DEVNULL, timeout=10)
            lines = out.decode().splitlines()[1:]
            return [ln.split()[0] for ln in lines if ln.strip()]
        except Exception:
            pass
    return []


def resolve_model(installed: list[str] | None = None) -> str:
    """The one function every call site should use. Never returns a tag that
    is guaranteed to 404.
    """
    explicit = os.environ.get("NEDSTER_MODEL") or os.environ.get("MODEL")
    if explicit:
        return explicit

    if installed is None:
        installed = list_installed_tags()

    if not installed:
        # Ollama unreachable. Return the preferred tag and let the caller's
        # error path report "daemon down" rather than "model missing".
        return PREFERRED_TAGS[0]

    bare = {t.split(":")[0] for t in installed}
    for tag in PREFERRED_TAGS:
        if tag in installed or tag.split(":")[0] in bare:
            return tag

    for tag in installed:
        if any(fam in tag.lower() for fam in ("qwen", "coder")):
            return tag

    return RESCUE_TAG


def supports_native_tools(model: str) -> bool:
    """Ask Ollama what the model can do; fall back to a family heuristic.

    A tag-prefix check ties capability to branding and silently marks renamed
    models tool-incapable.
    """
    try:
        import ollama

        info = ollama.Client(host="127.0.0.1").show(model)
        caps = info.get("capabilities") if isinstance(info, dict) else getattr(info, "capabilities", None)
        if caps:
            return "tools" in [str(c).lower() for c in caps]
    except Exception:
        pass
    lowered = model.lower()
    if lowered.split(":")[0] in LOCAL_BUILD_TAGS:
        return True
    return any(fam in lowered for fam in TOOL_CAPABLE_FAMILIES)


@dataclass
class RuntimeProfile:
    """What the boot sequence hands to the rest of the app."""

    model: str
    tier: Tier
    vram_gb: float
    native_tools: bool
    env: dict = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"model={self.model}  tier={self.tier.name}  vram={self.vram_gb:.1f}GB  "
            f"num_ctx={self.tier.num_ctx}  kv={self.tier.kv_cache_type}  "
            f"tools={'native' if self.native_tools else 'xml-fallback'}  "
            f"budget={self.tier.max_tool_iterations} iters"
        )


def build_profile(probe_tools: bool = True) -> RuntimeProfile:
    vram = probe_vram_gb()
    tier = resolve_tier(vram)
    model = resolve_model()
    env = {
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KV_CACHE_TYPE": tier.kv_cache_type,
        "OLLAMA_MAX_LOADED_MODELS": str(tier.max_loaded_models),
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_KEEP_ALIVE": "15m",
    }
    return RuntimeProfile(
        model=model,
        tier=tier,
        vram_gb=vram,
        native_tools=supports_native_tools(model) if probe_tools else True,
        env=env,
    )


def apply_env(profile: RuntimeProfile) -> None:
    """Export the tier's Ollama tuning. Set before the daemon loads weights;
    OLLAMA_KV_CACHE_TYPE is read at model-load time, not per-request."""
    for k, v in profile.env.items():
        os.environ.setdefault(k, v)
    os.environ["NEDSTER_MODEL"] = profile.model
    os.environ["NEDSTER_NUM_CTX"] = str(profile.tier.num_ctx)


if __name__ == "__main__":
    p = build_profile(probe_tools=False)
    print(p.describe())
    print(json.dumps(p.env, indent=2))
    print(f"\ntier note: {p.tier.notes}")
    print(f"installed tags: {list_installed_tags() or '(ollama unreachable)'}")
