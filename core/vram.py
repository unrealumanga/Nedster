"""
core/vram.py — VRAM lifecycle for an Ollama-backed agent.

CORRECTION TO THE DRAFT BLUEPRINT
---------------------------------
The proposed KVCacheLifecycleManager called:

    gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()

That frees nothing here. Nedster does not host the model weights — the `ollama
serve` daemon does, in a separate OS process with its own CUDA context. There is
no PyTorch allocator in this process holding the KV cache, so `empty_cache()`
has no cache to empty. (torch appears in requirements.txt at ~2GB installed and
is used for exactly one call in the whole repo: `torch.get_num_threads()` in
main.py:63.)

The blueprint also used `torch.mps.is_available()`, which is not the API —
it is `torch.backends.mps.is_available()`. That line would raise AttributeError
on every macOS run.

The real levers on Ollama-held VRAM are:
    ollama stop <model>            -> unload weights now
    OLLAMA_KEEP_ALIVE              -> idle unload timer (set before daemon start)
    OLLAMA_MAX_LOADED_MODELS       -> how many models may be resident at once
    OLLAMA_KV_CACHE_TYPE           -> read at model-load time, not per request
    num_ctx in the request options  -> the number that actually decides OOM

nedster.py:35 already does the right thing with `ollama stop` on exit. This
module generalises it and makes it measurable.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class VramSnapshot:
    total_mb: int
    used_mb: int
    free_mb: int

    @property
    def used_pct(self) -> float:
        return (self.used_mb / self.total_mb * 100.0) if self.total_mb else 0.0

    def __str__(self) -> str:
        return f"{self.used_mb / 1024:.1f}/{self.total_mb / 1024:.1f} GB ({self.used_pct:.0f}%)"


def snapshot() -> VramSnapshot | None:
    """Current VRAM state, or None when there is no NVIDIA GPU to ask."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip().splitlines()[0]
        total, used, free = (int(x.strip()) for x in out.split(","))
        return VramSnapshot(total_mb=total, used_mb=used, free_mb=free)
    except Exception:
        return None


def unload(model: str, timeout: int = 10) -> bool:
    """Evict a model's weights and KV cache from VRAM immediately.

    This is the operation the draft blueprint was reaching for. It is a
    daemon-level instruction, not a local allocator hint.
    """
    if not shutil.which("ollama"):
        return False
    try:
        proc = subprocess.run(["ollama", "stop", model],
                              capture_output=True, timeout=timeout)
        return proc.returncode == 0
    except Exception:
        return False


def loaded_models() -> list[dict]:
    """Models currently resident in VRAM, with their footprints."""
    try:
        import ollama

        data = ollama.Client(host="127.0.0.1").ps()
        models = data.get("models", []) if isinstance(data, dict) else getattr(data, "models", [])
        out = []
        for model in models:
            if isinstance(model, dict):
                name = model.get("model") or model.get("name")
                size_vram = model.get("size_vram") or 0
            else:
                name = getattr(model, "model", None) or getattr(model, "name", None)
                size_vram = getattr(model, "size_vram", 0) or 0
            out.append({"model": name, "size_vram": size_vram})
        return out
    except Exception:
        return []


class VramGuard:
    """Context manager for a phase that needs headroom.

    Unlike a torch cache flush, this can only act on whole models — so it
    unloads *other* resident models rather than pretending to trim a cache
    it does not own.

        with VramGuard(keep="nedster", min_free_mb=2000) as g:
            ...                      # g.freed_mb tells you what it reclaimed
    """

    def __init__(self, keep: str, min_free_mb: int = 1500, release_on_exit: bool = False):
        self.keep = keep
        self.min_free_mb = min_free_mb
        self.release_on_exit = release_on_exit
        self.freed_mb = 0
        self.before: VramSnapshot | None = None

    def __enter__(self) -> VramGuard:
        self.before = snapshot()
        if self.before and self.before.free_mb < self.min_free_mb:
            for entry in loaded_models():
                name = entry.get("model") or ""
                if (
                    name
                    and name.split(":")[0] != self.keep.split(":")[0]
                    and unload(name)
                ):
                    self.freed_mb += int(entry.get("size_vram") or 0) // (1024 * 1024)
        return self

    def __exit__(self, *_exc) -> bool:
        if self.release_on_exit:
            unload(self.keep)
        return False


def advise(profile_num_ctx: int, snap: VramSnapshot | None = None) -> list[str]:
    """Concrete, actionable warnings — the kind main.py:38 gestures at with
    'Suggest: ollama stop && ollama pull qwen3.5:9b-q3_k_m'."""
    snap = snap or snapshot()
    tips: list[str] = []
    if snap is None:
        tips.append("No NVIDIA GPU visible: Ollama will run on CPU. Expect 10-40x slower generation.")
        return tips
    if snap.used_pct > 92:
        tips.append(f"VRAM at {snap.used_pct:.0f}%. Run `ollama ps` and stop any model you are not using.")
    if snap.total_mb < 9000 and profile_num_ctx > 16384:
        tips.append(
            f"num_ctx={profile_num_ctx} on {snap.total_mb / 1024:.0f}GB will OOM mid-task. "
            "Cap it at 12288-16384; the KV cache grows linearly with context and is not compressible "
            "after the fact."
        )
    if snap.free_mb < 1200:
        tips.append("Under 1.2GB free: the next model load will fail rather than swap. Unload first.")
    return tips


if __name__ == "__main__":
    snap = snapshot()
    print(f"VRAM: {snap if snap else 'no NVIDIA GPU detected'}")
    print(f"Resident: {loaded_models() or '(none / ollama unreachable)'}")
    for tip in advise(24576, snap):
        print(f"  ! {tip}")
