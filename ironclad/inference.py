from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from .types import HardwareProfile


@dataclass(slots=True, frozen=True)
class LlamaCppPlan:
    """A loopback-only llama.cpp server plan tuned to a Nedster hardware tier."""

    command: tuple[str, ...]
    profile: str
    speculative_mode: str

    def shell_command(self) -> str:
        return shlex.join(self.command)


def llama_cpp_plan(
    profile: HardwareProfile,
    model: str | Path,
    *,
    draft_model: str | Path | None = None,
    port: int = 8080,
    executable: str = "llama-server",
) -> LlamaCppPlan:
    """Build a conservative, local-only launch command without starting a process."""
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    model_path = str(Path(model).expanduser())
    if not model_path:
        raise ValueError("model path is required")

    command = [
        executable,
        "--model",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(profile.context_tokens),
        "--parallel",
        str(profile.max_parallel_workers),
        "--cache-type-k",
        profile.kv_quantization,
        "--cache-type-v",
        profile.kv_quantization,
        "--flash-attn",
        "on",
        "--mmap",
        "--jinja",
    ]
    if draft_model:
        command.extend(
            [
                "--model-draft",
                str(Path(draft_model).expanduser()),
                "--spec-type",
                "draft",
            ]
        )
        mode = "draft-model"
    else:
        command.extend(["--spec-type", "ngram-mod"])
        mode = "prompt-lookup"
    return LlamaCppPlan(tuple(command), profile.name, mode)
