from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import asdict

import psutil

from .types import HardwareProfile, HardwareSnapshot

PROFILES = (
    HardwareProfile("edge-8g", (1.5, 4.0), (7.0, 9.0), 1, 32_768, "Q4_K_M / AWQ 4-bit", "q4_0"),
    HardwareProfile("forge-16g", (3.0, 8.0), (12.0, 20.0), 2, 65_536, "Q5_K_M / EXL2 5-bit", "q5_1"),
    HardwareProfile("titan-32g", (7.0, 14.0), (27.0, 40.0), 3, 131_072, "Q6_K / EXL2 8-bit", "q8_0"),
)


def _run(command: list[str], timeout: int = 5) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _nvidia() -> tuple[str, float, float] | None:
    if not shutil.which("nvidia-smi"):
        return None
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    name, total, free = [part.strip() for part in output.splitlines()[0].split(",", 2)]
    return name, round(float(total) / 1024, 2), round(float(free) / 1024, 2)


def _amd() -> tuple[str, float, float] | None:
    if not shutil.which("rocm-smi"):
        return None
    output = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    try:
        data = json.loads(output)
        gpu = next(iter(data.values()))
        name = str(gpu.get("Card series") or gpu.get("Card model") or "AMD ROCm GPU")
        total = float(gpu.get("VRAM Total Memory (B)", 0)) / 1024**3
        used = float(gpu.get("VRAM Total Used Memory (B)", 0)) / 1024**3
        return name, round(total, 2), round(max(0.0, total - used), 2)
    except (ValueError, TypeError, StopIteration, AttributeError):
        return None


def probe_hardware() -> HardwareSnapshot:
    memory = psutil.virtual_memory()
    accelerator = _nvidia() or _amd()
    is_apple = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
    if accelerator:
        name, total, free = accelerator
    elif is_apple:
        name = "Apple Silicon unified memory"
        total = memory.total / 1024**3
        free = memory.available / 1024**3
    else:
        name, total, free = "cpu", 0.0, 0.0
    return HardwareSnapshot(
        platform=f"{platform.system()} {platform.machine()}",
        cpu_count=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=round(memory.total / 1024**3, 2),
        ram_available_gb=round(memory.available / 1024**3, 2),
        accelerator=name,
        vram_total_gb=round(total, 2),
        vram_free_gb=round(free, 2),
        unified_memory=is_apple,
    )


def choose_profile(snapshot: HardwareSnapshot) -> HardwareProfile:
    budget = snapshot.vram_total_gb
    if snapshot.unified_memory:
        budget = min(snapshot.ram_total_gb * 0.72, snapshot.ram_available_gb)
    if budget >= 28:
        return PROFILES[2]
    if budget >= 13:
        return PROFILES[1]
    return PROFILES[0]


def doctor_report() -> dict[str, object]:
    snapshot = probe_hardware()
    profile = choose_profile(snapshot)
    return {
        "hardware": asdict(snapshot),
        "profile": asdict(profile),
        "runtimes": {
            "ollama": shutil.which("ollama"),
            "llama-server": shutil.which("llama-server"),
            "bubblewrap": shutil.which("bwrap") or shutil.which("bubblewrap"),
            "gvisor": shutil.which("runsc"),
            "docker": shutil.which("docker"),
            "podman": shutil.which("podman"),
            "firecracker": shutil.which("firecracker"),
        },
    }
