# skills/vram_governor.py
import subprocess, time
from skills.base import NedsterSkill

# Ordered most-capable -> least-capable
MODEL_LADDER = [
    {"model": "nedster:32b", "min_vram_mb": 26000, "context": 65536},
    {"model": "nedster:14b", "min_vram_mb": 12000, "context": 32768},
    {"model": "nedster", "min_vram_mb": 7000, "context": 16384},
]

class VRAMGovernor:
    def __init__(self, stability_window_s: int = 30):
        self.current_rung = 0
        self.last_downshift_ts = 0
        self.stability_window_s = stability_window_s

    def _free_vram_mb(self) -> int:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        ).decode().strip()
        return int(out.splitlines()[0])

    def check_and_adjust(self) -> dict:
        free_mb = self._free_vram_mb()
        rung = MODEL_LADDER[self.current_rung]
        now = time.time()

        if free_mb < rung["min_vram_mb"] and self.current_rung < len(MODEL_LADDER) - 1:
            self.current_rung += 1
            self.last_downshift_ts = now
            return {"action": "downshift", "free_vram_mb": free_mb, "new_model": MODEL_LADDER[self.current_rung]}

        if self.current_rung > 0 and (now - self.last_downshift_ts) > self.stability_window_s:
            better = MODEL_LADDER[self.current_rung - 1]
            if free_mb >= better["min_vram_mb"]:
                self.current_rung -= 1
                return {"action": "upshift", "free_vram_mb": free_mb, "new_model": MODEL_LADDER[self.current_rung]}

        return {"action": "none", "free_vram_mb": free_mb, "current_model": rung}


class CheckVRAMAndAdjustModel(NedsterSkill):
    name = "check_vram_and_adjust_model"
    description = "Poll free VRAM and downshift/upshift the active model rung accordingly."
    parameters = {"type": "object", "properties": {}}

    def __init__(self):
        self.governor = VRAMGovernor()

    async def run(self) -> dict:
        return self.governor.check_and_adjust()
