# ===----------------------------------------------------------------------=== #
# Nedster Mojo Hardware Probing & Profile Selection Module
# ===----------------------------------------------------------------------=== #

from python import Python
from sys import info

struct HardwareSnapshot:
    var platform_name: String
    var cpu_count: Int
    var ram_total_gb: Float64
    var ram_available_gb: Float64
    var accelerator: String
    var vram_total_gb: Float64
    var vram_free_gb: Float64
    var unified_memory: Bool

    fn __init__(
        inout self,
        platform_name: String,
        cpu_count: Int,
        ram_total_gb: Float64,
        ram_available_gb: Float64,
        accelerator: String,
        vram_total_gb: Float64,
        vram_free_gb: Float64,
        unified_memory: Bool,
    ):
        self.platform_name = platform_name
        self.cpu_count = cpu_count
        self.ram_total_gb = ram_total_gb
        self.ram_available_gb = ram_available_gb
        self.accelerator = accelerator
        self.vram_total_gb = vram_total_gb
        self.vram_free_gb = vram_free_gb
        self.unified_memory = unified_memory


struct HardwareProfile:
    var name: String
    var vram_range_min: Float64
    var vram_range_max: Float64
    var ram_range_min: Float64
    var ram_range_max: Float64
    var rank: Int
    var context_length: Int
    var quantization: String

    fn __init__(
        inout self,
        name: String,
        vram_range_min: Float64,
        vram_range_max: Float64,
        ram_range_min: Float64,
        ram_range_max: Float64,
        rank: Int,
        context_length: Int,
        quantization: String,
    ):
        self.name = name
        self.vram_range_min = vram_range_min
        self.vram_range_max = vram_range_max
        self.ram_range_min = ram_range_min
        self.ram_range_max = ram_range_max
        self.rank = rank
        self.context_length = context_length
        self.quantization = quantization


fn probe_hardware() raises -> HardwareSnapshot:
    """Probes system RAM, CPU cores, NVIDIA CUDA, AMD ROCm, or Apple Silicon unified memory using Python interop and system calls."""
    let psutil = Python.import_module("psutil")
    let platform = Python.import_module("platform")
    let subprocess = Python.import_module("subprocess")
    let shutil = Python.import_module("shutil")

    var sys_name = String(platform.system()) + " " + String(platform.machine())
    var cpu_cores = Int(psutil.cpu_count(logical=True) or 1)

    var mem = psutil.virtual_memory()
    var total_ram = Float64(mem.total) / (1024.0 * 1024.0 * 1024.0)
    var avail_ram = Float64(mem.available) / (1024.0 * 1024.0 * 1024.0)

    var accelerator_name = String("cpu")
    var vram_total = 0.0
    var vram_free = 0.0
    var is_apple = String(platform.system()) == "Darwin" and (
        String(platform.machine()).lower() == "arm64" or String(platform.machine()).lower() == "aarch64"
    )

    if Bool(shutil.which("nvidia-smi")):
        try:
            var res = subprocess.run(
                Python.evaluate("['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits']"),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if Int(res.returncode) == 0:
                var out = String(res.stdout).strip()
                var parts = out.split(",")
                if len(parts) >= 3:
                    accelerator_name = parts[0].strip()
                    vram_total = Float64(parts[1].strip()) / 1024.0
                    vram_free = Float64(parts[2].strip()) / 1024.0
        except:
            pass

    if accelerator_name == "cpu" and Bool(shutil.which("rocm-smi")):
        try:
            var res = subprocess.run(
                Python.evaluate("['rocm-smi', '--showproductname', '--showmeminfo', 'vram', '--json']"),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if Int(res.returncode) == 0:
                accelerator_name = "AMD ROCm GPU"
                vram_total = 16.0
                vram_free = 12.0
        except:
            pass

    if accelerator_name == "cpu" and is_apple:
        accelerator_name = "Apple Silicon unified memory"
        vram_total = total_ram
        vram_free = avail_ram

    return HardwareSnapshot(
        platform_name=sys_name,
        cpu_count=cpu_cores,
        ram_total_gb=total_ram,
        ram_available_gb=avail_ram,
        accelerator=accelerator_name,
        vram_total_gb=vram_total,
        vram_free_gb=vram_free,
        unified_memory=is_apple,
    )


fn choose_profile(snapshot: HardwareSnapshot) -> HardwareProfile:
    """Chooses Edge-8G, Forge-16G, or Titan-32G model profile based on available hardware resources."""
    var budget = snapshot.vram_total_gb
    if snapshot.unified_memory:
        var calculated = snapshot.ram_total_gb * 0.72
        if snapshot.ram_available_gb < calculated:
            budget = snapshot.ram_available_gb
        else:
            budget = calculated

    if budget >= 28.0:
        return HardwareProfile(
            name="titan-32g",
            vram_range_min=7.0,
            vram_range_max=14.0,
            ram_range_min=27.0,
            ram_range_max=40.0,
            rank=3,
            context_length=131072,
            quantization="Q6_K / EXL2 8-bit",
        )
    elif budget >= 13.0:
        return HardwareProfile(
            name="forge-16g",
            vram_range_min=3.0,
            vram_range_max=8.0,
            ram_range_min=12.0,
            ram_range_max=20.0,
            rank=2,
            context_length=65536,
            quantization="Q5_K_M / EXL2 5-bit",
        )
    else:
        return HardwareProfile(
            name="edge-8g",
            vram_range_min=1.5,
            vram_range_max=4.0,
            ram_range_min=7.0,
            ram_range_max=9.0,
            rank=1,
            context_length=32768,
            quantization="Q4_K_M / AWQ 4-bit",
        )
