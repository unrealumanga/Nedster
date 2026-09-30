# ===----------------------------------------------------------------------=== #
# Nedster Main CLI Launcher Entrypoint (Mojo Language)
# ===----------------------------------------------------------------------=== #

from sys import argv
from .tui import TerminalUI
from .orchestrator import AgentOrchestrator

fn main() raises:
    """Main CLI entrypoint for Nedster Mojo Agent."""
    var args = argv()
    var project_dir = "."

    if len(args) > 1:
        if args[1] == "run" and len(args) > 2:
            var query = args[2]
            var orch = AgentOrchestrator(project_dir)
            var result = orch.process_user_query(query)
            print(result)
            return
        elif args[1] == "doctor":
            var orch = AgentOrchestrator(project_dir)
            print("=== Nedster Mojo Hardware Doctor ===")
            print("Platform     : " + orch.hardware.platform_name)
            print("CPU Cores    : " + String(orch.hardware.cpu_count))
            print("Accelerator  : " + orch.hardware.accelerator)
            print("RAM Total    : " + String(orch.hardware.ram_total_gb) + " GB")
            print("RAM Available: " + String(orch.hardware.ram_available_gb) + " GB")
            print("VRAM Total   : " + String(orch.hardware.vram_total_gb) + " GB")
            print("Profile      : " + orch.profile.name)
            return

    # Default: launch interactive TUI
    var tui = TerminalUI(project_dir)
    tui.run_interactive_loop()
