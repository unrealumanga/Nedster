# ===----------------------------------------------------------------------=== #
# Nedster Interactive Terminal UI (TUI) & Console Loop
# ===----------------------------------------------------------------------=== #

from python import Python
from .orchestrator import AgentOrchestrator

struct TerminalUI:
    var orchestrator: AgentOrchestrator

    fn __init__(inout self, project_root: String = ".") raises:
        self.orchestrator = AgentOrchestrator(project_root)

    fn render_header(self):
        print("==================================================================")
        print(" 🔥 Nedster Agent [Mojo Edition] — Local AI & High-Frequency Crypto")
        print(" Hardware Profile : " + self.orchestrator.profile.name)
        print(" Accelerator      : " + self.orchestrator.hardware.accelerator)
        print(" RAM / VRAM       : " + String(self.orchestrator.hardware.ram_total_gb) + " GB RAM | " + String(self.orchestrator.hardware.vram_total_gb) + " GB VRAM")
        print("==================================================================")
        print(" Commands: /doctor, /market BTC/USDT, /help, /quit")
        print("")

    fn run_interactive_loop(self) raises:
        """Runs the quiet, chat-first monospace TUI loop."""
        let builtins = Python.import_module("builtins")
        self.render_header()

        while True:
            try:
                var prompt_text = String(builtins.input("nedster> ")).strip()
                if len(prompt_text) == 0:
                    continue

                if prompt_text == "/quit" or prompt_text == "exit":
                    print("Exiting Nedster Mojo. Goodbye!")
                    break
                elif prompt_text == "/doctor":
                    print("--- Hardware Doctor Report ---")
                    print("Platform: " + self.orchestrator.hardware.platform_name)
                    print("CPUs: " + String(self.orchestrator.hardware.cpu_count))
                    print("Profile: " + self.orchestrator.profile.name)
                    print("Context Window: " + String(self.orchestrator.profile.context_length) + " tokens")
                    print("Quantization: " + self.orchestrator.profile.quantization)
                elif prompt_text == "/help":
                    print("Type any coding query or market quote request (e.g., 'Analyze BTC/USDT' or 'List files').")
                else:
                    var response = self.orchestrator.process_user_query(prompt_text)
                    print("\n" + response + "\n")
            except:
                print("\nSession interrupted. Exiting...")
                break
