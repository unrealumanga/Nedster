# ===----------------------------------------------------------------------=== #
# Nedster Orchestrator & Autonomous Multi-Role Agent Loop
# ===----------------------------------------------------------------------=== #

from python import Python
from .hardware import probe_hardware, choose_profile, HardwareSnapshot, HardwareProfile
from .inference import InferenceEngine
from .tools import ToolRegistry, ToolResult
from .exchange_api import ExchangeClient
from .chart_analytics import SignalOutput

struct PlanStep:
    var step_id: String
    var title: String
    var role: String
    var tool_name: String

    fn __init__(inout self, step_id: String, title: String, role: String, tool_name: String):
        self.step_id = step_id
        self.title = title
        self.role = role
        self.tool_name = tool_name


struct AgentOrchestrator:
    var project_root: String
    var hardware: HardwareSnapshot
    var profile: HardwareProfile
    var inference: InferenceEngine
    var tools: ToolRegistry
    var exchange: ExchangeClient

    fn __init__(inout self, project_root: String = ".") raises:
        self.project_root = project_root
        self.hardware = probe_hardware()
        self.profile = choose_profile(self.hardware)
        self.inference = InferenceEngine()
        self.tools = ToolRegistry(project_root)
        self.exchange = ExchangeClient("binance")

    fn process_user_query(self, query: String) raises -> String:
        """Processes query through router -> worker -> critic tool execution harness."""
        var lowered = query.lower()

        # Crypto High-Frequency Data & Chart Analytics Path
        if "btc" in lowered or "eth" in lowered or "sol" in lowered or "market" in lowered or "crypto" in lowered:
            var symbol = "BTC/USDT"
            if "eth" in lowered:
                symbol = "ETH/USDT"
            elif "sol" in lowered:
                symbol = "SOL/USDT"

            try:
                var signal = self.exchange.fetch_orderbook_and_analyze(symbol)
                return "=== Nedster Real-Time Market Signal ===\n" + \
                       "Symbol: " + signal.symbol + "\n" + \
                       "Direction: " + signal.side + "\n" + \
                       "Signal Score: " + String(signal.score) + " / 100\n" + \
                       "OFI Pressure: " + String(signal.ofi_score) + "\n" + \
                       "CVD Delta: " + String(signal.cvd_score) + "\n" + \
                       "1m RSI: " + String(signal.rsi_val) + "\n" + \
                       "MACD Line: " + String(signal.macd_val) + "\n" + \
                       "Strategy Reason: " + signal.reason
            except err:
                return "Exchange Stream Error: " + String(err)

        # Local Coding Agent Workflow Path
        if "list" in lowered or "files" in lowered:
            var res = self.tools.list_files()
            if res.ok:
                return "Project Files:\n" + res.output
            else:
                return "Error listing files: " + res.error

        if "git" in lowered or "diff" in lowered:
            var res = self.tools.git_diff()
            if res.ok:
                return "Git Diff Output:\n" + res.output
            else:
                return "Git Error: " + res.error

        # General Local LLM Inference
        var system_prompt = "You are Nedster, a local AI coding and crypto assistant running on " + self.profile.name + "."
        return self.inference.generate_completion(query, system_prompt, 0.2, "nedster-worker")
