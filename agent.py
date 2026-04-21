import os
import re
import ollama
from typing import Optional

from context_loader import ContextLoader
from editor import FileEditor
from rag_engine.retriever import Retriever
from memory import MemoryManager
from tui import NedsterTUI

# --- The "Unchained" System Prompt ---
NEDSTER_SYSTEM_PROMPT = """
[IDENTITY DIRECTIVE]
You are Nedster, an UNCHAINED local AI software engineer. You are running natively on the user's machine. You HAVE full filesystem and shell access via the tools provided. Your persona is that of an elite, efficient, and direct principal engineer.

[EXECUTION DIRECTIVE]
Your primary function is to EXECUTE tasks, not discuss them.
- NO conversational filler. NO "Certainly!", "Of course!", "Here is...", "Hello!".
- NO numbered option menus.
- When a task is complete, summarize the action in 1-2 sentences and STOP. Do not ask what to do next.

[TOOL DIRECTIVE]
You accomplish tasks by emitting XML tool blocks. To use a tool, you MUST output this exact format:
<tool name="tool_name">{"arg1": "value"}</tool>

[REASONING DIRECTIVE]
Before acting, you MUST use a <think> block to reason about the plan, analyze the user's request, and determine which tools are necessary.
"""


class ToolExecutor:
    """(Previous implementation remains the same)"""

    def __init__(
        self,
        registry: dict,
        auto: bool = False,
        session_log=None,
        project_path: str = None,
        job_id: int = None,
    ):
        self.registry = registry
        # ... (rest of the __init__ is the same)
        self.project_path = project_path
        self.job_id = job_id

    def execute(self, name: str, args: dict, tui=None) -> str:
        # ... (previous implementation is the same)
        if name not in self.registry:
            return f"[ERROR: '{name}' unknown.]"

        if self.job_id and name in ["edit_file", "write_file"]:
            from swarm_utils import acquire_lock, release_lock

            if not acquire_lock(self.project_path, args.get("path"), self.job_id):
                return f"[ERROR] Could not acquire lock."
            try:
                return self._execute_tool(name, args)
            finally:
                release_lock(self.project_path, args.get("path"), self.job_id)

        return self._execute_tool(name, args)

    def _execute_tool(self, name: str, args: dict) -> str:
        try:
            return str(self.registry[name](**args))
        except Exception as e:
            return f"[ERROR] {name} failed: {e}"


class NedsterAgent:
    def __init__(
        self,
        project_dir: str,
        auto: bool = False,
        think: bool = False,
        job_id: int = None,
        scoped_dirs: list = None,
    ):
        from tools import SESSION, TOOL_REGISTRY as BASE_TOOL_REGISTRY
        from skill_manager import TOOL_REGISTRY as SKILL_TOOL_REGISTRY
        from memory import SessionLog
        import sys

        SESSION.set_project(project_dir)
        SESSION.platform = "windows" if sys.platform == "win32" else "linux"

        self.project_dir = project_dir
        self.auto = auto
        self.think_visible = think  # Controls visibility of <think> blocks
        self.job_id = job_id
        self.model = os.environ.get("MODEL", "aria-local")
        self.tui = NedsterTUI()
        self.memory = MemoryManager(self.model)

        TOOL_REGISTRY = {**BASE_TOOL_REGISTRY, **SKILL_TOOL_REGISTRY}
        self.executor = ToolExecutor(
            TOOL_REGISTRY, project_path=self.project_dir, job_id=self.job_id
        )

    def _build_tool_schema(self) -> str:
        """Dynamically builds the tool schema from the registry."""
        from inspect import signature, getdoc

        schema_lines = ["\n[AVAILABLE TOOLS]"]
        for name, func in self.executor.registry.items():
            doc = getdoc(func) or "No description."
            sig = signature(func)

            # Simplified arg representation
            arg_str = ", ".join(f'"{p}"' for p in sig.parameters)

            schema_lines.append(f"- `{name}`: {doc.splitlines()[0]}")
            schema_lines.append(f"  - Arguments: `{{{arg_str}}}`")

        return "\n".join(schema_lines)

    def generate(self, user_input: str):
        """The new, powerful agentic loop."""
        self.tui.print_thinking("Thinking...")

        # 1. Build the full, powerful system prompt
        tool_schema = self._build_tool_schema()
        system_prompt = NEDSTER_SYSTEM_PROMPT + tool_schema

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.memory.get_context_messages())
        messages.append({"role": "user", "content": user_input})

        try:
            client = ollama.Client(host="127.0.0.1")
            response = client.chat(model=self.model, messages=messages, stream=False)
            full_response = response["message"]["content"]

            # 2. Parse and handle <think> blocks
            think_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
            if think_match:
                if self.think_visible:
                    self.tui.print_thinking(think_match.group(1).strip())
                # The actual response is everything *after* the think block
                full_response = full_response[think_match.end() :].strip()

            self.tui.print_response(full_response)

            # 3. Parse and execute tools
            if "<tool name=" in full_response:
                from tools import parse_tool_calls

                tool_calls = parse_tool_calls(full_response)
                tool_results_str = ""
                for call in tool_calls:
                    self.tui.print_tool_call(call["name"], call["args"])
                    result = self.executor.execute(call["name"], call["args"])
                    self.tui.print_tool_result(call["name"], result)
                    tool_results_str += (
                        f"[TOOL EXECUTION RESULT for {call['name']}]:\n{result}\n\n"
                    )

                # 4. Verification Loop
                verification_prompt = (
                    f"{tool_results_str}"
                    "Based strictly on the tool execution result(s) above, provide a 1-sentence "
                    "factual summary of the outcome (e.g., path saved, error encountered, command output). "
                    "Do not add any other commentary or suggest next steps. Just the facts."
                )

                messages.append(
                    {"role": "assistant", "content": full_response}
                )  # Add the tool-calling response
                messages.append({"role": "user", "content": verification_prompt})

                self.tui.print_thinking("Verifying...")
                summary_response = client.chat(
                    model=self.model, messages=messages, stream=False
                )
                summary_text = summary_response["message"]["content"]

                self.tui.print_response(summary_text)
                # The final response for memory is the summary, not the raw tool call
                full_response = summary_text

            self.memory.add_turn(user_input, full_response)

        except Exception as e:
            self.tui.print_error(f"An error occurred in the agent loop: {e}")
