import asyncio
import inspect

from core.critic_loop import CriticLoop
from core.event_bus import EventBus
from core.indexer import IncrementalIndexer
from core.loop_adapter import run_with_loop
from core.model_registry import resolve_model
from core.plugin_manager import PluginManager
from core.replay_engine import ReplayEngine
from core.world_graph import WorldGraph
from memory import MemoryManager
from rag_engine.retriever import Retriever
from tui import NedsterTUI


def get_legacy_schema(func, name: str) -> dict:
    """Converts a standard Python function to an Ollama tool schema."""
    doc = inspect.getdoc(func) or "No description."
    sig = inspect.signature(func)
    
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name == "kwargs":
            continue
        
        param_type = "string"
        if param.annotation is int:
            param_type = "integer"
        elif param.annotation is bool:
            param_type = "boolean"
        elif param.annotation is float:
            param_type = "number"
            
        properties[param_name] = {"type": param_type}
        
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
            
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": doc.splitlines()[0] if doc else "No description",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


class ToolExecutor:
    def __init__(
        self,
        registry: dict,
        auto: bool = False,
        session_log=None,
        project_path: str = None,
        job_id: int = None,
    ):
        self.registry = registry
        self.project_path = project_path
        self.job_id = job_id

    def execute(self, name: str, args: dict, tui=None) -> str:
        if name not in self.registry:
            return f"[ERROR: '{name}' unknown.]"

        if self.job_id and name in ["edit_file", "write_file"]:
            from swarm_utils import acquire_lock, release_lock
            if not acquire_lock(self.project_path, args.get("path"), self.job_id):
                return "[ERROR] Could not acquire lock."
            try:
                return self._execute_tool(name, args)
            finally:
                release_lock(self.project_path, args.get("path"), self.job_id)

        return self._execute_tool(name, args)

    def _execute_tool(self, name: str, args: dict) -> str:
        try:
            tool = self.registry[name]
            # Check if it's an async NedsterSkill instance
            if hasattr(tool, "run") and asyncio.iscoroutinefunction(tool.run):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    return str(asyncio.run(tool.run(**args)))
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return str(pool.submit(asyncio.run, tool.run(**args)).result())
            else:
                # It's a standard Python function (BASE_TOOL_REGISTRY)
                return str(tool(**args))
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
        backend: str = "local"
    ):
        import sys

        from skill_manager import TOOL_REGISTRY as SKILL_TOOL_REGISTRY
        from tools import SESSION
        from tools import TOOL_REGISTRY as BASE_TOOL_REGISTRY

        SESSION.set_project(project_dir)
        SESSION.platform = "windows" if sys.platform == "win32" else "linux"

        self.project_dir = project_dir
        self.auto = auto
        self.think_visible = think
        self.job_id = job_id
        self.backend = backend
        self.model = resolve_model()
        self.tui = NedsterTUI()
        self.memory = MemoryManager(self.model)

        # Merge base tools and newly discovered skills
        self.tool_registry = {**BASE_TOOL_REGISTRY, **SKILL_TOOL_REGISTRY}
        self.executor = ToolExecutor(
            self.tool_registry, project_path=self.project_dir, job_id=self.job_id
        )

        # Initialize core RAG retriever
        try:
            self.rag = Retriever()
        except Exception:
            self.rag = None

        # Initialize event-driven system & plugin architectures
        self.plugin_manager = PluginManager()
        self.plugin_manager.discover()
        self.plugin_manager.wrap_existing_skills(self.tool_registry)

        self.world_graph = WorldGraph(self.project_dir)
        self.indexer = IncrementalIndexer(self.project_dir, self.world_graph, rag_engine=self.rag)
        self.indexer.start()

        self.replay_engine = ReplayEngine()
        
        mutation_tester = self.tool_registry.get("mutation_test_file")
        persona_skill = self.tool_registry.get("persona_review_diff")
        persona_router = getattr(persona_skill, "router", None) if persona_skill else None
        
        self.critic = CriticLoop(
            mutation_tester=mutation_tester,
            persona_router=persona_router,
        )

    def _sync_emit(self, event_type: str, payload=None):
        """Helper to safely emit async events from synchronous context."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.ensure_future(EventBus.emit(event_type, payload))
            else:
                loop.run_until_complete(EventBus.emit(event_type, payload))
        except RuntimeError:
            asyncio.run(EventBus.emit(event_type, payload))

    def _get_native_tools(self) -> list:
        """Builds a JSON schema list of tools natively supported by Ollama."""
        tools = []
        for name, tool in self.tool_registry.items():
            if hasattr(tool, "to_tool_schema"):
                # It's a NedsterSkill object
                tools.append(tool.to_tool_schema())
            else:
                # It's a legacy Python function
                tools.append(get_legacy_schema(tool, name))
        return tools

    def generate(self, user_input: str):
        """Run the bounded multi-turn classic agent loop."""
        return run_with_loop(self, user_input)
