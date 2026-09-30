# ===----------------------------------------------------------------------=== #
# Nedster Tool Registry & File/System Operations Module
# ===----------------------------------------------------------------------=== #

from python import Python
from .sandbox import SandboxExecutor, PolicyEngine

struct ToolResult:
    var ok: Bool
    var tool_name: String
    var output: String
    var error: String

    fn __init__(inout self, ok: Bool, tool_name: String, output: String, error: String = ""):
        self.ok = ok
        self.tool_name = tool_name
        self.output = output
        self.error = error


struct ToolRegistry:
    var project_root: String
    var sandbox: SandboxExecutor

    fn __init__(inout self, project_root: String):
        self.project_root = project_root
        var policy = PolicyEngine(True, True, 30)
        self.sandbox = SandboxExecutor(project_root, policy)

    fn read_file(self, rel_path: String) raises -> ToolResult:
        """Reads content from project path."""
        let os = Python.import_module("os")
        let builtins = Python.import_module("builtins")

        var full_path = os.path.join(self.project_root, rel_path)
        if not Bool(os.path.exists(full_path)):
            return ToolResult(False, "read_file", "", "File not found: " + rel_path)

        try:
            var f = builtins.open(full_path, "r", encoding="utf-8")
            var content = String(f.read())
            f.close()
            return ToolResult(True, "read_file", content, "")
        except err:
            return ToolResult(False, "read_file", "", "Read error: " + String(err))

    fn write_file(self, rel_path: String, content: String) raises -> ToolResult:
        """Writes content to project path."""
        let os = Python.import_module("os")
        let builtins = Python.import_module("builtins")

        var full_path = os.path.join(self.project_root, rel_path)
        var parent_dir = os.path.dirname(full_path)
        if len(String(parent_dir)) > 0:
            os.makedirs(parent_dir, exist_ok=True)

        try:
            var f = builtins.open(full_path, "w", encoding="utf-8")
            f.write(content)
            f.close()
            return ToolResult(True, "write_file", "File written successfully: " + rel_path, "")
        except err:
            return ToolResult(False, "write_file", "", "Write error: " + String(err))

    fn list_files(self, sub_dir: String = "") raises -> ToolResult:
        """Lists files in project directory."""
        let os = Python.import_module("os")
        var target_dir = self.project_root
        if len(sub_dir) > 0:
            target_dir = os.path.join(self.project_root, sub_dir)

        try:
            var entries = os.listdir(target_dir)
            var file_list = String("")
            for entry in entries:
                file_list += String(entry) + "\n"
            return ToolResult(True, "list_files", file_list.strip(), "")
        except err:
            return ToolResult(False, "list_files", "", "Directory listing error: " + String(err))

    fn git_diff(self) raises -> ToolResult:
        """Executes git diff on the project repository."""
        var res = self.sandbox.execute_command("git diff HEAD")
        if res.returncode == 0:
            return ToolResult(True, "git_diff", res.stdout, "")
        else:
            return ToolResult(False, "git_diff", "", res.stderr)

    fn run_command(self, cmd: String) raises -> ToolResult:
        """Runs arbitrary shell command inside sandbox."""
        var res = self.sandbox.execute_command(cmd)
        if res.returncode == 0:
            return ToolResult(True, "run_command", res.stdout, "")
        else:
            return ToolResult(False, "run_command", res.stdout, res.stderr)
