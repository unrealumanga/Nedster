# ===----------------------------------------------------------------------=== #
# Nedster Sandboxed Process Executor & Policy Authority Module
# ===----------------------------------------------------------------------=== #

from python import Python

struct PolicyEngine:
    var allow_write: Bool
    var allow_exec: Bool
    var max_command_timeout: Int

    fn __init__(inout self, allow_write: Bool = True, allow_exec: Bool = True, max_command_timeout: Int = 30):
        self.allow_write = allow_write
        self.allow_exec = allow_exec
        self.max_command_timeout = max_command_timeout

    fn is_command_permitted(self, command: String) -> Bool:
        if not self.allow_exec:
            return False
        # Block dangerous system commands
        if "rm -rf /" in command or "format " in command or "dd if=" in command:
            return False
        return True


struct SandboxResult:
    var returncode: Int
    var stdout: String
    var stderr: String
    var timed_out: Bool

    fn __init__(inout self, returncode: Int, stdout: String, stderr: String, timed_out: Bool = False):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


struct SandboxExecutor:
    var project_root: String
    var policy: PolicyEngine

    fn __init__(inout self, project_root: String, policy: PolicyEngine):
        self.project_root = project_root
        self.policy = policy

    fn execute_command(self, command: String, timeout_sec: Int = 30) raises -> SandboxResult:
        if not self.policy.is_command_permitted(command):
            return SandboxResult(-1, "", "Command rejected by security policy", False)

        let subprocess = Python.import_module("subprocess")
        try:
            var res = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=timeout_sec,
            )
            return SandboxResult(
                Int(res.returncode),
                String(res.stdout),
                String(res.stderr),
                False,
            )
        except err:
            return SandboxResult(-1, "", "Execution error or timeout: " + String(err), True)
