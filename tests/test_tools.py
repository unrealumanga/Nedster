import tempfile
import unittest
from pathlib import Path

from ironclad.config import PolicyConfig, SandboxConfig
from ironclad.sandbox import SandboxExecutor
from ironclad.security import CapabilityAuthority, PolicyEngine
from ironclad.tools import ToolRegistry
from ironclad.types import ToolAction


class ToolRegistryTests(unittest.TestCase):
    def test_project_path_boundary_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority = CapabilityAuthority(b"z" * 32)
            policy = PolicyEngine(PolicyConfig(autonomous=True, max_auto_risk="write"))
            sandbox = SandboxExecutor(SandboxConfig(backend="process", allow_unsafe_fallback=True), root)
            tools = ToolRegistry(root, authority, policy, sandbox)
            action = ToolAction("write_file", {"path": "src/demo.txt", "content": "ironclad"})
            request = tools.prepare(action)
            token, _ = tools.authorize(request)
            self.assertIsNotNone(token)
            result = tools.execute(request, token or "")
            self.assertTrue(result["created"])
            self.assertEqual((root / "src/demo.txt").read_text(), "ironclad")
            with self.assertRaises(PermissionError):
                tools.prepare(ToolAction("read_file", {"path": "../escape"}))

    def test_search_does_not_follow_external_symlink(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            target = Path(outside) / "secret.txt"
            target.write_text("NEVER_INDEX_THIS", encoding="utf-8")
            link = root / "linked.txt"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlinks require elevated privileges on this platform")
            authority = CapabilityAuthority(b"y" * 32)
            policy = PolicyEngine(PolicyConfig())
            sandbox = SandboxExecutor(SandboxConfig(), root)
            tools = ToolRegistry(root, authority, policy, sandbox)
            request = tools.prepare(
                ToolAction("search_text", {"pattern": "NEVER_INDEX_THIS", "glob": "**/*"})
            )
            token, _ = tools.authorize(request)
            result = tools.execute(request, token or "")
            self.assertEqual(result["matches"], [])


if __name__ == "__main__":
    unittest.main()
