import tempfile
import unittest
from pathlib import Path

from ironclad.config import PolicyConfig
from ironclad.security import CapabilityAuthority, PolicyEngine
from ironclad.types import RiskLevel, ToolRequest


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.root = str(Path(tempfile.gettempdir()).resolve())
        self.request = ToolRequest("read_file", {"path": "README.md"}, RiskLevel.READ, self.root)

    def test_capability_is_bound_and_single_use(self):
        authority = CapabilityAuthority(b"x" * 32)
        token = authority.mint(self.request)
        capability = authority.verify(token, self.request)
        self.assertEqual(capability.tool, "read_file")
        with self.assertRaises(PermissionError):
            authority.verify(token, self.request)

    def test_capability_rejects_argument_change(self):
        authority = CapabilityAuthority(b"x" * 32)
        token = authority.mint(self.request)
        changed = ToolRequest("read_file", {"path": "LICENSE"}, RiskLevel.READ, self.root)
        with self.assertRaises(PermissionError):
            authority.verify(token, changed)

    def test_policy_is_read_only_until_autonomy_enabled(self):
        policy = PolicyEngine(PolicyConfig(autonomous=False, max_auto_risk="write"))
        write = ToolRequest("write_file", {"path": "x", "content": "x"}, RiskLevel.WRITE, self.root)
        self.assertFalse(policy.evaluate(write)[0])
        policy.config.autonomous = True
        self.assertTrue(policy.evaluate(write)[0])


if __name__ == "__main__":
    unittest.main()
