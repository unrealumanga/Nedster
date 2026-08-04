import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ironclad.config import SandboxConfig
from ironclad.sandbox import SandboxExecutor


class SandboxTests(unittest.TestCase):
    def test_auto_uses_container_on_windows(self):
        def available(name):
            return "C:/docker.exe" if name == "docker" else None

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("ironclad.sandbox.platform.system", return_value="Windows"),
            patch("ironclad.sandbox.shutil.which", side_effect=available),
        ):
            sandbox = SandboxExecutor(SandboxConfig(), Path(directory))
        self.assertEqual(sandbox.backend, "docker")

    def test_process_fallback_is_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            sandbox = SandboxExecutor(SandboxConfig(backend="process"), Path(directory))
            with self.assertRaises(RuntimeError):
                sandbox.build_command(["python", "-V"])


if __name__ == "__main__":
    unittest.main()
