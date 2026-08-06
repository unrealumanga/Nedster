import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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

    def test_bubblewrap_mounts_only_rust_toolchain_directories_from_home(self):
        with tempfile.TemporaryDirectory() as project, tempfile.TemporaryDirectory() as home:
            home_path = Path(home)
            (home_path / ".cargo" / "bin").mkdir(parents=True)
            (home_path / ".rustup").mkdir()
            completed = Mock(returncode=0, stdout="cargo 1.0", stderr="")
            with (
                patch("ironclad.sandbox.Path.home", return_value=home_path),
                patch("ironclad.sandbox.subprocess.run", return_value=completed) as run,
            ):
                sandbox = SandboxExecutor(SandboxConfig(backend="bubblewrap"), Path(project))
                sandbox.execute(["cargo", "--version"])

            command = run.call_args.args[0]
            environment = run.call_args.kwargs["env"]
            self.assertIn(str(home_path / ".cargo" / "bin"), command)
            self.assertIn(str(home_path / ".rustup"), command)
            self.assertNotIn(str(home_path), command[command.index("--ro-bind") + 1 :: 2])
            self.assertEqual(environment["CARGO_HOME"], "/tmp/cargo-home")
            self.assertEqual(environment["RUSTUP_HOME"], str(home_path / ".rustup"))


if __name__ == "__main__":
    unittest.main()
