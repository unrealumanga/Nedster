import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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

    def test_write_file_normalizes_one_item_string_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = object.__new__(ToolRegistry)
            tools.project_root = Path(directory)

            result = tools._write_file(
                {"path": "src/main.rs", "content": ["fn main() {}\n"]}
            )

            self.assertTrue(result["created"])
            self.assertEqual(
                (Path(directory) / "src/main.rs").read_text(encoding="utf-8"),
                "fn main() {}\n",
            )

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

    def test_globs_are_recursive_support_braces_and_ignore_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bot" / "logs").mkdir(parents=True)
            (root / "bot" / "logs" / "trade.log").write_text("profit=12", encoding="utf-8")
            (root / "bot" / "report.json").write_text('{"profit": 12}', encoding="utf-8")
            (root / "venv" / "package").mkdir(parents=True)
            (root / "venv" / "package" / "noise.log").write_text("profit=999", encoding="utf-8")
            authority = CapabilityAuthority(b"x" * 32)
            tools = ToolRegistry(
                root,
                authority,
                PolicyEngine(PolicyConfig()),
                SandboxExecutor(SandboxConfig(), root),
            )

            request = tools.prepare(
                ToolAction("list_files", {"glob": "**/*.{log,json,txt,csv}"})
            )
            token, _ = tools.authorize(request)
            listed = tools.execute(request, token or "")
            self.assertEqual(listed["files"], ["bot/logs/trade.log", "bot/report.json"])
            self.assertEqual(listed["count"], 2)
            self.assertEqual(listed["total_matches"], 2)

            request = tools.prepare(ToolAction("search_text", {"pattern": "profit", "glob": "*.log"}))
            token, _ = tools.authorize(request)
            searched = tools.execute(request, token or "")
            self.assertEqual([item["path"] for item in searched["matches"]], ["bot/logs/trade.log"])

    def test_web_fetch_blocks_private_hosts(self):
        with self.assertRaises(PermissionError):
            ToolRegistry._public_https_url("https://127.0.0.1/secrets")
        with self.assertRaises(ValueError):
            ToolRegistry._public_https_url("http://example.com")

    def test_web_search_parses_public_results_without_an_api_key(self):
        tools = object.__new__(ToolRegistry)
        page = """
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fbtc">BTC price</a>
        <div class="result__snippet">A current public quote.</div>
        """
        with patch.object(
            ToolRegistry,
            "_get_public",
            return_value=("https://html.duckduckgo.com/html/?q=btc", "text/html", page),
        ):
            result = tools._web_search({"query": "BTC price", "limit": 3})

        self.assertEqual(result["provider"], "DuckDuckGo")
        self.assertEqual(result["results"][0]["url"], "https://example.com/btc")
        self.assertEqual(result["results"][0]["snippet"], "A current public quote.")

    def test_web_search_falls_back_to_bing_rss_when_duckduckgo_is_blocked(self):
        tools = object.__new__(ToolRegistry)
        rss = """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0"><channel><item>
          <title>Verified world headline</title>
          <link>http://www.bing.com/news/apiclick.aspx?url=https%3A%2F%2Fexample.com%2Fworld</link>
          <description>A current public report.</description>
          <pubDate>Wed, 30 Sep 2026 10:00:00 GMT</pubDate>
        </item></channel></rss>
        """
        with patch.object(
            ToolRegistry,
            "_get_public",
            side_effect=[
                (
                    "https://html.duckduckgo.com/html/?q=world+news",
                    "text/html",
                    '<div class="anomaly-modal">bot challenge</div>',
                ),
                (
                    "https://www.bing.com/news/search?format=rss&q=world+news",
                    "application/xml",
                    rss,
                ),
            ],
        ):
            result = tools._web_search({"query": "world news", "limit": 3})

        self.assertEqual(result["provider"], "Bing News")
        self.assertEqual(result["results"][0]["url"], "https://example.com/world")
        self.assertIn("Published: Wed, 30 Sep 2026", result["results"][0]["snippet"])
        self.assertEqual(result["provider_warnings"], ["DuckDuckGo: blocked by a bot challenge"])

    def test_market_price_cross_checks_two_public_providers(self):
        tools = object.__new__(ToolRegistry)
        with patch.object(
            ToolRegistry,
            "_get_public",
            side_effect=[
                (
                    "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
                    "application/json",
                    '{"symbol":"BTCUSDT","price":"64463.73000000"}',
                ),
                (
                    "https://api.exchange.coinbase.com/products/BTC-USDT/ticker",
                    "application/json",
                    '{"price":"64463.9","time":"2026-08-06T19:56:01Z"}',
                ),
            ],
        ):
            result = tools._market_price({"symbol": "BTC/USDT"})

        self.assertEqual(result["symbol"], "BTCUSDT")
        self.assertEqual([item["provider"] for item in result["quotes"]], ["Binance", "Coinbase"])
        self.assertLess(float(result["spread_percent"]), 0.01)

    def test_project_search_ranks_natural_language_passages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "architecture.md").write_text(
                "## Explicit non-goals\n\n- No automatic internet access.\n",
                encoding="utf-8",
            )
            tools = object.__new__(ToolRegistry)
            tools.project_root = root

            result = tools._project_search(
                {"query": "where is automatic internet access called a non-goal?"}
            )

            self.assertEqual(result["matches"][0]["path"], "docs/architecture.md")
            self.assertIn("automatic internet access", result["matches"][0]["text"])

    def test_project_search_prefers_implementation_for_code_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "architecture.md").write_text(
                "Citation grounding is verified and rejects unsupported values.\n",
                encoding="utf-8",
            )
            (root / "grounding.py").write_text(
                "class GroundingHarness:\n    def verify(self, answer):\n        return reject_unsupported(answer)\n",
                encoding="utf-8",
            )
            tools = object.__new__(ToolRegistry)
            tools.project_root = root

            result = tools._project_search(
                {"query": "Where is citation grounding verified and what does it reject?"}
            )

            self.assertEqual(result["matches"][0]["path"], "grounding.py")

    def test_run_tests_selects_a_python_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_demo.py").write_text("def test_demo(): pass\n", encoding="utf-8")
            tools = object.__new__(ToolRegistry)
            tools.project_root = root
            tools.sandbox = Mock()
            tools.sandbox.execute.return_value = Mock(
                backend="test",
                returncode=0,
                stdout="1 passed",
                stderr="",
                timed_out=False,
            )

            with patch("ironclad.tools.shutil.which", side_effect=lambda name: "/usr/bin/uv" if name == "uv" else None):
                result = tools._run_tests({})

            self.assertEqual(result["command"], ["uv", "run", "--with", "pytest", "pytest", "-q"])
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(result["tests_run"], 1)

    def test_git_diff_returns_structured_tracked_and_untracked_files(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = object.__new__(ToolRegistry)
            tools.project_root = Path(directory)
            completed = [
                Mock(returncode=0, stdout="diff body", stderr=""),
                Mock(returncode=0, stdout="1 file changed", stderr=""),
                Mock(returncode=0, stdout=" M tracked.py\n?? new.py\n", stderr=""),
            ]

            with patch("ironclad.tools.subprocess.run", side_effect=completed):
                result = tools._git_diff({})

            self.assertEqual(result["tracked_files"], ["tracked.py"])
            self.assertEqual(result["untracked_files"], ["new.py"])
            self.assertEqual(result["tracked_count"], 1)
            self.assertEqual(result["untracked_count"], 1)

    def test_run_command_checks_expected_user_facing_output(self):
        tools = object.__new__(ToolRegistry)
        tools.project_root = Path("/tmp").resolve()
        tools.sandbox = Mock()
        tools.sandbox.execute.return_value = Mock(
            backend="test",
            returncode=0,
            stdout="Result: 19\n",
            stderr="",
            timed_out=False,
        )

        result = tools._run_command(
            {
                "command": ["cargo", "run", "--", "add", "12", "7"],
                "expect_stdout_contains": "19",
            }
        )

        self.assertTrue(result["assertions_passed"])
        self.assertEqual(result["assertions"][1]["check"], "stdout_contains")

        normalized = tools._run_command(
            {
                "command": ["cargo", "run"],
                "expect_stdout_contains": ["19"],
            }
        )
        self.assertTrue(normalized["assertions_passed"])

    def test_run_command_recognizes_a_successful_test_command(self):
        tools = object.__new__(ToolRegistry)
        tools.project_root = Path("/tmp").resolve()
        tools.sandbox = Mock()
        tools.sandbox.execute.return_value = Mock(
            backend="test",
            returncode=0,
            stdout="running 0 tests\nrunning 6 tests\ntest result: ok. 6 passed",
            stderr="",
            timed_out=False,
        )

        result = tools._run_command(
            {"command": ["cargo", "test", "--", "--nocapture"], "expect_exit": 0}
        )

        self.assertEqual(result["command"], ["cargo", "test", "--", "--nocapture"])
        self.assertEqual(result["tests_run"], 6)
        self.assertTrue(result["assertions_passed"])

    def test_run_command_inserts_cargo_application_argument_separator(self):
        tools = object.__new__(ToolRegistry)
        tools.project_root = Path("/tmp").resolve()
        tools.sandbox = Mock()
        tools.sandbox.execute.return_value = Mock(
            backend="test",
            returncode=0,
            stdout="5.5\n",
            stderr="",
            timed_out=False,
        )

        result = tools._run_command(
            {
                "command": ["cargo", "run", "--add", "2.0", "3.5"],
                "expect_stdout_contains": "5.5",
            }
        )

        self.assertEqual(
            tools.sandbox.execute.call_args.args[0],
            ["cargo", "run", "--", "--add", "2.0", "3.5"],
        )
        self.assertEqual(result["command"], ["cargo", "run", "--", "--add", "2.0", "3.5"])
        self.assertTrue(result["assertions_passed"])

    def test_cargo_flags_remain_before_application_arguments(self):
        self.assertEqual(
            ToolRegistry._normalize_cargo_run(
                ["cargo", "run", "--quiet", "--release", "add", "2", "3"]
            ),
            ["cargo", "run", "--quiet", "--release", "--", "add", "2", "3"],
        )


if __name__ == "__main__":
    unittest.main()
