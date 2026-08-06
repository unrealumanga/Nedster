from ironclad.grounding import GroundingHarness


def test_grounding_collects_and_ranks_scoped_tool_evidence():
    evidence = [
        {
            "action": {"tool": "list_files", "arguments": {"glob": "**/*"}},
            "result": {"ok": True, "output": {"files": ["README.md", "app.py"]}},
        },
        {
            "action": {"tool": "read_file", "arguments": {"path": "README.md"}},
            "result": {
                "ok": True,
                "output": {
                    "path": "README.md",
                    "content": "The balanced profile uses 16,384 context tokens.",
                },
            },
        },
    ]

    sources = GroundingHarness(max_sources=1).collect(evidence, "balanced context tokens")

    assert len(sources) == 1
    assert sources[0].label == "README.md"
    assert sources[0].source_id == "S1"


def test_grounding_rejects_unknown_citations_and_unsupported_exact_values():
    harness = GroundingHarness()
    sources = harness.collect(
        [
            {
                "action": {"tool": "read_file", "arguments": {"path": "README.md"}},
                "result": {
                    "ok": True,
                    "output": {"path": "README.md", "content": "Nedster version 3.1.0 uses 16,384 tokens."},
                },
            }
        ],
        "Nedster version",
    )

    assert harness.verify("Nedster version 3.1.0 uses 16,384 tokens. [S1]", sources).valid
    check = harness.verify("Nedster version 9.9.9 uses 16,384 tokens. [S2]", sources)
    assert not check.valid
    assert any("unknown citations" in error for error in check.errors)


def test_grounding_requires_citations_for_factual_sentences():
    harness = GroundingHarness()
    sources = harness.collect(
        [
            {
                "action": {"tool": "read_file", "arguments": {"path": "README.md"}},
                "result": {"ok": True, "output": {"path": "README.md", "content": "Nedster is local-first."}},
            }
        ],
        "Nedster",
    )

    check = harness.verify("Nedster is a local-first coding assistant.", sources)
    assert not check.valid
    assert any("uncited sentence" in error for error in check.errors)


def test_grounding_deduplicates_sources_by_file_label():
    evidence = [
        {
            "action": {"tool": "search_text", "arguments": {"pattern": "profit"}},
            "result": {
                "ok": True,
                "output": {
                    "matches": [{"path": "bot.log", "line": 1, "text": "profit=12"}]
                },
            },
        },
        {
            "action": {"tool": "read_file", "arguments": {"path": "bot.log"}},
            "result": {
                "ok": True,
                "output": {"path": "bot.log", "content": "profit=12\ntrades=4"},
            },
        },
    ]

    sources = GroundingHarness().collect(evidence, "profit")

    assert [source.label for source in sources] == ["bot.log"]


def test_grounding_preserves_stderr_test_results_and_replacement_details():
    evidence = [
        {
            "action": {
                "tool": "replace_text",
                "arguments": {"path": "calculator.py", "old": "return 0", "new": "return left + right"},
            },
            "result": {
                "ok": True,
                "output": {"path": "calculator.py", "changed": True, "bytes": 63},
            },
        },
        {
            "action": {"tool": "run_tests", "arguments": {}},
            "result": {
                "ok": True,
                "output": {
                    "command": ["python3", "-m", "unittest"],
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "Ran 1 test\nOK",
                    "timed_out": False,
                },
            },
        },
    ]

    sources = GroundingHarness().collect(evidence, "fix calculator and run tests")

    rendered = GroundingHarness.render(sources)
    assert "returncode=0" in rendered
    assert "Ran 1 test" in rendered
    assert "replacement_text" in rendered


def test_grounding_prefers_a_later_successful_verifier_result_over_a_long_failure():
    evidence = [
        {
            "action": {"tool": "run_tests", "arguments": {}},
            "result": {
                "ok": True,
                "output": {
                    "command": ["cargo", "test"],
                    "returncode": 101,
                    "stdout": "",
                    "stderr": "very long compiler failure " * 100,
                    "timed_out": False,
                },
            },
        },
        {
            "action": {"tool": "run_tests", "arguments": {}},
            "result": {
                "ok": True,
                "output": {
                    "command": ["cargo", "test"],
                    "returncode": 0,
                    "stdout": "4 passed; 0 failed",
                    "stderr": "",
                    "timed_out": False,
                },
            },
        },
    ]

    sources = GroundingHarness().collect(evidence, "build the Rust calculator")

    cargo_test = next(source for source in sources if source.label == "command: cargo test")
    assert "returncode=0" in cargo_test.text
    assert "4 passed" in cargo_test.text
    assert "compiler failure" not in cargo_test.text
