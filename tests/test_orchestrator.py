from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ironclad.config import NedsterConfig
from ironclad.grounding import GroundingHarness, GroundingSource
from ironclad.orchestrator import (
    NedsterRuntime,
    _attach_single_source_citations,
    _normalize_citations,
    classify_interaction,
)
from ironclad.types import ModelRole, PlanStep, TaskPlan


@pytest.mark.parametrize("value", ["hello", "HI THERE!", "HOW ARE YOU?", "thank you"])
def test_casual_inputs_take_the_chat_path(value):
    assert classify_interaction(value) == "chat"


@pytest.mark.parametrize(
    "value",
    [
        "find out in ~/Workspace folder top crypto trading bots",
        "findout the best crypto trading apps in ~/Workspace",
        "rank the bots in this folder by profits in their logs",
        "findout three apps in this workspace; rank them by profitable decisions in logs",
        "inspect this project",
        "fix the failing tests",
        "read README.md",
        "whats the current BTCUSDT market price",
        "search the web for the latest Ollama release",
        "How many Python files are in the active project?",
        "Inspect the current git diff and summarize the changes.",
    ],
)
def test_local_work_takes_the_agent_path(value):
    assert classify_interaction(value) == "agent"


@pytest.mark.parametrize(
    "value",
    [
        "lets build a RUST calculator?",
        "let's create a CLI",
        "can you fix this project?",
        "could you implement the parser?",
        "I want you to update the tests",
        "help me debug this code",
        "finish the Rust calculator",
        "complete the implementation",
        "compile the application",
        "export the report",
        "render the final image",
        "please do it",
    ],
)
def test_conversational_action_requests_take_the_agent_path(value):
    assert classify_interaction(value) == "agent"


def test_general_questions_take_the_direct_ask_path():
    assert classify_interaction("Why does retrieval reduce hallucination?") == "ask"


def test_self_contained_direct_question_does_not_receive_stale_history():
    runtime = object.__new__(NedsterRuntime)
    runtime.config = NedsterConfig()
    runtime.conversation = [
        {"role": "user", "content": "unrelated old task"},
        {"role": "assistant", "content": "unrelated old answer"},
    ]

    assert runtime._conversation_prompt("What is 2+2?") == "What is 2+2?"


def test_citations_on_the_following_line_are_attached_to_the_sentence():
    assert _normalize_citations("The bot failed startup.\n[S3]") == "The bot failed startup [S3]."
    assert _normalize_citations("Python 3.11+.[S1]") == "Python 3.11+ [S1]."


def test_single_source_citation_repair_cites_each_uncited_sentence():
    repaired = _attach_single_source_citations(
        "There are 113 Python files. This was verified with total_matches=113.", "S1"
    )

    assert repaired == (
        "There are 113 Python files [S1]. This was verified with total_matches=113 [S1]."
    )


def test_invalid_critic_answer_falls_back_to_only_verified_findings():
    runtime = object.__new__(NedsterRuntime)
    runtime.grounding = GroundingHarness()
    sources = [GroundingSource("S1", "bot.log", "profit=12 and trades=4", 1.0)]

    answer = runtime._verified_partial_answer(
        {
            "findings": [
                "[S1] The log records profit=12 and trades=4.",
                "[S2] Another bot made profit=999.",
                "The candidate answer lacks required citation markers [S1].",
                "uncited sentence: invented ranking",
            ]
        },
        sources,
    )

    assert "profit=12" in answer
    assert "profit=999" not in answer
    assert "citation markers" not in answer
    assert "profit-based" not in answer
    assert runtime.grounding.verify(answer, sources).valid


def test_empty_verified_partial_answer_does_not_invent_project_inspection():
    runtime = object.__new__(NedsterRuntime)
    runtime.grounding = GroundingHarness()
    sources = [
        GroundingSource(
            "S1",
            "https://api.example.test/quote",
            '{"price": "42", "symbol": "TESTUSDT"}',
            1.0,
        )
    ]

    answer = runtime._verified_partial_answer({"findings": []}, sources)

    assert "active project" not in answer
    assert "retrieved evidence does not verify" in answer


def test_small_action_plan_is_merged_into_one_end_to_end_tool_step():
    runtime = object.__new__(NedsterRuntime)
    plan = TaskPlan(
        objective="Fix calculator.py and run tests",
        steps=[
            PlanStep("inspect", "Inspect", "Inspect", allowed_tools=["read_file"]),
            PlanStep(
                "edit",
                "Edit",
                "Edit",
                role=ModelRole.CODER,
                depends_on=["inspect"],
                allowed_tools=["replace_text"],
            ),
            PlanStep(
                "test",
                "Test",
                "Test",
                role=ModelRole.CODER,
                depends_on=["edit"],
                allowed_tools=["run_command"],
            ),
        ],
    )

    normalized = runtime._normalize_generated_plan(plan, "Fix calculator.py and run tests")

    assert len(normalized.steps) == 1
    assert normalized.steps[0].id == "complete-objective"
    assert {"read_file", "replace_text", "write_file", "run_command"}.issubset(
        normalized.steps[0].allowed_tools
    )


def test_greenfield_software_plan_requires_tests_and_a_behavioral_smoke_check():
    runtime = object.__new__(NedsterRuntime)

    normalized = runtime._normalize_generated_plan(
        TaskPlan(objective="lets build a Rust calculator", steps=[]),
        "lets build a Rust calculator",
    )

    criteria = normalized.steps[0].success_criteria
    assert any("automated tests" in criterion for criterion in criteria)
    assert any("expected-output assertion" in criterion for criterion in criteria)


@pytest.mark.asyncio
async def test_action_request_skips_the_router_and_uses_one_stateful_step():
    runtime = object.__new__(NedsterRuntime)
    runtime._emit = Mock()
    runtime._json_call = AsyncMock()

    plan = await runtime.plan("lets build a Rust calculator")

    runtime._json_call.assert_not_awaited()
    assert len(plan.steps) == 1
    assert {"read_file", "write_file", "run_command", "run_tests"}.issubset(
        plan.steps[0].allowed_tools
    )


def test_validation_can_repeat_only_after_a_project_mutation():
    before = NedsterRuntime._action_fingerprint("run_tests", {}, 0)
    unchanged = NedsterRuntime._action_fingerprint("run_tests", {}, 0)
    changed = NedsterRuntime._action_fingerprint("run_tests", {}, 1)

    assert before == unchanged
    assert before != changed

    read_before = NedsterRuntime._action_fingerprint("read_file", {"path": "src/main.rs"}, 0)
    read_after = NedsterRuntime._action_fingerprint("read_file", {"path": "src/main.rs"}, 1)
    assert read_before != read_after


def test_zero_test_success_is_not_meaningful_software_validation():
    zero_tests = [
        {
            "action": {"tool": "run_tests"},
            "result": {
                "ok": True,
                "output": {"returncode": 0, "timed_out": False, "tests_run": 0},
            },
        }
    ]
    real_tests = [
        {
            "action": {"tool": "run_tests"},
            "result": {
                "ok": True,
                "output": {"returncode": 0, "timed_out": False, "tests_run": 4},
            },
        }
    ]

    assert not NedsterRuntime._has_meaningful_test_success(zero_tests)
    assert NedsterRuntime._has_meaningful_test_success(real_tests)


def test_asserted_test_runner_command_is_meaningful_validation():
    evidence = [
        {
            "action": {"tool": "run_command"},
            "result": {
                "ok": True,
                "output": {
                    "returncode": 0,
                    "timed_out": False,
                    "assertions_passed": True,
                    "tests_run": 6,
                },
            },
        }
    ]

    assert NedsterRuntime._has_meaningful_test_success(evidence)


def test_user_facing_smoke_requires_asserted_command_output():
    unasserted = [
        {
            "action": {
                "tool": "run_command",
                "arguments": {"command": ["cargo", "build"], "expect_exit": 0},
            },
            "result": {"ok": True, "output": {"returncode": 0, "assertions_passed": None}},
        }
    ]
    asserted = [
        {
            "action": {
                "tool": "run_command",
                "arguments": {
                    "command": ["cargo", "run", "--", "add", "12", "7"],
                    "expect_stdout_contains": "19",
                },
            },
            "result": {"ok": True, "output": {"returncode": 0, "assertions_passed": True}},
        }
    ]

    assert not NedsterRuntime._has_successful_smoke_check(unasserted, "build a Rust calculator CLI")
    assert NedsterRuntime._has_successful_smoke_check(asserted, "build a Rust calculator CLI")


def test_calculator_banner_is_not_accepted_as_behavioral_smoke():
    banner_only = [
        {
            "action": {
                "tool": "run_command",
                "arguments": {
                    "command": ["cargo", "run"],
                    "expect_stdout_contains": "Calculator v0.1.0",
                },
            },
            "result": {"ok": True, "output": {"returncode": 0, "assertions_passed": True}},
        }
    ]

    assert not NedsterRuntime._has_successful_smoke_check(banner_only, "build a calculator")


def test_failed_smoke_after_passing_tests_reopens_the_edit_phase():
    evidence = [
        {
            "action": {"tool": "run_tests"},
            "result": {
                "ok": True,
                "output": {"returncode": 0, "tests_run": 5, "timed_out": False},
            },
        },
        {
            "action": {"tool": "run_command"},
            "result": {
                "ok": True,
                "output": {
                    "returncode": 0,
                    "assertions_passed": False,
                    "timed_out": False,
                },
            },
        },
    ]

    assert NedsterRuntime._has_failed_smoke_attempt_after_tests(evidence)


def test_stated_expected_output_can_verify_an_unasserted_smoke_command():
    evidence = [
        {
            "action": {
                "tool": "run_command",
                "arguments": {"command": ["cargo", "run", "--", "add", "12", "7"]},
            },
            "result": {
                "ok": True,
                "output": {
                    "returncode": 0,
                    "stdout": "19\n",
                    "stderr": "",
                    "timed_out": False,
                    "assertions_passed": None,
                },
            },
        }
    ]

    objective = "finish the calculator and verify the output is 19"
    assert NedsterRuntime._has_successful_smoke_check(evidence, objective)


def test_git_summary_cannot_certify_an_unrelated_action_step():
    runtime = object.__new__(NedsterRuntime)
    step = PlanStep(
        "complete-objective",
        "Complete objective",
        "fix the calculator",
        allowed_tools=["git_diff", "write_file", "run_tests"],
    )

    assert runtime._validated_git_verdict(step, [], []) is None


@pytest.mark.asyncio
async def test_chat_response_uses_one_text_call_and_skips_agent_pipeline():
    runtime = object.__new__(NedsterRuntime)
    runtime.config = NedsterConfig()
    runtime.conversation = []
    runtime.on_event = None
    runtime.ledger = Mock(append=Mock(return_value=1))
    runtime._text_call = AsyncMock(return_value="Hello! How can I help?")
    runtime.run = AsyncMock()

    result = await runtime.respond("hello")

    assert result["answer"] == "Hello! How can I help?"
    assert result["mode"] == "chat"
    assert result["grounded"] is False
    runtime._text_call.assert_awaited_once()
    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_stable_knowledge_answer_stays_on_the_low_cost_direct_path():
    runtime = object.__new__(NedsterRuntime)
    runtime.config = NedsterConfig()
    runtime.conversation = []
    runtime.on_event = None
    runtime.ledger = Mock(append=Mock(return_value=1))
    runtime._text_call = AsyncMock(return_value="Retrieval adds relevant evidence to the prompt.")
    runtime._execute_agent = AsyncMock()

    result = await runtime.respond("Why does retrieval reduce hallucination?")

    assert result["mode"] == "ask"
    assert result["grounded"] is False
    runtime._execute_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_unverified_work_claim_in_direct_answer_is_escalated_to_tools():
    runtime = object.__new__(NedsterRuntime)
    runtime.config = NedsterConfig()
    runtime.conversation = []
    runtime.on_event = None
    runtime.ledger = Mock(append=Mock(return_value=1))
    runtime._text_call = AsyncMock(return_value="I reviewed your project and fixed the issue.")
    runtime._execute_agent = AsyncMock(
        return_value={"answer": "Verified with tools.", "passed": True, "mode": "agent"}
    )

    result = await runtime.respond("What is the likely cause of this obscure issue?")

    assert result["answer"] == "Verified with tools."
    runtime._execute_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_unverified_work_claim_is_removed_from_chat_response():
    runtime = object.__new__(NedsterRuntime)
    runtime.config = NedsterConfig()
    runtime.conversation = []
    runtime.on_event = None
    runtime.ledger = Mock(append=Mock(return_value=1))
    runtime._text_call = AsyncMock(
        return_value="Nice to meet you.\n- Updated your project while we were chatting."
    )

    result = await runtime.respond("hello")

    assert result["answer"] == "Nice to meet you."
    assert result["grounded"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "draft",
    [
        "NEED_TOOLS: this requires current evidence",
        "I cannot access the information needed to answer that.",
        "I don't have access to those records.",
    ],
)
async def test_direct_inability_is_hidden_and_escalated_to_the_agent(draft):
    runtime = object.__new__(NedsterRuntime)
    runtime.config = NedsterConfig()
    runtime.conversation = []
    runtime.on_event = None
    runtime.ledger = Mock(append=Mock(return_value=1))
    runtime._text_call = AsyncMock(return_value=draft)
    runtime._execute_agent = AsyncMock(
        return_value={"answer": "Solved with tools.", "passed": True, "mode": "agent"}
    )

    result = await runtime.respond("Investigate this obscure problem")

    assert result["answer"] == "Solved with tools."
    runtime._execute_agent.assert_awaited_once_with(
        "Investigate this obscure problem", "Investigate this obscure problem"
    )


@pytest.mark.asyncio
async def test_local_access_question_is_answered_from_runtime_capabilities():
    runtime = object.__new__(NedsterRuntime)
    runtime.config = NedsterConfig()
    runtime.config.runtime.project_root = Path("/work/project")
    runtime.conversation = []

    result = await runtime.respond("how do i give you access to local files?")

    assert "can read and search" in result["answer"]
    assert "Edits and approved commands stay inside" in result["answer"]
    assert "/work/project" in result["answer"]
    assert "nedster --project" in result["answer"]


@pytest.mark.asyncio
async def test_model_question_reports_the_active_fleet_without_model_guessing():
    runtime = object.__new__(NedsterRuntime)
    runtime.config = NedsterConfig()
    runtime.conversation = []
    runtime.fleet = SimpleNamespace(
        oracle=None,
        as_rows=Mock(
            return_value=[
                ("router", "qwen3.5:4b-q4_K_M", "ollama"),
                ("coder", "qwen3.5:9b-q4_K_M", "ollama"),
            ]
        ),
    )

    result = await runtime.respond("which LLM you using?")

    assert "qwen3.5:4b-q4_K_M" in result["answer"]
    assert "qwen3.5:9b-q4_K_M" in result["answer"]
    assert "Cloud oracle: disabled" in result["answer"]


def test_explicit_parent_workspace_is_a_read_only_task_scope(tmp_path):
    workspace = tmp_path / "AI_Lab" / "Workspace"
    project = workspace / "Nedster"
    project.mkdir(parents=True)
    runtime = object.__new__(NedsterRuntime)
    runtime.tools = SimpleNamespace(project_root=project)

    root, error = runtime._requested_inspection_root(f"findout the best apps in {workspace}")

    assert error is None
    assert root == workspace


def test_missing_explicit_local_path_gets_a_truthful_remedy(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    runtime = object.__new__(NedsterRuntime)
    runtime.tools = SimpleNamespace(project_root=project)

    root, error = runtime._requested_inspection_root("search /missing/nedster-folder for logs")

    assert root is None
    assert "does not exist" in str(error)


@pytest.mark.asyncio
async def test_agent_followup_adds_criterion_and_resumes_the_original_objective():
    runtime = object.__new__(NedsterRuntime)
    runtime.config = NedsterConfig()
    runtime.conversation = []
    runtime.active_objective = "findout the best crypto apps in this workspace"
    runtime.run = AsyncMock(
        return_value={"answer": "Ranked from logs.", "passed": True, "mode": "agent"}
    )

    result = await runtime.respond("best is the one making profits in logs")

    resumed = runtime.run.await_args.args[0]
    assert "findout the best crypto apps" in resumed
    assert "Additional user criterion: best is the one making profits in logs" in resumed
    assert result["answer"] == "Ranked from logs."


@pytest.mark.asyncio
async def test_current_market_price_uses_a_cross_checked_tool_without_an_llm():
    runtime = object.__new__(NedsterRuntime)
    runtime.config = NedsterConfig()
    runtime.conversation = []
    runtime.on_event = None
    runtime.ledger = Mock(append=Mock(return_value=1))
    runtime.tools = Mock()
    runtime.tools.prepare.return_value = SimpleNamespace(risk=0)
    runtime.tools.authorize.return_value = ("token", "allowed")
    runtime.tools.execute.return_value = {
        "symbol": "BTCUSDT",
        "quotes": [
            {
                "provider": "Binance",
                "price": "64463.73000000",
                "quote_currency": "USDT",
                "source_url": "https://api.binance.com/quote",
            },
            {
                "provider": "Coinbase",
                "price": "64463.9",
                "quote_currency": "USDT",
                "source_url": "https://api.exchange.coinbase.com/quote",
            },
        ],
        "spread_percent": "0.000264",
        "observed_at": "2026-08-06T19:56:01+00:00",
    }
    runtime.run = AsyncMock()

    result = await runtime.respond("whats the current BTCUSDT market price")

    assert result["passed"] is True
    assert result["grounded"] is True
    assert "Binance 64,463.73 USDT" in result["answer"]
    assert "Coinbase 64,463.9 USDT" in result["answer"]
    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("objective", "expected_symbol"),
    [
        ("whats the btc price now ?", "BTCUSDT"),
        ("what is eth price now ?", "ETHUSDT"),
        ("current Ethereum price in USD", "ETHUSD"),
    ],
)
async def test_current_market_price_accepts_common_asset_shorthand(objective, expected_symbol):
    runtime = object.__new__(NedsterRuntime)
    runtime.config = NedsterConfig()
    runtime.conversation = []
    runtime.on_event = None
    runtime.ledger = Mock(append=Mock(return_value=1))
    runtime.tools = Mock()
    runtime.tools.prepare.return_value = SimpleNamespace(risk=0)
    runtime.tools.authorize.return_value = ("token", "allowed")
    runtime.tools.execute.return_value = {
        "symbol": expected_symbol,
        "quotes": [
            {
                "provider": "Exchange",
                "price": "2693.32",
                "quote_currency": "USD" if expected_symbol.endswith("USD") else "USDT",
                "source_url": "https://api.example.test/quote",
            }
        ],
        "spread_percent": "0",
        "observed_at": "2026-09-30T10:43:01+00:00",
    }
    runtime.run = AsyncMock()

    result = await runtime.respond(objective)

    requested = runtime.tools.prepare.call_args.args[0]
    assert requested.tool == "market_price"
    assert requested.arguments == {"symbol": expected_symbol}
    assert result["passed"] is True
    assert result["grounded"] is True
    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_current_news_is_rendered_directly_from_search_results_without_an_llm():
    runtime = object.__new__(NedsterRuntime)
    runtime.config = NedsterConfig()
    runtime.conversation = []
    runtime.on_event = None
    runtime.ledger = Mock(append=Mock(return_value=1))
    runtime.tools = Mock()
    runtime.tools.prepare.return_value = SimpleNamespace(risk=0)
    runtime.tools.authorize.return_value = ("token", "allowed")
    runtime.tools.execute.return_value = {
        "provider": "Bing News",
        "retrieved_at": "2026-09-30T11:00:00+00:00",
        "results": [
            {
                "title": f"Verified headline {index}",
                "url": f"https://example.test/news/{index}",
                "snippet": f"Published report {index}. Published: Wed, 30 Sep 2026 10:0{index}:00 GMT",
            }
            for index in range(1, 6)
        ],
    }
    runtime.run = AsyncMock()

    result = await runtime.respond("top 5 global news?")

    requested = runtime.tools.prepare.call_args.args[0]
    assert requested.tool == "web_search"
    assert requested.arguments == {"query": "latest world news headlines today", "limit": 5}
    assert result["passed"] is True
    assert result["grounded"] is True
    assert "Verified headline 1" in result["answer"]
    assert "https://example.test/news/5" in result["answer"]
    assert "INSUFFICIENT_CONTEXT" not in result["answer"]
    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_step_restarts_with_evidence_nogood_and_action_memory():
    runtime = object.__new__(NedsterRuntime)
    runtime.config = NedsterConfig()
    runtime._emit = Mock()
    step = PlanStep(id="solve", title="Solve", objective="Solve the task")
    inherited = [{"type": "worker_message", "message": "Earlier dependency evidence"}]
    first = {
        "verdict": {"passed": False, "next_action": "Try another source."},
        "evidence": [*inherited, {"type": "worker_message", "message": "First attempt"}],
    }
    second = {"verdict": {"passed": True}, "evidence": []}
    runtime._run_step = AsyncMock(side_effect=[first, second])

    result = await runtime._run_step_with_recovery(
        "Solve the task", step, inherited_evidence=inherited
    )

    assert result is second
    assert runtime._run_step.await_count == 2
    second_call = runtime._run_step.await_args_list[1]
    carried = second_call.kwargs["initial_evidence"]
    assert carried[0] == inherited[0]
    assert any(item.get("type") == "nogood" for item in carried)
    assert "Try another source" in carried[-1]["rule"]
    assert (
        runtime._run_step.await_args_list[0].kwargs["seen_action_fingerprints"]
        is second_call.kwargs["seen_action_fingerprints"]
    )
