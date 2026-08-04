"""
core/loop_adapter.py — wires AgentLoop into the existing NedsterAgent.

Integration is intentionally three lines in agent.py rather than a rewrite:

    from core.loop_adapter import run_with_loop      # top of file

    def generate(self, user_input: str):            # replace the body
        return run_with_loop(self, user_input)

Everything the old generate() did that was worth keeping is preserved here:
event emission, replay advisory lookup, replay storage, TUI output, memory turn.
What changes is that tool execution now iterates.
"""

from __future__ import annotations

import json
from contextlib import suppress

from core.agent_loop import BASE_IDENTITY, AgentLoop


def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0.0


def _advisory(agent, user_input: str) -> str:
    """Past-workflow suggestion, same as agent.py did, but only for workflows
    that actually succeeded. The old code stored every run as outcome="success",
    so this filter matters more than it looks."""
    try:
        suggestion = agent.replay_engine.suggest_for(user_input, _jaccard)
    except Exception:
        return ""
    if not suggestion:
        return ""
    return (
        "\n\n[ADVISORY — a similar past task succeeded with these steps. "
        "Adapt or ignore as the situation warrants.]\n"
        f"Task: {suggestion['from']}\n"
        f"Steps: {json.dumps(suggestion['suggested_steps'], indent=2)}\n"
        f"Similarity: {suggestion['confidence']:.2f}\n"
    )


def _num_ctx(agent) -> int | None:
    import os

    raw = os.environ.get("NEDSTER_NUM_CTX")
    return int(raw) if raw and raw.isdigit() else None


def run_with_loop(agent, user_input: str, max_iterations: int | None = None):
    """Drop-in replacement for NedsterAgent.generate()."""
    tui = agent.tui
    tui.print_thinking("Planning...")
    agent._sync_emit("BEFORE_PROMPT", {"user_input": user_input})

    if max_iterations is None:
        try:
            from core.model_registry import resolve_tier

            max_iterations = resolve_tier().max_tool_iterations
        except Exception:
            max_iterations = 8

    options = {}
    ctx = _num_ctx(agent)
    if ctx:
        options["num_ctx"] = ctx

    def chat_fn(messages, tools):
        if agent.backend == "local":
            import ollama

            client = ollama.Client(host="127.0.0.1")
            kwargs = {"model": agent.model, "messages": messages, "stream": False}
            if tools:
                kwargs["tools"] = tools
            if options:
                kwargs["options"] = options
            response = client.chat(**kwargs)
        else:
            from nedster_api import LiteLLMNedsterBridge

            bridge = LiteLLMNedsterBridge()
            response = bridge.generate(messages=messages, tools=tools)
        message = response.get("message", {}) if isinstance(response, dict) else response["message"]
        return dict(message) if not isinstance(message, dict) else message

    def on_step(kind, value):
        if kind == "assistant_text" and agent.think_visible:
            tui.print_thinking(str(value)[:2000])
        elif kind == "tool":
            tui.print_tool_call(value.name, value.args)
            tui.print_tool_result(value.name, value.result)
        elif kind == "iteration" and value > 1:
            tui.print_thinking(f"Step {value}/{max_iterations}...")

    loop = AgentLoop(
        chat_fn=chat_fn,
        execute_fn=agent.executor.execute,
        tools_schema=agent._get_native_tools(),
        max_iterations=max_iterations,
        emit_fn=agent._sync_emit,
        on_step=on_step,
    )

    result = loop.run(user_input, history=agent.memory.get_context_messages(),
                      base_prompt=BASE_IDENTITY, advisory=_advisory(agent, user_input))

    if result.outcome == "error":
        tui.print_error(f"Agent loop failed: {result.error}")
    else:
        if result.final_text:
            tui.print_response(result.final_text)
        if result.outcome == "budget_exhausted":
            tui.print_error(
                f"Iteration budget ({max_iterations}) reached after "
                f"{len(result.invocations)} tool calls. Re-run to continue, or raise the budget."
            )
        elif result.outcome == "stalled":
            tui.print_error("Loop stopped: the model repeated itself without progressing.")

    # Only successful runs enter the replay corpus.
    if result.succeeded and result.invocations:
        with suppress(Exception):
            agent.replay_engine.store(
                user_input=user_input,
                steps=result.steps_for_replay(),
                outcome="success",
                metrics={"iterations": result.iterations, "tools": len(result.invocations)},
            )

    with suppress(Exception):
        agent.memory.add_turn(user_input, result.final_text)

    agent._sync_emit("AFTER_PROMPT", {
        "user_input": user_input,
        "response": result.final_text,
        "outcome": result.outcome,
    })
    return result
