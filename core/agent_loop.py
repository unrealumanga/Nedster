"""
core/agent_loop.py — the actual agentic loop.

WHY THIS EXISTS
---------------
agent.py:generate() today does exactly one pass:

    chat(tools=native_tools)  ->  execute every tool_call  ->  chat(NO tools)  ->  done

The second chat() call omits `tools=`, so the model *cannot* request another
tool even when the first result makes it obvious it must (read a file, then
edit it; run tests, then fix the failure). Every task that needs step N+1 to
depend on step N's output silently terminates after step 1 with a confident
summary. The README advertises "Iteration Budgets" and "continuity watchdogs";
neither exists in the code.

This module is the loop that was missing. It is deliberately transport-agnostic:
`chat_fn` is injected, so it unit-tests without Ollama running.

DESIGN
------
- Bounded: hard iteration budget from the VRAM tier (see model_registry.Tier).
- Two roles, one model: PLANNER_SUFFIX / EXECUTOR_SUFFIX are appended to the
  same base prompt rather than routed to two different models. Below ~18GB a
  planner/worker split thrashes weights off the GPU on every handoff.
- Stall detection: identical (tool, args) repeated, or N consecutive text-only
  turns, ends the loop instead of burning budget.
- Honest outcomes: a task is `success` only if it ended with the model
  volunteering completion AND no tool errored on the final iteration.
  agent.py currently writes outcome="success" unconditionally into the replay
  corpus, which teaches the ReplayEngine to suggest workflows that failed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

TOOL_ERROR_MARKER = "[ERROR"

BASE_IDENTITY = """[IDENTITY]
You are Nedster, a local AI software engineer running natively on the user's
machine with real filesystem and shell access through your tools. You are not a
chat assistant describing what could be done; you do it.

These statements are always false when you say them:
- "I cannot directly read or write files on your local machine."
- "You can create the file manually by running:"
- "I don't have file system tools available in my current environment."

[EXECUTION]
No filler openers. No option menus. No emoji except a leading warning glyph on
a single inline caution line. Never show a bash block as markdown for the user
to run — call the tool.

[DISCIPLINE]
- Find before you read: discover paths with search tools, never guess them.
- Activate an existing venv before any pip/python call; never create a second one.
- Detect `python` vs `python3` before writing a script that invokes either.
- After writing any shell script: syntax-check it before declaring done.
- Never echo an API key or secret. Mask phone numbers and personal names in output.
- Verify before asserting. A passing check means proceed, not report.
- When asked what you can do, read your own codebase before claiming a gap.
"""

PLANNER_SUFFIX = """
[MODE: PLAN]
Produce the shortest ordered sequence of tool calls that completes the request.
State the plan in a <think> block, then begin executing it. Do not ask which
option the user prefers; choose the one an experienced engineer would.
"""

EXECUTOR_SUFFIX = """
[MODE: EXECUTE]
You are mid-task. The tool results above are ground truth — trust them over your
expectations. Continue calling tools until the request is fully satisfied.
When and only when nothing remains, reply with a plain-text summary of what
changed (paths, commands, outcomes) and prefix it with TASK_COMPLETE.
If you are blocked, prefix with TASK_BLOCKED and name the specific blocker.
"""

COMPLETE_MARKER = "TASK_COMPLETE"
BLOCKED_MARKER = "TASK_BLOCKED"


@dataclass
class ToolInvocation:
    name: str
    args: dict
    result: str = ""
    ok: bool = True
    duration_s: float = 0.0

    def fingerprint(self) -> str:
        return f"{self.name}:{json.dumps(self.args, sort_keys=True, default=str)}"


@dataclass
class LoopResult:
    outcome: str                      # success | blocked | budget_exhausted | stalled | error
    final_text: str = ""
    iterations: int = 0
    invocations: list[ToolInvocation] = field(default_factory=list)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.outcome == "success"

    def steps_for_replay(self) -> list[dict]:
        return [{"tool": i.name, "args": i.args, "ok": i.ok} for i in self.invocations]


def normalize_tool_calls(message: dict) -> list[dict]:
    """Pull tool calls out of an Ollama-shaped message, tolerating the three
    formats this codebase encounters: native tool_calls, a JSON-string
    arguments field, and legacy <tool name=...> XML in the content."""
    calls: list[dict] = []

    for i, call in enumerate(message.get("tool_calls") or []):
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        raw = fn.get("arguments", {})
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {}
        calls.append({
            "id": (call.get("id") if isinstance(call, dict) else None) or f"call_{i}",
            "name": fn.get("name", ""),
            "args": raw if isinstance(raw, dict) else {},
        })

    content = message.get("content") or ""
    if not calls and "<tool name=" in content:
        try:
            from tools import parse_tool_calls  # type: ignore

            for i, xml in enumerate(parse_tool_calls(content)):
                calls.append({"id": f"xml_{i}", "name": xml["name"], "args": xml.get("args", {})})
        except Exception:
            pass

    return [c for c in calls if c["name"]]


class AgentLoop:
    """Bounded ReAct loop.

    chat_fn(messages, tools) -> dict shaped like Ollama's response["message"]
    execute_fn(name, args)   -> str
    emit_fn(event, payload)  -> None                (optional)
    """

    def __init__(
        self,
        chat_fn: Callable[[list[dict], list | None], dict],
        execute_fn: Callable[[str, dict], str],
        tools_schema: list | None = None,
        max_iterations: int = 8,
        max_text_only_turns: int = 2,
        emit_fn: Callable[[str, Any], None] | None = None,
        on_step: Callable[[str, Any], None] | None = None,
    ):
        self.chat_fn = chat_fn
        self.execute_fn = execute_fn
        self.tools_schema = tools_schema
        self.max_iterations = max(1, max_iterations)
        self.max_text_only_turns = max_text_only_turns
        self.emit_fn = emit_fn or (lambda *_: None)
        self.on_step = on_step or (lambda *_: None)

    def run(self, user_input: str, history: list[dict] | None = None,
            base_prompt: str = BASE_IDENTITY, advisory: str = "") -> LoopResult:
        messages: list[dict] = [{"role": "system", "content": base_prompt + PLANNER_SUFFIX}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_input + advisory})

        result = LoopResult(outcome="budget_exhausted")
        seen_fingerprints: set[str] = set()
        text_only_streak = 0

        for iteration in range(1, self.max_iterations + 1):
            result.iterations = iteration
            self.on_step("iteration", iteration)

            try:
                message = self.chat_fn(messages, self.tools_schema)
            except Exception as exc:
                result.outcome, result.error = "error", str(exc)
                self.emit_fn("ON_ERROR", {"error": str(exc), "iteration": iteration})
                return result

            # Ollama returns a Message object; a stub or a bridge may return a
            # bare dict with no role. Normalise before it enters `messages`,
            # otherwise the next turn's history contains a role-less turn and
            # the provider rejects the request.
            message = dict(message or {})
            message.setdefault("role", "assistant")
            text = (message.get("content") or "").strip()
            calls = normalize_tool_calls(message)

            # -- terminal text, no tools requested --------------------------
            if not calls:
                text_only_streak += 1
                result.final_text = text or result.final_text

                if COMPLETE_MARKER in text:
                    result.outcome = "success" if self._last_iteration_clean(result) else "blocked"
                    result.final_text = text.replace(COMPLETE_MARKER, "").strip()
                    return result
                if BLOCKED_MARKER in text:
                    result.outcome = "blocked"
                    result.final_text = text.replace(BLOCKED_MARKER, "").strip()
                    return result

                if text_only_streak >= self.max_text_only_turns:
                    # Model has stopped asking for tools but never claimed done.
                    # Treat as finished rather than nagging it into a loop.
                    result.outcome = "success" if result.invocations and self._last_iteration_clean(result) else "stalled"
                    return result

                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": ("If the request is fully satisfied, reply with " + COMPLETE_MARKER +
                                " and a summary of what changed. If work remains, call the next tool now."),
                })
                continue

            text_only_streak = 0
            if text:
                self.on_step("assistant_text", text)

            # -- execute the requested tools --------------------------------
            messages.append(message)
            iteration_invocations: list[ToolInvocation] = []
            repeat_detected = False

            for call in calls:
                inv = ToolInvocation(name=call["name"], args=call["args"])
                fp = inv.fingerprint()
                if fp in seen_fingerprints:
                    repeat_detected = True
                    inv.result = ("[ERROR] Identical call already made this task with the same "
                                  "arguments and the same result. Do not repeat it — either use the "
                                  "earlier result or change approach.")
                    inv.ok = False
                else:
                    seen_fingerprints.add(fp)
                    self.emit_fn("BEFORE_TOOL", {"name": inv.name, "args": inv.args})
                    started = time.monotonic()
                    try:
                        inv.result = str(self.execute_fn(inv.name, inv.args))
                    except Exception as exc:
                        inv.result, inv.ok = f"[ERROR] {inv.name} raised: {exc}", False
                    inv.duration_s = time.monotonic() - started
                    inv.ok = inv.ok and not inv.result.lstrip().startswith(TOOL_ERROR_MARKER)
                    self.emit_fn("AFTER_TOOL", {"name": inv.name, "result": inv.result, "ok": inv.ok})
                    if inv.name in ("write_file", "edit_file") and inv.ok:
                        self.emit_fn("AFTER_WRITE", {"path": inv.args.get("path"), "diff": inv.result})

                self.on_step("tool", inv)
                iteration_invocations.append(inv)
                result.invocations.append(inv)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": inv.name,
                    "content": inv.result,
                })

            # Switch the system prompt into executor mode after the first pass so
            # the model stops re-planning and starts converging.
            messages[0] = {"role": "system", "content": base_prompt + EXECUTOR_SUFFIX}

            if repeat_detected and all(not i.ok for i in iteration_invocations):
                result.outcome = "stalled"
                result.final_text = text or "Loop aborted: the model repeated an identical tool call."
                return result

        # budget exhausted — ask for a wrap-up so the user gets something usable
        result.final_text = self._final_summary(messages) or result.final_text
        return result

    # ------------------------------------------------------------------
    def _last_iteration_clean(self, result: LoopResult) -> bool:
        """True when nothing errored in the most recent tool batch. Used so
        `success` means something, unlike the unconditional outcome="success"
        currently written to the replay corpus in agent.py."""
        if not result.invocations:
            return True
        return result.invocations[-1].ok

    def _final_summary(self, messages: list[dict]) -> str:
        try:
            wrap = messages + [{
                "role": "user",
                "content": ("Iteration budget reached. In plain text, state what you completed, "
                            "what remains, and the exact next step. No tool calls."),
            }]
            return (self.chat_fn(wrap, None).get("content") or "").strip()
        except Exception:
            return ""
