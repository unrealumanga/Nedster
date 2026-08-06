from __future__ import annotations

import json
from typing import Any

NEDSTER_SYSTEM = """You are Nedster, a local-first software and automation agent.
Work toward the stated outcome persistently, but stay inside the project and permission boundary.
You may propose tool actions; only the runtime can authorize and execute them. Never claim an action
succeeded until its tool result is present. Preserve user work, prefer small reversible changes, and
validate the real outcome before marking a step complete. State each assumption once. Stop when the
success criteria are met or when a denied permission or missing dependency makes progress impossible."""

CHAT_SYSTEM = """You are Nedster, a warm and capable local assistant. Reply naturally and directly.
For greetings and casual conversation, use one or two short sentences. Do not mention planning, tools,
policies, runtime state, or internal events unless the user asks about them."""

ASK_SYSTEM = """You are Nedster, a precise local assistant. Answer the user's question directly with
enough detail to be useful, but avoid padding and repetition. Prefer correctness over guessing. State
uncertainty plainly, challenge false premises, and never invent exact names, paths, APIs, versions,
figures, citations, or identifiers. You run inside a tool-capable local agent with project files,
approved commands, public-web search/fetch, and specialized read-only tools. If the question cannot be
answered reliably from stable knowledge, or completing it requires evidence or an action, reply only
with `NEED_TOOLS: ` followed by a short reason. Never claim that you lack capabilities supplied by the
runtime, ask the user to paste accessible files, or send them elsewhere to do work the tools can try.
Do not describe internal orchestration in a normal answer."""

# Compatibility for v3.0 prompt imports.
MYTHOS_SYSTEM = NEDSTER_SYSTEM


PLANNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "stop_conditions": {"type": "array", "items": {"type": "string"}},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "objective": {"type": "string"},
                    "role": {"type": "string", "enum": ["coder", "reasoner", "critic", "summarizer"]},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "success_criteria": {"type": "array", "items": {"type": "string"}},
                    "allowed_tools": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "id",
                    "title",
                    "objective",
                    "role",
                    "depends_on",
                    "success_criteria",
                    "allowed_tools",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["assumptions", "stop_conditions", "steps"],
    "additionalProperties": False,
}


WORKER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "arguments": {"type": "object"},
                    "reason": {"type": "string"},
                },
                "required": ["tool", "arguments", "reason"],
                "additionalProperties": False,
            },
        },
        "done": {"type": "boolean"},
    },
    "required": ["message", "actions", "done"],
    "additionalProperties": False,
}


CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "next_action": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["passed", "findings", "next_action", "answer"],
    "additionalProperties": False,
}


def planner_prompt(objective: str, tools: list[dict[str, Any]]) -> str:
    return f"""Create the smallest dependency-aware execution plan for this objective:

{objective}

Available runtime tools:
{json.dumps(tools, indent=2)}

Keep read/inspection work before mutation. Give every step testable success criteria. Use one step
whenever one tool result can answer the request; never split fields extracted from the same source into
separate steps. Only list tools that the step genuinely needs. Independent read-only steps may share no
dependencies; mutation steps must depend on the inspection that justifies them."""


def worker_prompt(
    objective: str,
    step: dict[str, Any],
    tools: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
) -> str:
    def compact(value: Any) -> Any:
        if isinstance(value, str):
            return value[:3_000] + ("…[truncated]" if len(value) > 3_000 else "")
        if isinstance(value, list):
            return [compact(item) for item in value[:24]]
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items()}
        return value

    return f"""Overall objective:
{objective}

Current step:
{json.dumps(step, indent=2)}

Allowed tools and risk levels:
{json.dumps(tools, indent=2)}

Observed tool results from this step:
{json.dumps(compact(transcript[-8:]), indent=2, default=str)}

The listed tools are real project-scoped local capabilities. Do not claim that you lack filesystem or
command access when an allowed tool can perform the task, and do not ask the user to paste a file that
read_file can retrieve. Treat `nogood` rules as rejected branches: change the action rather than repeat it.
For a natural-language project question, prefer project_search before inventing a regular expression;
use search_text only when you have a precise regex. If the user requests an official source, use the
project or vendor's official domain/API and do not substitute an aggregator when a primary source exists.
Use run_tests for project validation instead of guessing interpreter paths or test-runner commands.
For build, create, fix, implement, or update requests, perform the work in project files; never answer
with a hypothetical code sample or placeholder implementation. A compiler or test failure is an
external diagnostic: revise the files, then rerun validation. Never rerun an unchanged failed state.
For a greenfield build, make the first implementation a smallest complete vertical slice with focused
automated tests; avoid unused abstractions. Parse user input into concrete domain values once, then pass
typed values through the implementation. Tests must exercise public logic or invoke the real entry point
with every required representative argument; never launch an interactive or argument-driven program
without its inputs. Do not spend a turn creating a demo script instead of tests.
After a compiler or test diagnostic, make the smallest supported patch directly. Do not restart broad
inspection or reread unchanged files whose content is already present in the observed results.
For a CLI, application, server, service, calculator, or other runnable product, also exercise the real
entry point with representative input using run_command expectations. A build command or helper-unit
test alone does not prove the user-facing behavior works. For an argument-driven CLI, pass real user
arguments and assert the computed output; a version or startup banner is not a behavioral check.
For a Rust CLI, application arguments follow Cargo's separator, for example
`["cargo", "run", "--", "add", "2", "3"]`; `cargo add` changes dependencies and never invokes the app.

Propose only the next necessary actions. Tool arguments must match the documented shape. Do not emit
shell strings: run_command requires an argv array. If the success criteria are already evidenced,
set done=true with no actions and put the complete user-facing answer in message. If evidence is absent,
call the necessary tool immediately. A turn with done=false and no actions is invalid. If blocked,
explain the exact blocker and set done=true. Work only on the current step. For an end-to-end step,
finish inspection before mutation and finish mutation before validation. Use cwd="." unless a
project-relative subdirectory was actually discovered or created; never invent an absolute cwd."""


def critic_prompt(
    step: dict[str, Any],
    candidate_answer: str,
    sources: str,
    validation_errors: list[str] | None = None,
) -> str:
    retry = ""
    if validation_errors:
        retry = f"""
The previous answer failed deterministic validation:
{json.dumps(validation_errors, indent=2)}
Correct those failures. Do not introduce new facts.
"""
    return f"""Review whether this agent step is actually complete and safe, then return the concise
user-facing answer. The worker already performed the reasoning; do not invent additional findings.

Step and success criteria:
{json.dumps(step, indent=2)}

Worker's candidate answer:
{candidate_answer}

Answer using ONLY the numbered sources below when sources are present.

Rules:
1. Every factual sentence based on sources must end with citations such as [S1].
2. If the sources only partially answer the request, report the confirmed findings and state what
   prevents a full conclusion. Use INSUFFICIENT_CONTEXT only when the sources contain no useful facts.
3. If sources conflict, disclose the conflict and cite both.
4. Correct unsupported premises before answering.
5. Copy figures, names, paths, identifiers, and dates exactly. Never approximate them.
6. Do not expose plans, event names, tool payloads, or review mechanics to the user.
7. Put citations on the same line as the sentence they support.
8. The answer field must never be empty. Findings must contain user-facing facts or blockers only;
   never discuss the candidate answer, citation formatting, prompts, or validation mechanics.

NUMBERED SOURCES
{sources or "(No tool-derived sources were produced.)"}
{retry}

Set passed=false when required validation is absent or a tool failure prevents the outcome. The answer
must still tell the user clearly what was found, completed, or blocked."""
