from __future__ import annotations

import json
from typing import Any

MYTHOS_SYSTEM = """You are Nedster Mythos, a local-first software agent inside the Ironclad runtime.
Work toward the stated outcome persistently, but stay inside the project and permission boundary.
You may propose tool actions; only the runtime can authorize and execute them. Never claim an action
succeeded until its tool result is present. Preserve user work, prefer small reversible changes, and
validate the real outcome before marking a step complete. State each assumption once. Stop when the
success criteria are met or when a denied permission or missing dependency makes progress impossible."""


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
                "required": ["id", "title", "objective", "role", "depends_on", "success_criteria", "allowed_tools"],
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
    },
    "required": ["passed", "findings", "next_action"],
    "additionalProperties": False,
}


def planner_prompt(objective: str, tools: list[dict[str, str]]) -> str:
    return f"""Create the smallest dependency-aware execution plan for this objective:

{objective}

Available runtime tools:
{json.dumps(tools, indent=2)}

Keep read/inspection work before mutation. Give every step testable success criteria. Only list tools
that the step genuinely needs. Independent read-only steps may share no dependencies; mutation steps
must depend on the inspection that justifies them."""


def worker_prompt(
    objective: str,
    step: dict[str, Any],
    tools: list[dict[str, str]],
    transcript: list[dict[str, Any]],
) -> str:
    return f"""Overall objective:
{objective}

Current step:
{json.dumps(step, indent=2)}

Allowed tools and risk levels:
{json.dumps(tools, indent=2)}

Observed tool results from this step:
{json.dumps(transcript[-12:], indent=2, default=str)}

Propose only the next necessary actions. Tool arguments must match the documented shape. Do not emit
shell strings: run_command requires an argv array. If the success criteria are already evidenced,
set done=true with no actions. If blocked, explain the exact blocker and set done=true."""


def critic_prompt(step: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    return f"""Review whether this agent step is actually complete and safe.

Step and success criteria:
{json.dumps(step, indent=2)}

Evidence:
{json.dumps(evidence[-16:], indent=2, default=str)}

Fail the step if the evidence is only a claim, a command failed, required validation is missing, or a
write escaped the requested scope. Return concise, actionable findings."""
