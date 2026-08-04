from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any


class ModelRole(StrEnum):
    ROUTER = "router"
    CODER = "coder"
    REASONER = "reasoner"
    CRITIC = "critic"
    SUMMARIZER = "summarizer"
    ORACLE = "oracle"


class RiskLevel(IntEnum):
    READ = 0
    WRITE = 1
    EXECUTE = 2
    DESTRUCTIVE = 3
    EXTERNAL = 4


@dataclass(slots=True, frozen=True)
class HardwareSnapshot:
    platform: str
    cpu_count: int
    ram_total_gb: float
    ram_available_gb: float
    accelerator: str = "cpu"
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    unified_memory: bool = False


@dataclass(slots=True, frozen=True)
class HardwareProfile:
    name: str
    router_parameters_b: tuple[float, float]
    worker_parameters_b: tuple[float, float]
    max_parallel_workers: int
    context_tokens: int
    weight_quantization: str
    kv_quantization: str


@dataclass(slots=True)
class ModelSpec:
    name: str
    role: ModelRole
    provider: str = "ollama"
    size_bytes: int = 0
    keep_alive: str = "90s"
    thinking: bool | str = False


@dataclass(slots=True)
class PlanStep:
    id: str
    title: str
    objective: str
    role: ModelRole = ModelRole.CODER
    depends_on: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    status: str = "pending"

    @classmethod
    def from_dict(cls, value: dict[str, Any], index: int) -> PlanStep:
        raw_role = str(value.get("role", ModelRole.CODER.value))
        try:
            role = ModelRole(raw_role)
        except ValueError:
            role = ModelRole.CODER
        return cls(
            id=str(value.get("id") or f"step-{index + 1}"),
            title=str(value.get("title") or f"Step {index + 1}"),
            objective=str(value.get("objective") or value.get("title") or ""),
            role=role,
            depends_on=[str(item) for item in value.get("depends_on", [])],
            success_criteria=[str(item) for item in value.get("success_criteria", [])],
            allowed_tools=[str(item) for item in value.get("allowed_tools", [])],
        )


@dataclass(slots=True)
class TaskPlan:
    objective: str
    steps: list[PlanStep]
    assumptions: list[str] = field(default_factory=list)
    stop_conditions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any], objective: str) -> TaskPlan:
        steps = [PlanStep.from_dict(step, i) for i, step in enumerate(value.get("steps", []))]
        if not steps:
            steps = [
                PlanStep(
                    id="step-1",
                    title="Complete objective",
                    objective=objective,
                    success_criteria=["The requested outcome is implemented and validated."],
                )
            ]
        return cls(
            objective=objective,
            steps=steps,
            assumptions=[str(item) for item in value.get("assumptions", [])],
            stop_conditions=[str(item) for item in value.get("stop_conditions", [])],
        )


@dataclass(slots=True)
class ToolAction:
    tool: str
    arguments: dict[str, Any]
    reason: str = ""


@dataclass(slots=True)
class WorkerTurn:
    message: str
    actions: list[ToolAction]
    done: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkerTurn:
        actions = [
            ToolAction(
                tool=str(item.get("tool", "")),
                arguments=dict(item.get("arguments") or {}),
                reason=str(item.get("reason", "")),
            )
            for item in value.get("actions", [])
            if isinstance(item, dict) and item.get("tool")
        ]
        return cls(str(value.get("message", "")), actions, bool(value.get("done", False)))


@dataclass(slots=True, frozen=True)
class ToolRequest:
    tool: str
    arguments: dict[str, Any]
    risk: RiskLevel
    project_root: str
