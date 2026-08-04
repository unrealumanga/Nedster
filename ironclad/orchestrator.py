from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from .config import IroncladConfig
from .hardware import choose_profile, probe_hardware
from .ledger import EventLedger, load_or_create_ledger_key
from .models import ModelFleet, ModelScheduler
from .prompts import (
    CRITIC_SCHEMA,
    MYTHOS_SYSTEM,
    PLANNER_SCHEMA,
    WORKER_SCHEMA,
    critic_prompt,
    planner_prompt,
    worker_prompt,
)
from .providers import BackendError, OllamaBackend, OpenAIResponsesBackend
from .sandbox import SandboxExecutor
from .security import CapabilityAuthority, PolicyEngine
from .tools import ToolRegistry
from .types import ModelRole, PlanStep, TaskPlan, WorkerTurn

EventCallback = Callable[[dict[str, Any]], None]
READ_ONLY_TOOLS = frozenset({"read_file", "list_files", "search_text", "git_diff"})


class MythosOrchestrator:
    def __init__(self, config: IroncladConfig, on_event: EventCallback | None = None):
        self.config = config
        self.on_event = on_event
        self.hardware = probe_hardware()
        self.profile = choose_profile(self.hardware)
        self.ollama = OllamaBackend(config.models.ollama_url)
        self.openai = OpenAIResponsesBackend()
        self.fleet: ModelFleet | None = None
        authority = CapabilityAuthority()
        policy = PolicyEngine(config.policy)
        sandbox = SandboxExecutor(config.sandbox, config.runtime.project_root)
        self.tools = ToolRegistry(config.runtime.project_root, authority, policy, sandbox)
        key = load_or_create_ledger_key(config.runtime.state_dir, config.runtime.ledger_key_env)
        self.ledger = EventLedger(config.runtime.state_dir / "events.sqlite3", key)

    def _emit(self, kind: str, actor: str, **payload: Any) -> None:
        sequence = self.ledger.append(kind, actor, payload)
        event = {"seq": sequence, "kind": kind, "actor": actor, **payload}
        if self.on_event:
            self.on_event(event)

    async def initialize(self) -> ModelFleet:
        scheduler = ModelScheduler(self.config.models, self.profile, self.ollama)
        self.fleet = await scheduler.discover()
        self._emit(
            "runtime.ready",
            "ironclad",
            profile=self.profile.name,
            accelerator=self.hardware.accelerator,
            fleet=self.fleet.as_rows(),
            sandbox=self.tools.sandbox.backend,
        )
        return self.fleet

    async def _json_call(
        self,
        role: ModelRole,
        prompt: str,
        schema: dict[str, Any],
        prefer_oracle: bool = False,
    ) -> dict[str, Any]:
        if not self.fleet:
            await self.initialize()
        assert self.fleet is not None
        spec = self.fleet.oracle if prefer_oracle and self.fleet.oracle else self.fleet.for_role(role)
        if spec.provider == "openai":
            return await self.openai.complete_json(
                model=spec.name,
                system=MYTHOS_SYSTEM,
                prompt=prompt,
                schema=schema,
                reasoning_effort=self.config.models.cloud_reasoning_effort,
                reasoning_mode=self.config.models.cloud_reasoning_mode,
                reasoning_context=self.config.models.cloud_reasoning_context,
            )
        return await self.ollama.complete_json(
            model=spec.name,
            system=MYTHOS_SYSTEM,
            prompt=prompt,
            schema=schema,
            thinking=spec.thinking,
            keep_alive=spec.keep_alive,
        )

    async def plan(self, objective: str) -> TaskPlan:
        self._emit("objective.received", "user", objective=objective)
        try:
            value = await self._json_call(
                ModelRole.ROUTER,
                planner_prompt(objective, self.tools.descriptions()),
                PLANNER_SCHEMA,
            )
            plan = TaskPlan.from_dict(value, objective)
        except (BackendError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            self._emit("planner.fallback", "router", error=str(exc))
            plan = TaskPlan.from_dict({}, objective)
        self._emit("plan.created", "router", plan=asdict(plan))
        return plan

    def _tool_descriptions_for(self, step: PlanStep) -> list[dict[str, str]]:
        all_tools = self.tools.descriptions()
        if not step.allowed_tools:
            return all_tools
        allowed = set(step.allowed_tools)
        return [tool for tool in all_tools if tool["name"] in allowed]

    async def _review(self, step: PlanStep, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        prefer_oracle = bool(self.fleet and self.fleet.oracle and step.role == ModelRole.REASONER)
        return await self._json_call(
            ModelRole.CRITIC,
            critic_prompt(asdict(step), evidence),
            CRITIC_SCHEMA,
            prefer_oracle=prefer_oracle,
        )

    async def _run_step(self, objective: str, step: PlanStep) -> dict[str, Any]:
        step.status = "running"
        self._emit("step.started", step.role.value, step=asdict(step))
        evidence: list[dict[str, Any]] = []
        descriptions = self._tool_descriptions_for(step)
        allowed_names = {tool["name"] for tool in descriptions}

        for turn_number in range(1, self.config.runtime.max_worker_turns + 1):
            value = await self._json_call(
                step.role,
                worker_prompt(objective, asdict(step), descriptions, evidence),
                WORKER_SCHEMA,
                prefer_oracle=step.role == ModelRole.ORACLE,
            )
            turn = WorkerTurn.from_dict(value)
            self._emit("worker.turn", step.role.value, step_id=step.id, turn=turn_number, message=turn.message)
            evidence.append({"type": "worker_message", "message": turn.message})

            for action in turn.actions:
                if action.tool not in allowed_names:
                    result = {"ok": False, "error": f"tool is outside this step's manifest: {action.tool}"}
                    evidence.append({"action": asdict(action), "result": result})
                    self._emit("tool.denied", "policy", step_id=step.id, tool=action.tool, reason=result["error"])
                    continue
                try:
                    request = self.tools.prepare(action)
                    token, reason = self.tools.authorize(request)
                    if not token:
                        result = {"ok": False, "error": reason}
                        self._emit("tool.denied", "policy", step_id=step.id, tool=action.tool, reason=reason)
                    else:
                        self._emit("tool.authorized", "policy", step_id=step.id, tool=action.tool, risk=request.risk.name.lower())
                        output = await asyncio.to_thread(self.tools.execute, request, token)
                        result = {"ok": True, "output": output}
                        self._emit("tool.completed", action.tool, step_id=step.id, result=output)
                except Exception as exc:
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    self._emit("tool.failed", action.tool, step_id=step.id, error=result["error"])
                evidence.append({"action": asdict(action), "result": result})

            if turn.done and not turn.actions:
                break

        try:
            verdict = await self._review(step, evidence)
        except (BackendError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            verdict = {"passed": False, "findings": [f"critic unavailable: {exc}"], "next_action": "Run validation manually."}
        step.status = "completed" if verdict.get("passed") else "needs_review"
        self._emit("step.finished", "critic", step=asdict(step), verdict=verdict)
        return {"step": asdict(step), "verdict": verdict, "evidence": evidence}

    async def run(self, objective: str) -> dict[str, Any]:
        plan = await self.plan(objective)
        results: list[dict[str, Any]] = []
        completed: set[str] = set()
        pending = list(plan.steps)
        while pending:
            ready = [step for step in pending if set(step.depends_on).issubset(completed)]
            if not ready:
                self._emit("run.blocked", "router", reason="plan contains a dependency cycle")
                break
            parallel_limit = min(
                self.config.runtime.max_parallel_workers,
                self.profile.max_parallel_workers,
            )
            parallel = [
                step
                for step in ready
                if step.allowed_tools and set(step.allowed_tools).issubset(READ_ONLY_TOOLS)
            ][:parallel_limit]
            batch = parallel if len(parallel) > 1 else ready[:1]
            if len(batch) > 1:
                self._emit("swarm.parallel", "router", step_ids=[step.id for step in batch])
            batch_results = await asyncio.gather(*(self._run_step(objective, step) for step in batch))
            for step, result in zip(batch, batch_results, strict=True):
                results.append(result)
                completed.add(step.id)
                pending.remove(step)
                if step.status != "completed":
                    pending.clear()
                    break
        passed = bool(results) and all(item["verdict"].get("passed") for item in results)
        summary = {
            "objective": objective,
            "passed": passed,
            "completed_steps": len(completed),
            "total_steps": len(plan.steps),
            "results": results,
        }
        self._emit("run.finished", "ironclad", summary={k: v for k, v in summary.items() if k != "results"})
        return summary

    def close(self) -> None:
        self.ledger.close()
