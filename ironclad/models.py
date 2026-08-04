from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import ModelConfig
from .providers import OllamaBackend
from .types import HardwareProfile, ModelRole, ModelSpec

PARAMETERS = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


def parameter_count(name: str) -> float:
    matches = [float(value) for value in PARAMETERS.findall(name)]
    return matches[-1] if matches else 0.0


@dataclass(slots=True)
class ModelFleet:
    router: ModelSpec
    coder: ModelSpec
    reasoner: ModelSpec
    critic: ModelSpec
    oracle: ModelSpec | None = None

    def for_role(self, role: ModelRole) -> ModelSpec:
        if role == ModelRole.ORACLE and self.oracle:
            return self.oracle
        if role == ModelRole.ROUTER:
            return self.router
        if role == ModelRole.REASONER:
            return self.reasoner
        if role == ModelRole.CRITIC:
            return self.critic
        return self.coder

    def as_rows(self) -> list[tuple[str, str, str]]:
        rows = [
            ("router", self.router.name, self.router.provider),
            ("coder", self.coder.name, self.coder.provider),
            ("reasoner", self.reasoner.name, self.reasoner.provider),
            ("critic", self.critic.name, self.critic.provider),
        ]
        if self.oracle:
            rows.append(("oracle", self.oracle.name, self.oracle.provider))
        return rows


class ModelScheduler:
    def __init__(self, config: ModelConfig, profile: HardwareProfile, ollama: OllamaBackend):
        self.config = config
        self.profile = profile
        self.ollama = ollama

    @staticmethod
    def _name(item: dict[str, Any]) -> str:
        return str(item.get("model") or item.get("name") or "")

    def _score(self, name: str, role: ModelRole) -> float:
        lower = name.lower()
        params = parameter_count(lower)
        router_min, router_max = self.profile.router_parameters_b
        worker_min, worker_max = self.profile.worker_parameters_b
        if any(marker in lower for marker in ("embed", "rerank", "vision-only")):
            return -10_000
        score = 0.0
        if role in {ModelRole.ROUTER, ModelRole.CRITIC, ModelRole.SUMMARIZER}:
            target = (router_min + router_max) / 2
            score -= abs((params or target) - target) * 3
            score += 6 if any(x in lower for x in ("qwen", "nemotron", "gpt-oss")) else 0
        else:
            target = worker_max
            score -= abs((params or target) - target)
            score += 10 if any(x in lower for x in ("coder", "code", "devstral")) else 0
            score += 5 if any(x in lower for x in ("qwen", "gpt-oss", "deepseek")) else 0
        score += 4 if any(x in lower for x in ("instruct", "tools", "agent")) else 0
        score += 2 if any(x in lower for x in ("q4", "q5", "gguf")) else 0
        return score

    def _select(self, items: list[dict[str, Any]], role: ModelRole, configured: str) -> str:
        names = [self._name(item) for item in items if self._name(item)]
        if configured != "auto":
            for name in names:
                if name == configured or name.startswith(configured):
                    return name
            return configured
        if not names:
            raise RuntimeError("No local Ollama models were discovered")
        return max(names, key=lambda name: self._score(name, role))

    async def discover(self) -> ModelFleet:
        items = await self.ollama.list_models()
        router = self._select(items, ModelRole.ROUTER, self.config.router)
        coder = self._select(items, ModelRole.CODER, self.config.coder)
        reasoner = self._select(items, ModelRole.REASONER, self.config.reasoner)
        critic = self._select(items, ModelRole.CRITIC, self.config.critic)
        oracle = None
        if self.config.cloud_oracle_enabled:
            oracle = ModelSpec(
                self.config.cloud_model,
                ModelRole.ORACLE,
                provider="openai",
                thinking=True,
            )
        return ModelFleet(
            router=ModelSpec(router, ModelRole.ROUTER, keep_alive=self.config.router_keep_alive),
            coder=ModelSpec(coder, ModelRole.CODER, keep_alive=self.config.worker_keep_alive, thinking=self.config.local_thinking),
            reasoner=ModelSpec(reasoner, ModelRole.REASONER, keep_alive=self.config.worker_keep_alive, thinking=self.config.local_thinking),
            critic=ModelSpec(critic, ModelRole.CRITIC, keep_alive=self.config.worker_keep_alive),
            oracle=oracle,
        )
