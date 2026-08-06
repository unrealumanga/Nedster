from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx


class BackendError(RuntimeError):
    pass


def _json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start, end = stripped.find("{"), stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise BackendError("model returned JSON that is not an object")
    return value


class OllamaBackend:
    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.is_success
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return list(response.json().get("models") or [])

    async def complete(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        thinking: bool | str = False,
        keep_alive: str = "90s",
        num_predict: int | None = None,
        temperature: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": thinking,
            "keep_alive": keep_alive,
        }
        options: dict[str, Any] = {}
        if num_predict is not None:
            options["num_predict"] = num_predict
        if temperature is not None:
            options["temperature"] = temperature
        if options:
            payload["options"] = options
        if schema:
            payload["format"] = schema
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                message = response.json().get("message") or {}
                return str(message.get("content") or "")
        except httpx.HTTPError as exc:
            raise BackendError(f"Ollama request failed: {exc}") from exc

    async def complete_json(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        thinking: bool | str = False,
        keep_alive: str = "90s",
        num_predict: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        # Ollama's generate endpoint reliably applies JSON Schema to Qwen 3.5.
        # The chat endpoint may return prose even when the same schema is sent.
        payload: dict[str, Any] = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "format": schema,
            "stream": False,
            "think": thinking,
            "keep_alive": keep_alive,
        }
        options: dict[str, Any] = {}
        if num_predict is not None:
            options["num_predict"] = num_predict
        if temperature is not None:
            options["temperature"] = temperature
        if options:
            payload["options"] = options
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                return _json_from_text(str(response.json().get("response") or ""))
        except httpx.HTTPError as exc:
            raise BackendError(f"Ollama request failed: {exc}") from exc

    async def stream(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        thinking: bool | str = False,
        keep_alive: str = "90s",
    ) -> AsyncIterator[tuple[str, str]]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "think": thinking,
            "keep_alive": keep_alive,
        }
        async with (
            httpx.AsyncClient(timeout=None) as client,
            client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line).get("message") or {}
                if chunk.get("thinking"):
                    yield "thinking", str(chunk["thinking"])
                if chunk.get("content"):
                    yield "content", str(chunk["content"])


class OpenAIResponsesBackend:
    """Optional quality-first oracle. Nedster remains local-only unless explicitly enabled."""

    def __init__(self, api_key: str | None = None, timeout: float = 600.0):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    async def available(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _output_text(value: dict[str, Any]) -> str:
        if value.get("output_text"):
            return str(value["output_text"])
        fragments: list[str] = []
        for item in value.get("output") or []:
            if item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    fragments.append(str(content["text"]))
        return "".join(fragments)

    async def complete(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        reasoning_effort: str = "high",
        reasoning_mode: str = "standard",
        reasoning_context: str = "current_turn",
        verbosity: str = "medium",
        num_predict: int | None = None,
        **_: Any,
    ) -> str:
        if not self.api_key:
            raise BackendError("OPENAI_API_KEY is not configured")
        reasoning: dict[str, Any] = {
            "effort": reasoning_effort,
            "context": reasoning_context,
        }
        if reasoning_mode == "pro":
            reasoning["mode"] = "pro"
        text_config: dict[str, Any] = {"verbosity": verbosity}
        if schema:
            text_config["format"] = {
                "type": "json_schema",
                "name": "nedster_contract",
                "schema": schema,
                "strict": True,
            }
        payload = {
            "model": model,
            "instructions": system,
            "input": prompt,
            "reasoning": reasoning,
            "text": text_config,
            "store": False,
        }
        if num_predict is not None:
            payload["max_output_tokens"] = num_predict
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/responses", headers=headers, json=payload)
                response.raise_for_status()
                return self._output_text(response.json())
        except httpx.HTTPError as exc:
            raise BackendError(f"OpenAI Responses request failed: {exc}") from exc

    async def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        return _json_from_text(await self.complete(**kwargs))
