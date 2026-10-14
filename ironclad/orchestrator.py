from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from .claim_gate import screen_unverified
from .config import NedsterConfig
from .grounding import GroundingHarness, GroundingSource
from .hardware import choose_profile, probe_hardware
from .ledger import EventLedger, load_or_create_ledger_key
from .models import ModelFleet, ModelScheduler
from .prompts import (
    ASK_SYSTEM,
    CHAT_SYSTEM,
    CRITIC_SCHEMA,
    NEDSTER_SYSTEM,
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
from .types import ModelRole, PlanStep, TaskPlan, ToolAction, WorkerTurn

EventCallback = Callable[[dict[str, Any]], None]
READ_ONLY_TOOLS = frozenset(
    {
        "read_file",
        "list_files",
        "search_text",
        "project_search",
        "git_diff",
        "web_search",
        "web_fetch",
        "market_price",
    }
)
SINGLE_EVIDENCE_TOOLS = frozenset(
    {"git_diff", "list_files", "market_price", "project_search", "web_fetch"}
)

CASUAL_INPUTS = frozenset(
    {
        "hello",
        "hi",
        "hi there",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "how's it going",
        "thanks",
        "thank you",
        "who are you",
    }
)
LOCAL_TARGET = re.compile(
    r"(?:^|\s)(?:this\s+)?(?:project|repo(?:sitory)?|workspace|codebase|folder|directory|files?|logs?|"
    r"git|diff|changes?|working\s+tree)\b|[~/]|\.[A-Za-z0-9]{1,8}\b",
    re.IGNORECASE,
)
AGENT_ACTION = re.compile(
    r"^(?:(?:please|kindly)\s+|(?:can|could|would|will)\s+you\s+|"
    r"(?:let(?:'s|s)|let\s+us)\s+|(?:i\s+(?:want|need)\s+(?:you\s+)?to)\s+|"
    r"help\s+me\s+)?"
    r"(?:build|calculate|change|compile|complete|convert|create|debug|deploy|download|draft|edit|"
    r"export|fetch|finish|fix|generate|implement|import|install|migrate|organize|package|patch|"
    r"refactor|remove|rename|render|run|test|update|write)\b",
    re.IGNORECASE,
)
MUTATION_TOOLS = frozenset({"replace_text", "write_file"})
VALIDATION_TOOLS = frozenset({"run_command", "run_tests"})
STATE_SENSITIVE_TOOLS = frozenset(
    {"git_diff", "list_files", "project_search", "read_file", "run_command", "run_tests", "search_text"}
)
SOFTWARE_VALIDATION = re.compile(
    r"\b(?:api|app(?:lication)?|build|calculator|cli|code|compile|crate|debug|function|"
    r"implement|library|parser|program|rust|test)\b|\.(?:c|cc|cpp|go|java|js|jsx|py|rs|ts|tsx)\b",
    re.IGNORECASE,
)
USER_FACING_SOFTWARE = re.compile(
    r"\b(?:app(?:lication)?|cli|command[- ]line|program|server|service|tool)\b|"
    r"\bcalculator\b(?!\.[A-Za-z0-9])",
    re.IGNORECASE,
)
ARGUMENT_DRIVEN_SOFTWARE = re.compile(
    r"\b(?:cli|command[- ]line)\b|\bcalculator\b(?!\.[A-Za-z0-9])", re.IGNORECASE
)
CALCULATOR_PRODUCT = re.compile(r"\bcalculator\b(?!\.[A-Za-z0-9])", re.IGNORECASE)
EXPECTED_OUTPUT = re.compile(
    r"\b(?:output|result)\s+(?:is|equals?|should\s+be)\s+[`'\"]?([^\s,;.!?`'\"]+)",
    re.IGNORECASE,
)
LOCAL_INSPECTION = re.compile(
    r"\b(?:analy[sz]e|audit|check|compare|count|evaluate|explore|find|findout|how\s+many|inspect|list|"
    r"locate|look\s+for|rank|read|scan|search|show|where)\b",
    re.IGNORECASE,
)
EXPLICIT_FILE = re.compile(r"(?<![\w.-])([\w.-]+\.[A-Za-z0-9]{1,8})(?![\w.-])")
LOCAL_PATH = re.compile(
    r"(?<!\w)((?:[A-Za-z]:[\\/]|~[\\/]|/)(?:[^\s,;!?]+))"
)
FOLLOWUP_CRITERIA = re.compile(
    r"^(?:and\s+|also\s+)?(?:best\s+is|by\s+|based\s+on|continue|go\s+on|keep\s+going|"
    r"how\s+many|what\s+about|how\s+about|look\s+at|rank\s+by|the\s+one|those|them|"
    r"of\s+them|use\s+|with\s+)",
    re.IGNORECASE,
)
MODEL_QUESTION = re.compile(
    r"\b(?:(?:which|what)\s+(?:local\s+)?(?:llm|model)|(?:llm|model)\s+(?:are\s+you|you(?:'re|\s+are)?\s+using))\b",
    re.IGNORECASE,
)
ACCESS_QUESTION = re.compile(
    r"\b(?:what\s+can\s+you\s+do|(?:how\s+(?:do|can)\s+i\s+)?(?:give|grant|enable)\s+you\s+(?:local\s+)?access|"
    r"(?:can|do)\s+you\s+(?:access|read|write|search).*(?:files?|folders?|file\s*system)|"
    r"(?:local\s+)?(?:file\s*system|files?|folders?)\s+access)\b",
    re.IGNORECASE,
)
FRESHNESS_QUERY = re.compile(
    r"\b(?:current|currently|latest|live|now|real[ -]?time|right now|today(?:'s)?|"
    r"market price|exchange rate|weather|forecast|news|score|standings)\b",
    re.IGNORECASE,
)
WEB_QUERY = re.compile(
    r"\b(?:browse|fetch|internet|online|search the web|web search|website)\b|https://",
    re.IGNORECASE,
)
NEWS_QUERY = re.compile(r"\b(?:breaking\s+news|global\s+news|news|world\s+headlines?)\b", re.IGNORECASE)
NEWS_COUNT = re.compile(r"\btop\s+(\d{1,2})\b", re.IGNORECASE)
MARKET_PAIR = re.compile(r"\b([A-Z0-9]{2,12})[\s/_-]?(USDT|USDC|USD|EUR)\b", re.IGNORECASE)
MARKET_ASSET_ALIASES = {
    "ada": "ADA",
    "avalanche": "AVAX",
    "avax": "AVAX",
    "bitcoin": "BTC",
    "bnb": "BNB",
    "btc": "BTC",
    "cardano": "ADA",
    "doge": "DOGE",
    "dogecoin": "DOGE",
    "eth": "ETH",
    "ether": "ETH",
    "ethereum": "ETH",
    "link": "LINK",
    "litecoin": "LTC",
    "ltc": "LTC",
    "sol": "SOL",
    "solana": "SOL",
    "xrp": "XRP",
}
MARKET_ASSET = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, MARKET_ASSET_ALIASES), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
MARKET_QUOTE = re.compile(r"\b(?:in|to|against|quoted\s+in)\s+(USDT|USDC|USD|EUR)\b", re.IGNORECASE)
GIT_INSPECTION = re.compile(
    r"\b(?:git\s+diff|working\s+tree|uncommitted\s+changes?|changed\s+files?|current\s+diff)\b",
    re.IGNORECASE,
)
FILE_COUNT_QUERY = re.compile(
    r"\b(?:count|how\s+many)\b.*\b(?:files?|documents?|logs?)\b|"
    r"\b(?:files?|documents?|logs?)\b.*\b(?:count|how\s+many)\b",
    re.IGNORECASE,
)
EXPLICIT_URL = re.compile(r"https://[^\s,;!?]+", re.IGNORECASE)
TOOL_ESCALATION = re.compile(
    r"^\s*NEED_TOOLS\b|^\s*(?:I\s+(?:do\s+not|don't|cannot|can't|am\s+unable\s+to)|"
    r"I\s+have\s+no\s+access|As\s+an\s+AI[^.]{0,100}\b(?:cannot|can't|unable))\b",
    re.IGNORECASE,
)


def classify_interaction(value: str) -> str:
    normalized = " ".join(value.casefold().strip().rstrip("!?.,").split())
    if normalized in CASUAL_INPUTS:
        return "chat"
    if (
        AGENT_ACTION.search(value)
        or (LOCAL_TARGET.search(value) and LOCAL_INSPECTION.search(value))
        or FRESHNESS_QUERY.search(value)
        or WEB_QUERY.search(value)
        or re.search(r"\b(?:please\s+)?do\s+(?:this|that|it)\b", value, re.IGNORECASE)
    ):
        return "agent"
    return "ask"


def _market_symbol(value: str) -> str | None:
    asset = MARKET_ASSET.search(value)
    pair = MARKET_PAIR.search(value)
    if asset:
        normalized_base = MARKET_ASSET_ALIASES[asset.group(1).casefold()]
        if pair and MARKET_ASSET_ALIASES.get(pair.group(1).casefold()) == normalized_base:
            return f"{normalized_base}{pair.group(2).upper()}"
        quote = MARKET_QUOTE.search(value)
        return f"{normalized_base}{quote.group(1).upper() if quote else 'USDT'}"
    if pair and pair.group(1).casefold() not in {"in", "is", "of", "to"}:
        base, quote = pair.groups()
        normalized_base = MARKET_ASSET_ALIASES.get(base.casefold(), base.upper())
        return f"{normalized_base}{quote.upper()}"
    return None


def _result(
    objective: str,
    answer: str,
    *,
    mode: str = "ask",
    passed: bool = True,
    grounded: bool = True,
) -> dict[str, Any]:
    return {
        "objective": objective,
        "answer": answer,
        "passed": passed,
        "mode": mode,
        "grounded": grounded,
        "completed_steps": 0,
        "total_steps": 0,
        "results": [],
    }


def _normalize_citations(value: str) -> str:
    normalized = re.sub(
        r"([.!?])\s*\n+\s*((?:\[S\d+]\s*)+)",
        lambda match: f" {match.group(2).strip()}{match.group(1)}",
        value.strip(),
    )
    return re.sub(
        r"([.!?])((?:\[S\d+])+)",
        lambda match: f" {match.group(2)}{match.group(1)}",
        normalized,
    )


def _attach_single_source_citations(value: str, source_id: str) -> str:
    parts = re.split(r"((?<=[.!?])\s+|\n+)", value.strip())
    repaired: list[str] = []
    for part in parts:
        sentence = part.strip()
        if (
            sentence
            and not re.fullmatch(r"\s+", part)
            and len(sentence.split()) >= 5
            and not re.search(r"\[S\d+]", sentence)
            and not sentence.startswith(("Sources:", "INSUFFICIENT_CONTEXT"))
        ):
            if re.search(r"[.!?]$", sentence):
                sentence = re.sub(r"([.!?])$", rf" [{source_id}]\1", sentence)
            else:
                sentence += f" [{source_id}]"
            part = part.replace(part.strip(), sentence, 1)
        repaired.append(part)
    return _normalize_citations("".join(repaired))


class NedsterRuntime:
    def __init__(self, config: NedsterConfig, on_event: EventCallback | None = None):
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
        self.grounding = GroundingHarness(max_sources=config.runtime.max_grounding_sources)
        self.conversation: list[dict[str, str]] = []
        self.active_objective: str | None = None
        self.last_mode: str | None = None
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
            "nedster",
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
        constrained_prompt = (
            f"{prompt}\n\n"
            "Return ONLY one JSON object matching this schema. Do not add markdown or prose outside JSON.\n"
            f"{json.dumps(schema, separators=(',', ':'))}"
        )
        if spec.provider == "openai":
            return await self.openai.complete_json(
                model=spec.name,
                system=NEDSTER_SYSTEM,
                prompt=constrained_prompt,
                schema=schema,
                reasoning_effort=self.config.models.cloud_reasoning_effort,
                reasoning_mode=self.config.models.cloud_reasoning_mode,
                reasoning_context=self.config.models.cloud_reasoning_context,
            )
        return await self.ollama.complete_json(
            model=spec.name,
            system=NEDSTER_SYSTEM,
            prompt=constrained_prompt,
            schema=schema,
            # JSON-schema decoding is the constrained pass. Keeping thinking
            # outside it avoids Qwen placing valid JSON in the thinking field.
            thinking=False,
            keep_alive=spec.keep_alive,
            temperature=0.2,
        )

    async def _text_call(self, role: ModelRole, system: str, prompt: str, *, max_tokens: int) -> str:
        if not self.fleet:
            await self.initialize()
        assert self.fleet is not None
        spec = self.fleet.for_role(role)
        if spec.provider == "openai":
            return await self.openai.complete(
                model=spec.name,
                system=system,
                prompt=prompt,
                reasoning_effort="low",
                verbosity="low",
                num_predict=max_tokens,
            )
        return await self.ollama.complete(
            model=spec.name,
            system=system,
            prompt=prompt,
            thinking=False,
            keep_alive=spec.keep_alive,
            num_predict=max_tokens,
        )

    def _conversation_prompt(self, objective: str, *, include_history: bool = False) -> str:
        if not include_history:
            return objective
        budget = self.config.runtime.conversation_history_chars
        history: list[str] = []
        used = 0
        for message in reversed(self.conversation):
            rendered = f"{message['role'].title()}: {message['content']}"
            if used + len(rendered) > budget:
                break
            history.append(rendered)
            used += len(rendered)
        history.reverse()
        if not history:
            return objective
        return "Recent conversation:\n" + "\n".join(history) + f"\n\nUser: {objective}\nAssistant:"

    def _remember(self, user: str, assistant: str) -> None:
        self.conversation.extend(
            (
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            )
        )
        while (
            sum(len(item["content"]) for item in self.conversation)
            > self.config.runtime.conversation_history_chars
        ):
            self.conversation.pop(0)

    def _access_answer(self) -> str:
        root = self.config.runtime.project_root
        displayed_root = root.as_posix()
        suggested_root = (
            root.parent.as_posix()
            if root.parent.name.casefold() == "workspace"
            else "/path/to/folder"
        )
        return (
            f"I can read and search the active project or an explicitly named local folder. Edits and "
            f"approved commands stay inside the active project: `{displayed_root}`. To make another folder the "
            "full project, start me with "
            f"`nedster --project {suggested_root}`. "
            "Use `/permissions` to inspect the current write and command policy."
        )

    def _requested_inspection_root(self, objective: str) -> tuple[Path | None, str | None]:
        if not LOCAL_INSPECTION.search(objective):
            return None, None
        match = LOCAL_PATH.search(objective)
        if not match:
            return None, None
        raw = match.group(1).rstrip("'\"`.)]}")
        requested = Path(raw).expanduser()
        current = self.tools.project_root.resolve()
        if not requested.exists() and requested.name.casefold() == current.parent.name.casefold():
            requested = current.parent
        if not requested.exists():
            return None, f"I can inspect local files, but `{raw}` does not exist. Active project: `{current}`."
        if not requested.is_dir():
            return None, None
        resolved = requested.resolve()
        return (None, None) if resolved == current else (resolved, None)

    async def _models_answer(self) -> str:
        if not self.fleet:
            await self.initialize()
        assert self.fleet is not None
        grouped: dict[tuple[str, str], list[str]] = {}
        for role, name, provider in self.fleet.as_rows():
            grouped.setdefault((name, provider), []).append(role)
        models = "; ".join(
            f"{', '.join(roles)}: `{name}` via {provider}"
            for (name, provider), roles in grouped.items()
        )
        oracle = "enabled" if self.fleet.oracle else "disabled"
        return f"Active models — {models}. Cloud oracle: {oracle}."

    @staticmethod
    def _display_price(raw: str) -> str:
        value = format(Decimal(raw), "f")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        whole, separator, fraction = value.partition(".")
        return f"{int(whole):,}{separator}{fraction}"

    async def _market_response(self, objective: str, symbol: str) -> dict[str, Any] | None:
        action = ToolAction(
            tool="market_price",
            arguments={"symbol": symbol},
            reason="Retrieve and cross-check the current public market quote.",
        )
        try:
            request = self.tools.prepare(action)
            token, reason = self.tools.authorize(request)
            if not token:
                raise PermissionError(reason)
            self._emit("response.started", "user", mode="agent")
            self._emit("tool.authorized", "policy", tool=action.tool, risk="read")
            output = await asyncio.to_thread(self.tools.execute, request, token)
            self._emit("tool.completed", action.tool, result=output)
        except Exception as exc:
            self._emit("tool.failed", action.tool, error=f"{type(exc).__name__}: {exc}")
            return None

        quotes = list(output.get("quotes") or [])
        if not quotes:
            return None
        rendered_quotes = "; ".join(
            f"{item['provider']} {self._display_price(str(item['price']))} {item['quote_currency']}"
            for item in quotes
        )
        checked = str(output.get("observed_at") or "").replace("T", " ").replace("+00:00", " UTC")
        spread = str(output.get("spread_percent") or "0")
        caution = ""
        passed = True
        if Decimal(spread) > Decimal("1"):
            caution = f" The sources differ by {spread}%, so treat the quote as unsettled."
            passed = False
        answer = f"{output['symbol']}: {rendered_quotes}. Checked {checked}.{caution}"
        self._remember(objective, answer)
        self.last_mode = "agent"
        result = {
            "objective": objective,
            "answer": answer,
            "passed": passed,
            "mode": "agent",
            "grounded": True,
            "completed_steps": 1,
            "total_steps": 1,
            "results": [
                {
                    "step": {"id": "market-price", "title": f"Get {output['symbol']} quote"},
                    "verdict": {"passed": passed, "findings": [], "next_action": "", "answer": answer},
                    "answer": answer,
                    "sources": [
                        f"{item['provider']}: {item['source_url']}" for item in quotes
                    ],
                    "evidence": [
                        {"action": asdict(action), "result": {"ok": True, "output": output}}
                    ],
                }
            ],
        }
        self._emit("response.finished", "nedster", mode="agent")
        return result

    async def _news_response(self, objective: str) -> dict[str, Any]:
        count_match = NEWS_COUNT.search(objective)
        count = min(max(int(count_match.group(1)) if count_match else 5, 1), 10)
        query = "latest world news headlines today"
        action = ToolAction(
            tool="web_search",
            arguments={"query": query, "limit": count},
            reason="Retrieve current world-news headlines from public sources.",
        )
        self._emit("response.started", "user", mode="agent")
        try:
            request = self.tools.prepare(action)
            token, reason = self.tools.authorize(request)
            if not token:
                raise PermissionError(reason)
            self._emit("tool.authorized", "policy", tool=action.tool, risk="read")
            output = await asyncio.to_thread(self.tools.execute, request, token)
            self._emit("tool.completed", action.tool, result=output)
        except Exception as exc:
            self._emit("tool.failed", action.tool, error=f"{type(exc).__name__}: {exc}")
            answer = "I could not retrieve current news from the configured public search providers."
            self._remember(objective, answer)
            self.last_mode = "agent"
            result = _result(objective, answer, mode="agent", passed=False, grounded=True)
            self._emit("response.finished", "nedster", mode="agent")
            return result

        headlines = list(output.get("results") or [])[:count]
        if not headlines:
            answer = "I could not retrieve any current news results from the configured providers."
            passed = False
        else:
            retrieved = str(output.get("retrieved_at") or "unknown time").replace("T", " ")
            lines = [f"Latest world-news results (retrieved {retrieved}):"]
            for index, item in enumerate(headlines, 1):
                title = " ".join(str(item.get("title") or "Untitled result").split())
                url = str(item.get("url") or "")
                snippet = " ".join(str(item.get("snippet") or "").split())
                lines.append(f"{index}. {title}\n   {snippet}\n   {url}".rstrip())
            answer = "\n\n".join(lines)
            passed = len(headlines) == count
        self._remember(objective, answer)
        self.last_mode = "agent"
        result = {
            "objective": objective,
            "answer": answer,
            "passed": passed,
            "mode": "agent",
            "grounded": True,
            "completed_steps": 1,
            "total_steps": 1,
            "results": [
                {
                    "step": {"id": "current-news", "title": "Retrieve current world news"},
                    "verdict": {
                        "passed": passed,
                        "findings": [],
                        "next_action": "" if passed else "Try another public news provider.",
                        "answer": answer,
                    },
                    "answer": answer,
                    "sources": [str(item.get("url") or "") for item in headlines],
                    "evidence": [
                        {"action": asdict(action), "result": {"ok": True, "output": output}}
                    ],
                }
            ],
        }
        self._emit("response.finished", "nedster", mode="agent")
        return result

    def _continued_objective(self, objective: str, mode: str) -> str:
        active = getattr(self, "active_objective", None)
        if active and mode != "agent" and FOLLOWUP_CRITERIA.search(objective.strip()):
            return f"{active}\n\nAdditional user criterion: {objective}"
        return objective

    @staticmethod
    def _needs_tool_escalation(answer: str) -> bool:
        return not answer.strip() or bool(TOOL_ESCALATION.search(answer.strip()))

    async def _execute_agent(self, objective: str, user_objective: str) -> dict[str, Any]:
        self.active_objective = objective
        inspection_root, scope_error = self._requested_inspection_root(objective)
        if scope_error:
            self._remember(user_objective, scope_error)
            self.last_mode = "agent"
            return _result(user_objective, scope_error, mode="agent", passed=False)
        scoped_objective = objective
        original_root: Path | None = None
        if inspection_root:
            original_root = self.tools.project_root
            self.tools.project_root = inspection_root
            scoped_objective += (
                f"\n\nRuntime inspection root: {inspection_root}. Use paths relative to this root."
            )
        try:
            result = await self.run(scoped_objective)
        finally:
            if original_root is not None:
                self.tools.project_root = original_root
        self._remember(user_objective, str(result.get("answer") or ""))
        self.last_mode = "agent"
        return result

    async def respond(self, objective: str) -> dict[str, Any]:
        if ACCESS_QUESTION.search(objective):
            answer = self._access_answer()
            self._remember(objective, answer)
            self.last_mode = "ask"
            return _result(objective, answer)
        if MODEL_QUESTION.search(objective):
            answer = await self._models_answer()
            self._remember(objective, answer)
            self.last_mode = "ask"
            return _result(objective, answer)

        market_symbol = _market_symbol(objective)
        if market_symbol and FRESHNESS_QUERY.search(objective):
            market_result = await self._market_response(objective, market_symbol)
            if market_result:
                return market_result
        if NEWS_QUERY.search(objective) and FRESHNESS_QUERY.search(objective):
            return await self._news_response(objective)

        mode = classify_interaction(objective)
        continued = self._continued_objective(objective, mode)
        if continued != objective:
            mode = "agent"
        if mode == "agent":
            return await self._execute_agent(continued, objective)

        self._emit("response.started", "user", mode=mode)
        role = ModelRole.ROUTER if mode == "chat" else ModelRole.CODER
        system = CHAT_SYSTEM if mode == "chat" else ASK_SYSTEM
        max_tokens = (
            self.config.runtime.chat_max_tokens if mode == "chat" else self.config.runtime.answer_max_tokens
        )
        answer = (
            await self._text_call(role, system, self._conversation_prompt(objective), max_tokens=max_tokens)
        ).strip()
        if mode == "ask" and self._needs_tool_escalation(answer):
            self._emit("response.escalated", "nedster", reason=answer[:240] or "empty answer")
            return await self._execute_agent(objective, objective)
        screened = screen_unverified(answer)
        if screened.claims_unverified_work:
            if mode == "ask":
                self._emit("response.escalated", "nedster", reason="unverified work claim")
                return await self._execute_agent(objective, objective)
            answer = screened.without_work_claims
        if not answer:
            raise BackendError("the local model returned an empty answer")
        self._remember(objective, answer)
        self.last_mode = mode
        result = _result(objective, answer, mode=mode, grounded=False)
        self._emit("response.finished", "nedster", mode=mode)
        return result

    async def plan(self, objective: str) -> TaskPlan:
        self._emit("objective.received", "user", objective=objective)
        explicit_file = EXPLICIT_FILE.search(objective)
        if explicit_file and LOCAL_INSPECTION.search(objective):
            path = explicit_file.group(1)
            plan = TaskPlan(
                objective=objective,
                steps=[
                    PlanStep(
                        id="inspect-file",
                        title=f"Inspect {path}",
                        objective=objective,
                        role=ModelRole.REASONER,
                        success_criteria=[f"Answer is grounded in the retrieved contents of {path}."],
                        allowed_tools=["read_file"],
                    )
                ],
            )
            self._emit("plan.created", "router", plan=asdict(plan), deterministic=True)
            return plan
        if GIT_INSPECTION.search(objective):
            plan = TaskPlan(
                objective=objective,
                steps=[
                    PlanStep(
                        id="inspect-git",
                        title="Inspect current project changes",
                        objective=objective,
                        role=ModelRole.REASONER,
                        success_criteria=["The summary is grounded in the current git diff."],
                        allowed_tools=["git_diff"],
                    )
                ],
            )
            self._emit("plan.created", "router", plan=asdict(plan), deterministic=True)
            return plan
        if FILE_COUNT_QUERY.search(objective) and LOCAL_TARGET.search(objective):
            plan = TaskPlan(
                objective=objective,
                steps=[
                    PlanStep(
                        id="count-project-files",
                        title="Count matching project files",
                        objective=objective,
                        role=ModelRole.REASONER,
                        success_criteria=[
                            "The answer reports the deterministic total_matches value for the requested file pattern."
                        ],
                        allowed_tools=["list_files"],
                    )
                ],
            )
            self._emit("plan.created", "router", plan=asdict(plan), deterministic=True)
            return plan
        if LOCAL_TARGET.search(objective) and LOCAL_INSPECTION.search(objective):
            plan = TaskPlan(
                objective=objective,
                steps=[
                    PlanStep(
                        id="inspect-project",
                        title="Inspect and compare project evidence",
                        objective=objective,
                        role=ModelRole.REASONER,
                        success_criteria=[
                            "The answer is based on relevant files or logs retrieved from the active project."
                        ],
                        allowed_tools=[
                            "project_search",
                            "list_files",
                            "search_text",
                            "read_file",
                            "git_diff",
                        ],
                    )
                ],
            )
            self._emit("plan.created", "router", plan=asdict(plan), deterministic=True)
            return plan
        explicit_url = EXPLICIT_URL.search(objective)
        if explicit_url:
            plan = TaskPlan(
                objective=objective,
                steps=[
                    PlanStep(
                        id="fetch-url",
                        title="Fetch the requested public source",
                        objective=objective,
                        role=ModelRole.REASONER,
                        success_criteria=[
                            "The answer is grounded in the explicitly requested URL and retrieval time."
                        ],
                        allowed_tools=["web_fetch"],
                    )
                ],
            )
            self._emit("plan.created", "router", plan=asdict(plan), deterministic=True)
            return plan
        if FRESHNESS_QUERY.search(objective) or WEB_QUERY.search(objective):
            plan = TaskPlan(
                objective=objective,
                steps=[
                    PlanStep(
                        id="web-research",
                        title="Research current public sources",
                        objective=objective,
                        role=ModelRole.REASONER,
                        success_criteria=[
                            "The answer cites current public evidence and includes its retrieval time."
                        ],
                        allowed_tools=["web_search", "web_fetch", "market_price"],
                    )
                ],
            )
            self._emit("plan.created", "router", plan=asdict(plan), deterministic=True)
            return plan
        if AGENT_ACTION.search(objective):
            plan = self._normalize_generated_plan(
                TaskPlan(objective=objective, steps=[]), objective
            )
            self._emit("plan.created", "router", plan=asdict(plan), deterministic=True)
            return plan
        try:
            value = await self._json_call(
                ModelRole.ROUTER,
                planner_prompt(objective, self.tools.descriptions()),
                PLANNER_SCHEMA,
            )
            plan = TaskPlan.from_dict(value, objective)
            plan = self._normalize_generated_plan(plan, objective)
        except (BackendError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            self._emit("planner.fallback", "router", error=str(exc))
            plan = TaskPlan.from_dict({}, objective)
            plan = self._normalize_generated_plan(plan, objective)
        self._emit("plan.created", "router", plan=asdict(plan))
        return plan

    def _normalize_generated_plan(self, plan: TaskPlan, objective: str) -> TaskPlan:
        if not AGENT_ACTION.search(objective):
            return plan
        local_agent_tools = [
            "read_file",
            "list_files",
            "replace_text",
            "write_file",
            "run_command",
            "run_tests",
        ]
        if re.search(r"\b(?:download|install|online|web|website)\b", objective, re.IGNORECASE):
            local_agent_tools.extend(("web_search", "web_fetch"))
        planned_tools = {
            tool for step in plan.steps for tool in step.allowed_tools if tool != "market_price"
        }
        allowed_tools = list(dict.fromkeys([*local_agent_tools, *sorted(planned_tools)]))
        success_criteria = [
            criterion for step in plan.steps for criterion in step.success_criteria
        ]
        success_criteria.append(
            "The requested outcome is completed and validated with real tool evidence."
        )
        if SOFTWARE_VALIDATION.search(objective):
            success_criteria.append(
                "Focused automated tests for the requested behavior pass after the latest edit."
            )
        if USER_FACING_SOFTWARE.search(objective):
            success_criteria.append(
                "The real entry point passes a representative expected-output assertion."
            )
        return TaskPlan(
            objective=objective,
            steps=[
                PlanStep(
                    id="complete-objective",
                    title="Complete and validate the objective",
                    objective=objective,
                    role=ModelRole.CODER,
                    success_criteria=list(dict.fromkeys(success_criteria)),
                    allowed_tools=allowed_tools,
                )
            ],
            assumptions=plan.assumptions,
            stop_conditions=plan.stop_conditions,
        )

    def _tool_descriptions_for(self, step: PlanStep) -> list[dict[str, Any]]:
        all_tools = self.tools.descriptions()
        if not step.allowed_tools:
            return all_tools
        allowed = set(step.allowed_tools)
        return [tool for tool in all_tools if tool["name"] in allowed]

    @staticmethod
    def _action_fingerprint(tool: str, arguments: dict[str, Any], mutation_epoch: int) -> str:
        value: dict[str, Any] = {"tool": tool, "arguments": arguments}
        if tool in STATE_SENSITIVE_TOOLS:
            value["mutation_epoch"] = mutation_epoch
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _mutation_epoch(evidence: list[dict[str, Any]]) -> int:
        return sum(
            1
            for item in evidence
            if (item.get("action") or {}).get("tool") in MUTATION_TOOLS
            and (item.get("result") or {}).get("ok")
            and ((item.get("result") or {}).get("output") or {}).get("changed")
        )

    @staticmethod
    def _has_meaningful_test_success(evidence: list[dict[str, Any]]) -> bool:
        last_mutation = max(
            (
                index
                for index, item in enumerate(evidence)
                if (item.get("action") or {}).get("tool") in MUTATION_TOOLS
                and (item.get("result") or {}).get("ok")
                and ((item.get("result") or {}).get("output") or {}).get("changed")
            ),
            default=-1,
        )
        for item in reversed(evidence[last_mutation + 1 :]):
            tool = (item.get("action") or {}).get("tool")
            if tool not in {"run_tests", "run_command"}:
                continue
            result = item.get("result") or {}
            output = result.get("output") or {}
            if not result.get("ok") or not isinstance(output, dict):
                continue
            if output.get("returncode") != 0 or output.get("timed_out"):
                continue
            tests_run = output.get("tests_run")
            if tool == "run_command" and tests_run is None:
                continue
            return tests_run is None or int(tests_run) > 0
        return False

    @staticmethod
    def _has_successful_smoke_check(
        evidence: list[dict[str, Any]], objective: str = ""
    ) -> bool:
        last_mutation = max(
            (
                index
                for index, item in enumerate(evidence)
                if (item.get("action") or {}).get("tool") in MUTATION_TOOLS
                and (item.get("result") or {}).get("ok")
                and ((item.get("result") or {}).get("output") or {}).get("changed")
            ),
            default=-1,
        )
        for item in evidence[last_mutation + 1 :]:
            action = item.get("action") or {}
            result = item.get("result") or {}
            output = result.get("output") or {}
            if (
                action.get("tool") != "run_command"
                or not result.get("ok")
                or not isinstance(output, dict)
                or output.get("returncode") != 0
                or output.get("timed_out")
            ):
                continue
            arguments = action.get("arguments") or {}
            command = list(
                map(str, output.get("command") or arguments.get("command") or [])
            )
            if not command:
                continue
            executable = Path(command[0]).name.casefold()
            subcommand = command[1].casefold() if len(command) > 1 else ""
            if (
                (executable == "cargo" and subcommand in {"build", "check", "clippy", "fmt", "test"})
                or (executable in {"go", "npm"} and subcommand == "test")
                or executable in {"pytest", "ruff"}
                or (executable.startswith("python") and "pytest" in command)
            ):
                continue
            expected_output = str(
                arguments.get("expect_stdout_contains")
                or arguments.get("expect_stderr_contains")
                or ""
            )
            if output.get("assertions_passed") is False:
                continue
            if output.get("assertions_passed") is not True and not expected_output:
                objective_match = EXPECTED_OUTPUT.search(objective)
                expected_output = objective_match.group(1) if objective_match else ""
            if not expected_output:
                continue
            actual_output = f"{output.get('stdout') or ''}\n{output.get('stderr') or ''}"
            if output.get("assertions_passed") is not True and expected_output not in actual_output:
                continue
            if ARGUMENT_DRIVEN_SOFTWARE.search(objective):
                if executable == "cargo" and subcommand == "run":
                    if "--" not in command or command.index("--") == len(command) - 1:
                        continue
                elif len(command) < 2:
                    continue
            if CALCULATOR_PRODUCT.search(objective) and not re.search(r"[-+]?\d", expected_output):
                continue
            return True
        return False

    @staticmethod
    def _has_failed_smoke_attempt_after_tests(evidence: list[dict[str, Any]]) -> bool:
        test_index = -1
        for index, item in enumerate(evidence):
            action = item.get("action") or {}
            result = item.get("result") or {}
            output = result.get("output") or {}
            if action.get("tool") not in {"run_tests", "run_command"}:
                continue
            tests_run = output.get("tests_run") if isinstance(output, dict) else None
            if (
                result.get("ok")
                and output.get("returncode") == 0
                and tests_run is not None
                and int(tests_run) > 0
            ):
                test_index = index
        for item in evidence[test_index + 1 :]:
            if (item.get("action") or {}).get("tool") != "run_command":
                continue
            result = item.get("result") or {}
            output = result.get("output") or {}
            if not result.get("ok"):
                return True
            if (
                isinstance(output, dict)
                and (
                    output.get("returncode") != 0
                    or output.get("timed_out")
                    or output.get("assertions_passed") is False
                )
            ):
                return True
        return False

    def _verified_partial_answer(
        self, verdict: dict[str, Any], sources: list[GroundingSource]
    ) -> str:
        confirmed: list[str] = []
        review_meta = re.compile(
            r"\b(?:candidate answer|citation markers?|deterministic validation|grounded in source|"
            r"previous validation|requires? explicit citations?|answer is empty)\b",
            re.IGNORECASE,
        )
        for raw in verdict.get("findings") or []:
            finding = _normalize_citations(str(raw).strip().lstrip("- "))
            if review_meta.search(finding) or not re.search(r"\[S\d+]", finding):
                continue
            if self.grounding.verify(finding, sources).valid:
                confirmed.append(finding)
            if len(confirmed) >= 3:
                break
        if not confirmed:
            return (
                "INSUFFICIENT_CONTEXT — The retrieved evidence does not verify the requested "
                "conclusion."
            )
        bullets = "\n".join(f"- {finding}" for finding in confirmed)
        return (
            "INSUFFICIENT_CONTEXT — The retrieved evidence does not fully support the requested "
            f"conclusion. Confirmed findings:\n{bullets}"
        )

    async def _review(
        self,
        step: PlanStep,
        evidence: list[dict[str, Any]],
        candidate_answer: str,
    ) -> tuple[dict[str, Any], list[GroundingSource]]:
        prefer_oracle = bool(self.fleet and self.fleet.oracle and step.role == ModelRole.REASONER)
        sources = self.grounding.collect(evidence, step.objective)
        if not sources and (step.allowed_tools or LOCAL_TARGET.search(step.objective)):
            return (
                {
                    "passed": False,
                    "findings": ["No relevant project evidence was retrieved."],
                    "next_action": "Inspect the requested project source and retry once.",
                    "answer": "INSUFFICIENT_CONTEXT",
                },
                [],
            )
        deterministic = self._validated_action_verdict(step, evidence, sources)
        if (
            AGENT_ACTION.search(step.objective)
            and USER_FACING_SOFTWARE.search(step.objective)
            and not self._has_successful_smoke_check(evidence, step.objective)
        ):
            return (
                {
                    "passed": False,
                    "findings": ["The user-facing behavior has not passed an asserted smoke check."],
                    "next_action": (
                        "Run the built entry point with representative inputs and use run_command "
                        "output assertions to verify the expected result."
                    ),
                    "answer": "The project files were created, but the runnable behavior is not yet verified.",
                },
                sources,
            )
        if deterministic is not None:
            return deterministic, sources
        deterministic = self._validated_git_verdict(step, evidence, sources)
        if deterministic is not None:
            return deterministic, sources
        verdict = await self._json_call(
            ModelRole.CRITIC,
            critic_prompt(asdict(step), candidate_answer, self.grounding.render(sources)),
            CRITIC_SCHEMA,
            prefer_oracle=prefer_oracle,
        )
        verdict["answer"] = _normalize_citations(str(verdict.get("answer") or candidate_answer))
        check = self.grounding.verify(verdict["answer"], sources)
        if (
            len(step.allowed_tools) == 1
            and step.allowed_tools[0] in SINGLE_EVIDENCE_TOOLS
            and check.valid
            and "INSUFFICIENT_CONTEXT" not in verdict["answer"]
        ):
            verdict["passed"] = True
            verdict["findings"] = []
            verdict["next_action"] = ""
            return verdict, sources
        citation_only = check.errors and all(
            error == "grounded answer has no source citations" or error.startswith("uncited sentence:")
            for error in check.errors
        )
        if len(sources) == 1 and citation_only:
            repaired = _attach_single_source_citations(verdict["answer"], sources[0].source_id)
            repaired_check = self.grounding.verify(repaired, sources)
            if repaired_check.valid:
                verdict["answer"] = repaired
                verdict["passed"] = True
                verdict["findings"] = []
                verdict["next_action"] = ""
                return verdict, sources
        if not check.valid:
            verdict = await self._json_call(
                ModelRole.CRITIC,
                critic_prompt(
                    asdict(step), candidate_answer, self.grounding.render(sources), list(check.errors)
                ),
                CRITIC_SCHEMA,
                prefer_oracle=prefer_oracle,
            )
            verdict["answer"] = _normalize_citations(str(verdict.get("answer") or ""))
            check = self.grounding.verify(verdict["answer"], sources)
        if not check.valid:
            verdict["passed"] = False
            verdict["answer"] = self._verified_partial_answer(verdict, sources)
            if step.id == "inspect-file" and "Confirmed findings:\n" in verdict["answer"]:
                confirmed = verdict["answer"].split("Confirmed findings:\n", 1)[1]
                if self.grounding.verify(confirmed, sources).valid:
                    verdict["passed"] = True
                    verdict["answer"] = confirmed
                    verdict["findings"] = []
                    verdict["next_action"] = ""
            verdict["findings"] = [*map(str, verdict.get("findings") or []), *check.errors]
            verdict["next_action"] = "Gather stronger project evidence and retry once."
        return verdict, sources

    def _validated_action_verdict(
        self,
        step: PlanStep,
        evidence: list[dict[str, Any]],
        sources: list[GroundingSource],
    ) -> dict[str, Any] | None:
        """Accept software work only when deterministic tests and behavior prove it."""
        changes: list[tuple[str, dict[str, Any]]] = []
        successful_test: dict[str, Any] | None = None
        successful_smoke: dict[str, Any] | None = None
        last_mutation = max(
            (
                index
                for index, item in enumerate(evidence)
                if (item.get("action") or {}).get("tool") in MUTATION_TOOLS
                and (item.get("result") or {}).get("ok")
                and ((item.get("result") or {}).get("output") or {}).get("changed")
            ),
            default=-1,
        )
        for index, item in enumerate(evidence):
            action = item.get("action") or {}
            result = item.get("result") or {}
            output = result.get("output") or {}
            if not result.get("ok") or not isinstance(output, dict):
                continue
            if action.get("tool") in {"write_file", "replace_text"} and output.get("changed"):
                changes.append((str(output.get("path") or "project file"), output))
            if (
                action.get("tool") in {"run_tests", "run_command"}
                and index > last_mutation
                and output.get("returncode") == 0
                and not output.get("timed_out")
                and (
                    (
                        action.get("tool") == "run_tests"
                        and output.get("tests_run") is None
                    )
                    or (
                        output.get("tests_run") is not None
                        and int(output.get("tests_run")) > 0
                    )
                )
            ):
                successful_test = output
            if index > last_mutation and self._has_successful_smoke_check([item], step.objective):
                successful_smoke = item
        if successful_test is None:
            return None

        command = " ".join(map(str, successful_test.get("command") or [])) or "project test runner"
        test_source = next(
            (source for source in sources if source.label == f"command: {command}"), None
        )
        if test_source is None:
            return None
        answer_parts: list[str] = []
        if changes:
            change_source = next(
                (source for source in sources if source.label.startswith("change: ")), None
            )
            if change_source is None:
                return None
            answer_parts.append(
                f"Implemented the requested project changes. [{change_source.source_id}]"
            )
        if USER_FACING_SOFTWARE.search(step.objective):
            if successful_smoke is None:
                return None
            smoke_action = successful_smoke.get("action") or {}
            smoke_command = " ".join(
                map(str, (smoke_action.get("arguments") or {}).get("command") or [])
            )
            smoke_source = next(
                (source for source in sources if source.label == f"command: {smoke_command}"), None
            )
            if smoke_source is None:
                return None
            answer_parts.append(
                f"The runnable behavior passed its expected-output check. [{smoke_source.source_id}]"
            )
        answer_parts.append(
            f"Tests passed with exit code 0 using `{command}`. [{test_source.source_id}]"
        )
        answer = "\n".join(answer_parts)
        if not self.grounding.verify(answer, sources).valid:
            return None
        return {
            "passed": True,
            "findings": [],
            "next_action": "",
            "answer": answer,
        }

    def _validated_git_verdict(
        self,
        step: PlanStep,
        evidence: list[dict[str, Any]],
        sources: list[GroundingSource],
    ) -> dict[str, Any] | None:
        if step.allowed_tools != ["git_diff"]:
            return None
        source = next((item for item in sources if item.label == "working tree diff"), None)
        if source is None:
            return None
        output = next(
            (
                (item.get("result") or {}).get("output")
                for item in reversed(evidence)
                if (item.get("action") or {}).get("tool") == "git_diff"
                and (item.get("result") or {}).get("ok")
            ),
            None,
        )
        if not isinstance(output, dict):
            return None
        tracked = list(map(str, output.get("tracked_files") or []))
        untracked = list(map(str, output.get("untracked_files") or []))
        if not tracked and not untracked:
            answer = f"The git working tree is clean. [{source.source_id}]"
        else:
            tracked_text = ", ".join(tracked) if tracked else "none"
            untracked_text = ", ".join(untracked) if untracked else "none"
            answer = (
                f"The working tree has {len(tracked)} tracked changes and {len(untracked)} "
                f"untracked files. [{source.source_id}]\n"
                f"Tracked: {tracked_text}. [{source.source_id}]\n"
                f"Untracked: {untracked_text}. [{source.source_id}]"
            )
        if not self.grounding.verify(answer, sources).valid:
            return None
        return {"passed": True, "findings": [], "next_action": "", "answer": answer}

    async def _run_step(
        self,
        objective: str,
        step: PlanStep,
        *,
        initial_evidence: list[dict[str, Any]] | None = None,
        seen_action_fingerprints: set[str] | None = None,
    ) -> dict[str, Any]:
        step.status = "running"
        self._emit("step.started", step.role.value, step=asdict(step))
        evidence = list(initial_evidence or [])
        candidate_answer = next(
            (
                str(item.get("message") or "")
                for item in reversed(evidence)
                if item.get("type") == "worker_message" and str(item.get("message") or "").strip()
            ),
            "",
        )
        descriptions = self._tool_descriptions_for(step)
        allowed_names = {tool["name"] for tool in descriptions}
        stalled_turns = 0
        restarts = 0
        seen_actions = seen_action_fingerprints if seen_action_fingerprints is not None else set()
        mutation_epoch = self._mutation_epoch(evidence)
        has_tool_evidence = any(
            bool((item.get("result") or {}).get("ok")) and bool(item.get("action"))
            for item in evidence
        )

        turn_limit = self.config.runtime.max_worker_turns
        if descriptions and allowed_names.issubset(READ_ONLY_TOOLS):
            turn_limit = min(turn_limit, 4)
        for turn_number in range(1, turn_limit + 1):
            try:
                value = await self._json_call(
                    step.role,
                    worker_prompt(objective, asdict(step), descriptions, evidence),
                    WORKER_SCHEMA,
                    prefer_oracle=step.role == ModelRole.ORACLE,
                )
            except (BackendError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                self._emit(
                    "worker.failed",
                    step.role.value,
                    step_id=step.id,
                    turn=turn_number,
                    error=f"invalid structured response: {exc}",
                )
                evidence.append(
                    {
                        "type": "nogood",
                        "rule": "The previous response was not valid structured output; emit exactly the required object shape.",
                    }
                )
                if restarts >= self.config.runtime.max_worker_restarts:
                    break
                restarts += 1
                continue

            turn = WorkerTurn.from_dict(value)
            if turn.message.strip():
                candidate_answer = turn.message.strip()
            self._emit(
                "worker.turn", step.role.value, step_id=step.id, turn=turn_number, message=turn.message
            )
            evidence.append({"type": "worker_message", "message": turn.message})

            if not turn.done and not turn.actions:
                stalled_turns += 1
                self._emit(
                    "worker.stalled",
                    step.role.value,
                    step_id=step.id,
                    turn=turn_number,
                    reason="model returned no action and did not finish",
                )
                if stalled_turns >= 2:
                    if restarts >= self.config.runtime.max_worker_restarts:
                        break
                    restarts += 1
                    stalled_turns = 0
                    evidence.append(
                        {
                            "type": "nogood",
                            "rule": "The previous branch stalled without a tool action; choose a different relevant tool action now.",
                        }
                    )
                continue
            stalled_turns = 0

            if turn.done and not turn.actions and step.allowed_tools and not has_tool_evidence:
                evidence.append(
                    {
                        "type": "nogood",
                        "rule": "Completion without local tool evidence was rejected; inspect the project before answering.",
                    }
                )
                self._emit(
                    "worker.stalled",
                    step.role.value,
                    step_id=step.id,
                    turn=turn_number,
                    reason="completion claimed without required tool evidence",
                )
                if restarts >= self.config.runtime.max_worker_restarts:
                    break
                restarts += 1
                continue

            if (
                turn.done
                and not turn.actions
                and AGENT_ACTION.search(objective)
                and "run_tests" in allowed_names
                and SOFTWARE_VALIDATION.search(objective)
                and not self._has_meaningful_test_success(evidence)
            ):
                evidence.append(
                    {
                        "type": "nogood",
                        "rule": (
                            "Completion was rejected because changed software lacks a successful "
                            "behavior test. Add or repair focused tests, then run run_tests."
                        ),
                    }
                )
                self._emit(
                    "worker.stalled",
                    step.role.value,
                    step_id=step.id,
                    turn=turn_number,
                    reason="software mutation lacks successful behavior tests",
                )
                continue

            if (
                turn.done
                and not turn.actions
                and AGENT_ACTION.search(objective)
                and USER_FACING_SOFTWARE.search(objective)
                and not self._has_successful_smoke_check(evidence, objective)
            ):
                evidence.append(
                    {
                        "type": "nogood",
                        "rule": (
                            "Completion was rejected because the runnable entry point lacks an "
                            "asserted smoke check. Use run_command with representative arguments "
                            "and expect_stdout_contains (plus expect_exit when needed)."
                        ),
                    }
                )
                self._emit(
                    "worker.stalled",
                    step.role.value,
                    step_id=step.id,
                    turn=turn_number,
                    reason="user-facing software lacks an asserted smoke check",
                )
                continue

            single_evidence_complete = False
            for action in turn.actions:
                if (
                    AGENT_ACTION.search(objective)
                    and SOFTWARE_VALIDATION.search(objective)
                    and self._has_meaningful_test_success(evidence)
                ):
                    behavior_verified = (
                        not USER_FACING_SOFTWARE.search(objective)
                        or self._has_successful_smoke_check(evidence, objective)
                    )
                    if behavior_verified:
                        single_evidence_complete = True
                        break
                    smoke_failed = self._has_failed_smoke_attempt_after_tests(evidence)
                    requested_command = list(
                        map(str, (action.arguments or {}).get("command") or [])
                    )
                    requested_path = str((action.arguments or {}).get("path") or "")
                    edits_test_file = bool(
                        re.search(
                            r"(?:^|/)(?:tests?|specs?)(?:/|$)|(?:^|/)(?:test_[^/]+|[^/]+_test\.[^/]+)$",
                            requested_path,
                            re.IGNORECASE,
                        )
                    )
                    executable = Path(requested_command[0]).name.casefold() if requested_command else ""
                    subcommand = requested_command[1].casefold() if len(requested_command) > 1 else ""
                    non_behavior_command = (
                        not requested_command
                        or (
                            executable == "cargo"
                            and subcommand
                            in {"add", "build", "check", "clippy", "fmt", "test"}
                        )
                        or (executable in {"go", "npm"} and subcommand == "test")
                        or executable in {"pytest", "ruff"}
                    )
                    if (
                        (
                            not smoke_failed
                            and (action.tool != "run_command" or non_behavior_command)
                        )
                        or (
                            smoke_failed
                            and (action.tool not in MUTATION_TOOLS or edits_test_file)
                        )
                    ):
                        result = {
                            "ok": False,
                            "error": (
                                "automated tests pass; run only the missing expected-output smoke check"
                                if not smoke_failed
                                else "the expected-output smoke check failed; edit the implementation before running another command"
                            ),
                        }
                        evidence.append({"action": asdict(action), "result": result})
                        self._emit(
                            "worker.stalled",
                            step.role.value,
                            step_id=step.id,
                            turn=turn_number,
                            reason=result["error"],
                        )
                        continue
                fingerprint = self._action_fingerprint(
                    action.tool, action.arguments, mutation_epoch
                )
                if fingerprint in seen_actions:
                    if action.tool in READ_ONLY_TOOLS:
                        duplicate_error = (
                            "duplicate read rejected for the unchanged project state; use the "
                            "content already present in evidence and make the next required edit"
                        )
                    elif action.tool in VALIDATION_TOOLS:
                        duplicate_error = (
                            "duplicate verification rejected for the unchanged project state; "
                            "change the implementation or validation strategy first"
                        )
                    else:
                        duplicate_error = "duplicate action rejected; change the tool or arguments"
                    result = {
                        "ok": False,
                        "error": duplicate_error,
                    }
                    evidence.append({"action": asdict(action), "result": result})
                    self._emit(
                        "worker.stalled",
                        step.role.value,
                        step_id=step.id,
                        turn=turn_number,
                        reason=result["error"],
                    )
                    continue
                seen_actions.add(fingerprint)
                if action.tool not in allowed_names:
                    result = {"ok": False, "error": f"tool is outside this step's manifest: {action.tool}"}
                    evidence.append({"action": asdict(action), "result": result})
                    self._emit(
                        "tool.denied", "policy", step_id=step.id, tool=action.tool, reason=result["error"]
                    )
                    continue
                try:
                    request = self.tools.prepare(action)
                    token, reason = self.tools.authorize(request)
                    if not token:
                        result = {"ok": False, "error": reason}
                        self._emit("tool.denied", "policy", step_id=step.id, tool=action.tool, reason=reason)
                    else:
                        self._emit(
                            "tool.authorized",
                            "policy",
                            step_id=step.id,
                            tool=action.tool,
                            risk=request.risk.name.lower(),
                        )
                        output = await asyncio.to_thread(self.tools.execute, request, token)
                        result = {"ok": True, "output": output}
                        has_tool_evidence = True
                        if action.tool in MUTATION_TOOLS and output.get("changed"):
                            mutation_epoch += 1
                        single_evidence_complete = (
                            len(allowed_names) == 1 and action.tool in SINGLE_EVIDENCE_TOOLS
                        )
                        self._emit("tool.completed", action.tool, step_id=step.id, result=output)
                except Exception as exc:
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    self._emit("tool.failed", action.tool, step_id=step.id, error=result["error"])
                evidence.append({"action": asdict(action), "result": result})
                output = result.get("output") or {}
                if (
                    result.get("ok")
                    and action.tool in VALIDATION_TOOLS
                    and isinstance(output, dict)
                    and (
                        output.get("returncode") != 0
                        or output.get("timed_out")
                        or output.get("assertions_passed") is False
                    )
                ):
                    diagnostic = str(output.get("stderr") or output.get("stdout") or "")[-2_000:]
                    retry_rule = (
                        "Change the implementation before rerunning it."
                        if action.tool == "run_tests"
                        else "Change the implementation or validation command before retrying."
                    )
                    evidence.append(
                        {
                            "type": "nogood",
                            "rule": (
                                "External verification rejected the current project state. "
                                f"{retry_rule} Diagnostic: {diagnostic}"
                            ),
                        }
                    )
                if (
                    result.get("ok")
                    and action.tool in {"run_tests", "run_command"}
                    and isinstance(output, dict)
                    and output.get("returncode") == 0
                    and output.get("tests_run") == 0
                ):
                    evidence.append(
                        {
                            "type": "nogood",
                            "rule": (
                                "The test command passed but executed zero behavior tests. "
                                "Add focused tests for the requested outcome before completion."
                            ),
                        }
                    )
                if not result.get("ok"):
                    evidence.append(
                        {
                            "type": "nogood",
                            "rule": f"The last {action.tool} attempt failed; inspect its diagnostic and change the arguments or tool.",
                        }
                    )

            if single_evidence_complete:
                break

            if (
                AGENT_ACTION.search(objective)
                and SOFTWARE_VALIDATION.search(objective)
                and self._has_meaningful_test_success(evidence)
                and (
                    not USER_FACING_SOFTWARE.search(objective)
                    or self._has_successful_smoke_check(evidence, objective)
                )
            ):
                break

            if turn.done and not turn.actions:
                break

        try:
            verdict, sources = await self._review(step, evidence, candidate_answer)
        except (BackendError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            sources = self.grounding.collect(evidence, step.objective)
            verdict = {
                "passed": False,
                "findings": [f"critic unavailable: {exc}"],
                "next_action": "Run validation manually.",
                "answer": candidate_answer or "I could not produce a verified answer.",
            }
        step.status = "completed" if verdict.get("passed") else "needs_review"
        self._emit("step.finished", "critic", step=asdict(step), verdict=verdict)
        answer = str(verdict.get("answer") or candidate_answer).strip()
        if sources and answer != "INSUFFICIENT_CONTEXT":
            answer += "\n\nSources:\n" + "\n".join(self.grounding.source_labels(sources))
        return {
            "step": asdict(step),
            "verdict": verdict,
            "answer": answer,
            "sources": self.grounding.source_labels(sources),
            "evidence": evidence,
        }

    async def _run_step_with_recovery(
        self,
        objective: str,
        step: PlanStep,
        inherited_evidence: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        evidence = list(inherited_evidence or [])
        seen_actions: set[str] = set()
        mutation_epoch = 0
        for item in evidence:
            action = item.get("action") or {}
            if action:
                seen_actions.add(
                    self._action_fingerprint(
                        str(action.get("tool") or ""),
                        dict(action.get("arguments") or {}),
                        mutation_epoch,
                    )
                )
            if (
                action.get("tool") in MUTATION_TOOLS
                and (item.get("result") or {}).get("ok")
                and ((item.get("result") or {}).get("output") or {}).get("changed")
            ):
                mutation_epoch += 1
        result: dict[str, Any] = {}
        attempts = max(1, self.config.runtime.max_step_attempts)
        for attempt in range(1, attempts + 1):
            result = await self._run_step(
                objective,
                step,
                initial_evidence=evidence,
                seen_action_fingerprints=seen_actions,
            )
            if result.get("verdict", {}).get("passed") or attempt >= attempts:
                return result
            verdict = result.get("verdict") or {}
            diagnostic = str(verdict.get("next_action") or "").strip()
            if not diagnostic:
                diagnostic = "; ".join(map(str, verdict.get("findings") or []))[:500]
            evidence = list(result.get("evidence") or [])
            evidence.append(
                {
                    "type": "nogood",
                    "rule": (
                        "External review rejected the previous attempt. Preserve its useful evidence, "
                        f"do not repeat rejected actions, and address this diagnostic: {diagnostic}"
                    ),
                }
            )
            self._emit(
                "step.restarted",
                "nedster",
                step_id=step.id,
                attempt=attempt + 1,
                diagnostic=diagnostic,
            )
        return result

    async def run(self, objective: str) -> dict[str, Any]:
        plan = await self.plan(objective)
        results: list[dict[str, Any]] = []
        evidence_by_step: dict[str, list[dict[str, Any]]] = {}
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
            inherited_by_step = {
                step.id: [
                    item
                    for dependency in step.depends_on
                    for item in evidence_by_step.get(dependency, [])
                ]
                for step in batch
            }
            batch_results = await asyncio.gather(
                *(
                    self._run_step_with_recovery(
                        objective,
                        step,
                        inherited_evidence=inherited_by_step[step.id],
                    )
                    for step in batch
                )
            )
            for step, result in zip(batch, batch_results, strict=True):
                results.append(result)
                evidence_by_step[step.id] = list(result.get("evidence") or [])
                completed.add(step.id)
                pending.remove(step)
                if step.status != "completed":
                    pending.clear()
                    break
        passed = bool(results) and all(item["verdict"].get("passed") for item in results)
        answers = [
            str(item.get("answer") or "").strip() for item in results if str(item.get("answer") or "").strip()
        ]
        if len(answers) == 1:
            answer = answers[0]
        elif answers and any(item.get("step", {}).get("depends_on") for item in results):
            answer = answers[-1]
        elif answers:
            answer = "\n\n".join(
                f"{item['step']['title']}\n{item['answer']}" for item in results if item.get("answer")
            )
        else:
            answer = "I could not produce a verified answer from the available evidence."
        summary = {
            "objective": objective,
            "answer": answer,
            "mode": "agent",
            "passed": passed,
            "grounded": True,
            "completed_steps": len(completed),
            "total_steps": len(plan.steps),
            "results": results,
        }
        self._emit("run.finished", "nedster", summary={k: v for k, v in summary.items() if k != "results"})
        return summary

    def close(self) -> None:
        self.ledger.close()


# Compatibility for v3.0 integrations. New code should import NedsterRuntime.
MythosOrchestrator = NedsterRuntime
