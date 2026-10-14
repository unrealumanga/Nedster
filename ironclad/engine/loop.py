"""The native engine: one tool-capable conversation loop.

Message layout per session: ``[system prompt, pinned task state, *history]``. History is append-only
between compactions so the rendered prefix stays stable. The outcome is decided by ``acceptance.evaluate``
from tool records; model prose never upgrades it.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, cast

from ironclad.contracts import (
    AcceptanceResult,
    ChatMessage,
    ChatTransport,
    EngineEvent,
    EventKind,
    JobHandle,
    SessionStore,
    TextDelta,
    ToolCall,
    ToolCallsReady,
    ToolResult,
    ToolRuntime,
    ToolSpec,
    ToolStatus,
    TransportError,
    TurnFinished,
)
from ironclad.engine import acceptance
from ironclad.engine.acceptance import StopReason
from ironclad.engine.context import (
    SHOW_OUTPUT,
    SHOW_OUTPUT_SPEC,
    ContextManager,
    Estimator,
    OutputVault,
    estimate_tokens,
    run_show_output,
    tool_message,
)
from ironclad.engine.prompt import build_system_prompt, project_facts
from ironclad.engine.state import Budget, TaskState, ToolRecord

REPEAT_LIMIT = 3
VALIDATION_ARG = "validation"
EVENT_OUTPUT_CHARS = 4_000  # bounded tool output carried on tool.finished for UIs
EVENT_DATA_KEYS = ("exit_code", "tests_run", "tests_failed", "diff", "backend", "isolation")
VALIDATABLE_TOOLS = frozenset({"run_command"})
COMPACTED: EventKind = cast(EventKind, "context.compacted")  # proposed addition to contracts.EventKind
JOB_STOP_TIMEOUT_S = 10.0

Stop = tuple[StopReason, str]


@dataclass(slots=True)
class _Session:
    session_id: str
    messages: list[ChatMessage]  # [system, pinned, *history]
    state: TaskState
    vault: OutputVault
    context: ContextManager
    records: list[ToolRecord] = field(default_factory=list)  # all runs, for task state


@dataclass(slots=True)
class _Run:
    run_id: str
    started: float
    records: list[ToolRecord] = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    invalid: dict[str, int] = field(default_factory=dict)
    prose: str = ""


def _signature(call: ToolCall) -> str:
    return call.name + json.dumps(dict(call.arguments), sort_keys=True, default=str)


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")[:200]


def _groups(calls: Sequence[ToolCall], mutating: Callable[[ToolCall], bool]) -> Iterator[list[ToolCall]]:
    """Consecutive read-only calls form one concurrent group; each mutating call is its own group."""
    batch: list[ToolCall] = []
    for call in calls:
        if mutating(call):
            if batch:
                yield batch
                batch = []
            yield [call]
        else:
            batch.append(call)
    if batch:
        yield batch


class Engine:
    def __init__(
        self,
        transport: ChatTransport,
        tools: ToolRuntime,
        *,
        model: str,
        store: SessionStore | None = None,
        budget: Budget | None = None,
        emit: Callable[[EngineEvent], None] | None = None,
        project_root: str | Path | None = None,
        estimator: Estimator = estimate_tokens,
        think: bool | str = False,
        temperature: float | None = None,
    ) -> None:
        self.transport = transport
        self.tools = tools
        self.model = model
        self.store = store
        self.budget = budget or Budget()
        self._emit = emit or (lambda _event: None)
        self.project_root = project_root
        self.estimator = estimator
        self.think = think
        self.temperature = temperature
        self.last_answer = ""
        self._sessions: dict[str, _Session] = {}
        self._running: set[JobHandle] = set()
        self._task: asyncio.Task[Stop] | None = None
        self._augmented: set[str] = set()
        self._validating: set[str] = set()

    # ------------------------------------------------------------------ public API

    def cancel(self) -> None:
        """Stop the current run: running jobs are cancelled and awaited, the outcome is CANCELLED."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    def messages(self, session_id: str) -> list[ChatMessage]:
        """Current active context (system, pinned state, history) of a session."""
        return list(self._session(session_id).messages)

    async def run(self, session_id: str, user_text: str) -> AcceptanceResult:
        session = self._session(session_id)
        run = _Run(uuid.uuid4().hex[:12], time.monotonic())
        self._event("turn.started", session, run, {"user_text": user_text})
        self._task = asyncio.create_task(self._drive(session, run, user_text))
        try:
            stop, detail = await self._task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            stop, detail = "cancelled", ""
        finally:
            self._task = None
        return self._finish(session, run, stop, detail)

    # ------------------------------------------------------------------ session setup

    def _session(self, session_id: str) -> _Session:
        if session_id in self._sessions:
            return self._sessions[session_id]
        store = self.store
        history = store.messages(session_id) if store else []
        state = TaskState.from_dict(store.get_state(session_id, "task_state") if store else None)
        facts = store.get_state(session_id, "project_facts") if store else None
        if not facts:
            facts = project_facts(self.project_root)
            if store:
                store.set_state(session_id, "project_facts", facts)
        vault = OutputVault(session_id, store)
        messages = [
            ChatMessage(role="system", content=build_system_prompt(facts)),
            ChatMessage(role="system", content=state.render()),
            *history,
        ]
        context = ContextManager(self.budget.context_tokens, vault, self.estimator)
        session = _Session(session_id, messages, state, vault, context)
        self._sessions[session_id] = session
        return session

    def _specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for spec in self.tools.specs():
            props = dict(spec.parameters.get("properties", {}))
            if spec.name in VALIDATABLE_TOOLS and VALIDATION_ARG not in props:
                props[VALIDATION_ARG] = {
                    "type": "boolean",
                    "description": "true when this command validates your change (tests, build, lint)",
                }
                spec = replace(spec, parameters={**spec.parameters, "properties": props})
                self._augmented.add(spec.name)
            specs.append(spec)
        if all(s.name != SHOW_OUTPUT for s in specs):
            specs.append(SHOW_OUTPUT_SPEC)
        self._validating = {s.name for s in specs if s.validates}
        return specs

    # ------------------------------------------------------------------ the loop

    async def _drive(self, session: _Session, run: _Run, user_text: str) -> Stop:
        fresh = len(session.messages) == 2
        session.state.note_user(user_text)
        if fresh:  # nothing sent yet for this session: the pinned state may still change freely
            session.messages[1] = ChatMessage(role="system", content=session.state.render())
        self._append(session, ChatMessage(role="user", content=user_text))
        specs = self._specs()
        try:
            async with asyncio.timeout(self.budget.deadline_s):
                return await self._loop(session, run, specs)
        except TimeoutError:
            return "budget", "deadline exceeded"
        except TransportError as exc:
            self._event("error", session, run, {"message": str(exc), "remedy": "check the model server"})
            return "transport_error", f"model transport failed: {exc}"

    async def _loop(self, session: _Session, run: _Run, specs: list[ToolSpec]) -> Stop:
        by_name = {spec.name: spec for spec in specs}
        budget = self.budget
        while True:
            if budget.max_steps is not None and run.steps >= budget.max_steps:
                return "budget", f"step budget exhausted ({budget.max_steps} model calls)"
            if budget.max_tokens is not None and run.tokens >= budget.max_tokens:
                return "budget", f"token budget exhausted ({run.tokens}/{budget.max_tokens})"
            self._maybe_compact(session, run, specs)
            message, done_reason = await self._model_turn(session, run, specs)
            run.steps += 1
            self._append(session, message)
            if not message.tool_calls:
                run.prose = message.content
                if done_reason == "cancelled":
                    return "cancelled", ""
                if done_reason in ("length", "error"):
                    return "model_error", f"model stopped early ({done_reason})"
                return "final", ""
            stop = await self._run_calls(session, run, message.tool_calls, by_name)
            if stop is not None:
                return stop

    async def _model_turn(self, session: _Session, run: _Run, specs: list[ToolSpec]) -> tuple[ChatMessage, str]:
        stream: Any = self.transport.chat(
            model=self.model,
            messages=list(session.messages),
            tools=specs,
            think=self.think,
            temperature=self.temperature,
        )
        if inspect.isawaitable(stream):
            stream = await stream
        content: list[str] = []
        thinking: list[str] = []
        calls: tuple[ToolCall, ...] = ()
        final: TurnFinished | None = None
        try:
            async for event in stream:
                if isinstance(event, TextDelta):
                    (thinking if event.kind == "thinking" else content).append(event.text)
                    self._event("model.delta", session, run, {"text": event.text, "kind": event.kind})
                elif isinstance(event, ToolCallsReady):
                    calls = calls + tuple(event.calls)
                elif isinstance(event, TurnFinished):
                    final = event
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
        if final is None:
            message = ChatMessage(role="assistant", content="".join(content), thinking="".join(thinking))
            done_reason = "error" if not calls else "stop"
        else:
            usage = final.usage
            run.tokens += usage.prompt_tokens + usage.completion_tokens
            self._event("model.usage", session, run, asdict(usage))
            message = final.message
            done_reason = final.done_reason
        return (
            ChatMessage(
                role="assistant",
                content=message.content or "".join(content),
                tool_calls=list(calls or message.tool_calls),
                thinking=message.thinking or "".join(thinking),
            ),
            done_reason,
        )

    # ------------------------------------------------------------------ tool execution

    async def _run_calls(
        self, session: _Session, run: _Run, calls: Sequence[ToolCall], by_name: Mapping[str, ToolSpec]
    ) -> Stop | None:
        limit = self.budget.max_tool_calls
        allowed = len(calls) if limit is None else max(0, min(len(calls), limit - run.tool_calls))
        runnable, skipped = list(calls[:allowed]), list(calls[allowed:])

        def mutating(call: ToolCall) -> bool:
            spec = by_name.get(call.name)
            return spec.mutates if spec is not None else call.name != SHOW_OUTPUT

        answered: set[str] = set()
        try:
            for group in _groups(runnable, mutating):
                if len(group) == 1:
                    records = [await self._exec_one(session, run, group[0], mutating(group[0]))]
                else:
                    async with asyncio.TaskGroup() as tg:
                        tasks = [tg.create_task(self._exec_one(session, run, c, False)) for c in group]
                    records = [task.result() for task in tasks]
                for record in records:
                    run.tool_calls += 1
                    answered.add(record.call.id)
                    self._append(session, self._tool_message(session, run, record))
        except asyncio.CancelledError:
            for call in calls:
                if call.id not in answered:
                    self._append(session, self._placeholder(call, "cancelled", "not completed: run was stopped"))
            raise
        for call in skipped:
            self._append(session, self._placeholder(call, "cancelled", "not executed: tool-call budget exhausted"))
        if skipped:
            return "budget", f"tool-call budget exhausted ({limit} calls)"
        for signature, count in run.invalid.items():
            if count >= REPEAT_LIMIT:
                name = signature.split("{", 1)[0]
                return "repeated_invalid", f"same invalid {name} call repeated {count} times"
        return None

    def _tool_message(self, session: _Session, run: _Run, record: ToolRecord) -> ChatMessage:
        message = tool_message(record.result, session.vault, self.budget.max_output_chars)
        if record.result.status in (ToolStatus.INVALID, ToolStatus.DENIED):
            count = run.invalid.get(_signature(record.call), 0)
            hint = "Correct the arguments." if count < REPEAT_LIMIT - 1 else "Repeating this exact call again stops the task."
            message = ChatMessage(role="tool", content=f"{message.content}\n{hint}", tool_call_id=message.tool_call_id, name=message.name)
        return message

    @staticmethod
    def _placeholder(call: ToolCall, status: str, text: str) -> ChatMessage:
        return ChatMessage(role="tool", content=f"[{status}] {text}", tool_call_id=call.id, name=call.name)

    async def _exec_one(self, session: _Session, run: _Run, call: ToolCall, mutating: bool) -> ToolRecord:
        validation = call.name in self._validating or (
            call.name in self._augmented and call.arguments.get(VALIDATION_ARG) is True
        )
        self._event("tool.started", session, run, {"call_id": call.id, "tool": call.name, "arguments": dict(call.arguments)})
        before = self.tools.revision()
        if call.name == SHOW_OUTPUT:
            result = run_show_output(session.vault, call, self.budget.max_output_chars)
        else:
            result = await self._start_and_wait(self._runtime_call(call))
        record = ToolRecord(call, result, before, self.tools.revision(), mutating, validation)
        run.records.append(record)
        session.records.append(record)
        session.state.note_record(record)
        if result.status in (ToolStatus.INVALID, ToolStatus.DENIED):
            sig = _signature(call)
            run.invalid[sig] = run.invalid.get(sig, 0) + 1
        self._event(
            "tool.finished",
            session,
            run,
            {
                "call_id": call.id,
                "tool": call.name,
                "status": result.status.value,
                "summary": _first_line(result.output),
                "mutated_paths": list(result.mutated_paths),
                "elapsed_ms": result.elapsed_ms,
                "output": result.output[:EVENT_OUTPUT_CHARS],
                **{key: result.data[key] for key in EVENT_DATA_KEYS if key in result.data},
            },
        )
        return record

    def _runtime_call(self, call: ToolCall) -> ToolCall:
        if call.name in self._augmented and VALIDATION_ARG in call.arguments:
            args = {k: v for k, v in call.arguments.items() if k != VALIDATION_ARG}
            return ToolCall(call.id, call.name, args)
        return call

    async def _start_and_wait(self, call: ToolCall) -> ToolResult:
        try:
            handle = self.tools.start(call)
        except Exception as exc:  # a runtime bug must not kill the loop
            return ToolResult(call.id, call.name, ToolStatus.FAILED, f"tool runtime error: {exc}")
        self._running.add(handle)
        try:
            return await handle.wait()
        except asyncio.CancelledError:
            await self._stop_job(handle)
            raise
        except Exception as exc:
            return ToolResult(call.id, call.name, ToolStatus.FAILED, f"tool runtime error: {exc}")
        finally:
            self._running.discard(handle)

    @staticmethod
    async def _stop_job(handle: JobHandle) -> None:
        """Cancel and wait until the job has really stopped (bounded so a broken job cannot hang us)."""
        try:
            await asyncio.wait_for(handle.cancel(), JOB_STOP_TIMEOUT_S)
            await asyncio.wait_for(handle.wait(), JOB_STOP_TIMEOUT_S)
        except Exception:  # best effort; the job reports its own failure
            pass

    # ------------------------------------------------------------------ context + persistence

    def _maybe_compact(self, session: _Session, run: _Run, specs: list[ToolSpec]) -> None:
        if not session.context.over(session.messages, specs):
            return
        protect = self._failing_validation(session)
        compaction = session.context.compact(session.messages, specs, protect=protect)
        if compaction is None:  # nothing left to mask: keep the prefix instead of churning the pinned state
            return
        current = acceptance.evaluate(run.records)
        session.state.validation = acceptance.validation_status(session.records)
        session.state.obligations = acceptance.obligations(run.records, current) if run.records else []
        session.messages[1] = ChatMessage(role="system", content=session.state.render())
        if self.store is not None:
            replace_messages = getattr(self.store, "replace_messages", None)
            if callable(replace_messages):
                replace_messages(session.session_id, session.messages[2:])
            self.store.set_state(session.session_id, "task_state", session.state.to_dict())
        data = {
            "elided": compaction.elided,
            "handles": list(compaction.handles),
            "tokens_before": compaction.tokens_before,
            "tokens_after": compaction.tokens_after,
        }
        self._event(COMPACTED, session, run, data)

    @staticmethod
    def _failing_validation(session: _Session) -> set[str]:
        for record in reversed(session.records):
            verdict = acceptance.validation_verdict(record)
            if verdict is not None:
                return {record.call.id} if verdict == "fail" else set()
        return set()

    def _append(self, session: _Session, message: ChatMessage) -> None:
        session.messages.append(message)
        if self.store is not None:
            self.store.append_message(session.session_id, message)

    def _finish(self, session: _Session, run: _Run, stop: StopReason, detail: str) -> AcceptanceResult:
        result = acceptance.evaluate(run.records, stop, detail)
        session.state.validation = acceptance.validation_status(session.records)
        session.state.obligations = acceptance.obligations(run.records, result)
        if self.store is not None:
            self.store.set_state(session.session_id, "task_state", session.state.to_dict())
        prose = run.prose.strip()
        summary = acceptance.render_summary(run.records, result)
        self.last_answer = f"{prose}\n\n{summary}" if prose else summary
        self._event(
            "acceptance",
            session,
            run,
            {
                "outcome": result.outcome.value,
                "reasons": list(result.reasons),
                "changed_paths": list(result.changed_paths),
                "validated_at": result.validated_at.counter if result.validated_at else None,
            },
        )
        self._event("turn.finished", session, run, {"answer": self.last_answer, "outcome": result.outcome.value})
        return replace(result, answer=self.last_answer)

    def _event(self, kind: EventKind, session: _Session, run: _Run, data: Mapping[str, Any]) -> None:
        self._emit(EngineEvent(kind, session.session_id, run.run_id, dict(data), time.time_ns()))
