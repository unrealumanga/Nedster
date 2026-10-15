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
    Outcome,
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
from ironclad.engine.acceptance import (
    DECLARE_NO_CHANGE,
    DECLARE_NO_CHANGE_SPEC,
    StopReason,
    declare_no_change_result,
)
from ironclad.engine.context import (
    SHOW_OUTPUT,
    SHOW_OUTPUT_SPEC,
    ContextManager,
    Estimator,
    OutputVault,
    SummaryData,
    completion_reserve,
    estimate_tokens,
    run_show_output,
    tool_message,
)
from ironclad.engine.intent import Intent, classify
from ironclad.engine.prompt import (
    ProjectInstructions,
    SoulText,
    build_system_prompt,
    project_facts,
    session_instructions,
    session_soul,
)
from ironclad.engine.roles import Role, get_role
from ironclad.engine.state import Budget, TaskState, ToolRecord

REPEAT_LIMIT = 3
VALIDATION_ARG = "validation"
EVENT_OUTPUT_CHARS = 4_000  # bounded tool output carried on tool.finished for UIs
EVENT_DATA_KEYS = ("exit_code", "tests_run", "tests_failed", "diff", "backend", "isolation")
VALIDATABLE_TOOLS = acceptance.FLAGGABLE_TOOLS
TRUNCATED = ("length", "error")  # done reasons whose tool calls may be cut off: never executed
COMPACTION_STATE = "compaction_state"  # session state: {"compactions": n, "summary": SummaryData.to_dict()}
CONTEXT_EXHAUSTED = "context budget exhausted: the request cannot fit the model window even after compaction"
COMPACTED: EventKind = cast(EventKind, "context.compacted")  # proposed addition to contracts.EventKind
JOB_STOP_TIMEOUT_S = 10.0
ENGINE_TOOLS = frozenset({SHOW_OUTPUT, DECLARE_NO_CHANGE})  # answered by the engine itself; never mutate
TRANSCRIPT_TEXT_CHARS = 2_000
AUTO_VERIFY_TOOL = "run_tests"
AUTO_VERIFY_TEXT = "Running the project's tests to verify the change."

Stop = tuple[StopReason, str]


@dataclass(slots=True)
class _Session:
    session_id: str
    messages: list[ChatMessage]  # [system, pinned, *history]
    state: TaskState
    vault: OutputVault
    context: ContextManager
    facts: dict[str, str] = field(default_factory=dict)  # kept so a role switch can re-render the prefix
    instructions: ProjectInstructions = field(default_factory=ProjectInstructions)
    soul: SoulText = field(default_factory=SoulText)
    records: list[ToolRecord] = field(default_factory=list)  # all runs, for task state
    compactions: int = 0  # every compaction (mask or whole-group), persisted so a restart keeps counting


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
    intent: Intent = Intent.UNKNOWN  # what this user message asks for; decides whether "no change" can pass
    auto_verified: int = 0  # engine-initiated run_tests calls in this run


def _signature(call: ToolCall) -> str:
    return call.name + json.dumps(dict(call.arguments), sort_keys=True, default=str)


def _latest_change(records: Sequence[ToolRecord]) -> int | None:
    """Highest workspace revision produced by a change (not by a validation) among all session records."""
    rejected = (ToolStatus.INVALID, ToolStatus.DENIED)
    bumps = [
        r.revision_after.counter
        for r in records
        if (r.result.status is ToolStatus.OK and r.result.mutated_paths)
        or (r.bumped and r.result.status not in rejected and acceptance.validation_verdict(r) is None)
    ]
    return max(bumps) if bumps else None


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


INTERRUPTED = "[cancelled] not completed: session was interrupted"


def repair_history(history: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Answer every assistant tool call that has no tool message (e.g. the process was killed mid-run).

    Placeholders go right after the call's block, before the next non-tool message, so pairing holds.
    """
    repaired: list[ChatMessage] = []
    pending: list[ToolCall] = []

    def flush() -> None:
        repaired.extend(
            ChatMessage(role="tool", content=INTERRUPTED, tool_call_id=c.id, name=c.name) for c in pending
        )
        pending.clear()

    for message in history:
        if message.role == "tool":
            match = next((c for c in pending if c.id == message.tool_call_id), None)
            if match is not None:
                pending.remove(match)
        else:
            flush()
            if message.role == "assistant":
                pending.extend(message.tool_calls)
        repaired.append(message)
    flush()
    return repaired


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
        completion_reserve_tokens: int | None = None,
        auto_verify: int = 2,
        soul: str = "off",
        role: str = "code",
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
        self.completion_reserve_tokens = completion_reserve_tokens
        self.auto_verify = max(0, auto_verify)  # max engine-run validations per run when the model skips them
        self.soul_mode = soul
        self._role: Role = get_role(role)
        self._apply_role()
        self.last_answer = ""
        self.instructions: ProjectInstructions = ProjectInstructions()
        self.soul: SoulText = SoulText()
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

    @property
    def role(self) -> Role:
        return self._role

    @property
    def role_name(self) -> str:
        return self._role.name

    @property
    def role_label(self) -> str:
        return self._role.status_label()

    def set_role(self, name: str) -> None:
        """Switch the active role: re-gate the runtime (remove-only) and re-render each session's prefix.

        The step budget and recorded history are never reset, so a switch cannot launder work already done.
        """
        self._role = get_role(name)
        self._apply_role()
        for session in self._sessions.values():
            session.messages[0] = ChatMessage(
                role="system",
                content=build_system_prompt(
                    session.facts, session.instructions, session.soul, self._role.prompt
                ),
            )

    async def run(self, session_id: str, user_text: str, *, intent: Intent | None = None) -> AcceptanceResult:
        """Run one user message. ``intent`` (e.g. from ``--expect``) overrides the deterministic classifier."""
        session = self._session(session_id)
        run = _Run(uuid.uuid4().hex[:12], time.monotonic(), intent=classify(user_text, intent))
        if self._role.read_only:  # a read-only role can only inspect/answer; never a change request
            run.intent = Intent.QUESTION
        session.state.intent = run.intent.value
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
        history = self._load_history(session_id)
        state = TaskState.from_dict(store.get_state(session_id, "task_state") if store else None)
        facts = store.get_state(session_id, "project_facts") if store else None
        if not facts:
            facts = project_facts(self.project_root)
            if store:
                store.set_state(session_id, "project_facts", facts)
        instructions = session_instructions(store, session_id, self.project_root)
        self.instructions = instructions
        soul = session_soul(store, session_id, self.soul_mode)
        self.soul = soul
        vault = OutputVault(session_id, store)
        messages = [
            ChatMessage(role="system", content=build_system_prompt(facts, instructions, soul, self._role.prompt)),
            ChatMessage(role="system", content=state.render()),
            *history,
        ]
        reserve = self.completion_reserve_tokens
        if reserve is None:
            reserve = completion_reserve(self.budget.context_tokens)
        context = ContextManager(self.budget.context_tokens, vault, self.estimator, reserve_tokens=reserve)
        saved = (store.get_state(session_id, COMPACTION_STATE) if store else None) or {}
        context.summary = SummaryData.from_dict(saved.get("summary"))
        session = _Session(
            session_id,
            messages,
            state,
            vault,
            context,
            facts=dict(facts),
            instructions=instructions,
            soul=soul,
            compactions=int(saved.get("compactions", 0)),
        )
        self._sessions[session_id] = session
        return session

    def _load_history(self, session_id: str) -> list[ChatMessage]:
        store = self.store
        if store is None:
            return []
        history = store.messages(session_id)
        repaired = repair_history(history)
        if len(repaired) == len(history):
            return history
        if repaired[: len(history)] == history:
            for message in repaired[len(history) :]:
                store.append_message(session_id, message)
        else:
            replace_messages = getattr(store, "replace_messages", None)
            if callable(replace_messages):
                replace_messages(session_id, repaired)
        return repaired

    def _role_allows(self, name: str) -> bool:
        return self._role.tools is None or name in self._role.tools

    def _apply_role(self) -> None:
        setter = getattr(self.tools, "set_role", None)
        if callable(setter):
            setter(self._role.name, self._role.tools)

    def _effective_steps(self) -> int | None:
        limits = [n for n in (self.budget.max_steps, self._role.step_budget) if n is not None]
        return min(limits) if limits else None

    def _specs(self) -> list[ToolSpec]:
        allowed = self._role.tools
        specs: list[ToolSpec] = []
        for spec in self.tools.specs():
            if allowed is not None and spec.name not in allowed:
                continue
            props = dict(spec.parameters.get("properties", {}))
            if spec.name in VALIDATABLE_TOOLS and VALIDATION_ARG not in props:
                props[VALIDATION_ARG] = {
                    "type": "boolean",
                    "description": "true only when this command runs the project's tests with a known runner "
                    "(pytest, unittest, npm test, node --test, cargo test, go test); other commands never validate",
                }
                spec = replace(spec, parameters={**spec.parameters, "properties": props})
                self._augmented.add(spec.name)
            specs.append(spec)
        if all(s.name != SHOW_OUTPUT for s in specs) and (allowed is None or SHOW_OUTPUT in allowed):
            specs.append(SHOW_OUTPUT_SPEC)
        if all(s.name != DECLARE_NO_CHANGE for s in specs) and (allowed is None or DECLARE_NO_CHANGE in allowed):
            specs.append(DECLARE_NO_CHANGE_SPEC)
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
        step_limit = self._effective_steps()
        while True:
            if step_limit is not None and run.steps >= step_limit:
                return "budget", f"step budget exhausted ({step_limit} model calls)"
            if budget.max_tokens is not None and run.tokens >= budget.max_tokens:
                return "budget", f"token budget exhausted ({run.tokens}/{budget.max_tokens})"
            self._maybe_compact(session, run, specs)
            if not session.context.fits(session.messages, specs):
                return "budget", CONTEXT_EXHAUSTED
            message, done_reason = await self._model_turn(session, run, specs)
            run.steps += 1
            self._append(session, message)
            if message.tool_calls and done_reason in (*TRUNCATED, "cancelled"):
                why = "run was stopped" if done_reason == "cancelled" else f"model output was truncated ({done_reason})"
                for call in message.tool_calls:
                    self._append(session, self._placeholder(call, "cancelled", f"not executed: {why}"))
                run.prose = message.content
                if done_reason == "cancelled":
                    return "cancelled", ""
                return "model_error", f"model stopped early ({done_reason}); its tool calls were not executed"
            if not message.tool_calls:
                run.prose = message.content
                if done_reason == "cancelled":
                    return "cancelled", ""
                if done_reason in TRUNCATED:
                    return "model_error", f"model stopped early ({done_reason})"
                if await self._auto_verify(session, run, by_name):
                    continue
                return "final", ""
            stop = await self._run_calls(session, run, message.tool_calls, by_name)
            if stop is not None:
                return stop

    async def _auto_verify(self, session: _Session, run: _Run, by_name: Mapping[str, ToolSpec]) -> bool:
        """The model finished after changing files without validating: run the tests ourselves.

        Evidence comes from the real tool record (same policy, sandbox and runner parsing as a model call), so this
        can only move UNVALIDATED to COMPLETED or INCOMPLETE. Returns True when the tests failed and the model gets
        another turn to repair; the last allowed attempt never loops.
        """
        if run.auto_verified >= self.auto_verify or AUTO_VERIFY_TOOL not in by_name:
            return False
        if acceptance.evaluate(run.records, intent=run.intent).outcome is not Outcome.UNVALIDATED:
            return False
        run.auto_verified += 1
        call = ToolCall(f"auto-verify-{run.run_id}-{run.auto_verified}", AUTO_VERIFY_TOOL, {})
        self._append(session, ChatMessage(role="assistant", content=AUTO_VERIFY_TEXT, tool_calls=[call]))
        await self._run_calls(session, run, [call], by_name)
        record = next((r for r in reversed(run.records) if r.call.id == call.id), None)
        if record is None or acceptance.validation_verdict(record) != "fail":
            return False
        return run.auto_verified < self.auto_verify

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
            return spec.mutates if spec is not None else call.name not in ENGINE_TOOLS

        answered: set[int] = set()  # indexes into calls (ids are not guaranteed unique)
        tasks: list[asyncio.Task[ToolRecord]] = []

        def answer(index: int, record: ToolRecord) -> None:
            run.tool_calls += 1
            answered.add(index)
            self._append(session, self._tool_message(session, run, record))

        offset = 0
        try:
            for group in _groups(runnable, mutating):
                if len(group) == 1:
                    answer(offset, await self._exec_one(session, run, group[0], mutating(group[0])))
                else:
                    tasks = []
                    async with asyncio.TaskGroup() as tg:
                        tasks = [tg.create_task(self._exec_one(session, run, c, False)) for c in group]
                    for i, task in enumerate(tasks):
                        answer(offset + i, task.result())
                    tasks = []
                offset += len(group)
        except asyncio.CancelledError:
            # Calls of the interrupted concurrent group that did finish keep their real results.
            for i, task in enumerate(tasks):
                if task.done() and not task.cancelled() and task.exception() is None:
                    answer(offset + i, task.result())
            for index, call in enumerate(calls):
                if index not in answered:
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
        if not self._role_allows(call.name):  # enforced here too, so a fake/loose runtime cannot bypass the role
            result = ToolResult(
                call.id, call.name, ToolStatus.DENIED, f"{call.name} is not allowed in {self._role.name} mode"
            )
            mutating, validation = False, False
        elif call.name == SHOW_OUTPUT:
            result = run_show_output(session.vault, call, self.budget.max_output_chars)
        elif call.name == DECLARE_NO_CHANGE:
            result = declare_no_change_result(call.arguments, call.id)
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
        compaction = session.context.compact(
            session.messages,
            specs,
            protect=protect,
            records=session.records,
            latest_change=_latest_change(session.records),
        )
        if compaction is None:  # nothing left to drop: keep the prefix instead of churning the pinned state
            return
        session.compactions += 1
        current = acceptance.evaluate(run.records, intent=run.intent)
        session.state.validation = acceptance.validation_status(session.records)
        session.state.obligations = acceptance.obligations(run.records, current) if run.records else []
        session.messages[1] = ChatMessage(role="system", content=session.state.render())
        if self.store is not None:
            replace_messages = getattr(self.store, "replace_messages", None)
            if callable(replace_messages):
                replace_messages(session.session_id, session.messages[2:])
            self.store.set_state(session.session_id, "task_state", session.state.to_dict())
            self.store.set_state(
                session.session_id,
                COMPACTION_STATE,
                {"compactions": session.compactions, "summary": session.context.summary.to_dict()},
            )
        data = {
            "elided": compaction.elided,
            "handles": list(compaction.handles),
            "tokens_before": compaction.tokens_before,
            "tokens_after": compaction.tokens_after,
            "compaction": session.compactions,
            "stage": compaction.stage,
            "dropped_groups": compaction.dropped_groups,
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
            self._transcript(session, message)

    def _transcript(self, session: _Session, message: ChatMessage) -> None:
        """Display history for /resume: separate from (and never rewritten by) model-history compaction."""
        append = getattr(self.store, "append_transcript", None)
        if not callable(append):
            return
        meta: dict[str, Any] = {}
        text = message.content or ""
        if message.role == "assistant":
            if message.tool_calls:
                meta["tool_calls"] = [call.name for call in message.tool_calls]
            if message.thinking:
                meta["thinking"] = message.thinking[:TRANSCRIPT_TEXT_CHARS]
            if not text and not meta:
                return
        elif message.role == "tool":
            meta.update(tool=message.name, call_id=message.tool_call_id)
            text = _first_line(text)
        append(session.session_id, message.role, text[:TRANSCRIPT_TEXT_CHARS], meta)

    def _finish(self, session: _Session, run: _Run, stop: StopReason, detail: str) -> AcceptanceResult:
        result = acceptance.evaluate(run.records, stop, detail, intent=run.intent)
        session.state.validation = acceptance.validation_status(session.records)
        session.state.obligations = acceptance.obligations(run.records, result)
        if self.store is not None:
            self.store.set_state(session.session_id, "task_state", session.state.to_dict())
        prose = run.prose.strip()
        summary = acceptance.render_summary(run.records, result)
        self.last_answer = f"{prose}\n\n{summary}" if prose else summary
        append = getattr(self.store, "append_transcript", None) if self.store is not None else None
        if callable(append):
            append(session.session_id, "assistant", summary[:TRANSCRIPT_TEXT_CHARS], {"outcome": result.outcome.value})
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
