"""Engine-side records and the pinned task state.

``ToolRecord`` is the unit of evidence the acceptance layer reasons about. ``TaskState`` is the
structured snapshot rendered into the pinned message; it is refreshed only at compaction so the
prompt prefix stays stable between compactions.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ironclad.contracts import ToolCall, ToolResult, ToolStatus, WorkspaceRevision

MAX_CONSTRAINTS = 8
MAX_CONSTRAINT_CHARS = 400


@dataclass(slots=True)
class Budget:
    """Hard limits for one ``Engine.run``. ``None`` disables a limit."""

    max_steps: int | None = 40  # model calls
    max_tool_calls: int | None = 120
    max_tokens: int | None = None  # prompt + completion tokens reported by the transport
    deadline_s: float | None = None  # wall-clock seconds from the start of the run
    context_tokens: int = 32_768  # model context window used for compaction decisions
    max_output_chars: int = 12_000  # bound on one tool message


@dataclass(slots=True, frozen=True)
class ToolRecord:
    """One executed (or rejected) call with the workspace revision around it."""

    call: ToolCall
    result: ToolResult
    revision_before: WorkspaceRevision
    revision_after: WorkspaceRevision
    mutating: bool = False  # ToolSpec.mutates of the called tool
    validation: bool = False  # run_command the model flagged as validation

    @property
    def bumped(self) -> bool:
        return self.revision_after.counter > self.revision_before.counter


@dataclass(slots=True)
class TaskState:
    objective: str = ""
    constraints: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)  # path -> latest sha ("" when unknown)
    validation: str = "not run"
    obligations: list[str] = field(default_factory=list)
    updated_ns: int = 0

    def note_user(self, text: str) -> None:
        """First user message is the objective; later ones are kept verbatim as constraints."""
        text = text.strip()
        if not text:
            return
        if not self.objective:
            self.objective = text[:MAX_CONSTRAINT_CHARS]
        else:
            self.constraints.append(text[:MAX_CONSTRAINT_CHARS])
            del self.constraints[:-MAX_CONSTRAINTS]
        self.updated_ns = time.time_ns()

    def note_record(self, record: ToolRecord) -> None:
        result = record.result
        if result.status is ToolStatus.OK and result.mutated_paths:
            sha = str(result.data.get("sha256", "") or "")
            for path in result.mutated_paths:
                self.files[path] = sha
        self.updated_ns = time.time_ns()

    def render(self) -> str:
        lines = ["[task state — maintained by the engine from tool records]"]
        lines.append(f"Objective: {self.objective or '(none yet)'}")
        if self.constraints:
            lines.append("Later user instructions (most recent last):")
            lines.extend(f"- {c}" for c in self.constraints)
        if self.files:
            lines.append("Files changed this session:")
            lines.extend(f"- {p} {sha}".rstrip() for p, sha in sorted(self.files.items()))
        lines.append(f"Validation: {self.validation}")
        if self.obligations:
            lines.append("Open obligations:")
            lines.extend(f"- {o}" for o in self.obligations)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> TaskState:
        if not data:
            return cls()
        return cls(
            objective=str(data.get("objective", "")),
            constraints=[str(c) for c in data.get("constraints", [])],
            files={str(k): str(v) for k, v in dict(data.get("files", {})).items()},
            validation=str(data.get("validation", "not run")),
            obligations=[str(o) for o in data.get("obligations", [])],
            updated_ns=int(data.get("updated_ns", 0)),
        )
