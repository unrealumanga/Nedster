"""Built-in role presets: which tools a run may use and how it should work.

A role can only *remove* tools from the runtime; it never grants a permission the policy withholds. The
allowed set is enforced at dispatch (the composite runtime and the engine both refuse a call the role
excludes), not merely described in the prompt. The prompt fragment is fixed text appended after the
instructions, so it is part of the stable per-session prefix and can never change tool permissions.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_ROLE",
    "READ_ONLY_TOOLS",
    "ROLES",
    "Role",
    "get_role",
    "list_roles",
    "role_names",
]

DEFAULT_ROLE = "code"
READ_ONLY_MAX_STEPS = 24  # a read-only role never needs more model calls than this

# Read-only tool names across the whole runtime (files, search, git diffs, engine output handles).
READ_ONLY_TOOLS = frozenset(
    {
        "read_file",
        "list_files",
        "search_text",
        "git_diff",
        "git_status",
        "git_log",
        "git_show",
        "show_output",
    }
)


@dataclass(frozen=True, slots=True)
class Role:
    """One workflow preset. ``tools`` is ``None`` when the role adds no restriction beyond the policy.

    A non-``None`` ``tools`` is intersected with what exists; it is a hard whitelist for that run. The
    fragment reaches the prompt after the fixed instructions and never alters policy or sandbox settings.
    """

    name: str
    label: str
    description: str
    prompt: str
    tools: frozenset[str] | None = None  # None = every tool the policy allows
    step_budget: int | None = None  # extra ceiling on model calls for this run

    @property
    def read_only(self) -> bool:
        return self.tools is not None

    def status_label(self) -> str:
        return f"{self.label} · read-only" if self.read_only else self.label


ROLES: dict[str, Role] = {
    "code": Role(
        name="code",
        label="CODE",
        description="Read, edit, run and validate (default).",
        prompt=(
            "Mode: code. You may use every tool the policy allows. Make the smallest change that satisfies the "
            "request, then validate it with run_tests."
        ),
    ),
    "plan": Role(
        name="plan",
        label="PLAN",
        description="Read-only: investigate, then return a numbered plan; never edits or runs commands.",
        prompt=(
            "Mode: plan. Read-only: you may inspect the project but you cannot edit files or run commands; such "
            "calls are denied. Investigate what the change needs, then finish with a numbered plan only. Each step "
            "names the files it touches, the change to make, and the test command that would verify it. Do not "
            "write code."
        ),
        tools=READ_ONLY_TOOLS,
        step_budget=READ_ONLY_MAX_STEPS,
    ),
    "review": Role(
        name="review",
        label="REVIEW",
        description="Read-only: review the code and report findings with file:line evidence and uncertainty.",
        prompt=(
            "Mode: review. Read-only: you may inspect the project but you cannot edit files or run commands. Review "
            "the code you actually read and report findings, each citing file:line and the evidence for it, and "
            "stating your uncertainty. Change nothing."
        ),
        tools=READ_ONLY_TOOLS,
        step_budget=READ_ONLY_MAX_STEPS,
    ),
    "explain": Role(
        name="explain",
        label="EXPLAIN",
        description="Read-only: inspect the code first, then give a grounded answer and say what you did not verify.",
        prompt=(
            "Mode: explain. Read-only: inspect the project before answering; you cannot edit files or run commands. "
            "Read the relevant files first and base the answer on what the tools actually showed. Say what you did "
            "not verify. Change nothing."
        ),
        tools=READ_ONLY_TOOLS,
        step_budget=READ_ONLY_MAX_STEPS,
    ),
}


def role_names() -> tuple[str, ...]:
    return tuple(ROLES)


def list_roles() -> tuple[Role, ...]:
    return tuple(ROLES.values())


def get_role(name: str | None) -> Role:
    """A ``Role`` for ``name``; ``None``/empty -> the default. Raises ``ValueError`` on an unknown name."""
    key = (name or "").strip().lower()
    if not key:
        return ROLES[DEFAULT_ROLE]
    if key not in ROLES:
        raise ValueError(f"unknown role {name!r}; choose one of {', '.join(role_names())}")
    return ROLES[key]
