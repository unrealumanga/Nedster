"""Task model for the todo CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Task:
    id: int
    title: str
    done: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(id=int(data["id"]), title=str(data["title"]), done=bool(data.get("done", False)))
