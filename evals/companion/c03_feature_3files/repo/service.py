"""Storage and operations for tasks. Tasks live in a JSON file."""

from __future__ import annotations

import json
from pathlib import Path

from models import Task


class TaskService:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._tasks: list[Task] = self._load()

    def _load(self) -> list[Task]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        return [Task.from_dict(item) for item in raw]

    def _save(self) -> None:
        self.path.write_text(json.dumps([t.to_dict() for t in self._tasks], indent=2), encoding="utf-8")

    def add(self, title: str) -> Task:
        title = title.strip()
        if not title:
            raise ValueError("title must not be empty")
        next_id = max((t.id for t in self._tasks), default=0) + 1
        task = Task(id=next_id, title=title)
        self._tasks.append(task)
        self._save()
        return task

    def complete(self, task_id: int) -> Task:
        for task in self._tasks:
            if task.id == task_id:
                task.done = True
                self._save()
                return task
        raise KeyError(f"no task with id {task_id}")

    def list(self, include_done: bool = True) -> list[Task]:
        tasks = [t for t in self._tasks if include_done or not t.done]
        return sorted(tasks, key=lambda t: t.id)
