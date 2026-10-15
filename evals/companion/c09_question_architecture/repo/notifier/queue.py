"""A tiny in-process job queue between the web thread and the worker."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class Job:
    channel: str
    alert: object
    attempts: int = 0


class JobQueue:
    def __init__(self):
        self._jobs: deque[Job] = deque()

    def enqueue(self, job: Job) -> None:
        self._jobs.append(job)

    def pop(self) -> Job | None:
        return self._jobs.popleft() if self._jobs else None

    def __len__(self) -> int:
        return len(self._jobs)


QUEUE = JobQueue()
