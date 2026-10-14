"""Decides what to do with a parsed alert and puts the work on the queue."""

from __future__ import annotations

from notifier.dedupe import Deduper
from notifier.queue import QUEUE, Job
from notifier.rules import apply_rules


class Router:
    def __init__(self, queue=QUEUE, deduper=None):
        self.queue = queue
        self.deduper = deduper or Deduper()

    def route(self, alert) -> int:
        """Returns how many jobs were queued."""
        if self.deduper.is_duplicate(alert):
            return 0
        channels = apply_rules(alert)
        for channel in channels:
            self.queue.enqueue(Job(channel=channel, alert=alert))
        return len(channels)


ROUTER = Router()
