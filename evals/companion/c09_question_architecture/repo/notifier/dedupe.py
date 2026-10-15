"""Suppress repeats of the same alert inside a time window."""

from __future__ import annotations

import time

from notifier.config import SETTINGS


class Deduper:
    def __init__(self, clock=time.monotonic):
        self._seen: dict[str, float] = {}
        self._clock = clock

    def is_duplicate(self, alert) -> bool:
        now = self._clock()
        last = self._seen.get(alert.fingerprint)
        if last is not None and now - last < SETTINGS["dedupe_window_seconds"]:
            return True
        self._seen[alert.fingerprint] = now
        return False
