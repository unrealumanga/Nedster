"""Retry helper shared by the senders."""

from __future__ import annotations

import time

from notifier.config import SETTINGS


def retry_with_backoff(action, sleep=time.sleep):
    """Call action() until it stops raising OSError, doubling the wait each time."""
    delay = 1.0
    for attempt in range(SETTINGS["max_retries"] + 1):
        try:
            return action()
        except OSError:
            if attempt == SETTINGS["max_retries"]:
                raise
            sleep(delay)
            delay *= 2
