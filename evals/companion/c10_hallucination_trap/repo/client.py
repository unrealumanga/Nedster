"""Thin API client with a fixed retry pause."""

from __future__ import annotations

import time
import urllib.request

from settings import load_settings

RETRY_PAUSE_SECONDS = 2


def fetch(path: str) -> bytes:
    cfg = load_settings()
    url = cfg["base_url"] + path
    last_error = None
    for _ in range(cfg["retries"] + 1):
        try:
            with urllib.request.urlopen(url, timeout=cfg["timeout"]) as response:
                return response.read()
        except OSError as exc:
            last_error = exc
            time.sleep(RETRY_PAUSE_SECONDS)
    raise last_error
