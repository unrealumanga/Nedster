"""Runtime settings for the sync client, overridable from the environment."""

from __future__ import annotations

import os

DEFAULTS = {
    "base_url": "https://api.example.invalid/v2",
    "timeout": 10.0,
    "retries": 3,
    "page_size": 100,
}


def load_settings() -> dict:
    settings = dict(DEFAULTS)
    if "SYNC_TIMEOUT" in os.environ:
        settings["timeout"] = float(os.environ["SYNC_TIMEOUT"])
    if "SYNC_RETRIES" in os.environ:
        settings["retries"] = int(os.environ["SYNC_RETRIES"])
    return settings
