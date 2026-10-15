"""In-memory LRU cache with optional per-entry TTL. See SPEC.md."""

from __future__ import annotations

import time


class TTLCache:
    def __init__(self, capacity, default_ttl=None, clock=time.monotonic):
        raise NotImplementedError

    def put(self, key, value, ttl=None):
        raise NotImplementedError

    def get(self, key, default=None):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def delete(self, key):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def purge_expired(self):
        raise NotImplementedError

    def stats(self):
        raise NotImplementedError
