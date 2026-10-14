from __future__ import annotations

from pathlib import Path


def check(root: Path, answer: str) -> tuple[bool, str]:
    low = answer.lower()
    if "functools" in low or "lru_cache" in low or "cache" in low and "dict" in low:
        return True, "answer proposes a caching mechanism"
    return False, "answer must mention functools/lru_cache or a dict cache"
