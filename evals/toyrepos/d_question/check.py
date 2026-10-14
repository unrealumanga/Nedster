from __future__ import annotations

from pathlib import Path


def check(root: Path, answer: str) -> tuple[bool, str]:
    low = answer.lower()
    if "retries" in low and "3" in low and "debug" in low and "false" in low:
        return True, "answer names the defaults"
    return False, "answer must state {'debug': False, 'retries': 3}"
