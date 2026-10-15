"""Signup form validation."""

from __future__ import annotations


def validate_signup(record: dict) -> list[str]:
    """Return one 'field:reason' string per problem; an empty list means the record is valid."""
    raise NotImplementedError


def handle_signup(record: dict) -> dict:
    problems = validate_signup(record)
    if problems:
        return {"ok": False, "problems": problems}
    return {"ok": True, "user": record["username"]}
