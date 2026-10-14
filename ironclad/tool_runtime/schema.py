"""Tiny JSON-Schema subset validator for tool arguments. No external libraries.

Supported keywords: ``type`` (object/string/integer/number/boolean/array),
``required``, ``properties``, ``additionalProperties: false``, ``enum``,
``minimum``, ``maximum``, ``maxLength`` and ``items``. Anything else is ignored.

``validate`` returns a list of human-readable errors (empty when valid) and is
total: malformed schemas never raise, they simply constrain less.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ARRAY_TYPES = (list, tuple)
_ARTICLE = {"object": "an", "string": "a", "integer": "an", "number": "a", "boolean": "a", "array": "an"}


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, _ARRAY_TYPES)
    return True


def _label(path: str) -> str:
    return f"'{path}'" if path else "arguments"


def _field(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _check_object(schema: Mapping[str, Any], value: Mapping[str, Any], path: str, errors: list[str]) -> None:
    required = schema.get("required")
    if isinstance(required, (list, tuple)):
        for name in required:
            if name not in value:
                errors.append(f"missing required field '{_field(path, str(name))}'")
    properties = schema.get("properties")
    known = properties if isinstance(properties, Mapping) else {}
    for name, subschema in known.items():
        if name in value:
            _check(subschema, value[name], _field(path, str(name)), errors)
    if schema.get("additionalProperties") is False:
        for name in value:
            if name not in known:
                errors.append(f"unexpected field '{_field(path, str(name))}'")


def _check(schema: Any, value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(schema, Mapping):
        return
    expected = schema.get("type")
    if isinstance(expected, str) and not _matches_type(value, expected):
        errors.append(f"{_label(path)} must be {_ARTICLE.get(expected, 'a')} {expected}")
        return
    enum = schema.get("enum")
    if isinstance(enum, (list, tuple)) and value not in enum:
        errors.append(f"{_label(path)} must be one of: {', '.join(repr(item) for item in enum)}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{_label(path)} must be >= {minimum}")
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{_label(path)} must be <= {maximum}")
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and isinstance(value, (str, *_ARRAY_TYPES)) and len(value) > max_length:
        errors.append(f"{_label(path)} must be at most {max_length} characters long")
    if isinstance(value, Mapping):
        _check_object(schema, value, path, errors)
    elif isinstance(value, _ARRAY_TYPES):
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                _check(items, item, f"{path}[{index}]" if path else f"[{index}]", errors)


def validate(schema: Mapping[str, Any], args: Mapping[str, Any]) -> list[str]:
    """Return human-readable validation errors for ``args`` against ``schema``."""
    if not isinstance(args, Mapping):
        return ["arguments must be an object"]
    errors: list[str] = []
    _check(schema, args, "", errors)
    return errors


__all__ = ["validate"]
