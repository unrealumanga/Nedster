from __future__ import annotations

import pytest

from ironclad.tool_runtime.schema import validate

OBJECT = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "count": {"type": "integer", "minimum": 1, "maximum": 10},
        "flag": {"type": "boolean"},
        "path": {"type": "string", "maxLength": 5},
        "mode": {"enum": ["r", "w"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "inner": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
    },
    "required": ["content"],
    "additionalProperties": False,
}


def test_valid_arguments_pass():
    args = {"content": "hi", "count": 3, "flag": True, "tags": ["a", "b"], "mode": "r", "inner": {"x": 1}}
    assert validate(OBJECT, args) == []
    assert validate(OBJECT, {"content": ""}) == []


def test_missing_required_field():
    assert validate(OBJECT, {}) == ["missing required field 'content'"]


def test_missing_nested_required_field():
    assert validate(OBJECT, {"content": "x", "inner": {}}) == ["missing required field 'inner.x'"]


def test_type_errors():
    assert validate(OBJECT, {"content": 3}) == ["'content' must be a string"]
    assert validate(OBJECT, {"content": "x", "flag": 1}) == ["'flag' must be a boolean"]
    assert validate(OBJECT, {"content": "x", "tags": "no"}) == ["'tags' must be an array"]
    assert validate(OBJECT, {"content": "x", "inner": []}) == ["'inner' must be an object"]


def test_integer_rejects_bool_and_float():
    assert validate(OBJECT, {"content": "x", "count": True}) == ["'count' must be an integer"]
    assert validate(OBJECT, {"content": "x", "count": 1.5}) == ["'count' must be an integer"]


def test_array_items_are_checked_with_index():
    assert validate(OBJECT, {"content": "x", "tags": ["a", 2]}) == ["'tags[1]' must be a string"]


def test_additional_properties_false():
    assert validate(OBJECT, {"content": "x", "nope": 1}) == ["unexpected field 'nope'"]


def test_enum_minimum_maximum_maxlength():
    assert validate(OBJECT, {"content": "x", "mode": "z"})[0].startswith("'mode' must be one of:")
    assert validate(OBJECT, {"content": "x", "count": 0}) == ["'count' must be >= 1"]
    assert validate(OBJECT, {"content": "x", "count": 11}) == ["'count' must be <= 10"]
    assert validate(OBJECT, {"content": "x", "path": "toolong"}) == ["'path' must be at most 5 characters long"]
    short_list = {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string", "maxLength": 5}}}}
    assert validate(short_list, {"tags": ["toolong"]}) == ["'tags[0]' must be at most 5 characters long"]


def test_arguments_must_be_an_object():
    assert validate(OBJECT, "nope") == ["arguments must be an object"]
    assert validate(OBJECT, None) == ["arguments must be an object"]


def test_additional_properties_true_is_allowed():
    schema = {"type": "object", "properties": {}, "additionalProperties": True}
    assert validate(schema, {"anything": 1}) == []


@pytest.mark.parametrize(
    "schema",
    [
        {},
        {"type": "unknown-kind"},
        {"type": "string", "required": "content"},
        {"type": "object", "properties": ["a", "b"]},
        {"type": "object", "properties": {"a": "not-a-mapping"}},
        {"type": "array", "items": 7},
        {"type": "integer", "minimum": "one"},
        {"enum": "not-a-list"},
        {"type": "string", "maxLength": "5"},
    ],
)
def test_malformed_schemas_never_raise(schema):
    assert isinstance(validate(schema, {"a": 1}), list)
    assert isinstance(validate(schema, ["x"]), list)


def test_errors_are_readably_joined():
    errors = validate(OBJECT, {"count": 0, "extra": True})
    assert len(errors) == 3
    assert all(isinstance(item, str) and item for item in errors)
