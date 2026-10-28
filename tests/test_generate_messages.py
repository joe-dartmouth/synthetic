"""The generation request: its settings, its structured-output schema and its body."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest

from bko.datasets.cells import LABEL
from bko.datasets.uci import CATEGORICAL, CODEBOOK, FRACTIONAL, FRAME_COLUMNS
from bko.generate.messages import (
    LEVELS,
    SETTINGS,
    Call,
    Level,
    Settings,
    reaches,
    record_schema,
    text_of,
)


BOUNDS = frozenset(
    {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minItems"}
)
"""Schema keywords that bound a number or an array; structured outputs cannot hold them."""

REJECTED = frozenset({"temperature", "top_p", "top_k", "thinking", "tool_choice", "tools"})
"""Request parameters the request leaves out; `fallbacks` is checked beside them."""


def _objects(node: object) -> Iterator[dict[str, object]]:
    """Yield every dict in a schema, depth first."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from _objects(value)


def _record(sex: int = 0, label: int = 1) -> Any:
    schema: Any = record_schema(sex, label)
    return schema["properties"]["records"]["items"]


def _fields(sex: int = 0, label: int = 1) -> Any:
    return _record(sex, label)["properties"]


@pytest.mark.parametrize("field", ["max_tokens", "per_call"])
def test_settings_refuse_a_non_positive_token_limit_or_chunk(field):
    with pytest.raises(ValueError, match=field):
        replace(SETTINGS, **{field: 0})


def test_each_level_reaches_itself_and_every_level_below_it():
    assert tuple(Level) == LEVELS
    assert reaches(Level.ONTOLOGY, Level.FINDINGS)
    assert reaches(Level.CELL, Level.CELL)
    assert not reaches(Level.COHERENT, Level.FINDINGS)


def test_the_schema_closes_every_object_and_requires_exactly_the_frame_columns():
    schema = record_schema(0, 1)
    objects = [node for node in _objects(schema) if node.get("type") == "object"]
    assert len(objects) == 2
    assert all(node["additionalProperties"] is False for node in objects)
    assert schema["required"] == ["records"]
    record = _record()
    assert record["required"] == list(FRAME_COLUMNS)
    assert list(_fields()) == list(FRAME_COLUMNS)


def test_coded_fields_are_integer_enums_of_their_codebook():
    fields = _fields()
    for name in CATEGORICAL:
        if name == "sex":
            continue
        assert fields[name] == {"type": "integer", "enum": sorted(CODEBOOK[name])}, name


@pytest.mark.parametrize(("sex", "label"), [(0, 1), (1, 0)])
def test_the_cells_sex_and_target_are_pinned_to_one_value(sex, label):
    fields = _fields(sex, label)
    assert fields["sex"] == {"type": "integer", "enum": [sex]}
    assert fields[LABEL] == {"type": "integer", "enum": [label]}


def test_whole_fields_are_integers_and_oldpeak_a_number():
    fields = _fields()
    assert FRACTIONAL == ("oldpeak",)
    assert fields["oldpeak"] == {"type": "number"}
    for name in ("age", "trestbps", "chol", "thalach"):
        assert fields[name] == {"type": "integer"}, name


def test_the_schema_has_no_numeric_bound():
    for node in _objects(record_schema(1, 1)):
        assert not BOUNDS & node.keys(), node


def test_settings_are_summary_at_g4s_effort():
    assert Settings("claude-haiku-5-5", "medium", 16000, 20, Level.SUMMARY) == SETTINGS


def test_call_params_carry_model_tokens_system_user_effort_and_the_cells_schema():
    call = Call(SETTINGS, 1, 0, "the system", "the user")
    assert call.params() == {
        "model": "claude-haiku-5-5",
        "max_tokens": 16000,
        "system": "the system",
        "messages": [{"role": "user", "content": "the user"}],
        "output_config": {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": record_schema(1, 0)},
        },
    }


def test_the_request_carries_no_parameter_the_model_rejects():
    params: Any = Call(SETTINGS, 0, 1, "s", "u").params()
    assert not (REJECTED | {"fallbacks"}) & params.keys()
    assert params["messages"][-1]["role"] == "user"


def test_text_of_joins_text_blocks_and_ignores_thinking():
    raw = {
        "content": [
            {"type": "thinking", "thinking": "x"},
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]
    }
    assert text_of(raw) == "ab"


def test_text_of_gives_none_without_a_text_block():
    assert text_of({"content": []}) is None
