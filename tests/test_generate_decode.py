"""`bko.generate.decode`: a reply's items in order, then one placeholder per asked record short."""

from __future__ import annotations

import pytest

from bko.generate import decode as decode_module
from bko.generate.decode import Reason, Undecoded, decode
from bko.generate.messages import SETTINGS, Reply, Usage
from tests._generate_fixtures import records_text


def _reply(text: str | None, *, stop_reason: str | None = "end_turn") -> Reply:
    """A reply carrying `text`, stopped for `stop_reason`."""
    return Reply(
        request_id="req_test",
        model=SETTINGS.model,
        stop_reason=stop_reason,
        text=text,
        usage=Usage(1000, 500, 0, 0),
        raw={},
    )


def _records(count: int) -> list[object]:
    """`count` distinct records."""
    return [{"age": 40 + position} for position in range(count)]


def test_k_decoded_records_come_back_in_order_with_no_placeholder():
    records = _records(3)
    assert decode(_reply(records_text(records)), 3) == tuple(records)


def test_fewer_than_asked_are_followed_by_short_placeholders():
    records = _records(2)
    decoded = decode(_reply(records_text(records)), 5)
    assert decoded == (*records, *(Undecoded(Reason.SHORT),) * 3)


def test_more_than_asked_all_pass_through():
    records = _records(4)
    assert decode(_reply(records_text(records)), 2) == tuple(records)


def test_non_object_items_pass_through_unchanged():
    items: list[object] = [1, "a", None, [2], {"age": 50}]
    assert decode(_reply(records_text(items)), 5) == tuple(items)


def test_a_value_is_never_coerced():
    text = '{"records": [{"age": "63", "fbs": true, "chol": NaN, "thalach": Infinity, "x": 1}]}'
    (record,) = decode(_reply(text), 1)
    assert isinstance(record, dict)
    assert record["age"] == "63"
    assert record["fbs"] is True
    assert record["chol"] != record["chol"]
    assert record["thalach"] == float("inf")
    assert record["x"] == 1


def test_no_text_gives_all_placeholders():
    assert decode(_reply(None), 3) == (Undecoded(Reason.NO_TEXT),) * 3


def test_invalid_json_gives_all_placeholders():
    assert decode(_reply('{"records": [{"age": 63}'), 2) == (Undecoded(Reason.NOT_JSON),) * 2


def test_a_fenced_response_is_not_decoded():
    text = f"```json\n{records_text(_records(2))}\n```"
    assert decode(_reply(text), 2) == (Undecoded(Reason.NOT_JSON),) * 2


def test_a_five_thousand_digit_integer_gives_placeholders():
    text = '{"records": [{"age": ' + "9" * 5000 + "}]}"
    assert decode(_reply(text), 2) == (Undecoded(Reason.NOT_JSON),) * 2


def test_deeply_nested_json_gives_placeholders(monkeypatch: pytest.MonkeyPatch):
    """The decoder's `RecursionError` on deep nesting becomes placeholders.

    Python 3.14 bounds recursion by the C stack, so the depth that overflows depends on the
    machine; the decoder is made to raise as an overflow does rather than relying on one depth.
    """

    def overflow(text: str) -> object:
        raise RecursionError(text[:10])

    monkeypatch.setattr(decode_module.json, "loads", overflow)
    text = '{"records": ' + "[" * 10 + "]" * 10 + "}"
    assert decode(_reply(text), 2) == (Undecoded(Reason.NOT_JSON),) * 2


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "null",
        "3",
        '"records"',
        '{"rows": []}',
        '{"records": {"age": 63}}',
        '{"records": null}',
    ],
)
def test_a_root_that_is_not_an_object_with_a_records_list_gives_all_placeholders(text: str):
    assert decode(_reply(text), 2) == (Undecoded(Reason.NOT_RECORDS),) * 2


@pytest.mark.parametrize(
    ("stop_reason", "reason"),
    [
        ("refusal", Reason.REFUSED),
        ("max_tokens", Reason.TRUNCATED),
        ("pause_turn", Reason.STOPPED),
        ("stop_sequence", Reason.STOPPED),
        (None, Reason.STOPPED),
    ],
)
def test_refusal_truncation_and_other_stops_give_all_placeholders_with_their_reason(
    stop_reason: str | None, reason: Reason
):
    reply = _reply(records_text(_records(2)), stop_reason=stop_reason)
    assert decode(reply, 2) == (Undecoded(reason),) * 2
