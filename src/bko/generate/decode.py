"""A reply read into candidate records: its items as decoded, then one placeholder per one short.

Every record a call asked for reaches `bko.validate`, either as the item the reply decodes to,
unfiltered and unrepaired, or as one `Undecoded`, which validation rejects under `fields`. Items
decode only when the model stopped on `end_turn`. ADR `research/record-generation-design` records
why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from bko.generate.messages import Reply


class Reason(StrEnum):
    """Why an asked record has no decoded candidate."""

    REFUSED = "refused"
    TRUNCATED = "truncated"
    STOPPED = "stopped"
    NO_TEXT = "no-text"
    NOT_JSON = "not-json"
    NOT_RECORDS = "not-records"
    SHORT = "short"


@dataclass(frozen=True, slots=True)
class Undecoded:
    """One record a call asked for and did not decode; `bko.validate` rejects it under `fields`.

    Attributes:
        reason: Why the record has no decoded candidate.
    """

    reason: Reason


_STOPS: dict[str, Reason] = {"refusal": Reason.REFUSED, "max_tokens": Reason.TRUNCATED}


def decode(reply: Reply, asked: int) -> tuple[object, ...]:
    """The items the reply decodes to, in order, then one `Undecoded` per asked record short.

    Never raises on the reply's content and never coerces an item (ADR
    `research/record-validation-design`, Decision, Input).

    Args:
        reply: One call's reply.
        asked: How many records the call asked for.

    Returns:
        Every decoded item, then `max(0, asked - decoded)` placeholders, each carrying the reason
        the reply fell short.
    """
    items, reason = _items(reply)
    missing = max(0, asked - len(items))
    return (*items, *(Undecoded(reason or Reason.SHORT) for _ in range(missing)))


def _items(reply: Reply) -> tuple[list[object], Reason | None]:
    if reply.stop_reason != "end_turn":
        return [], _STOPS.get(reply.stop_reason or "", Reason.STOPPED)
    if reply.text is None:
        return [], Reason.NO_TEXT
    try:
        payload = json.loads(reply.text)
    except ValueError, RecursionError:
        return [], Reason.NOT_JSON
    items = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return [], Reason.NOT_RECORDS
    return items, None
