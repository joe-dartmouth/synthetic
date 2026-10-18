"""The hand-off to `bko.validate`: every asked record arrives, decoded or as a placeholder.

Runs the real `validate` on `decode`'s output, so a change to either side's counting rule fails
here (ADR `research/record-validation-design`, Decision, Input).
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from bko.generate.decode import decode
from bko.generate.messages import SETTINGS, Reply, Usage
from bko.validate.checks import FIELDS, Rejection, validate
from tests._generate_fixtures import GUIDANCE, records_text
from tests._validate_fixtures import BASE


RECORD = {**BASE, "sex": GUIDANCE.sex, "target": GUIDANCE.label}
"""A valid record of the fixture cell."""


def _reply(text: str | None, *, stop_reason: str = "end_turn") -> Reply:
    """A reply carrying `text`, stopped for `stop_reason`."""
    return Reply(
        request_id="req_test",
        model=SETTINGS.model,
        stop_reason=stop_reason,
        text=text,
        usage=Usage(1000, 500, 0, 0),
        raw={},
    )


@pytest.mark.parametrize(("asked", "decoded"), [(5, 5), (5, 3), (20, 0), (2, 0)])
def test_k_of_n_decoded_hands_validation_n_candidates_with_the_rest_failing_fields_alone(
    asked: int, decoded: int
):
    candidates = decode(_reply(records_text([RECORD] * decoded)), asked)
    result = validate(candidates)
    assert result.generated == asked
    assert result.passed == decoded
    assert result.rejected == tuple(Rejection(p, (FIELDS,)) for p in range(decoded, asked))
    assert not any(isinstance(candidate, Mapping) for candidate in candidates[decoded:])


def test_a_refusal_hands_validation_one_failure_per_asked_record():
    result = validate(decode(_reply(records_text([RECORD] * 3), stop_reason="refusal"), 3))
    assert result.generated == 3
    assert result.passed == 0
    assert result.rejected == tuple(Rejection(p, (FIELDS,)) for p in range(3))


def test_over_delivery_counts_every_decoded_item_as_generated():
    result = validate(decode(_reply(records_text([RECORD] * 4)), 2))
    assert result.generated == 4
    assert result.passed == 4
