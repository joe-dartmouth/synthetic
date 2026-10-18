"""`bko.generate.spend`: each token kind priced at its own rate, an unpriced model refused.

The oracles are hand-written from the Opus 5.5 prices `PRICES` records: 1M input tokens cost 4.00
USD, 1M output 20.00, 1M cache writes 5.00 and 1M cache reads 0.20.
"""

from __future__ import annotations

import pytest

from bko.generate.calls import Batch, Generation
from bko.generate.errors import GenerationError
from bko.generate.messages import SETTINGS, Usage
from bko.generate.spend import cost, price_of
from tests._generate_fixtures import CELL, reply


MILLION = 1_000_000


def test_input_output_and_cache_tokens_are_priced_separately():
    assert cost(Usage(MILLION, 0, 0, 0), SETTINGS.model) == pytest.approx(4.00)
    assert cost(Usage(0, MILLION, 0, 0), SETTINGS.model) == pytest.approx(20.00)
    assert cost(Usage(0, 0, MILLION, 0), SETTINGS.model) == pytest.approx(5.00)
    assert cost(Usage(0, 0, 0, MILLION), SETTINGS.model) == pytest.approx(0.20)
    assert cost(Usage(MILLION, MILLION, MILLION, MILLION), SETTINGS.model) == pytest.approx(29.20)


def test_absent_cache_fields_price_as_zero():
    # `bko.generate.api.reply_of` reads an absent cache count as 0.
    assert cost(Usage(1682, 195, 0, 0), SETTINGS.model) == pytest.approx(0.010628)


def test_an_unpriced_model_is_refused():
    with pytest.raises(GenerationError, match="claude-unpriced"):
        price_of("claude-unpriced")
    with pytest.raises(GenerationError, match="claude-unpriced"):
        cost(Usage(1, 1, 0, 0), "claude-unpriced")


def test_spend_sums_over_calls():
    batches = tuple(
        Batch(
            index=index,
            cell=CELL,
            asked=1,
            candidates=({},),
            reply=reply([{}]),
            usd=cost(Usage(MILLION, MILLION // 2, 0, 0), SETTINGS.model),
        )
        for index in (1, 2, 3)
    )
    assert Generation(batches).usd == pytest.approx(3 * (4.00 + 10.00))
