"""`bko.generate.spend`: each token kind priced at its own rate, an unpriced model refused.

The oracles are hand-written from the Haiku 5.5 prices `PRICES` records, read 2026-10-19 from the
pricing page's row for prompts up to 100,000 tokens: 1M input tokens cost 0.10 USD, 1M output 0.50,
1M cache writes 0.125 and 1M cache reads 0.01.
"""

from __future__ import annotations

import pytest

from bko.generate.calls import Batch, Generation
from bko.generate.errors import GenerationError
from bko.generate.messages import SETTINGS, Usage
from bko.generate.spend import PRICES, Price, cost, price_of
from tests._generate_fixtures import CELL, reply


MILLION = 1_000_000
HAIKU = "claude-haiku-5-5"


def test_input_output_and_cache_tokens_are_priced_separately():
    assert cost(Usage(MILLION, 0, 0, 0), HAIKU) == pytest.approx(0.10)
    assert cost(Usage(0, MILLION, 0, 0), HAIKU) == pytest.approx(0.50)
    assert cost(Usage(0, 0, MILLION, 0), HAIKU) == pytest.approx(0.125)
    assert cost(Usage(0, 0, 0, MILLION), HAIKU) == pytest.approx(0.01)
    assert cost(Usage(MILLION, MILLION, MILLION, MILLION), HAIKU) == pytest.approx(0.735)


def test_absent_cache_fields_price_as_zero():
    # `bko.generate.api.reply_of` reads an absent cache count as 0.
    assert cost(Usage(1682, 195, 0, 0), HAIKU) == pytest.approx(0.0002657)


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
            usd=cost(Usage(MILLION, MILLION // 2, 0, 0), HAIKU),
        )
        for index in (1, 2, 3)
    )
    assert Generation(batches).usd == pytest.approx(3 * (0.10 + 0.25))


def test_the_settings_model_is_the_only_priced_model():
    assert SETTINGS.model == HAIKU
    assert set(PRICES) == {HAIKU}
    assert price_of(HAIKU) == Price(input=0.10, output=0.50, cache_write=0.125, cache_read=0.01)
