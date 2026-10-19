"""What a call costs: its billed tokens priced at the model's per-token rates.

ADR `research/record-generation-design` records why spend is priced from `PRICES` rather than read
from the API, which reports tokens and no price.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from bko.generate.errors import GenerationError
from bko.generate.messages import Usage


@dataclass(frozen=True, slots=True)
class Price:
    """A model's prices, in USD per million tokens.

    Attributes:
        input: Uncached input tokens.
        output: Output tokens, thinking included.
        cache_write: Input tokens written to the 5-minute cache.
        cache_read: Input tokens read from the cache.
    """

    input: float
    output: float
    cache_write: float
    cache_read: float


PRICES: Final = MappingProxyType(
    {"claude-opus-5-5": Price(input=4.0, output=20.0, cache_write=5.0, cache_read=0.2)}
)
"""Each priced model's rates, read 2026-10-18 from
`platform.claude.com/docs/en/about-claude/pricing.md` (Model pricing, Claude Opus 5.5)."""


def price_of(model: str) -> Price:
    """Return `model`'s prices.

    Args:
        model: The model id.

    Returns:
        Its rates from `PRICES`.

    Raises:
        GenerationError: If `PRICES` holds no rates for `model`.
    """
    try:
        return PRICES[model]
    except KeyError:
        msg = f"no price for model {model}: add its rates to bko.generate.spend.PRICES"
        raise GenerationError(msg) from None


def cost(usage: Usage, model: str) -> float:
    """Return what `usage` costs on `model`, in USD.

    `output_tokens` already counts thinking, so thinking is never added on top.

    Args:
        usage: One call's billed tokens.
        model: The model the call is priced at.

    Returns:
        Each token kind times its rate, summed.

    Raises:
        GenerationError: As `price_of`.
    """
    price = price_of(model)
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_creation_input_tokens * price.cache_write
        + usage.cache_read_input_tokens * price.cache_read
    ) / 1e6
