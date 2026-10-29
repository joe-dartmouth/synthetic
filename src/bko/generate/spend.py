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


FREE: Final = Price(input=0.0, output=0.0, cache_write=0.0, cache_read=0.0)
"""An on-device model's rates: it bills nothing."""

PRICES: Final = MappingProxyType(
    {
        "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_write=0.125, cache_read=0.01),
        "gemma-4-26b-a4b-q8_0": FREE,
        "nemotron-3-nano-30b-a3b-q8_0": FREE,
        "olmo-3-7b-instruct-q8_0": FREE,
    }
)
"""Each priced model's rates. Claude Haiku 5.5's were read 2026-10-19 from
`platform.claude.com/docs/en/about-claude/pricing.md` (Model pricing, prompts up to 100,000
tokens; a call's prompt is a few thousand). The local models (`bko.generate.local.LOCAL_MODELS`)
run on this machine and bill nothing."""


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
