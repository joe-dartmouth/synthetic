"""NHAMCS's four-character ICD-10-CM diagnosis tokens and their reversible integer encoding.

The ED file writes each diagnosis as four characters: an ICD-10-CM code cut to its first four, the
decimal point implied after the third and a missing fourth character dashed (`I509`, `I10-`),
NCHS's own `ZZZ0` and `ZZZ2` to `ZZZ5` for an entry that is no diagnosis, and `BLANK` for an
empty field. A frame keeps every column int64, so `token_number` reads a token as a base-37 number
over `DIGITS`, `-` the digit 0, and `BLANK` as `UNKNOWN`; `token_text` reverses it. ADR
`research/nhamcs-frame` records the choice.
"""

from __future__ import annotations

import re
from typing import Final


UNKNOWN: Final = 0
"""The number of `BLANK`, the token of a diagnosis field the visit left empty."""

BLANK: Final = "-9  "
"""An empty diagnosis field as the file writes it: `-9` and two spaces."""

NOT_ICD: Final = frozenset({"ZZZ0", "ZZZ2", "ZZZ3", "ZZZ4", "ZZZ5"})
"""NCHS's codes for an entry that is no diagnosis, such as an illegible one."""

DIGITS: Final = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"""The base-37 digits of a token, in value order: `-` is 0, `0` is 1 and `Z` is 36."""

WIDTH: Final = 4
"""The characters of every token."""

CODE: Final = re.compile(r"[A-Z][0-9][0-9A-Z][0-9A-Z-]")
"""A token that is an ICD-10-CM code cut to four characters: a letter, a digit, a digit or letter,
then a digit, a letter or a dash."""


def is_token(token: str) -> bool:
    """Whether `token` is one the file's diagnosis fields may hold."""
    return token == BLANK or token in NOT_ICD or CODE.fullmatch(token) is not None


def token_number(token: str) -> int:
    """The number that encodes `token`.

    Raises:
        ValueError: If `token` is not a diagnosis token (`is_token`).
    """
    if not is_token(token):
        msg = f"{token!r} is not an NHAMCS diagnosis token"
        raise ValueError(msg)
    if token == BLANK:
        return UNKNOWN
    number = 0
    for char in token:
        number = number * len(DIGITS) + DIGITS.index(char)
    return number


def token_text(number: int) -> str:
    """The token `number` encodes.

    Raises:
        ValueError: If no token encodes to `number`.
    """
    if number == UNKNOWN:
        return BLANK
    chars: list[str] = []
    rest = number
    for _ in range(WIDTH):
        rest, digit = divmod(rest, len(DIGITS))
        chars.append(DIGITS[digit])
    token = "".join(reversed(chars))
    if rest or number < 0 or not is_token(token):
        msg = f"{number} is not an encoded NHAMCS diagnosis token"
        raise ValueError(msg)
    return token
