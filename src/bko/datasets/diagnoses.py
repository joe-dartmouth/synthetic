"""Diabetes 130's ICD-9 diagnosis tokens and their reversible integer encoding.

A token is `?` (no diagnosis), a numeric code such as `8` or `250.41`, a V code such as `V45.81`
or an E code such as `E888.1`, as `diag_1`, `diag_2` and `diag_3` write them. The encoding keeps
a frame all int64: `?` is `UNKNOWN`; a numeric code is 1_000_000 + whole x 1000 + tail, a V code
2_000_000 + the same, an E code 3_000_000 + the same, where the tail is 0 with no decimal,
100 + d with one decimal digit d, and 200 + dd with two digits dd. ADR
`research/diabetes-diagnosis-codes` records the choice.
"""

from __future__ import annotations

import re
from typing import Final


UNKNOWN: Final = 0
"""The number of `?`, the token of a diagnosis field the encounter left empty."""

NUMERIC_CODE: Final = r"([0-9]{1,3})(?:\.([0-9]{1,2}))?"
"""A numeric ICD-9 code: its whole part and, optionally, one or two decimal digits."""

DIAGNOSIS_PATTERN: Final = (
    rf"\?|{NUMERIC_CODE}|V[0-9]{{2}}(?:\.[0-9]{{1,2}})?|E[0-9]{{3}}(?:\.[0-9])?"
)
"""Every token a diagnosis field may hold, matched whole."""

_PARTS: Final = re.compile(r"([VE]?)([0-9]+)(?:\.([0-9]+))?")
_NONE: Final = "?"
_KINDS: Final = {"": 1, "V": 2, "E": 3}
_PREFIXES: Final = {kind: prefix for prefix, kind in _KINDS.items()}
_WIDTHS: Final = {2: 2, 3: 3}
"""The digits of a V or E code's whole part, zero-padded."""
_KIND_BASE: Final = 1_000_000


def code_number(token: str) -> int:
    """The number that encodes `token`.

    Raises:
        ValueError: If `token` is not a diagnosis token (`DIAGNOSIS_PATTERN`).
    """
    if token == _NONE:
        return UNKNOWN
    prefix, whole, digits = _split(token)
    tail = 0 if not digits else 100 * len(digits) + int(digits)
    return _KINDS[prefix] * _KIND_BASE + int(whole) * 1000 + tail


def code_text(number: int) -> str:
    """The token `number` encodes; a numeric code's whole part is unpadded, as the archive has it.

    Raises:
        ValueError: If no token encodes to `number`.
    """
    if number == UNKNOWN:
        return _NONE
    kind, rest = divmod(number, _KIND_BASE)
    whole, tail = divmod(rest, 1000)
    width = tail // 100
    prefix = _PREFIXES.get(kind)
    head = str(whole) if kind == 1 else f"{prefix}{whole:0{_WIDTHS.get(kind, 0)}d}"
    text = head if width == 0 else f"{head}.{tail % 100:0{width}d}"
    if prefix is None or not _is_token(text) or code_number(text) != number:
        msg = f"{number} is not an encoded ICD-9 code"
        raise ValueError(msg)
    return text


def canonical(token: str) -> str:
    """`token` as MONDO's `ICD9:` xrefs write it, a numeric code's whole part padded to 3 digits.

    `?`, V and E codes are returned unchanged.

    Raises:
        ValueError: If `token` is not a diagnosis token (`DIAGNOSIS_PATTERN`).
    """
    if token == _NONE:
        return token
    prefix, whole, digits = _split(token)
    head = f"{int(whole):03d}" if not prefix else f"{prefix}{whole}"
    return head if not digits else f"{head}.{digits}"


def _split(token: str) -> tuple[str, str, str]:
    """`token`'s kind prefix, whole part and decimal digits, after checking it."""
    match = _PARTS.fullmatch(token) if _is_token(token) else None
    if match is None:
        msg = f"{token!r} is not an ICD-9 code"
        raise ValueError(msg)
    return match[1], match[2], match[3] or ""


def _is_token(token: str) -> bool:
    return re.fullmatch(DIAGNOSIS_PATTERN, token) is not None
