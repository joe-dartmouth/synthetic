"""The reversible integer encoding of Diabetes 130's ICD-9 diagnosis tokens."""

from __future__ import annotations

import pytest

from bko.datasets.diagnoses import UNKNOWN, canonical, code_number, code_text


@pytest.mark.parametrize(
    ("token", "number"),
    [
        ("?", 0),
        ("428", 1_428_000),
        ("250.41", 1_250_241),
        ("250.4", 1_250_104),
        ("250.04", 1_250_204),
        ("8", 1_008_000),
        ("V45", 2_045_000),
        ("V45.81", 2_045_281),
        ("V57.1", 2_057_101),
        ("E888", 3_888_000),
        ("E888.1", 3_888_101),
    ],
)
def test_code_number_round_trips_every_form(token: str, number: int) -> None:
    assert code_number(token) == number
    assert code_text(number) == token


def test_the_unknown_token_is_zero() -> None:
    assert code_number("?") == UNKNOWN == 0


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("8", "008"),
        ("38.1", "038.1"),
        ("428", "428"),
        ("250.41", "250.41"),
        ("V45", "V45"),
        ("E888.1", "E888.1"),
    ],
)
def test_canonical_pads_the_whole_part_to_three_digits(token: str, expected: str) -> None:
    assert canonical(token) == expected
    assert code_number(canonical(token)) == code_number(token)


@pytest.mark.parametrize(
    "token", ["", "4280", "250.413", "V4", "V45.123", "E88", "E888.12", "X12", " 428", "٤٢٨"]
)
def test_a_token_off_the_pattern_is_refused(token: str) -> None:
    with pytest.raises(ValueError, match="not an ICD-9 code"):
        code_number(token)
    with pytest.raises(ValueError, match="not an ICD-9 code"):
        canonical(token)


@pytest.mark.parametrize(
    "number", [-1, 1, 999_999, 1_428_001, 1_428_110, 2_100_000, 3_888_200, 4_000_000]
)
def test_a_number_the_encoding_never_gives_is_refused(number: int) -> None:
    with pytest.raises(ValueError, match="not an encoded ICD-9 code"):
        code_text(number)
