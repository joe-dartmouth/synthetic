"""`bko.gaps.quota`: coverage, expected size and the recommended count, on hand-computed fixtures.

ADR `research/share-based-quota` records the rules.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from bko.datasets.uci import UCI_FRAME
from bko.gaps import quota as quota_module
from bko.gaps.observe import Cell
from bko.gaps.quota import (
    QuotaRule,
    asks,
    coverage,
    expected_size,
    fused_shares,
    quota,
    recommended,
)
from tests._gaps_fixtures import ANGINA, STENOSIS


@pytest.mark.parametrize(
    ("share", "count"),
    [
        # 0.5^29 + 0.9^29 = 0.0471 <= 0.05, and 0.9^28 = 0.0523 alone exceeds it; 0.015 < 0.02.
        ({"a": 0.5, "b": 0.1, "c": 0.015}, 29),
        # 0.98^149 = 0.0494, 0.98^148 = 0.0504.
        ({"a": 0.02}, 149),
        # 3 * 0.7^12 = 0.0415, 3 * 0.7^11 = 0.0593.
        ({"a": 0.3, "b": 0.3, "c": 0.3}, 12),
        # 0.5^5 = 0.03125, 0.5^4 = 0.0625.
        ({"a": 0.5}, 5),
        # A concept of share 1 is never missed after one row: 0.98's 149 decides.
        ({"a": 1.0, "b": 0.02}, 149),
        ({}, 0),
    ],
    ids=["two-of-three", "floor", "three-equal", "half", "certain-beside-floor", "none"],
)
def test_coverage_gives_the_hand_computed_count(share, count):
    assert coverage(share) == count


def test_coverage_of_concepts_all_below_the_floor_is_0():
    assert coverage({"a": 0.019, "b": 0.0}) == 0


def test_a_share_one_ulp_below_the_floor_reads_as_the_floor():
    # A share is the float nearest a ratio of counts, so 0.02's lower neighbour is read as 1/50.
    below = math.nextafter(0.02, 0.0)
    assert below < 0.02
    assert coverage({"a": below}) == coverage({"a": 0.02}) == 149


@pytest.mark.parametrize(
    "share",
    [
        # A miss of exactly 1 - 19/20 = 1/20 meets MISS_CHANCE after one row; read in floats,
        # 1 - 0.95 is 0.05000000000000004 and would take two.
        {"a": 0.95},
        # Two misses of 1/40 sum to exactly 1/20; read in floats, each is above 0.025.
        {"a": 0.975, "b": 0.975},
    ],
    ids=["one-twentieth", "two-fortieths"],
)
def test_coverage_meets_the_miss_chance_exactly_at_the_boundary(share):
    assert coverage(share) == 1


def test_a_nan_share_is_refused():
    with pytest.raises(ValueError, match="NaN"):
        coverage({"a": 0.5, "b": math.nan})


def test_expected_size_scales_the_other_label_by_the_neighbours_odds():
    # 10 * (6 + 1/2) / (12 + 1/2) = 5.2, rounded half up to 5; N1 6 is two other sexes' 4 and 2.
    counts: dict[Cell, int] = {
        ("a", "b", 1): 3,
        ("a", "b", 0): 10,
        ("a", "c", 1): 4,
        ("a", "c", 0): 7,
        ("a", "d", 1): 2,
        ("a", "d", 0): 5,
        ("z", "c", 1): 99,
    }
    assert expected_size(("a", "b", 1), counts) == 5


def test_expected_size_of_an_exact_count_is_that_count():
    # 7 * (1 + 1/2) / (3 + 1/2) = 3 exactly.
    counts: dict[Cell, int] = {
        ("a", "b", 1): 0,
        ("a", "b", 0): 7,
        ("a", "c", 1): 1,
        ("a", "c", 0): 3,
    }
    assert expected_size(("a", "b", 1), counts) == 3


def test_expected_size_of_a_cell_with_no_other_label_is_0():
    counts: dict[Cell, int] = {
        ("a", "b", 1): 4,
        ("a", "b", 0): 0,
        ("a", "c", 1): 6,
        ("a", "c", 0): 12,
    }
    assert expected_size(("a", "b", 1), counts) == 0


def test_expected_size_with_empty_neighbours_reads_the_splits_label_counts():
    # The other sex holds no record, so N1 and N0 are the split's 20 and 180:
    # 10 * (20 + 1/2) / (180 + 1/2) = 1.136, rounded to 1.
    counts: dict[Cell, int] = {
        ("a", "b", 1): 0,
        ("a", "b", 0): 10,
        ("a", "c", 1): 0,
        ("a", "c", 0): 0,
        ("z", "b", 1): 20,
        ("z", "b", 0): 170,
    }
    assert expected_size(("a", "b", 1), counts) == 1


def test_recommended_is_the_larger_count_less_the_cells():
    assert recommended(2, 29, 5) == 27
    assert recommended(2, 5, 29) == 27
    assert recommended(40, 29, 5) == 0


ROWS = (ANGINA, STENOSIS)


def _train() -> pd.DataFrame:
    """UCI-shaped rows: `<55` women of label 1 none, men of label 1 three, `cp` varied."""
    sizes = {
        (40, 0, 0): 4,
        (40, 0, 1): 0,
        (40, 1, 0): 5,
        (40, 1, 1): 3,
        (60, 0, 0): 6,
        (60, 0, 1): 7,
        (60, 1, 0): 2,
        (60, 1, 1): 9,
    }
    rows = [
        {"age": age, "sex": sex, "target": target, "cp": 1 + index % 4}
        for (age, sex, target), size in sizes.items()
        for index in range(size)
    ]
    return pd.DataFrame(rows).astype("int64")


def test_fused_shares_read_each_cell_against_its_own_label_group():
    train = _train()
    young_men = (("<55", "male", 1), ("<55", "male", 0))
    found = fused_shares(train, young_men, name="t", rows=ROWS, frame=UCI_FRAME)
    # Label 1: angina in 6 of 19 records, stenosis in all; the whole split's would be 12 and 19
    # of 36. The cell's 3 records show angina once, weighed against the group's 20:
    # (3 * 1/3 + 20 * 6/19) / 23 = 139/437.
    assert found[young_men[0]] == pytest.approx({"HP:9000003": 139 / 437, "MONDO:9000004": 1.0})
    # Label 0: angina in 6 of 17 records, no stenosis; the cell's 5 show angina twice:
    # (5 * 2/5 + 20 * 6/17) / 25 = 154/425.
    assert found[young_men[1]] == pytest.approx({"HP:9000003": 154 / 425, "MONDO:9000004": 0.0})


@pytest.mark.parametrize(
    ("rule", "asked"),
    [
        # The cell holds 3 records; its shares 139/437 and 1 give coverage 8, as 0.682^8 = 0.047.
        (QuotaRule.COVERAGE, 5),
        (QuotaRule.COVERAGE_2, 13),
        (QuotaRule.COVERAGE_4, 29),
        (QuotaRule.COVERAGE_8, 61),
    ],
)
def test_each_coverage_rule_asks_its_multiple_of_coverage_less_the_count(rule, asked):
    cell = ("<55", "male", 1)
    found = asks(_train(), [cell], rule, name="t", rows=ROWS, frame=UCI_FRAME)[cell]
    assert (found.count, found.coverage, found.asked) == (3, 8, asked)


@pytest.mark.parametrize(
    ("rule", "asked"),
    [
        # Shares 6/19 and 1: 0.684^7 = 0.070 and 0.684^8 = 0.048, so coverage is 8.
        (QuotaRule.COVERAGE, 8),
        (QuotaRule.COVERAGE_2, 16),
        (QuotaRule.COVERAGE_4, 32),
        (QuotaRule.COVERAGE_8, 64),
        (QuotaRule.EXPECTED, 3),
        (QuotaRule.RECOMMENDED, 8),
    ],
)
def test_a_cell_of_no_record_gets_a_count_from_each_function(rule, asked):
    train, cell = _train(), ("<55", "female", 1)
    found = asks(train, [cell], rule, name="t", rows=ROWS, frame=UCI_FRAME)[cell]
    # n(<55, female, 0) = 4, N1 = 3, N0 = 5: 4 * 3.5 / 5.5 = 2.55, rounded to 3.
    assert (found.count, found.coverage, found.expected, found.asked) == (0, 8, 3, asked)


def test_recommended_asks_the_expected_size_where_it_exceeds_coverage():
    # Stenosis alone, at share 1, is shown by the first row: coverage 1 against expected 3.
    cell = ("<55", "female", 1)
    rows = (STENOSIS,)
    found = asks(_train(), [cell], QuotaRule.RECOMMENDED, name="t", rows=rows, frame=UCI_FRAME)
    assert (found[cell].coverage, found[cell].expected, found[cell].asked) == (1, 3, 3)


def test_an_unknown_rule_is_refused():
    cell, unknown = ("<55", "female", 1), cast(QuotaRule, "no-such-rule")
    with pytest.raises(ValueError, match="no-such-rule"):
        asks(_train(), [cell], unknown, name="t", rows=ROWS, frame=UCI_FRAME)


def test_no_cell_asks_nothing_and_reads_no_share(monkeypatch):
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError

    monkeypatch.setattr(quota_module, "fused_shares", refuse)
    assert asks(_train(), [], QuotaRule.COVERAGE, name="t", rows=ROWS, frame=UCI_FRAME) == {}


def test_a_cell_asked_0_leaves_the_quota_and_fill_to_is_refused():
    train, cells = _train(), [("<55", "female", 1), ("<55", "male", 1)]
    found = asks(train, cells, QuotaRule.EXPECTED, name="t", rows=ROWS, frame=UCI_FRAME)
    # Men: n(<55, male, 0) = 5, N1 = 0, N0 = 4: 5 * 0.5 / 4.5 = 0.56, rounded to 1, less 3.
    assert found[cells[1]].expected == 1
    assert found[cells[1]].asked == 0
    assert quota(train, cells, QuotaRule.EXPECTED, name="t", rows=ROWS, frame=UCI_FRAME) == {
        cells[0]: 3
    }
    with pytest.raises(ValueError, match="fill-to"):
        asks(train, cells, QuotaRule.FILL_TO, name="t", rows=ROWS, frame=UCI_FRAME)


def test_the_module_imports_nothing_from_generate_or_evaluate():
    tree = ast.parse(Path(quota_module.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not {name for name in imported if name.startswith(("bko.generate", "bko.evaluate"))}
