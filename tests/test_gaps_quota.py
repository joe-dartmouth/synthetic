"""`bko.gaps.quota`: coverage, expected size and the recommended count, on hand-computed fixtures.

ADR `research/share-based-quota` records the rules.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pandas as pd
import pytest

from bko.datasets.uci import UCI_FRAME
from bko.gaps import quota as quota_module
from bko.gaps.observe import Cell
from bko.gaps.quota import (
    MULTIPLES,
    QuotaRule,
    asks,
    coverage,
    expected_size,
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
    assert recommended(40, 29, 5) == 0


def test_the_multiples_are_coverage_times_2_4_and_8():
    assert dict(MULTIPLES) == {
        QuotaRule.COVERAGE: 1,
        QuotaRule.COVERAGE_2: 2,
        QuotaRule.COVERAGE_4: 4,
        QuotaRule.COVERAGE_8: 8,
    }
    train = _train()
    cell = ("<55", "male", 1)
    base = asks(train, [cell], QuotaRule.COVERAGE, name="t", rows=ROWS, frame=UCI_FRAME)[cell]
    assert base.coverage > 0
    for rule, times in MULTIPLES.items():
        found = asks(train, [cell], rule, name="t", rows=ROWS, frame=UCI_FRAME)[cell]
        assert found.coverage == base.coverage
        assert found.asked == max(0, times * base.coverage - found.count)


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


def test_a_cell_of_no_record_gets_a_finite_count_from_each_function():
    train, cell = _train(), ("<55", "female", 1)
    for rule in QuotaRule:
        if rule is QuotaRule.FILL_TO:
            continue
        found = asks(train, [cell], rule, name="t", rows=ROWS, frame=UCI_FRAME)[cell]
        assert found.count == 0
        # n(<55, female, 0) = 4, N1 = 3, N0 = 5: 4 * 3.5 / 5.5 = 2.55, rounded to 3.
        assert found.expected == 3
        # The label group's shares: angina in 6 of 19 label-1 records, stenosis in all 19.
        assert found.coverage == coverage({"HP:9000003": 6 / 19, "MONDO:9000004": 1.0})
        assert all(math.isfinite(value) for value in (found.coverage, found.asked))
    assert recommended(0, coverage({}), expected_size(cell, {cell: 0})) == 0


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
