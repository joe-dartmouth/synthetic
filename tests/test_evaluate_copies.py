"""Added rows that copy a test record are counted, and the split loses every record they copy."""

from __future__ import annotations

import pandas as pd
import pytest

from bko.baselines.records import no_rows, number
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.copies import copies, decontaminated
from tests._evaluate_fixtures import GRADED


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def test_a_row_equal_to_a_test_record_is_counted():
    test = split(GRADED, seed=0, edges=UCI_FRAME.edges).test
    assert copies(number(pd.concat([test.iloc[[0]], test.iloc[[0]]])), test, frame=UCI_FRAME) == 2


def test_a_row_differing_in_one_column_is_not_a_copy():
    test = split(GRADED, seed=0, edges=UCI_FRAME.edges).test
    row = test.iloc[[0]].assign(chol=test["chol"].iloc[0] + 1)
    assert copies(number(row), test, frame=UCI_FRAME) == 0


def test_the_split_loses_every_record_some_frame_copies():
    test = split(GRADED, seed=0, edges=UCI_FRAME.edges).test
    clean = decontaminated(
        test, [number(test.iloc[[2]]), no_rows(frame=UCI_FRAME)], frame=UCI_FRAME
    )
    assert list(clean.index) == [index for index in test.index if index != test.index[2]]


def test_nothing_copied_leaves_the_split_whole():
    test = split(GRADED, seed=0, edges=UCI_FRAME.edges).test
    assert decontaminated(test, [no_rows(frame=UCI_FRAME)], frame=UCI_FRAME).equals(test)
