"""Fidelity: identical rows score 1, every column pair counts, and nothing reaches stdout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bko.datasets.uci import UCI_FRAME
from bko.evaluate.fidelity import MEASURES, fidelity, quality_report
from bko.evaluate.scores import Measure
from tests._baselines_fixtures import full_records


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

FRAME = full_records(8)


def test_identical_rows_score_one_and_print_nothing(capsys):
    scores = fidelity(FRAME, FRAME.sample(frac=1.0, random_state=0), frame=UCI_FRAME)
    assert scores == pytest.approx(
        {Measure.QUALITY: 1.0, Measure.COLUMN_SHAPES: 1.0, Measure.COLUMN_PAIR_TRENDS: 1.0}
    )
    assert tuple(scores) == MEASURES
    assert capsys.readouterr().out == ""


def test_a_column_outside_the_real_range_loses_only_its_shape():
    shifted = FRAME.assign(chol=FRAME["chol"] + 1000)
    assert fidelity(FRAME, shifted, frame=UCI_FRAME)[Measure.COLUMN_SHAPES] == pytest.approx(
        13 / 14
    )


def test_every_column_pair_counts():
    pairs = quality_report(FRAME, FRAME, frame=UCI_FRAME).get_details("Column Pair Trends")
    assert int(pairs["Score"].notna().sum()) == 14 * 13 // 2


def test_a_table_over_fifty_thousand_rows_scores_the_same_every_time():
    """SDMetrics subsamples column pairs above 50,000 rows unless told not to; nothing may."""
    large = pd.concat([FRAME] * 6251, ignore_index=True)
    synthetic = large.sample(frac=1.0, random_state=1).reset_index(drop=True)
    synthetic.loc[: len(synthetic) // 3, "sex"] = 1 - synthetic.loc[: len(synthetic) // 3, "sex"]
    readings = []
    for seed in (0, 1):
        np.random.seed(seed)  # the global state SDMetrics' unseeded subsample draws from
        readings.append(fidelity(large, synthetic, frame=UCI_FRAME))
    assert len(large) > 50_000
    assert readings[0] == readings[1]
