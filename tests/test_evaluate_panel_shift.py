"""`panel` regenerates the prior-shift arm, which adds no rows, as it does the controls."""

from __future__ import annotations

import pytest

from bko.evaluate.shift import prior_shift
from tests.test_evaluate_panel import Spy, folder_and_store, paneled
from tests.test_evaluate_shift import ROWS_ARMS, ROWS_RUN


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def test_panel_regenerates_prior_shift(tmp_path, caplog):
    folder, store = folder_and_store(tmp_path, caplog, ROWS_RUN, ROWS_ARMS)
    spies = {name: Spy(runner) for name, runner in ROWS_ARMS.items()}
    code, text = paneled(tmp_path, folder, store, experiment=ROWS_RUN, arms=spies)
    assert code == 0
    assert spies["prior-shift"].calls == len(ROWS_RUN.holdout_seeds)
    assert spies["prior-shift"].runner is prior_shift
    assert ",prior-shift,augmented,brier-rf," in text
