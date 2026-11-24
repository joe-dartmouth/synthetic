"""The prior-shift arm: a logit offset per requested subgroup, solved on out-of-fold predictions."""

from __future__ import annotations

import dataclasses
import io
import logging
import math
from typing import Final, cast

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from bko.baselines.request import Request, Strata
from bko.datasets.cells import LABEL
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME
from bko.evaluate import shift as shift_module
from bko.evaluate.__main__ import main
from bko.evaluate.classifier import Model, build
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import PRIOR_SHIFT_ARM
from bko.evaluate.run import run, score_holdout
from bko.evaluate.scores import Measure, Source, render
from bko.evaluate.shift import offset, out_of_fold, prior_shift, shifted
from tests._baselines_fixtures import EDGE, full_records
from tests._evaluate_bko_fixtures import NARROW
from tests._evaluate_fixtures import FAST_ARMS, GRADED, SMALL
from tests.test_evaluate_shared import rescorer, run_folder


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

YOUNG_WOMEN: Final = f"<{EDGE}/female"

ARMS: Final = {**FAST_ARMS, PRIOR_SHIFT_ARM: prior_shift}

WITH: Final = dataclasses.replace(
    SMALL,
    arms=("none", "smote", PRIOR_SHIFT_ARM),
    candidate="smote",
    comparators=("none",),
    replica_arms=("smote",),
    criteria=(),
)
"""Two holdouts of `GRADED` with a control, a row-adding arm with a replica, and `prior-shift`."""

WITHOUT: Final = dataclasses.replace(WITH, arms=("none", "smote"))


def _people(ages: list[int], sexes: list[int], labels: list[int], start: int = 0) -> pd.DataFrame:
    """Records holding only the cell columns, indexed from `start`."""
    index = pd.RangeIndex(start, start + len(ages))
    return pd.DataFrame({"age": ages, "sex": sexes, LABEL: labels}, index=index)


def test_two_predictions_offset_to_one_half_by_ln_2():
    assert offset(np.array([0.2, 0.5]), 0.5) == pytest.approx(0.6931471805599453, abs=1e-9)


def test_one_prediction_gives_the_closed_form():
    assert offset(np.array([0.3]), 0.6) == pytest.approx(logit(0.6) - logit(0.3), abs=1e-9)


def test_predictions_of_0_and_1_give_offset_0_at_one_half():
    # Clipped to 1e-12, the mean moves by about 1e-12 per unit of offset, so a float mean resolves
    # the offset to about 1e-4 only.
    assert offset(np.array([0.0, 1.0]), 0.5) == pytest.approx(0.0, abs=1e-3)


def test_an_empty_array_gives_offset_0():
    assert offset(np.array([]), 0.5) == 0.0  # ruff: ignore[float-equality-comparison]


def test_an_empty_cell_gives_a_finite_offset(caplog):
    train = _people([40, 40, 40, 60], [0, 0, 0, 0], [0, 0, 0, 1])
    request = Request({(f"<{EDGE}", "female", 1): 3}, frame=UCI_FRAME)
    counts = cast("dict[str, list[int]]", prior_shift(train, request).params["shift"])
    assert counts == {YOUNG_WOMEN: [3, 3]}
    held_out = pd.Series([0.25, 0.25, 0.25, 0.9], index=train.index)
    test = _people([41], [0], [1], start=10)
    fitted = pd.Series([0.25], index=test.index)
    with caplog.at_level(logging.INFO):
        moved = shifted(
            fitted, held_out, train, test, counts, edges=UCI_FRAME.edges, holdout=7, model=Model.LR
        )
    assert math.isfinite(moved.iloc[0])
    assert moved.iloc[0] == pytest.approx(0.5, abs=1e-9)
    assert (
        f"shift holdout=7 model=lr subgroup={YOUNG_WOMEN} records=3 target=0.500000"
        f" offset={math.log(3):.6f}"
    ) in caplog.messages


def test_a_subgroup_with_no_training_record_gets_offset_0(caplog):
    train = _people([60, 60], [1, 1], [0, 1])
    request = Request({(f"<{EDGE}", "female", 1): 2}, frame=UCI_FRAME)
    counts = cast("dict[str, list[int]]", prior_shift(train, request).params["shift"])
    assert counts == {YOUNG_WOMEN: [0, 2]}
    test = _people([40], [0], [1], start=10)
    fitted = pd.Series([0.3], index=test.index)
    held_out = pd.Series([0.4, 0.6], index=train.index)
    with caplog.at_level(logging.INFO):
        moved = shifted(
            fitted, held_out, train, test, counts, edges=UCI_FRAME.edges, holdout=0, model=Model.LR
        )
    assert moved.iloc[0] == 0.3  # ruff: ignore[float-equality-comparison]
    assert (
        f"shift holdout=0 model=lr subgroup={YOUNG_WOMEN} records=0 target=1.000000 offset=0.000000"
    ) in caplog.messages


def test_the_target_counts_the_training_records_plus_the_requests_rows():
    train = _people([40] * 5 + [60] * 2, [0] * 5 + [1] * 2, [0, 0, 0, 1, 1, 0, 1])
    request = Request(
        {(f"<{EDGE}", "female", 0): 4, (f"<{EDGE}", "female", 1): 6, (f">={EDGE}", "male", 1): 1},
        frame=UCI_FRAME,
    )
    baseline = prior_shift(train, request)
    assert baseline.params == {"shift": {YOUNG_WOMEN: [7, 8], f">={EDGE}/male": [1, 2]}}
    assert baseline.added.empty
    assert list(baseline.added.columns) == list(UCI_FRAME.columns)
    assert baseline.weights.tolist() == [1.0] * len(train)
    assert baseline.weights.index.equals(train.index)


def test_a_request_not_keyed_by_cell_is_refused():
    train = _people([40, 60], [0, 1], [0, 1])
    request = Request({(1,): 3}, strata=Strata.LABEL, frame=UCI_FRAME)
    with pytest.raises(EvaluationError, match="keyed by cell"):
        prior_shift(train, request)


def test_records_outside_a_shifted_subgroup_keep_their_probability():
    train = _people([40, 40, 60, 60], [0, 0, 1, 1], [0, 1, 0, 1])
    counts = {YOUNG_WOMEN: [1, 5]}
    held_out = pd.Series([0.2, 0.4, 0.5, 0.5], index=train.index)
    test = _people([40, 60, 40, 60], [0, 0, 1, 1], [1, 0, 1, 0], start=10)
    fitted = pd.Series([0.1, 0.2, 0.3, 0.4], index=test.index)
    moved = shifted(
        fitted, held_out, train, test, counts, edges=UCI_FRAME.edges, holdout=0, model=Model.LR
    )
    assert moved.iloc[0] > 0.1
    assert moved.iloc[1:].tolist() == [0.2, 0.3, 0.4]
    assert moved.index.equals(fitted.index)


def test_out_of_fold_predicts_every_training_record_from_a_fit_without_it():
    train = full_records(10).assign(weight=1.0)
    found = out_of_fold(train, factory=build, frame=UCI_FRAME, model=Model.LR, seed=3)
    assert found.index.equals(train.index)
    assert found.between(0.0, 1.0).all()
    again = out_of_fold(train, factory=build, frame=UCI_FRAME, model=Model.LR, seed=3)
    assert found.equals(again)


def test_out_of_fold_refuses_a_label_of_fewer_than_2_records():
    train = full_records(10).assign(weight=1.0)
    lone = train[(train[LABEL] == 0) | (train.index == train.index[train[LABEL] == 1][0])]
    with pytest.raises(EvaluationError, match="folds"):
        out_of_fold(lone, factory=build, frame=UCI_FRAME, model=Model.LR, seed=0)


def _others(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.split(",")[1] != PRIOR_SHIFT_ARM]


@pytest.fixture(scope="module")
def with_shift() -> str:
    return render(run(GRADED, WITH, ARMS, frame=UCI_FRAME))


def test_every_other_arms_rows_are_byte_equal_with_and_without_prior_shift(with_shift):
    without = render(run(GRADED, WITHOUT, ARMS, frame=UCI_FRAME))
    assert _others(with_shift) == without.splitlines()
    assert len(_others(with_shift)) < len(with_shift.splitlines())


def test_prior_shift_adds_no_rows(with_shift):
    rows = [
        line.split(",")
        for line in with_shift.splitlines()
        if line.split(",")[1:4] == [PRIOR_SHIFT_ARM, Source.ADDED.value, Measure.ROWS.value]
    ]
    assert [float(row[5]) for row in rows] == [0.0, 0.0]


def test_prior_shift_with_offsets_forced_to_zero_equals_nones_scores(monkeypatch):
    monkeypatch.setattr(shift_module, "offset", lambda probabilities, target: 0.0)
    text = render(run(GRADED, WITH, ARMS, frame=UCI_FRAME))
    lines = [line.split(",") for line in text.splitlines()[1:]]
    augmented = Source.AUGMENTED.value
    mine = [row[:1] + row[2:] for row in lines if row[1] == PRIOR_SHIFT_ARM and row[2] == augmented]
    nones = [row[:1] + row[2:] for row in lines if row[1] == "none" and row[2] == augmented]
    assert mine
    assert mine == nones


def test_prior_shift_scores_each_model(with_shift, caplog):
    measures = {
        line.split(",")[3]
        for line in with_shift.splitlines()
        if line.split(",")[1:3] == [PRIOR_SHIFT_ARM, Source.AUGMENTED.value]
    }
    assert {"recall", "recall-hgb", "auc-pr", "auc-pr-hgb"} <= measures
    with caplog.at_level(logging.INFO):
        score_holdout(split(GRADED, seed=0, edges=UCI_FRAME.edges), WITH, ARMS, frame=UCI_FRAME)
    shifts = [m for m in caplog.messages if m.startswith("shift holdout=0 ")]
    assert {m.split(" model=")[1].split(" ")[0] for m in shifts} == {"lr", "hgb"}
    assert {m.split(" subgroup=")[1].split(" ")[0] for m in shifts} == {
        f"<{EDGE}/female",
        f">={EDGE}/female",
    }


def test_prior_shift_cannot_be_a_replica_arm():
    with pytest.raises(ValueError, match="add no rows"):
        dataclasses.replace(WITH, replica_arms=("smote", PRIOR_SHIFT_ARM))


ROWS_RUN: Final = dataclasses.replace(
    NARROW,
    arms=("none", PRIOR_SHIFT_ARM),
    candidate=PRIOR_SHIFT_ARM,
    comparators=("none",),
    criteria=(),
)
"""A folder's run of `none` and `prior-shift` on the UCI fixture."""

ROWS_ARMS: Final = {"none": FAST_ARMS["none"], PRIOR_SHIFT_ARM: prior_shift}


def test_rows_rescores_a_folder_that_holds_prior_shift(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog, experiment=ROWS_RUN, arms=ROWS_ARMS)
    assert f",{PRIOR_SHIFT_ARM},augmented," in (folder / "scores.csv").read_text(encoding="utf-8")
    store = tmp_path / "run-rows" / "rows.jsonl"
    store.parent.mkdir()
    argv = ["rows", "--folder", str(folder), "--rows", str(store)]
    argv += ["--data-dir", str(tmp_path / "data")]
    caplog.clear()
    with caplog.at_level(logging.INFO):
        code = main(argv, context=rescorer(experiment=ROWS_RUN, arms=ROWS_ARMS), out=io.StringIO())
    assert code == 0
    assert any(" matches_folder=true" in message for message in caplog.messages)
