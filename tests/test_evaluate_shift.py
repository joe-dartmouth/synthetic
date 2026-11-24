"""The prior-shift arm: a logit offset per requested subgroup, solved on out-of-fold predictions."""

from __future__ import annotations

import dataclasses
import io
import logging
import math
import warnings
from typing import Final, cast

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit
from sklearn.exceptions import ConvergenceWarning

from bko.baselines.request import Request, Strata
from bko.datasets.cells import LABEL
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME
from bko.evaluate import run as run_module
from bko.evaluate import shift as shift_module
from bko.evaluate.__main__ import main
from bko.evaluate.classifier import Factory, Model, build, build_mlp
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import PRIOR_SHIFT_ARM
from bko.evaluate.run import run, score_holdout
from bko.evaluate.scores import Measure, Source, render
from bko.evaluate.shift import BOUND, offset, out_of_fold, prior_shift, shifted
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
            fitted,
            held_out,
            train,
            test,
            counts,
            edges=UCI_FRAME.edges,
            holdout=7,
            model=Model.LR,
            folds=5,
        )
    assert math.isfinite(moved.iloc[0])
    assert moved.iloc[0] == pytest.approx(0.5, abs=1e-9)
    assert (
        f"shift holdout=7 model=lr subgroup={YOUNG_WOMEN} records=3 target=0.500000"
        f" offset={math.log(3):.6f} negatives=3 positives=3 folds=5"
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
            fitted,
            held_out,
            train,
            test,
            counts,
            edges=UCI_FRAME.edges,
            holdout=0,
            model=Model.LR,
            folds=5,
        )
    assert moved.iloc[0] == 0.3  # ruff: ignore[float-equality-comparison]
    assert (
        f"shift holdout=0 model=lr subgroup={YOUNG_WOMEN} records=0 target=1.000000 offset=0.000000"
        " negatives=0 positives=2 folds=5"
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
        fitted,
        held_out,
        train,
        test,
        counts,
        edges=UCI_FRAME.edges,
        holdout=0,
        model=Model.LR,
        folds=5,
    )
    assert moved.iloc[0] == pytest.approx(0.5957252408878008, abs=1e-9)
    assert moved.iloc[1:].tolist() == [0.2, 0.3, 0.4]
    assert moved.index.equals(fitted.index)


def test_the_shift_reads_no_test_label():
    train = _people([40, 40, 60, 60], [0, 0, 1, 1], [0, 1, 0, 1])
    counts = {YOUNG_WOMEN: [1, 5]}
    held_out = pd.Series([0.2, 0.4, 0.5, 0.5], index=train.index)
    test = _people([40, 40, 60], [0, 0, 1], [1, 0, 1], start=10)
    fitted = pd.Series([0.1, 0.2, 0.3], index=test.index)
    flipped = test.assign(**{LABEL: 1 - test[LABEL]})
    kwargs = {"edges": UCI_FRAME.edges, "holdout": 0, "model": Model.LR, "folds": 5}
    moved = shifted(fitted, held_out, train, test, counts, **kwargs)
    assert moved.equals(shifted(fitted, held_out, train, flipped, counts, **kwargs))
    assert moved.iloc[0] != 0.1  # ruff: ignore[float-equality-comparison]


def test_a_zero_offset_keeps_a_probability_that_logit_and_expit_move(monkeypatch):
    # expit(logit(0.1)) is 0.10000000000000002, so only skipping a zero offset keeps 0.1.
    monkeypatch.setattr(shift_module, "offset", lambda predicted, target: 0.0)
    train = _people([40, 40], [0, 0], [0, 1])
    test = _people([40], [0], [1], start=10)
    fitted = pd.Series([0.1], index=test.index)
    held_out = pd.Series([0.3, 0.6], index=train.index)
    moved = shifted(
        fitted,
        held_out,
        train,
        test,
        {YOUNG_WOMEN: [1, 3]},
        edges=UCI_FRAME.edges,
        holdout=0,
        model=Model.LR,
        folds=5,
    )
    assert moved.iloc[0] == 0.1  # ruff: ignore[float-equality-comparison]


def _bound_warnings(caplog, held: list[float], counts: list[int]) -> list[str]:
    train = _people([40] * len(held), [0] * len(held), [1] * len(held))
    test = _people([40], [0], [1], start=10)
    with caplog.at_level(logging.INFO):
        shifted(
            pd.Series([0.3], index=test.index),
            pd.Series(held, index=train.index),
            train,
            test,
            {YOUNG_WOMEN: counts},
            edges=UCI_FRAME.edges,
            holdout=4,
            model=Model.LR,
            folds=2,
        )
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_an_unreachable_target_logs_a_bound_warning(caplog):
    # expit saturates at 1.0, so the bisection stops near 37.8 with a mean 3e-16 short of 1.
    warned = _bound_warnings(caplog, [0.25] * 3, [0, 3])
    assert (
        f"shift holdout=4 model=lr subgroup={YOUNG_WOMEN} records=3 target=1.000000"
        " offset=37.835413 negatives=0 positives=3 folds=2"
    ) in caplog.messages
    assert warned == [
        (
            f"shift bound holdout=4 model=lr subgroup={YOUNG_WOMEN} offset=37.835413"
            " target=1.000000 achieved=1.000000"
        )
    ]


def test_a_target_of_0_logs_a_bound_warning_at_the_search_bound(caplog):
    assert _bound_warnings(caplog, [0.25] * 3, [3, 0]) == [
        (
            f"shift bound holdout=4 model=lr subgroup={YOUNG_WOMEN} offset=-{BOUND:.6f}"
            " target=0.000000 achieved=0.000000"
        )
    ]


def test_a_reachable_target_logs_no_bound_warning(caplog):
    assert _bound_warnings(caplog, [0.25] * 3, [1, 3]) == []


def test_out_of_fold_predicts_every_training_record_from_a_fit_without_it():
    train = full_records(10).assign(weight=1.0)
    found, folds = out_of_fold(train, factory=build, frame=UCI_FRAME, model=Model.LR, seed=3)
    assert folds == 5
    assert found.index.equals(train.index)
    assert found.between(0.0, 1.0).all()
    again, _ = out_of_fold(train, factory=build, frame=UCI_FRAME, model=Model.LR, seed=3)
    assert found.equals(again)
    other, _ = out_of_fold(train, factory=build, frame=UCI_FRAME, model=Model.LR, seed=4)
    assert not found.equals(other)


class _Memorizer:
    """A model that records the records it fits on and predicts, predicting 0.5 for each."""

    def __init__(self, fits: list[tuple[set[int], set[int]]]) -> None:
        self.fits = fits
        self.seen: set[int] = set()

    def fit(self, features: pd.DataFrame, _labels: pd.Series, **_params: object) -> _Memorizer:
        self.seen = set(features.index)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.fits.append((self.seen, set(features.index)))
        return np.full((len(features), 2), 0.5)


def _spied(train: pd.DataFrame) -> list[tuple[set[int], set[int]]]:
    fits: list[tuple[set[int], set[int]]] = []
    found, folds = out_of_fold(
        train,
        factory=cast("Factory", lambda _frame: _Memorizer(fits)),
        frame=UCI_FRAME,
        model=Model.LR,
        seed=0,
    )
    assert folds == len(fits)
    assert found.index.equals(train.index)
    return fits


def test_each_fold_fits_without_the_records_it_predicts():
    train = full_records(10).assign(weight=1.0)
    fits = _spied(train)
    assert len(fits) == 5
    assert all(not (seen & held) for seen, held in fits)
    assert all(seen | held == set(train.index) for seen, held in fits)
    assert set().union(*(held for _, held in fits)) == set(train.index)


def test_a_label_of_fewer_than_5_records_sets_the_fold_count():
    train = full_records(10).assign(weight=1.0)
    three = train[(train[LABEL] == 0) | train.index.isin(train.index[train[LABEL] == 1][:3])]
    assert len(_spied(three)) == 3


def test_out_of_fold_refuses_a_label_of_fewer_than_2_records():
    train = full_records(10).assign(weight=1.0)
    lone = train[(train[LABEL] == 0) | (train.index == train.index[train[LABEL] == 1][0])]
    with pytest.raises(EvaluationError, match=r"holdout 0's prior-shift arm, model lr: .*folds"):
        out_of_fold(lone, factory=build, frame=UCI_FRAME, model=Model.LR, seed=0)


def test_out_of_fold_refuses_a_training_set_missing_a_label():
    train = full_records(10).assign(weight=1.0)
    negatives = train[train[LABEL] == 0]
    with pytest.raises(EvaluationError, match="need 2 folds"):
        out_of_fold(negatives, factory=build, frame=UCI_FRAME, model=Model.LR, seed=0)


class _Unconverged(_Memorizer):
    """A model whose fit warns that it did not converge."""

    def fit(self, features: pd.DataFrame, _labels: pd.Series, **_params: object) -> _Memorizer:
        warnings.warn("no convergence", ConvergenceWarning, stacklevel=2)
        return self


def test_a_fold_fit_refusal_names_the_holdout_the_arm_the_model_and_the_fold():
    train = full_records(10).assign(weight=1.0)
    with pytest.raises(
        EvaluationError,
        match=r"^holdout 3's prior-shift arm, model lr, fold 0: the classifier did not converge",
    ):
        out_of_fold(
            train,
            factory=cast("Factory", lambda _frame: _Unconverged([])),
            frame=UCI_FRAME,
            model=Model.LR,
            seed=3,
        )


def test_a_fixed_epoch_fold_fit_logs_whether_it_converged(caplog):
    train = full_records(10).assign(weight=1.0)
    with caplog.at_level(logging.INFO):
        out_of_fold(train, factory=build_mlp, frame=UCI_FRAME, model=Model.MLP, seed=3)
    folds = [m for m in caplog.messages if m.startswith("shift fold ")]
    assert folds == [f"shift fold holdout=3 model=mlp fold={k} converged=false" for k in range(5)]


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


def test_run_arms_maps_prior_shift_to_its_runner():
    assert run_module.ARMS[PRIOR_SHIFT_ARM] is prior_shift


def test_prior_shift_with_offsets_forced_to_zero_equals_nones_scores(monkeypatch):
    monkeypatch.setattr(shift_module, "offset", lambda predicted, target: 0.0)
    text = render(run(GRADED, WITH, ARMS, frame=UCI_FRAME))
    lines = [line.split(",") for line in text.splitlines()[1:]]
    augmented = Source.AUGMENTED.value
    mine = [row[:1] + row[2:] for row in lines if row[1] == PRIOR_SHIFT_ARM and row[2] == augmented]
    nones = [row[:1] + row[2:] for row in lines if row[1] == "none" and row[2] == augmented]
    assert mine
    assert mine == nones


def _flagged_recalls(rows: list[str], arm: str) -> list[float]:
    return [
        float(row.split(",")[5])
        for row in rows
        if row.split(",")[1:3] == [arm, Source.AUGMENTED.value]
        and row.split(",")[3] in {"recall", "recall-hgb"}
        and row.split(",")[4] == "flagged-subgroups"
    ]


def test_a_large_offset_flags_every_shifted_test_record(monkeypatch):
    monkeypatch.setattr(shift_module, "offset", lambda predicted, target: 30.0)
    parts = split(GRADED, seed=0, edges=UCI_FRAME.edges)
    rows = render(score_holdout(parts, WITH, ARMS, frame=UCI_FRAME)).splitlines()
    shifted_recalls = _flagged_recalls(rows, PRIOR_SHIFT_ARM)
    assert shifted_recalls == [1.0, 1.0]
    assert min(_flagged_recalls(rows, "none")) < 1.0


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
