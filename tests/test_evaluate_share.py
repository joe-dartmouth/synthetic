"""`Experiment.synthetic_share`: a share of each cell's training rows replaced by synthetic rows."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from fractions import Fraction

import pandas as pd
import pytest

import bko.evaluate.run_scores as run_scores_module
import bko.evaluate.shared as shared_module
from bko.baselines.errors import ShortfallError
from bko.baselines.methods import Runner
from bko.baselines.records import Baseline, number
from bko.baselines.request import Request
from bko.datasets.cells import CELL_COLUMNS, cell_keys
from bko.evaluate.classifier import Model, build
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import E1, REAL_ARM, Experiment, Removal, gap_request
from bko.evaluate.prepare import prepared
from bko.evaluate.provenance import experiment_line
from bko.evaluate.run import score_holdout
from bko.evaluate.run_folder import ASSUMED
from bko.evaluate.scores import Measure, Scope, Source
from tests._evaluate_fixtures import FAST_ARMS
from tests._frame_fixtures import TOY, TOY_PARTS
from tests.test_evaluate_shared import edit_log, refused, run_folder, shared


FULL = ("<45", "male", 1)
"""A toy cell holding 20 training rows at seed 0."""

SHARES = (Fraction(1, 50), Fraction(1, 10), Fraction(1, 4), Fraction(1, 2), Fraction(1))

SHARE_RUN = Experiment(
    holdout_seeds=(0, 1),
    arms=(REAL_ARM, "smote"),
    candidate="smote",
    comparators=(),
    replica_arms=(),
    decontaminate=False,
    flagged_cell=FULL,
    synthetic_share=Fraction(1, 4),
)
"""A toy share run: a quarter of each cell's training rows replaced, `real` beside SMOTE."""


def test_a_share_beside_a_removal_is_refused():
    message = r"a synthetic share cannot stand beside a removal or a training fraction below 1"
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(SHARE_RUN, removal=Removal(FULL, 0.5))
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(SHARE_RUN, training_fraction=Fraction(1, 2))


@pytest.mark.parametrize("share", [Fraction(0), Fraction(3, 2), 0.25, 1, True])
def test_a_share_outside_the_unit_interval_is_refused(share):
    with pytest.raises(ValueError, match=r"synthetic share .* must be a Fraction in \(0, 1\]"):
        dataclasses.replace(SHARE_RUN, synthetic_share=share)


def test_a_share_needs_a_flagged_cell():
    with pytest.raises(ValueError, match=r"a synthetic share needs a flagged cell"):
        dataclasses.replace(SHARE_RUN, flagged_cell=None)


@pytest.mark.parametrize("arm", ["prior-shift", "class-weight"])
def test_an_arm_reading_the_whole_split_is_refused_under_a_share(arm):
    arms = (*SHARE_RUN.arms, arm)
    removing = dataclasses.replace(
        SHARE_RUN, arms=arms, synthetic_share=None, removal=Removal(FULL, 0.5)
    )
    assert arm in removing.arms
    with pytest.raises(ValueError, match=rf"\['{arm}'\] read the whole training split"):
        dataclasses.replace(removing, removal=None, synthetic_share=Fraction(1, 4))


def test_the_tree_arm_is_refused_under_a_share():
    # Its code-share error reads `Prepared.removed`, which a share leaves empty.
    with pytest.raises(ValueError, match=r"bko-tree reads the removed rows' code shares"):
        dataclasses.replace(SHARE_RUN, arms=(*SHARE_RUN.arms, "bko-tree"))


def test_real_is_admitted_under_a_share():
    admitted = Experiment(
        holdout_seeds=(0, 1),
        arms=(REAL_ARM, "smote"),
        candidate="smote",
        comparators=(),
        replica_arms=(),
        decontaminate=False,
        flagged_cell=FULL,
        synthetic_share=Fraction(1, 2),
    )
    assert REAL_ARM in admitted.arms
    assert admitted.synthetic_share == Fraction(1, 2)
    with pytest.raises(ValueError, match=r"needs a removal or a synthetic share"):
        dataclasses.replace(SHARE_RUN, synthetic_share=None)


def test_prepared_keeps_the_whole_split_and_names_the_dropped_rows(caplog):
    caplog.set_level(logging.INFO, logger="bko.evaluate.prepare")
    ready = prepared(TOY_PARTS, SHARE_RUN, edges=TOY.edges)
    assert ready.ablated.train.equals(TOY_PARTS.train)
    assert ready.removed.empty
    # Ten cells of 20 training rows drop 5 each and two of 5 drop 1 each.
    assert len(ready.ablated.train) == 210
    assert len(ready.dropped) == 52
    assert set(ready.dropped.index) <= set(TOY_PARTS.train.index)
    assert ready.dropped.equals(TOY_PARTS.train.loc[ready.dropped.index])
    assert len(ready.kept) == 158
    assert set(ready.kept.index) == set(TOY_PARTS.train.index) - set(ready.dropped.index)
    ids = ",".join(str(record) for record in sorted(ready.dropped.index))
    digest = hashlib.sha256(ids.encode("ascii")).hexdigest()
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("share ")]
    assert line == f"share holdout=0 share=1/4 kept=158 dropped=52 dropped_sha256={digest}"


def test_no_share_drops_no_row(caplog):
    caplog.set_level(logging.INFO, logger="bko.evaluate.prepare")
    plain = dataclasses.replace(SHARE_RUN, arms=("none", "smote"), synthetic_share=None)
    ready = prepared(TOY_PARTS, plain, edges=TOY.edges)
    assert ready.dropped.empty
    assert list(ready.dropped.columns) == list(TOY_PARTS.train.columns)
    assert ready.kept is ready.ablated.train
    assert "share holdout=" not in caplog.text


@pytest.mark.parametrize("share", SHARES)
def test_the_test_split_and_its_fingerprint_are_untouched_at_every_share(share):
    experiment = dataclasses.replace(SHARE_RUN, synthetic_share=share)
    ready = prepared(TOY_PARTS, experiment, edges=TOY.edges)
    assert ready.ablated.test.equals(TOY_PARTS.test)
    assert ready.ablated.fingerprint == TOY_PARTS.fingerprint
    assert ready.ablated.train.equals(TOY_PARTS.train)
    assert ready.sized.train.equals(TOY_PARTS.train)


def test_the_experiment_line_names_a_share_only_when_set():
    plain = experiment_line("x", E1, ["lr"], frame="uci")
    assert "synthetic_share" not in plain
    arms = tuple(arm for arm in E1.arms if arm != "class-weight")
    comparators = tuple(arm for arm in E1.comparators if arm != "class-weight")
    cell = dataclasses.replace(E1, arms=arms, comparators=comparators, flagged_cell=FULL)
    named = experiment_line("x", cell, ["lr"], frame="uci")
    for share, spelled in [(Fraction(1, 50), "1/50"), (Fraction(1), "1/1")]:
        sharing = dataclasses.replace(cell, synthetic_share=share)
        line = experiment_line("x", sharing, ["lr"], frame="uci")
        assert f" training_fraction=1 synthetic_share={spelled} decontaminate=" in line
        assert line.replace(f"synthetic_share={spelled} ", "") == named


def test_an_older_experiment_line_reads_at_the_assumed_synthetic_share(tmp_path, caplog):
    assert ASSUMED["synthetic_share"] == "none"
    folder = run_folder(tmp_path, "run", caplog)
    edit_log(
        folder,
        "experiment ",
        lambda line: line.replace(
            "training_fraction=1 ", "training_fraction=1 synthetic_share=none ", 1
        ),
    )
    code, text = shared(tmp_path, folder)
    assert code == 0
    assert text == (folder / "scores.csv").read_text(encoding="utf-8")
    edit_log(
        folder,
        "experiment ",
        lambda line: line.replace("synthetic_share=none ", "synthetic_share=1/2 ", 1),
    )
    error = refused(tmp_path, caplog, folder)
    assert error.endswith("differ: {'synthetic_share': ('1/2', 'none')}")


QUARTER: dict[tuple[str, str, int], int] = {
    (band, sex, label): 1 if (band, label) == ("45-69", 1) else 5
    for band in ("<45", "45-69", ">=70")
    for sex in ("female", "male")
    for label in (0, 1)
}
"""Each toy cell's dropped rows at a quarter: 5 of 20, and 1 of the 5 in each `45-69` cell with
the condition."""

ARMS = (REAL_ARM, "smote", "label-bootstrap")
SCORED = dataclasses.replace(SHARE_RUN, arms=ARMS, comparators=("label-bootstrap",))
"""`SHARE_RUN` with the label bootstrap beside SMOTE."""


def _trainings(
    experiment: Experiment, monkeypatch, arms: dict[str, Runner] | None = None
) -> tuple[dict[str, pd.DataFrame], list]:
    """Each arm's training set at holdout 0 of the toy split, and the holdout's scores."""
    seen: list[pd.DataFrame] = []
    real = run_scores_module.probabilities

    def spy(training, test, **options):
        seen.append(training)
        return real(training, test, **options)

    monkeypatch.setattr(run_scores_module, "probabilities", spy)
    runners = arms or {name: FAST_ARMS[name] for name in experiment.arms if name != REAL_ARM}
    scores = score_holdout(TOY_PARTS, experiment, runners, models={Model.LR: build}, frame=TOY)
    return dict(zip(experiment.arms, seen, strict=True)), scores


def _real_ids(training: pd.DataFrame) -> set[int]:
    """The training set's real record ids: an added row's id is negative."""
    return {int(record) for record in training.index if record > 0}


@pytest.mark.parametrize("share", SHARES[1:])
def test_each_arms_training_set_is_the_training_size_at_every_share(share, monkeypatch):
    experiment = dataclasses.replace(SCORED, synthetic_share=share)
    trainings, _ = _trainings(experiment, monkeypatch)
    dropped = set(prepared(TOY_PARTS, experiment, edges=TOY.edges).dropped.index)
    assert dropped
    for arm, training in trainings.items():
        assert len(training) == 210, arm
        if arm != REAL_ARM:
            assert _real_ids(training) == set(TOY_PARTS.train.index) - dropped, arm


def test_the_request_asks_each_cells_dropped_count():
    assert gap_request(TOY_PARTS.train, SHARE_RUN, seed=0, frame=TOY).quota == QUARTER
    tenth = dataclasses.replace(SHARE_RUN, synthetic_share=Fraction(1, 10))
    expected = {cell: 1 if rows == 1 else 2 for cell, rows in QUARTER.items()}
    assert gap_request(TOY_PARTS.train, tenth, seed=0, frame=TOY).quota == expected
    # With no share, the flagged cell alone is restored to `fill_to`.
    restored = dataclasses.replace(SHARE_RUN, arms=("smote",), synthetic_share=None, fill_to=26)
    assert gap_request(TOY_PARTS.train, restored, seed=0, frame=TOY).quota == {FULL: 6}


def test_a_share_that_drops_no_row_is_refused():
    fiftieth = dataclasses.replace(SHARE_RUN, synthetic_share=Fraction(1, 50))
    with pytest.raises(EvaluationError, match=r"the synthetic share 1/50 drops no training row"):
        gap_request(TOY_PARTS.train, fiftieth, seed=0, frame=TOY)


def test_every_reader_rebuilds_a_share_runs_request(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="bko.evaluate.run")
    _trainings(SCORED, monkeypatch)
    used = {}
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith("quota holdout=0 "):
            _, _, stratum, rows = message.split(" ")
            band, sex, label = stratum.removeprefix("stratum=").split("/")
            used[band, sex, int(label)] = int(rows.removeprefix("rows="))
    assert used == QUARTER
    # `shared` rebuilds the request from the prepared split, as membership's donor reuse does.
    rebuilt: list[Request] = []

    def spy(others, seed, gap, replica):
        rebuilt.append(gap)
        return []

    monkeypatch.setattr(shared_module, "excluded", spy)
    assert shared_module._exclude([], TOY_PARTS, SCORED, TOY) == []
    assert [gap.quota for gap in rebuilt] == [used]


@pytest.mark.parametrize("share", SHARES[1:])
def test_real_at_any_share_trains_on_the_whole_split(share, monkeypatch):
    experiment = dataclasses.replace(SCORED, synthetic_share=share)
    training = _trainings(experiment, monkeypatch)[0][REAL_ARM]
    columns = list(TOY.columns)
    found = training.loc[:, columns].sort_values(columns).reset_index(drop=True)
    whole = TOY_PARTS.train.loc[:, columns].sort_values(columns).reset_index(drop=True)
    assert found.equals(whole)
    assert training["weight"].tolist() == pytest.approx([1.0] * 210)


def test_share_1_trains_on_synthetic_rows_alone(monkeypatch):
    experiment = dataclasses.replace(SCORED, synthetic_share=Fraction(1))
    trainings, _ = _trainings(experiment, monkeypatch)
    for arm in ("smote", "label-bootstrap"):
        assert len(trainings[arm]) == 210
        assert _real_ids(trainings[arm]) == set()
    assert len(trainings[REAL_ARM]) == 210


def _shorting(runner: Runner) -> Runner:
    """`runner`'s rows but those of the request's first cell, reported short on every attempt."""

    def short(train: pd.DataFrame, request: Request) -> Baseline:
        full = runner(train, request)
        cell, *_ = request.quota
        keys = cell_keys(full.added, edges=TOY.edges).loc[:, list(CELL_COLUMNS)]
        keep = [tuple(key) != cell for key in keys.itertuples(index=False)]
        missing = {"/".join(map(str, cell)): request.quota[cell]}
        kept = Baseline(number(full.added.loc[keep]), full.weights, {"short": missing})
        raise ShortfallError(f"short of {missing}", missing, kept.added, kept)

    return short


def test_an_arm_short_of_a_cells_count_scores_its_partial_fill(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="bko.evaluate")
    arms = {"smote": _shorting(FAST_ARMS["smote"]), "label-bootstrap": FAST_ARMS["label-bootstrap"]}
    trainings, scores = _trainings(SCORED, monkeypatch, arms)
    # The first cell, `<45/female/0`, drops 5 rows, which the short arm never adds.
    assert len(trainings["smote"]) == 205
    assert len(trainings["label-bootstrap"]) == 210
    rows = {
        s.arm: s.value
        for s in scores
        if (s.source, s.measure, s.scope) == (Source.ADDED, Measure.ROWS, Scope.ALL)
    }
    assert rows == {REAL_ARM: 52.0, "smote": 47.0, "label-bootstrap": 52.0}
    assert (
        "shortfall holdout=0 arm=smote attempt=5 cell=<45/female/0 short=5 quota=5" in caplog.text
    )
    assert "added=47 " in caplog.text
    assert 'params={"short": {"<45/female/0": 5}}' in caplog.text


@pytest.mark.parametrize(("labels", "held"), [((), 0), ((1,), 1)])
def test_a_share_1_fill_of_fewer_than_two_labels_names_its_holdout_and_arm(
    labels, held, monkeypatch
):
    experiment = dataclasses.replace(SCORED, synthetic_share=Fraction(1))
    full = FAST_ARMS["smote"]

    def thin(train: pd.DataFrame, request: Request) -> Baseline:
        added = full(train, request).added
        return Baseline(added.loc[added[TOY.label].isin(labels)], full(train, request).weights, {})

    arms = {"smote": thin, "label-bootstrap": FAST_ARMS["label-bootstrap"]}
    message = rf"holdout 0 arm smote trains on \d+ rows of {held} labels"
    with pytest.raises(EvaluationError, match=message):
        _trainings(experiment, monkeypatch, arms)


def test_a_fiftieth_drops_only_the_cells_large_enough(monkeypatch, caplog):
    # `FULL` doubled to 40 training rows drops round(0.8) = 1 at 1/50; every other cell, of 20
    # or 5 rows, drops none and is left out of the request.
    cell = cell_keys(TOY_PARTS.train, edges=TOY.edges).loc[:, list(CELL_COLUMNS)]
    inside = [tuple(key) == FULL for key in cell.itertuples(index=False)]
    extra = TOY_PARTS.train.loc[inside].copy()
    extra.index += 1000
    assert not set(extra.index) & (set(TOY_PARTS.train.index) | set(TOY_PARTS.test.index))
    parts = dataclasses.replace(TOY_PARTS, train=pd.concat([TOY_PARTS.train, extra]))
    experiment = dataclasses.replace(SCORED, synthetic_share=Fraction(1, 50))
    caplog.set_level(logging.INFO, logger="bko.evaluate")
    seen: list[pd.DataFrame] = []
    real = run_scores_module.probabilities

    def spy(training, test, **options):
        seen.append(training)
        return real(training, test, **options)

    monkeypatch.setattr(run_scores_module, "probabilities", spy)
    runners = {name: FAST_ARMS[name] for name in experiment.arms if name != REAL_ARM}
    score_holdout(parts, experiment, runners, models={Model.LR: build}, frame=TOY)
    quotas = [r.getMessage() for r in caplog.records if r.getMessage().startswith("quota ")]
    assert quotas == ["quota holdout=0 stratum=<45/male/1 rows=1"]
    (gone,) = prepared(parts, experiment, edges=TOY.edges).dropped.index
    assert gone in set(TOY_PARTS.train.index[inside]) | set(extra.index)
    trainings = dict(zip(experiment.arms, seen, strict=True))
    for arm, training in trainings.items():
        assert len(training) == 230, arm
    # Every arm keeps the same 229 real rows; `real` adds the dropped row back under a new id.
    for arm, training in trainings.items():
        assert _real_ids(training) == set(parts.train.index) - {gone}, arm
