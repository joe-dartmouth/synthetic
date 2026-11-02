"""The harness on the graded frame: every arm on every holdout, and no test record for an arm."""

from __future__ import annotations

import dataclasses
import logging
import re
from collections import Counter

import numpy as np
import pytest

import bko.evaluate.run as run_module
from bko.baselines.errors import BaselineError, OutputContractError, ShortfallError
from bko.baselines.records import Baseline, derived_seed, number, unit_weights
from bko.datasets.split import split
from bko.datasets.uci import FRAME_COLUMNS, FRAME_DTYPES, UCI_FRAME
from bko.evaluate.classifier import Model, build
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import E1, gap_request
from bko.evaluate.prepare import held_out_ratio
from bko.evaluate.run import run, score_holdout
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Source, render
from tests._baselines_fixtures import EDGE, cell_list
from tests._evaluate_fixtures import FAST_ARMS, GRADED, SMALL, Spy


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def test_a_control_arm_ignores_the_request():
    train = split(GRADED, seed=0, edges=UCI_FRAME.edges).train
    baseline = FAST_ARMS["none"](train, gap_request(train, E1, seed=0, frame=UCI_FRAME))
    assert baseline.added.empty


@pytest.fixture(scope="module")
def scores():
    return run(GRADED, SMALL, FAST_ARMS, frame=UCI_FRAME)


def test_every_arm_is_scored_on_every_holdout(scores):
    counts = Counter((score.holdout, score.arm, score.source) for score in scores)
    for seed in SMALL.holdout_seeds:
        assert counts[seed, HELD_OUT_ARM, Source.HELD_OUT] == 4
        for arm in SMALL.arms:
            assert counts[seed, arm, Source.AUGMENTED] == 12
        for arm in SMALL.replica_arms:
            assert counts[seed, arm, Source.REPLICA] == 10
            assert counts[seed, arm, Source.ADDED] == 4
        for arm in ("none", "class-weight"):
            assert counts[seed, arm, Source.ADDED] == 2


def test_every_row_adding_arm_adds_the_same_rows(scores):
    rows = {(s.holdout, s.arm): s.value for s in scores if s.measure is Measure.ROWS}
    assert {value for (_, arm), value in rows.items() if arm in SMALL.replica_arms} == {20.0}
    assert (rows[0, "none"], rows[0, "class-weight"]) == pytest.approx((0.0, 0.0))


REGRESSION = [
    (measure, scope)
    for scope in Scope
    for measure in (Measure.RECALL, Measure.PRECISION, Measure.AUC_PR)
]
BOOSTED = [
    (measure, scope)
    for scope in Scope
    for measure in (Measure.RECALL_HGB, Measure.PRECISION_HGB, Measure.AUC_PR_HGB)
]


def test_each_arms_boosting_rows_follow_its_regression_rows(scores):
    for seed in SMALL.holdout_seeds:
        for arm in SMALL.arms:
            at = [
                position
                for position, s in enumerate(scores)
                if (s.holdout, s.arm, s.source) == (seed, arm, Source.AUGMENTED)
            ]
            assert at == list(range(at[0], at[0] + 12))
            assert [(scores[i].measure, scores[i].scope) for i in at] == REGRESSION + BOOSTED


def test_replica_utility_reads_the_regression_only(scores):
    replica = {s.measure for s in scores if s.source is Source.REPLICA}
    assert Measure.RECALL in replica
    assert not replica & {measure for measure, _ in BOOSTED}


def test_dropping_the_boosting_rows_leaves_the_regression_rows_byte_for_byte():
    experiment = dataclasses.replace(
        SMALL,
        arms=("none", "smote"),
        candidate="smote",
        comparators=("none",),
        replica_arms=("smote",),
    )
    both = render(run(GRADED, experiment, FAST_ARMS, frame=UCI_FRAME))
    alone = render(run(GRADED, experiment, FAST_ARMS, models={Model.LR: build}, frame=UCI_FRAME))
    assert "-hgb," in both
    stripped = "".join(line for line in both.splitlines(keepends=True) if "-hgb," not in line)
    assert stripped == alone


def test_models_without_the_regression_are_refused():
    with pytest.raises(EvaluationError, match="lack the regression"):
        score_holdout(
            split(GRADED, seed=0, edges=UCI_FRAME.edges),
            SMALL,
            FAST_ARMS,
            models={Model.HGB: build},
            frame=UCI_FRAME,
        )


def test_the_held_out_ratio_is_the_largest_cells():
    assert held_out_ratio(
        split(GRADED, seed=0, edges=UCI_FRAME.edges), edges=UCI_FRAME.edges
    ) == pytest.approx(5 / 18)


def test_class_weights_reach_the_fit_through_the_harness():
    spy = Spy()
    experiment = dataclasses.replace(
        SMALL,
        arms=("class-weight",),
        candidate="class-weight",
        comparators=(),
        replica_arms=(),
    )
    run(GRADED, experiment, FAST_ARMS, models={Model.LR: spy}, frame=UCI_FRAME)
    labels, weights = spy.models[0].seen
    total, positives = len(labels), int(labels.sum())
    expected = np.where(labels == 1, total / (2 * positives), total / (2 * (total - positives)))
    assert weights == pytest.approx(expected)


def test_no_arm_sees_a_test_record():
    seen = []

    def spying(train, request):
        seen.append(set(train.index))
        return FAST_ARMS["smote"](train, request)

    experiment = dataclasses.replace(
        SMALL, arms=("smote",), candidate="smote", comparators=(), replica_arms=("smote",)
    )
    run(GRADED, experiment, {"smote": spying}, frame=UCI_FRAME)
    assert len(seen) == 2 * len(SMALL.holdout_seeds)
    for position, indexes in enumerate(seen):
        parts = split(GRADED, seed=SMALL.holdout_seeds[position // 2], edges=UCI_FRAME.edges)
        assert indexes == set(parts.train.index)
        assert not indexes & set(parts.test.index)


def test_an_arm_without_a_runner_is_refused():
    with pytest.raises(EvaluationError, match="no runner"):
        run(GRADED, SMALL, {"none": FAST_ARMS["none"]}, frame=UCI_FRAME)


def test_a_refusing_arm_stops_the_run_after_every_attempt():
    calls = []

    def refuse(train, request):
        calls.append(request.seed)
        msg = "refused"
        raise BaselineError(msg)

    with pytest.raises(BaselineError, match="refused"):
        run(GRADED, SMALL, {**FAST_ARMS, "smote": refuse}, frame=UCI_FRAME)
    assert len(calls) == SMALL.attempts


ONLY_SMOTE = dataclasses.replace(
    SMALL, arms=("smote",), candidate="smote", comparators=(), replica_arms=()
)


def shorting(calls, *, dropped=None, claimed=None, key=None, until=None, replica_only=False):
    """A runner that keeps SMOTE's rows but `dropped` of its first cell's and reports a shortfall.

    `dropped` None drops the whole cell; `claimed` is the shortfall it reports (`dropped` by
    default) under `key` (the cell's by default). From attempt `until` on, or on a gap request when
    `replica_only`, it returns SMOTE's whole fill.
    """

    def short(train, request):
        calls.append(request.seed)
        full = FAST_ARMS["smote"](train, request)
        if (until is not None and len(calls) >= until) or (replica_only and request.rows < 50):
            return full
        cell, *_ = request.quota
        cut = request.quota[cell] if dropped is None else dropped
        strata = cell_list(full.added)
        positions = [i for i, s in enumerate(strata) if s == cell][-cut:]
        keep = [i not in positions for i in range(len(strata))]
        missing = {key or "/".join(map(str, cell)): cut if claimed is None else claimed}
        kept = Baseline(number(full.added.loc[keep]), full.weights, {"short": missing})
        msg = f"short of {missing}"
        raise ShortfallError(msg, missing, kept.added, kept)

    return short


def test_a_generator_short_at_its_limit_on_the_last_attempt_fills_partially(caplog):
    assert SMALL.attempts > 1
    calls = []
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        scores = run(GRADED, ONLY_SMOTE, {"smote": shorting(calls)}, frame=UCI_FRAME)
    assert len(calls) == SMALL.attempts * len(SMALL.holdout_seeds)
    assert {score.holdout for score in scores} == set(SMALL.holdout_seeds)
    lines = [m for m in caplog.messages if m.startswith("shortfall ")]
    assert [m.split()[1] for m in lines] == [f"holdout={seed}" for seed in SMALL.holdout_seeds]
    assert all(
        re.fullmatch(
            rf"shortfall holdout=\d+ arm=smote attempt={SMALL.attempts} cell=\S+ "
            r"short=(\d+) quota=\1",
            m,
        )
        for m in lines
    )
    arms = [m for m in caplog.messages if m.startswith("arm holdout=")]
    assert len(arms) == len(SMALL.holdout_seeds)
    assert all(f" attempt={SMALL.attempts} " in m and '"short": {' in m for m in arms)


def test_a_partly_filled_cell_is_checked_against_what_it_kept(caplog):
    calls = []
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        run(GRADED, ONLY_SMOTE, {"smote": shorting(calls, dropped=1)}, frame=UCI_FRAME)
    lines = [m for m in caplog.messages if m.startswith("shortfall ")]
    assert lines
    assert all(" short=1 quota=" in m and not m.endswith(" quota=1") for m in lines)


@pytest.mark.parametrize(
    ("change", "message"),
    [({"dropped": 2, "claimed": 1}, "rows"), ({"key": "nowhere"}, "does not fit its quota")],
    ids=["kept-rows-disagree", "unknown-cell"],
)
def test_a_shortfall_that_breaks_the_contract_is_refused_and_logged(caplog, change, message):
    with (
        caplog.at_level(logging.INFO, logger="bko.evaluate"),
        pytest.raises(OutputContractError, match=message),
    ):
        run(GRADED, ONLY_SMOTE, {"smote": shorting([], **change)}, frame=UCI_FRAME)
    assert any(m.startswith("contract holdout=0 arm=smote attempt=") for m in caplog.messages)
    assert not any(m.startswith("shortfall ") for m in caplog.messages)


def test_a_shortfall_before_the_last_attempt_is_retried(caplog):
    calls = []
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        run(GRADED, ONLY_SMOTE, {"smote": shorting(calls, until=2)}, frame=UCI_FRAME)
    assert any(m.startswith("retry holdout=0 arm=smote attempt=1 ") for m in caplog.messages)
    assert not any(m.startswith("shortfall ") for m in caplog.messages)


def test_a_replica_short_at_its_limit_still_refuses(caplog):
    replica = dataclasses.replace(ONLY_SMOTE, replica_arms=("smote",))
    calls = []
    with caplog.at_level(logging.INFO, logger="bko.evaluate"), pytest.raises(ShortfallError):
        run(GRADED, replica, {"smote": shorting(calls, replica_only=True)}, frame=UCI_FRAME)
    assert any(m.startswith("refused holdout=0 arm=smote-replica ") for m in caplog.messages)


def test_a_shortfall_with_no_rows_built_still_refuses():
    def short(train, request):
        msg = "short"
        raise ShortfallError(msg, {"x": 1}, number(train.iloc[:0]))

    with pytest.raises(ShortfallError, match="short"):
        run(GRADED, SMALL, {**FAST_ARMS, "smote": short}, frame=UCI_FRAME)


ONE_RECORD = GRADED.drop(
    GRADED.index[[cell == (f"<{EDGE}", "female", 1) for cell in cell_list(GRADED)]][1:]
)
"""`GRADED` with one record left in `<55 female 1`: a structural shortfall for SMOTE."""


def test_a_structural_shortfall_fills_partially_on_attempt_1(caplog):
    assert SMALL.attempts > 1
    calls = []

    def counting(train, request):
        calls.append(request.seed)
        return FAST_ARMS["smote"](train, request)

    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        scores = run(ONE_RECORD, ONLY_SMOTE, {"smote": counting}, frame=UCI_FRAME)
    assert len(calls) == len(SMALL.holdout_seeds)
    assert {score.holdout for score in scores} == set(SMALL.holdout_seeds)
    assert not any(m.startswith("retry ") for m in caplog.messages)
    lines = [m for m in caplog.messages if m.startswith("shortfall ")]
    assert len(lines) == len(SMALL.holdout_seeds)
    assert all(
        re.fullmatch(
            rf"shortfall holdout=\d+ arm=smote attempt=1 cell=<{EDGE}/female/1 "
            r"short=(\d+) quota=\1",
            m,
        )
        for m in lines
    )
    arms = [m for m in caplog.messages if m.startswith("arm holdout=")]
    assert all(" attempt=1 " in m and '"short": {' in m for m in arms)


def test_a_plain_refusal_on_a_structural_request_keeps_its_retries(caplog):
    calls = []

    def flaky(train, request):
        calls.append(request.seed)
        if len(calls) % 2 == 1:
            msg = "past its budget"
            raise BaselineError(msg)
        return FAST_ARMS["smote"](train, request)

    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        run(ONE_RECORD, ONLY_SMOTE, {"smote": flaky}, frame=UCI_FRAME)
    assert len(calls) == 2 * len(SMALL.holdout_seeds)
    retries = [m for m in caplog.messages if m.startswith("retry ")]
    assert [m.split()[1:4] for m in retries] == [
        [f"holdout={seed}", "arm=smote", "attempt=1"] for seed in SMALL.holdout_seeds
    ]
    shortfalls = [m for m in caplog.messages if m.startswith("shortfall ")]
    assert len(shortfalls) == len(SMALL.holdout_seeds)
    assert all(" attempt=2 " in m for m in shortfalls)
    arms = [m for m in caplog.messages if m.startswith("arm holdout=")]
    assert all(" attempt=2 " in m for m in arms)


def test_a_structural_shortfall_on_a_replica_still_refuses(caplog):
    replica = dataclasses.replace(ONLY_SMOTE, replica_arms=("smote",), threshold=2, fill_to=3)
    with caplog.at_level(logging.INFO, logger="bko.evaluate"), pytest.raises(ShortfallError):
        run(ONE_RECORD, replica, {"smote": FAST_ARMS["smote"]}, frame=UCI_FRAME)
    assert any(m.startswith("refused holdout=0 arm=smote-replica ") for m in caplog.messages)


def test_a_refusal_is_retried_with_a_derived_seed():
    seeds = []

    def flaky(train, request):
        seeds.append(request.seed)
        if request.seed in SMALL.holdout_seeds:
            msg = "refused"
            raise BaselineError(msg)
        return FAST_ARMS["smote"](train, request)

    experiment = dataclasses.replace(
        SMALL, arms=("smote",), candidate="smote", comparators=(), replica_arms=()
    )
    scores = run(GRADED, experiment, {"smote": flaky}, frame=UCI_FRAME)
    expected = [s for seed in SMALL.holdout_seeds for s in (seed, derived_seed(seed, "attempt", 1))]
    assert seeds == expected
    assert {score.holdout for score in scores} == set(SMALL.holdout_seeds)


def test_a_contract_break_is_not_retried():
    calls = []

    def broken(train, request):
        calls.append(request.seed)
        msg = "broken"
        raise OutputContractError(msg)

    with pytest.raises(OutputContractError, match="broken"):
        run(GRADED, SMALL, {**FAST_ARMS, "smote": broken}, frame=UCI_FRAME)
    assert len(calls) == 1


def test_a_rerun_writes_the_same_bytes():
    experiment = dataclasses.replace(
        SMALL,
        arms=("none", "smote", "adasyn", "ctgan", "tvae"),
        candidate="smote",
        comparators=(),
        replica_arms=("smote",),
    )
    first = render(run(GRADED, experiment, FAST_ARMS, frame=UCI_FRAME))
    np.random.seed(1)
    assert render(run(GRADED, experiment, FAST_ARMS, frame=UCI_FRAME)) == first


def test_a_runner_that_changes_its_input_changes_nothing_else():
    def vandal(train, request):
        baseline = FAST_ARMS["smote"](train, request)
        train.loc[:, "target"] = 0
        train.drop(train.index[:10], inplace=True)
        return baseline

    experiment = dataclasses.replace(
        SMALL, arms=("smote", "none"), candidate="smote", comparators=(), replica_arms=("smote",)
    )
    honest = render(run(GRADED, experiment, FAST_ARMS, frame=UCI_FRAME))
    assert (
        render(run(GRADED, experiment, {**FAST_ARMS, "smote": vandal}, frame=UCI_FRAME)) == honest
    )


def test_every_retry_is_logged_with_its_own_derived_seed(caplog):
    seeds = []

    def twice_refused_replica(train, request):
        if request.rows == len(train):
            seeds.append(request.seed)
            if len(seeds) < 3:
                msg = "refused"
                raise BaselineError(msg)
        return FAST_ARMS["smote"](train, request)

    experiment = dataclasses.replace(
        SMALL, arms=("smote",), candidate="smote", comparators=(), replica_arms=("smote",)
    )
    with caplog.at_level(logging.INFO, logger="bko.evaluate.run"):
        run(GRADED, experiment, {"smote": twice_refused_replica}, frame=UCI_FRAME)
    first = SMALL.holdout_seeds[0]
    derived = [derived_seed(first, "attempt", attempt) for attempt in (1, 2)]
    assert seeds[:3] == [first, *derived]
    assert f"retry holdout={first} arm=smote-replica attempt=2 reason=refused" in caplog.text
    assert f"arm holdout={first} arm=smote-replica attempt=3 seed={derived[1]}" in caplog.text


PAIR = dataclasses.replace(
    SMALL, arms=("none", "smote"), candidate="smote", comparators=("none",), replica_arms=()
)


def fits_on(monkeypatch: pytest.MonkeyPatch) -> list[list[int]]:
    """Record the test index every classifier in the harness is scored on."""
    seen: list[list[int]] = []
    real = run_module.probabilities

    def spy(training, test, **options):
        seen.append(list(test.index))
        return real(training, test, **options)

    monkeypatch.setattr(run_module, "probabilities", spy)
    return seen


def test_utility_reads_the_split_without_a_copied_record(monkeypatch):
    parts = split(GRADED, seed=0, edges=UCI_FRAME.edges)
    cells = cell_list(parts.test)
    copied = parts.test[[cell == (f">={EDGE}", "female", 1) for cell in cells]].iloc[[0]]

    def copier(train, request):
        baseline = FAST_ARMS["smote"](train, request)
        added = baseline.added.copy()
        added.loc[-19, list(FRAME_COLUMNS)] = copied.loc[:, list(FRAME_COLUMNS)].to_numpy()[0]
        return dataclasses.replace(baseline, added=added.astype(dict(FRAME_DTYPES)))

    seen = fits_on(monkeypatch)
    scores = score_holdout(
        parts, PAIR, {"none": FAST_ARMS["none"], "smote": copier}, frame=UCI_FRAME
    )
    assert seen
    assert all(copied.index[0] not in index for index in seen)
    found = {(s.arm, s.measure): s.value for s in scores if s.source is Source.ADDED}
    assert found["smote", Measure.TEST_COPIES] == pytest.approx(1.0)
    assert found["none", Measure.TEST_COPIES] == pytest.approx(0.0)


def test_a_holdout_with_no_copy_scores_the_whole_test_split(monkeypatch):
    parts = split(GRADED, seed=0, edges=UCI_FRAME.edges)
    seen = fits_on(monkeypatch)
    score_holdout(parts, PAIR, FAST_ARMS, frame=UCI_FRAME)
    assert {len(index) for index in seen} == {len(parts.test)}


def test_a_drain_writes_candidates_rows_per_holdout():
    scores = run(
        GRADED,
        PAIR,
        FAST_ARMS,
        drains={"smote": lambda seed, source: {Measure.GENERATED: 1.0}},
        frame=UCI_FRAME,
    )
    rows = [s for s in scores if s.source is Source.CANDIDATES]
    assert sorted((s.holdout, s.arm, s.measure) for s in rows) == [
        (0, "smote", Measure.GENERATED),
        (1, "smote", Measure.GENERATED),
    ]


def test_a_replica_arms_drain_is_read_after_each_fill():
    counted = iter(range(1, 5))
    experiment = dataclasses.replace(PAIR, replica_arms=("smote",))

    asked: list[tuple[int, Source]] = []

    def drain(seed: int, source: Source) -> dict[Measure, float]:
        asked.append((seed, source))
        return {Measure.GENERATED: float(next(counted))}

    scores = run(GRADED, experiment, FAST_ARMS, drains={"smote": drain}, frame=UCI_FRAME)
    assert asked == [
        (0, Source.CANDIDATES),
        (0, Source.REPLICA_CANDIDATES),
        (1, Source.CANDIDATES),
        (1, Source.REPLICA_CANDIDATES),
    ]
    found = {(s.holdout, s.source): s.value for s in scores if s.measure is Measure.GENERATED}
    assert found == {
        (0, Source.CANDIDATES): 1.0,
        (0, Source.REPLICA_CANDIDATES): 2.0,
        (1, Source.CANDIDATES): 3.0,
        (1, Source.REPLICA_CANDIDATES): 4.0,
    }


def test_an_arm_filling_no_replica_is_drained_once_per_holdout_beside_one_that_does():
    calls: Counter[str] = Counter()

    def counter(name: str):
        def drain(seed: int, source: Source) -> dict[Measure, float]:
            calls[name] += 1
            return {Measure.GENERATED: float(calls[name])}

        return drain

    experiment = dataclasses.replace(PAIR, arms=("none", "smote", "tvae"), replica_arms=("smote",))
    drains = {name: counter(name) for name in ("smote", "tvae")}
    scores = run(GRADED, experiment, FAST_ARMS, drains=drains, frame=UCI_FRAME)
    found = {
        (s.holdout, s.arm, s.source): s.value for s in scores if s.measure is Measure.GENERATED
    }
    assert found == {
        (0, "smote", Source.CANDIDATES): 1.0,
        (0, "smote", Source.REPLICA_CANDIDATES): 2.0,
        (0, "tvae", Source.CANDIDATES): 1.0,
        (1, "smote", Source.CANDIDATES): 3.0,
        (1, "smote", Source.REPLICA_CANDIDATES): 4.0,
        (1, "tvae", Source.CANDIDATES): 2.0,
    }


def test_training_copies_are_counted_per_arm_and_fill(caplog):
    parts = split(GRADED, seed=0, edges=UCI_FRAME.edges)
    copying = dataclasses.replace(PAIR, replica_arms=("smote",))
    asked: list[int] = []

    def verbatim(train, request):
        rows = sum(request.quota.values())
        asked.append(rows)
        added = number(train.loc[:, list(FRAME_COLUMNS)].iloc[:rows])
        return Baseline(added=added, weights=unit_weights(train), params={})

    with caplog.at_level(logging.INFO, logger="bko.evaluate.run"):
        score_holdout(
            parts, copying, {"none": FAST_ARMS["none"], "smote": verbatim}, frame=UCI_FRAME
        )
    gap, replica = asked
    assert f"training holdout=0 arm=smote fill=gap rows={gap} copies={gap}" in caplog.text
    assert (
        f"training holdout=0 arm=smote fill=replica rows={replica} copies={replica}" in caplog.text
    )
    assert "training holdout=0 arm=none fill=gap rows=0 copies=0" in caplog.text


def test_smote_rows_copy_fewer_training_rows_than_it_adds(caplog):
    with caplog.at_level(logging.INFO, logger="bko.evaluate.run"):
        score_holdout(
            split(GRADED, seed=0, edges=UCI_FRAME.edges), PAIR, FAST_ARMS, frame=UCI_FRAME
        )
    found = re.search(r"training holdout=0 arm=smote fill=gap rows=(\d+) copies=(\d+)", caplog.text)
    assert found is not None
    rows, copied = map(int, found.groups())
    assert copied < rows
