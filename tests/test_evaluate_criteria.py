"""The criteria: Nadeau and Bengio's interval, paired by holdout, and three-valued verdicts."""

from __future__ import annotations

import dataclasses
import math

import pytest

from bko.evaluate.criteria import (
    CONTRAST_COLUMNS,
    Interval,
    Verdict,
    corrected_interval,
    judge,
    refuting_confidence,
    table,
    verdict,
)
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import E1
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

SEEDS = (0, 1, 2, 3)
EXPERIMENT = dataclasses.replace(
    E1, holdout_seeds=SEEDS, arms=("none", "smote"), comparators=("none",), replica_arms=()
)


def width(interval: Interval) -> float:
    return interval.high - interval.low


def test_the_interval_is_nadeau_and_bengios():
    interval = corrected_interval([0.1, 0.2, 0.3, 0.4], ratio=1 / 3, confidence=0.95, family=1)
    assert interval.mean == pytest.approx(0.25)
    assert (interval.low, interval.high) == pytest.approx((-0.06379, 0.56379), abs=1e-5)
    assert interval.holdouts == 4
    assert interval.refuting_high == pytest.approx(interval.high)


def test_refuting_confidence_splits_the_miss_rate_among_the_family():
    assert refuting_confidence(0.95, 1) == pytest.approx(0.95)
    assert refuting_confidence(0.95, 12) == pytest.approx(1 - 0.05 / 12)


def test_the_refuting_bound_is_the_interval_at_the_family_wise_level():
    contrast = [0.1, 0.2, 0.3, 0.4]
    family = corrected_interval(contrast, ratio=1 / 3, confidence=0.95, family=12)
    level = corrected_interval(
        contrast, ratio=1 / 3, confidence=refuting_confidence(0.95, 12), family=1
    )
    assert family.refuting_high == pytest.approx(level.high)
    assert (family.low, family.high) == pytest.approx((-0.06379, 0.56379), abs=1e-5)


def test_the_correction_widens_a_plain_t_interval_by_sqrt_one_plus_j_ratio():
    plain = corrected_interval([0.1, 0.2, 0.3, 0.4], ratio=0.0, confidence=0.95, family=1)
    corrected = corrected_interval([0.1, 0.2, 0.3, 0.4], ratio=1 / 3, confidence=0.95, family=1)
    assert width(corrected) / width(plain) == pytest.approx(math.sqrt(1 + 4 / 3))


def test_a_constant_contrast_has_a_point_interval():
    interval = corrected_interval([0.25] * 4, ratio=0.25, confidence=0.95, family=12)
    bounds = (interval.low, interval.high, interval.refuting_high)
    assert bounds == pytest.approx((0.25, 0.25, 0.25))


def test_a_criterion_without_intervals_is_refused():
    with pytest.raises(EvaluationError, match="at least one interval"):
        verdict([])


@pytest.mark.parametrize("contrast", [[0.1], [0.1, math.nan, 0.2]], ids=["one holdout", "nan"])
def test_an_interval_needs_two_finite_contrasts(contrast):
    with pytest.raises(EvaluationError):
        corrected_interval(contrast, ratio=0.25, confidence=0.95, family=1)


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ([(0.1, 0.3), (0.01, 0.2)], Verdict.MET),
        ([(0.1, 0.3), (-0.3, -0.01)], Verdict.NOT_MET),
        ([(0.1, 0.3), (-0.1, 0.2)], Verdict.INCONCLUSIVE),
        ([(0.0, 0.2)], Verdict.INCONCLUSIVE),
        ([(-0.2, 0.0)], Verdict.INCONCLUSIVE),
    ],
    ids=["all above", "one below", "one straddles", "touches zero", "tops out at zero"],
)
def test_the_verdict_reads_every_interval(bounds, expected):
    assert verdict([Interval(0.0, low, high, high, 4) for low, high in bounds]) is expected


def test_not_met_reads_the_refuting_bound():
    above = Interval(0.2, 0.1, 0.3, 0.35, 4)
    below = Interval(-0.15, -0.3, -0.01, 0.02, 4)
    assert verdict([above, below]) is Verdict.INCONCLUSIVE
    refuted = dataclasses.replace(below, refuting_high=-0.001)
    assert verdict([above, refuted]) is Verdict.NOT_MET


def series(arm, source, measure, scope, values):
    return [
        Score(seed, arm, source, measure, scope, value)
        for seed, value in zip(SEEDS, values, strict=True)
    ]


def scored(gain, *, privacy=0.1):
    rows = series(HELD_OUT_ARM, Source.HELD_OUT, Measure.HELD_OUT_RATIO, Scope.ALL, [0.25] * 4)
    rows += series(HELD_OUT_ARM, Source.HELD_OUT, Measure.DCR_P5, Scope.ALL, [0.1] * 4)
    rows += series("bko", Source.ADDED, Measure.DCR_P5, Scope.ALL, [privacy] * 4)
    for measure in (Measure.RECALL, Measure.AUC_PR):
        for scope in Scope:
            rows += series("none", Source.AUGMENTED, measure, scope, [0.5] * 4)
            rows += series("bko", Source.AUGMENTED, measure, scope, [0.5 + g for g in gain])
    return rows


def verdicts(contrasts):
    return {contrast.criterion: contrast.verdict for contrast in contrasts}


@pytest.mark.parametrize(
    ("gain", "expected"),
    [
        ([0.05, 0.06, 0.05, 0.06], Verdict.MET),
        ([-0.1] * 4, Verdict.NOT_MET),
        ([0.0, 0.05, 0.0, 0.05], Verdict.INCONCLUSIVE),
    ],
    ids=["separated", "worse", "straddles zero"],
)
def test_separation_needs_every_interval_above_the_margin(gain, expected):
    contrasts = judge(scored(gain), EXPERIMENT)
    assert verdicts(contrasts) == {
        "primary": expected,
        "secondary": expected,
        "privacy": Verdict.MET,
    }
    assert len(contrasts) == 5


def test_a_loss_inside_the_family_wise_bound_is_inconclusive():
    contrast = [-0.04, -0.02, -0.04, -0.02]
    gain = [value + EXPERIMENT.margin for value in contrast]
    contrasts = judge(scored(gain), EXPERIMENT)
    separation = [c for c in contrasts if c.criterion != "privacy"]
    assert all(c.interval.high < 0 < c.interval.refuting_high for c in separation)
    assert verdicts(contrasts) == {
        "primary": Verdict.INCONCLUSIVE,
        "secondary": Verdict.INCONCLUSIVE,
        "privacy": Verdict.MET,
    }


@pytest.mark.parametrize(("privacy", "expected"), [(0.1, Verdict.MET), (0.05, Verdict.NOT_MET)])
def test_privacy_compares_with_a_share_of_the_held_out_distance(privacy, expected):
    assert verdicts(judge(scored([0.05] * 4, privacy=privacy), EXPERIMENT))["privacy"] is expected


def test_a_missing_candidate_is_refused():
    rows = [score for score in scored([0.05] * 4) if score.arm != "bko"]
    with pytest.raises(EvaluationError, match="no arm 'bko'"):
        judge(rows, EXPERIMENT)


def test_another_candidate_is_judged_against_the_rest():
    rows = [
        dataclasses.replace(score, arm="smote") if score.arm == "bko" else score
        for score in scored([0.05] * 4)
    ]
    contrasts = judge(rows, EXPERIMENT, candidate="smote")
    assert {contrast.arm for contrast in contrasts} == {"smote"}
    assert {contrast.against for contrast in contrasts} == {"none"}


def test_unpaired_holdouts_are_refused():
    rows = [s for s in scored([0.05] * 4) if not (s.arm == "bko" and s.holdout == 3)]
    with pytest.raises(EvaluationError, match="different holdouts"):
        judge(rows, EXPERIMENT)


def test_scores_from_other_holdouts_are_refused():
    with pytest.raises(EvaluationError, match="not the experiment's"):
        judge(scored([0.05] * 4), dataclasses.replace(EXPERIMENT, holdout_seeds=(0, 1)))


def test_a_candidate_without_added_rows_has_no_privacy_series():
    rows = [s for s in scored([0.05] * 4) if s.source is not Source.ADDED]
    with pytest.raises(EvaluationError, match="hold no dcr-p5"):
        judge(rows, EXPERIMENT)


def test_a_candidate_without_comparators_is_refused():
    lonely = dataclasses.replace(EXPERIMENT, comparators=())
    with pytest.raises(EvaluationError, match="no comparator"):
        judge(scored([0.05] * 4), lonely)


def test_differing_counts_the_holdouts_where_the_scores_differ():
    contrasts = judge(scored([0.0, 0.05, 0.0, 0.05]), EXPERIMENT)
    assert {contrast.differing for contrast in contrasts if contrast.criterion == "primary"} == {2}


def test_the_table_has_one_row_per_contrast():
    lines = table(judge(scored([0.05, 0.06, 0.05, 0.06]), EXPERIMENT)).splitlines()
    assert lines[0] == ",".join(CONTRAST_COLUMNS)
    assert len(lines) == 6
    assert lines[1].startswith("primary,all,met,bko,none,recall,")


def test_a_gain_inside_the_margin_is_not_separation():
    gain = [0.01] * 4
    margined = dataclasses.replace(EXPERIMENT, margin=0.02)
    assert verdicts(judge(scored(gain), margined))["primary"] is Verdict.NOT_MET
    assert verdicts(judge(scored(gain), EXPERIMENT))["primary"] is Verdict.MET


def test_the_largest_held_out_ratio_widens_every_interval():
    gain = [0.05, 0.06, 0.05, 0.06]
    rows = [
        dataclasses.replace(s, value=0.5)
        if s.measure is Measure.HELD_OUT_RATIO and s.holdout == 3
        else s
        for s in scored(gain)
    ]
    first = judge(rows, EXPERIMENT)[0]
    contrast = [0.5 + g - 0.5 - EXPERIMENT.margin for g in gain]
    expected = corrected_interval(contrast, ratio=0.5, confidence=EXPERIMENT.confidence, family=2)
    assert (first.interval.low, first.interval.high) == pytest.approx((expected.low, expected.high))


def test_a_series_missing_a_holdout_on_both_sides_is_refused():
    def kept(score):
        recall = score.measure is Measure.RECALL and score.scope is Scope.ALL
        return not (recall and score.source is Source.AUGMENTED and score.holdout == 3)

    rows = [score for score in scored([0.05] * 4) if kept(score)]
    with pytest.raises(EvaluationError, match="different holdouts"):
        judge(rows, EXPERIMENT)
