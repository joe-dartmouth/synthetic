"""The scores CSV: written in order, read back exactly, and refused when malformed."""

from __future__ import annotations

import math

import pytest

from bko.evaluate.errors import ScoresFormatError
from bko.evaluate.membership import SCOPES
from bko.evaluate.scores import SCORE_COLUMNS, Measure, Scope, Score, Source, parse, render


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

SCORES = [
    Score(0, "none", Source.HELD_OUT, Measure.HELD_OUT_RATIO, Scope.ALL, 0.25),
    Score(0, "smote", Source.AUGMENTED, Measure.RECALL, Scope.FLAGGED, math.nan),
    Score(1, "smote", Source.AUGMENTED, Measure.AUC_PR, Scope.ALL, 0.1 + 0.2),
]


def test_render_writes_one_row_per_score_in_order():
    lines = render(SCORES).splitlines()
    assert lines[0] == ",".join(SCORE_COLUMNS)
    assert lines[1] == "0,none,held-out,held-out-ratio,all,0.25"
    assert lines[2] == "0,smote,augmented,recall,flagged-subgroups,"


def test_parse_reads_render_back_exactly():
    back = parse(render(SCORES))
    assert back[0] == SCORES[0]
    assert back[2] == SCORES[2]
    assert back[1].measure is Measure.RECALL
    assert math.isnan(back[1].value)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a,b\n1,2\n",
        render(SCORES).replace("recall", "accuracy"),
        render(SCORES[:1] * 2),
        "holdout,arm,source,measure,scope,value\nx,none,held-out,rows,all,1\n",
        "holdout,arm,source,measure,scope,value\n1.5,none,held-out,rows,all,1\n",
        "holdout,arm,source,measure,scope,value\n1,,held-out,rows,all,1\n",
    ],
    ids=[
        "empty",
        "columns",
        "unknown measure",
        "repeated key",
        "bad holdout",
        "fractional holdout",
        "empty arm",
    ],
)
def test_a_malformed_scores_file_is_refused(text):
    with pytest.raises(ScoresFormatError):
        parse(text)


@pytest.mark.parametrize("arm", ["None", "NA", "null", "nan"])
def test_an_arm_named_like_a_missing_value_reads_back_as_itself(arm):
    score = Score(0, arm, Source.ADDED, Measure.ROWS, Scope.ALL, 1.0)
    assert parse(render([score])) == [score]


def test_a_fills_candidates_and_its_replica_candidates_are_distinct_rows():
    rows = [
        Score(0, "bko", Source.CANDIDATES, Measure.GENERATED, Scope.ALL, 20.0),
        Score(0, "bko", Source.REPLICA_CANDIDATES, Measure.GENERATED, Scope.ALL, 238.0),
    ]
    assert render(rows).splitlines()[2] == "0,bko,replica-candidates,generated,all,238.0"
    assert parse(render(rows)) == rows


def test_the_boosting_models_measures_read_back_and_an_old_file_still_parses():
    boosted = [
        *SCORES,
        Score(1, "smote", Source.AUGMENTED, Measure.RECALL_HGB, Scope.ALL, 0.5),
        Score(1, "smote", Source.AUGMENTED, Measure.PRECISION_HGB, Scope.ALL, 0.25),
        Score(1, "smote", Source.AUGMENTED, Measure.AUC_PR_HGB, Scope.FLAGGED, math.nan),
    ]
    text = render(boosted)
    assert text.splitlines()[4] == "1,smote,augmented,recall-hgb,all,0.5"
    assert parse(text)[3:5] == boosted[3:5]
    old = "".join(line for line in text.splitlines(keepends=True) if "-hgb," not in line)
    assert old == render(SCORES)
    assert parse(old)[0] == SCORES[0]


PANEL_SUFFIXES = ("rf", "et", "mlp", "nb")
NEW_MEASURES = [
    *(
        f"{base}-{suffix}"
        for suffix in PANEL_SUFFIXES
        for base in ("recall", "precision", "auc-pr")
    ),
    *(
        f"{base}{suffix}"
        for base in ("brier", "calibration-in-the-large")
        for suffix in ("", "-hgb", *(f"-{suffix}" for suffix in PANEL_SUFFIXES))
    ),
]


def test_parse_round_trips_every_new_measure():
    rows = [
        Score(2, "bko", Source.AUGMENTED, Measure(name), scope, 0.125)
        for name in NEW_MEASURES
        for scope in Scope
    ]
    text = render(rows)
    assert [line.split(",")[3] for line in text.splitlines()[1::2]] == NEW_MEASURES
    assert parse(text) == rows
    assert Measure.CALIBRATION.value == "calibration-in-the-large"
    assert Measure.BRIER_NB.value == "brier-nb"


def test_parse_refuses_a_measure_of_an_unknown_model():
    with pytest.raises(ScoresFormatError, match="recall-xgb"):
        parse(render(SCORES).replace("auc-pr", "recall-xgb"))


PRIVACY_MEASURES = [
    *(f"mi-{statistic}-{scope}" for statistic in ("auc", "tpr01") for scope in SCOPES),
    *(f"mi-tpr001-{scope}" for scope in ("outside", "label")),
    "nndr-p5",
    "nndr-median",
    "nndr-ties",
    "nndr-exact",
]


def test_parse_round_trips_every_privacy_measure():
    rows = [
        Score(0, "real", Source.ADDED, Measure(name), Scope.ALL, 0.5) for name in PRIVACY_MEASURES
    ]
    rows.append(Score(0, "none", Source.HELD_OUT, Measure.NNDR_P5, Scope.ALL, math.nan))
    back = parse(render(rows))
    assert back[:-1] == rows[:-1]
    assert math.isnan(back[-1].value)


@pytest.mark.parametrize("name", ["mi-auc-all", "mi-tpr001-cell", "mi-tpr05-outside"])
def test_parse_refuses_a_privacy_measure_the_reading_does_not_write(name):
    with pytest.raises(ScoresFormatError, match=name):
        parse(render(SCORES).replace("auc-pr", name))
