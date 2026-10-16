"""The scores CSV: written in order, read back exactly, and refused when malformed."""

from __future__ import annotations

import math

import pytest

from bko.evaluate.errors import ScoresFormatError
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
    ],
    ids=["empty", "columns", "unknown measure", "repeated key", "bad holdout"],
)
def test_a_malformed_scores_file_is_refused(text):
    with pytest.raises(ScoresFormatError):
        parse(text)
