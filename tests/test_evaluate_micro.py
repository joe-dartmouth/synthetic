"""The `bko-micro` arm and its readers: trace, attribution and donor reuse name every donor.

ADR `research/sampler-donor-limits` fixes the draw; ADR `research/record-explainability` the
trace's `averaged` ids and the `donor-mean` attribution.
"""

from __future__ import annotations

import dataclasses
import io
import json
from collections import Counter
from pathlib import Path
from typing import Final

import pytest

from bko.evaluate.__main__ import main
from bko.evaluate.attribution import Attribution, arm_kind, attribution, classify
from bko.evaluate.context import Context
from bko.evaluate.direct_arms import CAPPED_ARM, DIRECT_ARMS, MICRO_ARM
from bko.evaluate.donor_attacks import donations
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.trace_lines import Trace
from bko.generate.direct import Draw
from tests.test_evaluate_attribution import COLUMNS, MAPPED, _direct, _write
from tests.test_evaluate_direct import SMALL_G2, g2_context
from tests.test_evaluate_trace import _scored, never


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

MICRO_G2: Final = dataclasses.replace(
    SMALL_G2,
    arms=("none", DIRECT_ARM, MICRO_ARM),
    candidate=MICRO_ARM,
    comparators=("none", DIRECT_ARM),
    replica_arms=(),
)
"""Two holdouts, one baseline, `bko-direct` and the candidate `bko-micro`."""

NUMERIC: Final = frozenset({"trestbps", "chol", "thalach", "oldpeak"})
"""UCI's numeric fields but `age`: the fields `bko-micro` averages."""


def _context() -> Context:
    """`MICRO_G2` registered as `g2`, with a send that fails."""
    ctx, _ = g2_context()
    return dataclasses.replace(ctx, runs={"g2": NamedRun(MICRO_G2)}, send=never)


@pytest.fixture(scope="module")
def scored(tmp_path_factory) -> Path:
    """A finished `g2` folder of `MICRO_G2`, its data root beside it."""
    base = tmp_path_factory.mktemp("micro")
    folder = base / "run"
    argv = ["score", "g2", "--candidates", str(folder / "holdout-candidates.jsonl")]
    _scored(folder, _context(), [*argv, "--data-dir", str(base / "data")])
    return folder


def _trace(folder: Path, arm: str) -> list[dict]:
    out = io.StringIO()
    data = folder.parent / "data"
    argv = ["trace", "g2", "--folder", str(folder), "--data-dir", str(data), "--holdout", "0"]
    code = main([*argv, "--arm", arm], context=_context(), out=out)
    assert code == 0
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_the_micro_arm_draws_on_the_capped_limits():
    capped, micro = DIRECT_ARMS[CAPPED_ARM], DIRECT_ARMS[MICRO_ARM]
    assert micro.microaggregate is not None
    assert dataclasses.replace(micro, microaggregate=None) == capped


def test_the_trace_names_every_donor(scored):
    lines = _trace(scored, MICRO_ARM)
    assert lines
    for line in lines:
        averaged = line["averaged"]
        assert set(averaged) == NUMERIC
        for field, ids in averaged.items():
            assert len(set(ids)) == len(ids) == 3
            assert ids[0] == line["sources"][field]


def test_a_default_trace_line_keeps_its_bytes():
    line = Trace(
        0,
        CAPPED_ARM,
        "gap",
        0,
        cell="<55/female/1",
        record={"chol": 240, "cp": 4},
        sources={"chol": 3, "cp": 3},
        fallback=("chol",),
    ).line()
    assert line == (
        '{"arm": "bko-capped", "call": null, "cell": "<55/female/1", "closed": [], '
        '"concepts": [], "failed": [], "fallback": ["chol"], "fill": "gap", "holdout": 0, '
        '"outcome": null, "record": {"chol": 240, "cp": 4}, "record_id": 0, "request": null, '
        '"sources": {"chol": 3, "cp": 3}, "untraced": null}\n'
    )


def test_a_scored_direct_arms_trace_holds_no_averaged_ids(scored):
    lines = _trace(scored, DIRECT_ARM)
    assert lines
    assert all("averaged" not in line for line in lines)


DRAWN: Final = Draw(
    ("<55", "female", 1),
    {"a": 1, "b": 2.0, "c": 3.5},
    frozenset(),
    frozenset(),
    frozenset(),
    {"a": 7, "b": 7, "c": 8},
    {"b": (7, 4, 5)},
)
"""A micro draw: unit `a`, `b` averaged over records 7, 4 and 5; unit `c` copied from 8."""


def test_donor_reuse_counts_every_averaged_id():
    groups = (("a", "b"), ("c",))
    found = donations([dict(DRAWN.record)] * 2, [DRAWN] * 2, groups)
    assert found == Counter({7: 2, 4: 2, 5: 2, 8: 2})
    changed = {**DRAWN.record, "b": 9.0}
    assert donations([changed], [DRAWN], groups) == Counter({8: 1})


def test_an_averaged_field_is_attributed_donor_mean():
    line = _direct(MICRO_ARM, averaged={"chol": [13, 21, 34], "cp": [11, 2, 3]})
    found = classify(line, mapped=MAPPED)
    assert found["chol"] is Attribution.MEAN
    assert found["cp"] is Attribution.KNOWLEDGE
    assert found["age"] is Attribution.DONOR
    assert list(Attribution).index(Attribution.MEAN) == 1
    assert Attribution.MEAN == "donor-mean"


def test_a_trace_line_whose_averaged_is_not_an_object_is_refused():
    with pytest.raises(EvaluationError, match=r"averaged is not a JSON object: \[13, 21, 34\]"):
        arm_kind(_direct(MICRO_ARM, averaged=[13, 21, 34]))


def _attributed(tmp_path: Path, lines: list[dict]) -> list[str]:
    path = _write(tmp_path / "trace.jsonl", lines)
    return attribution([path], run="g2", mapped=MAPPED, columns=COLUMNS).splitlines()


def test_only_a_microaggregating_arms_fill_writes_a_donor_mean_row(tmp_path):
    micro = _direct(MICRO_ARM, averaged={"chol": [13, 21, 34]})
    rows = _attributed(tmp_path, [micro, _direct()])
    assert [row for row in rows if ",donor-mean," in row] == [
        f"0,{MICRO_ARM},direct,donor-mean,field,1"
    ]
    assert sum(row.startswith(f"0,{MICRO_ARM},") for row in rows) == len(Attribution)
    assert sum(row.startswith(f"0,{DIRECT_ARM},") for row in rows) == len(Attribution) - 1


def test_averaged_ids_on_an_arm_that_averages_nothing_are_refused(tmp_path):
    with pytest.raises(EvaluationError, match=f"{CAPPED_ARM} fill has 1 donor-mean fields"):
        _attributed(tmp_path, [_direct(CAPPED_ARM, averaged={"chol": [13, 21, 34]})])


def test_attribution_counts_donor_means_on_a_scored_micro_folder(scored, tmp_path):
    paths = [
        _write(tmp_path / f"{arm}.jsonl", _trace(scored, arm)) for arm in (MICRO_ARM, DIRECT_ARM)
    ]
    out = io.StringIO()
    argv = ["attribution", "g2", "--trace", *map(str, paths)]
    assert main(argv, context=_context(), out=out) == 0
    means = [row.split(",") for row in out.getvalue().splitlines() if ",donor-mean," in row]
    assert [(arm, unit) for _, arm, _, _, unit, _ in means] == [(MICRO_ARM, "field")]
    records = len(_trace(scored, MICRO_ARM))
    assert int(means[0][-1]) == len(NUMERIC) * records > 0
