"""`Experiment.quota_rule`: each flagged cell asked its rule's count, and every reader reads it.

ADR `research/share-based-quota` records the rule; `bko.gaps.quota` holds the counts.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import re
from fractions import Fraction
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from bko.baselines.request import Stratum
from bko.datasets.cells import cell_counts
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM, E1, Experiment, gap_request
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.provenance import experiment_line
from bko.evaluate.run_folder import ASSUMED
from bko.gaps.quota import QuotaRule
from bko.ontology.schema import Codes
from tests._frame_fixtures import SHORT, TOY, TOY_TRAIN
from tests._gaps_fixtures import ANGINA, STENOSIS
from tests.test_evaluate_direct import SMALL_G2, g2_context
from tests.test_evaluate_flagged_cell import FULL, TOY_RUN
from tests.test_evaluate_shared import edit_log, refused, run_folder, shared
from tests.test_evaluate_trace import _scored, never


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

TOY_ROWS: Final = (dataclasses.replace(ANGINA, when=(Codes("sex", frozenset({0})),)), STENOSIS)
"""Two concepts the toy frame can show: one read off `sex`, one off the label."""

LINE: Final = re.compile(
    r"quota-rule holdout=(?P<holdout>\d+) fill=gap key=(?P<key>\S+) rule=(?P<rule>\S+) "
    r"count=(?P<count>\d+) coverage=(?P<coverage>\d+) expected=(?P<expected>\d+) "
    r"asked=(?P<asked>\d+)"
)


def _lines(messages: list[str]) -> list[dict[str, str]]:
    return [found.groupdict() for m in messages if (found := LINE.fullmatch(m))]


def test_the_experiment_line_names_a_quota_rule_only_when_set():
    plain = experiment_line("x", E1, ["lr"], frame="uci")
    assert "quota_rule" not in plain
    ruled = experiment_line(
        "x", dataclasses.replace(E1, quota_rule=QuotaRule.COVERAGE), ["lr"], frame="uci"
    )
    assert " fill_to=20 quota_rule=coverage budget=" in ruled
    assert ruled.replace("quota_rule=coverage ", "") == plain
    cut = dataclasses.replace(E1, cutoff=0.25, quota_rule=QuotaRule.EXPECTED)
    assert " fill_to=20 cutoff=0.25 quota_rule=expected-size budget=" in experiment_line(
        "x", cut, ["lr"], frame="uci"
    )


def test_an_older_experiment_line_reads_at_the_assumed_quota_rule(tmp_path, caplog):
    assert ASSUMED["quota_rule"] == "fill-to"
    folder = run_folder(tmp_path, "run", caplog)
    edit_log(
        folder,
        "experiment ",
        lambda line: line.replace(" fill_to=20 ", " fill_to=20 quota_rule=fill-to ", 1),
    )
    code, text = shared(tmp_path, folder)
    assert code == 0
    assert text == (folder / "scores.csv").read_text(encoding="utf-8")
    edit_log(
        folder,
        "experiment ",
        lambda line: line.replace("quota_rule=fill-to ", "quota_rule=coverage ", 1),
    )
    error = refused(tmp_path, caplog, folder)
    assert error.endswith("differ: {'quota_rule': ('coverage', 'fill-to')}")


def test_gap_request_with_the_default_rule_equals_fill_to_minus_count(caplog):
    caplog.set_level(logging.INFO)
    filled = dataclasses.replace(TOY_RUN, fill_to=26)
    counts = {
        (str(band), str(sex), int(label)): int(count)
        for band, sex, label, count, _ in cell_counts(TOY_TRAIN, edges=TOY.edges).itertuples(
            index=False
        )
    }
    expected = {cell: 26 - counts[cell] for cell in SHORT}
    plain = gap_request(TOY_TRAIN, filled, seed=0, frame=TOY)
    assert plain.quota == expected
    given = gap_request(TOY_TRAIN, filled, seed=0, frame=TOY, rows=TOY_ROWS, name="toy")
    assert (given.quota, given.strata, given.seed) == (plain.quota, plain.strata, plain.seed)
    assert not _lines([r.getMessage() for r in caplog.records])


def test_a_share_rule_without_rows_is_refused():
    ruled = dataclasses.replace(TOY_RUN, quota_rule=QuotaRule.COVERAGE)
    with pytest.raises(EvaluationError, match="needs the data's field rows"):
        gap_request(TOY_TRAIN, ruled, seed=0, frame=TOY)


def test_a_rule_beside_a_flagged_cell_is_refused():
    message = "cannot stand beside a flagged cell or a synthetic share"
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(TOY_RUN, flagged_cell=FULL, quota_rule=QuotaRule.COVERAGE)
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(
            TOY_RUN,
            flagged_cell=FULL,
            synthetic_share=Fraction(1, 4),
            quota_rule=QuotaRule.RECOMMENDED,
        )
    named = dataclasses.replace(TOY_RUN, flagged_cell=FULL)
    assert named.quota_rule is QuotaRule.FILL_TO


def _quota(experiment: Experiment) -> dict[Stratum, int]:
    """`experiment`'s gap request's quota on the toy split at seed 3; empty when none is asked."""
    try:
        request = gap_request(TOY_TRAIN, experiment, seed=3, frame=TOY, rows=TOY_ROWS, name="toy")
    except EvaluationError as error:
        if "nothing to fill" not in str(error):
            raise
        return {}
    return dict(request.quota)


def test_the_quota_rule_line_is_logged_only_off_the_default(caplog):
    caplog.set_level(logging.INFO)
    for rule in QuotaRule:
        caplog.clear()
        ruled = dataclasses.replace(TOY_RUN, quota_rule=rule)
        quota = _quota(ruled)
        lines = _lines([r.getMessage() for r in caplog.records])
        if rule is QuotaRule.FILL_TO:
            assert not lines
            continue
        assert [line["key"] for line in lines] == ["/".join(map(str, cell)) for cell in SHORT]
        for line in lines:
            assert (line["holdout"], line["rule"]) == ("3", str(rule))
            count, covered, sized = (int(line[k]) for k in ("count", "coverage", "expected"))
            value = {
                QuotaRule.EXPECTED: sized,
                QuotaRule.RECOMMENDED: max(covered, sized),
            }.get(rule, covered * {"coverage": 1, "coverage-x2": 2, "coverage-x4": 4}.get(rule, 8))
            assert int(line["asked"]) == max(0, value - count)
        asked = {
            cell: int(line["asked"])
            for cell, line in zip(SHORT, lines, strict=True)
            if int(line["asked"])
        }
        assert quota == asked


RULED_G2: Final = dataclasses.replace(SMALL_G2, replica_arms=(), quota_rule=QuotaRule.COVERAGE)
"""`SMALL_G2` with no replica, each flagged cell asked its coverage."""


def _context(runs_dir: Path | None = None) -> Context:
    """`RULED_G2` registered as `g2`, with a send that fails."""
    ctx, _ = g2_context()
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(RULED_G2)}, send=never)
    return ctx if runs_dir is None else dataclasses.replace(ctx, runs_dir=runs_dir)


@pytest.fixture(scope="module")
def scored(tmp_path_factory) -> Path:
    """A finished `g2` folder of `RULED_G2`, its data root beside it."""
    base = tmp_path_factory.mktemp("quota")
    folder = base / "run"
    argv = ["score", "g2", "--candidates", str(folder / "holdout-candidates.jsonl")]
    _scored(folder, _context(), [*argv, "--data-dir", str(base / "data")])
    return folder


def _messages(folder: Path) -> list[str]:
    return [
        line.split(": ", 1)[1]
        for line in (folder / "scores.log").read_text(encoding="utf-8").splitlines()
    ]


def test_a_coverage_rule_run_scores_end_to_end(scored):
    messages = _messages(scored)
    (experiment,) = [m for m in messages if m.startswith("experiment ")]
    assert " quota_rule=coverage " in experiment
    lines = _lines(messages)
    flagged = {m for m in messages if m.startswith("flagged holdout=")}
    assert len(flagged) == len(RULED_G2.holdout_seeds)
    for holdout in RULED_G2.holdout_seeds:
        mine = [line for line in lines if line["holdout"] == str(holdout)]
        (cells,) = [m for m in flagged if m.startswith(f"flagged holdout={holdout} ")]
        assert [line["key"] for line in mine] == cells.split("cells=")[1].split(",")
        asked = {line["key"]: int(line["asked"]) for line in mine if int(line["asked"])}
        assert asked
        drawn = {
            found["key"]: int(found["rows"])
            for m in messages
            if (
                found := re.match(
                    rf"direct holdout={holdout} fill=gap key=(?P<key>\S+) rows=(?P<rows>\d+) ", m
                )
            )
        }
        assert drawn == asked
        for arm, added in ((DIRECT_ARM, sum(asked.values())), ("none", 0)):
            (line,) = [m for m in messages if m.startswith(f"arm holdout={holdout} arm={arm} ")]
            assert f" added={added} " in line


def test_shared_rescores_a_run_under_a_share_rule(scored):
    out = io.StringIO()
    argv = ["shared", "g2", "--folder", str(scored), "--data-dir", str(scored.parent / "data")]
    assert main(argv, context=_context(), out=out) == 0
    assert out.getvalue() == (scored / "scores.csv").read_text(encoding="utf-8")


def test_trace_reads_a_run_under_a_share_rule(scored):
    out = io.StringIO()
    argv = ["trace", "g2", "--folder", str(scored), "--data-dir", str(scored.parent / "data")]
    argv += ["--holdout", "0", "--arm", DIRECT_ARM]
    assert main(argv, context=_context(), out=out) == 0
    found = [json.loads(line) for line in out.getvalue().splitlines()]
    asked = {
        line["key"]: int(line["asked"])
        for line in _lines(_messages(scored))
        if line["holdout"] == "0" and int(line["asked"])
    }
    assert len(found) == sum(asked.values())
    for record in found:
        assert record["request"]["quota"] == asked[record["cell"]]


def test_contradictions_reads_a_run_under_a_share_rule(scored):
    data = ["--data-dir", str(scored.parent / "data")]
    store = scored.parent / "rows.jsonl"
    if not store.exists():
        argv = ["rows", "g2", "--folder", str(scored), "--rows", str(store), *data]
        assert main(argv, context=_context(), out=io.StringIO()) == 0
    out = io.StringIO()
    argv = ["contradictions", "g2", "--rows", str(store), *data]
    assert main(argv, context=_context(scored.parent), out=out) == 0
    table = pd.read_csv(io.StringIO(out.getvalue()), keep_default_na=False)
    direct = table[(table["arm"] == DIRECT_ARM) & (table["kind"] == "empty-cell")]
    assert list(direct["holdout"]) == list(RULED_G2.holdout_seeds)
    assert (direct["count"].astype(int) == 0).all()
