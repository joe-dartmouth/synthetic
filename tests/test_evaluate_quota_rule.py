"""`Experiment.quota_rule`: each flagged cell asked its rule's count, and every reader reads it.

ADR `research/share-based-quota` records the rule; `bko.gaps.quota` holds the counts.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import math
import re
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Final, cast

import pandas as pd
import pytest

from bko.datasets.cells import cell_counts
from bko.evaluate import reference as reference_module
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import DIRECT_ARM, E1, REAL_ARM, gap_request
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.provenance import experiment_line
from bko.evaluate.run_folder import ASSUMED
from bko.evaluate.scores import Measure, parse
from bko.gaps.quota import QuotaRule
from bko.ontology.schema import Codes
from tests._frame_fixtures import SHORT, TOY, TOY_TRAIN
from tests._gaps_fixtures import ANGINA, STENOSIS
from tests.test_evaluate_direct import SMALL_G2, g2_context
from tests.test_evaluate_donor_attacks import DONORS, _read, _stored_direct
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
    plain = gap_request(TOY_TRAIN, filled, seed=0, frame=TOY, rows=(), source_name="")
    assert plain.quota == expected
    given = gap_request(TOY_TRAIN, filled, seed=0, frame=TOY, rows=TOY_ROWS, source_name="toy")
    assert (given.quota, given.strata, given.seed) == (plain.quota, plain.strata, plain.seed)
    assert not _lines([r.getMessage() for r in caplog.records])


def test_a_share_rule_without_rows_is_refused():
    ruled = dataclasses.replace(TOY_RUN, quota_rule=QuotaRule.COVERAGE)
    with pytest.raises(EvaluationError, match="needs the data's field rows"):
        gap_request(TOY_TRAIN, ruled, seed=0, frame=TOY, rows=(), source_name="")


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
    # A synthetic share with no flagged cell is refused too, and the rule's refusal comes first.
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(TOY_RUN, synthetic_share=Fraction(1, 4), quota_rule=QuotaRule.COVERAGE)
    named = dataclasses.replace(TOY_RUN, flagged_cell=FULL)
    assert named.quota_rule is QuotaRule.FILL_TO


def test_a_quota_rule_given_by_value_reads_as_the_rule():
    ruled = dataclasses.replace(TOY_RUN, quota_rule=cast(QuotaRule, "coverage"))
    assert ruled.quota_rule is QuotaRule.COVERAGE
    assert ruled == dataclasses.replace(TOY_RUN, quota_rule=QuotaRule.COVERAGE)
    with pytest.raises(ValueError, match="no-such-rule"):
        dataclasses.replace(TOY_RUN, quota_rule=cast(QuotaRule, "no-such-rule"))
    message = "cannot stand beside a flagged cell or a synthetic share"
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(TOY_RUN, flagged_cell=FULL, quota_rule=cast(QuotaRule, "coverage"))


TOY_ASKS: Final = {
    # Each SHORT cell holds 5 training records. Expected size: n(45-69, sex, 0) = 20 and the
    # other sex's 5 and 20 give 20 * 5.5 / 20.5 = 5.37, so 5. Coverage: angina, read off women,
    # has fused share (5 * 5/5 + 20 * 45/90) / 25 = 0.6 among women, 0.4^4 = 0.026, so 4; among
    # men (0 + 20 * 45/90) / 25 = 0.4, 0.6^6 = 0.047, so 6. Stenosis, at share 1, adds nothing.
    QuotaRule.COVERAGE: (4, 6, 0, 1),
    QuotaRule.COVERAGE_2: (4, 6, 3, 7),
    QuotaRule.COVERAGE_4: (4, 6, 11, 19),
    QuotaRule.COVERAGE_8: (4, 6, 27, 43),
    QuotaRule.RECOMMENDED: (4, 6, 0, 1),
}
"""Each rule's coverage of the women's and the men's SHORT cell, then the rows each is asked."""


@pytest.mark.parametrize("rule", list(TOY_ASKS))
def test_each_rule_logs_its_hand_computed_counts(caplog, rule):
    caplog.set_level(logging.INFO)
    ruled = dataclasses.replace(TOY_RUN, quota_rule=rule)
    request = gap_request(TOY_TRAIN, ruled, seed=3, frame=TOY, rows=TOY_ROWS, source_name="toy")
    women, men, asked_women, asked_men = TOY_ASKS[rule]
    lines = _lines([r.getMessage() for r in caplog.records])
    assert [line["key"] for line in lines] == ["45-69/female/1", "45-69/male/1"]
    found = [
        tuple(int(line[k]) for k in ("holdout", "count", "coverage", "expected", "asked"))
        for line in lines
    ]
    assert found == [(3, 5, women, 5, asked_women), (3, 5, men, 5, asked_men)]
    assert {line["rule"] for line in lines} == {str(rule)}
    asked = {SHORT[0]: asked_women, SHORT[1]: asked_men}
    assert dict(request.quota) == {cell: n for cell, n in asked.items() if n}


def test_a_rule_that_asks_no_cell_a_row_has_nothing_to_fill(caplog):
    caplog.set_level(logging.INFO)
    ruled = dataclasses.replace(TOY_RUN, quota_rule=QuotaRule.EXPECTED)
    with pytest.raises(
        EvaluationError, match=r"^at holdout 3, .*expected-size count; nothing to fill$"
    ):
        gap_request(TOY_TRAIN, ruled, seed=3, frame=TOY, rows=TOY_ROWS, source_name="toy")
    lines = _lines([r.getMessage() for r in caplog.records])
    assert [(line["key"], line["expected"], line["asked"]) for line in lines] == [
        ("45-69/female/1", "5", "0"),
        ("45-69/male/1", "5", "0"),
    ]


def test_the_default_rule_logs_no_quota_rule_line_and_names_the_seed_when_empty(caplog):
    caplog.set_level(logging.INFO)
    gap_request(TOY_TRAIN, TOY_RUN, seed=3, frame=TOY, rows=TOY_ROWS, source_name="toy")
    assert not _lines([r.getMessage() for r in caplog.records])
    low = dataclasses.replace(TOY_RUN, threshold=5, fill_to=5)
    with pytest.raises(EvaluationError, match=r"^at holdout 3, no cell .*; nothing to fill$"):
        gap_request(TOY_TRAIN, low, seed=3, frame=TOY, rows=(), source_name="")


def test_a_request_rebuilt_without_log_writes_no_line(caplog):
    caplog.set_level(logging.INFO)
    ruled = dataclasses.replace(TOY_RUN, quota_rule=QuotaRule.COVERAGE)
    quiet = gap_request(
        TOY_TRAIN, ruled, seed=3, frame=TOY, rows=TOY_ROWS, source_name="toy", log=False
    )
    assert not _lines([r.getMessage() for r in caplog.records])
    logged = gap_request(TOY_TRAIN, ruled, seed=3, frame=TOY, rows=TOY_ROWS, source_name="toy")
    assert (quiet.quota, quiet.seed) == (logged.quota, logged.seed)
    assert len(_lines([r.getMessage() for r in caplog.records])) == len(SHORT)


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


def _keys(caplog) -> list[tuple[str, str]]:
    """Each logged `quota-rule` line's holdout and key, in order."""
    return [(line["holdout"], line["key"]) for line in _lines(caplog.messages)]


def test_shared_rescores_a_run_under_a_share_rule(scored, caplog):
    caplog.set_level(logging.INFO)
    out = io.StringIO()
    argv = ["shared", "g2", "--folder", str(scored), "--data-dir", str(scored.parent / "data")]
    assert main(argv, context=_context(), out=out) == 0
    assert out.getvalue() == (scored / "scores.csv").read_text(encoding="utf-8")
    assert _keys(caplog) == [(line["holdout"], line["key"]) for line in _lines(_messages(scored))]


def test_shared_with_another_folder_logs_one_set_of_rule_lines_per_holdout(
    scored, tmp_path, caplog
):
    other = tmp_path / "other"
    shutil.copytree(scored, other)
    caplog.set_level(logging.INFO)
    out = io.StringIO()
    argv = ["shared", "g2", "--folder", str(scored), "--with", str(other)]
    argv += ["--data-dir", str(scored.parent / "data")]
    assert main(argv, context=_context(), out=out) == 0
    expected = [(line["holdout"], line["key"]) for line in _lines(_messages(scored))]
    assert expected
    assert _keys(caplog) == expected


def test_trace_reads_a_run_under_a_share_rule(scored, caplog):
    caplog.set_level(logging.INFO)
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
    assert not _keys(caplog)


def test_contradictions_reads_a_run_under_a_share_rule(scored, tmp_path, caplog):
    data = ["--data-dir", str(scored.parent / "data")]
    store = tmp_path / "rows.jsonl"
    argv = ["rows", "g2", "--folder", str(scored), "--rows", str(store), *data]
    assert main(argv, context=_context(), out=io.StringIO()) == 0
    caplog.set_level(logging.INFO)
    caplog.clear()
    out = io.StringIO()
    argv = ["contradictions", "g2", "--rows", str(store), *data]
    assert main(argv, context=_context(scored.parent), out=out) == 0
    table = pd.read_csv(io.StringIO(out.getvalue()), keep_default_na=False)
    direct = table[(table["arm"] == DIRECT_ARM) & (table["kind"] == "empty-cell")]
    assert list(direct["holdout"]) == list(RULED_G2.holdout_seeds)
    assert (direct["count"].astype(int) == 0).all()
    assert not _keys(caplog)


@pytest.fixture
def e1_reads_real(monkeypatch):
    """`e1`, which the privacy context's experiment stands in for, read against `real`."""
    monkeypatch.setattr(
        reference_module, "REFERENCE", {**reference_module.REFERENCE, "e1": REAL_ARM}
    )


@pytest.mark.usefixtures("e1_reads_real")
def test_privacy_rebuilds_a_share_rules_request_for_donor_reuse(tmp_path, caplog):
    ruled = dataclasses.replace(DONORS, quota_rule=QuotaRule.COVERAGE_8)
    store = _stored_direct(tmp_path, caplog, experiment=ruled)
    scored = [line["asked"] for line in _lines(_messages(tmp_path / "run"))]
    assert scored
    caplog.set_level(logging.INFO)
    code, text = _read(tmp_path, store, experiment=ruled)
    assert code == 0
    values = {(s.holdout, s.arm, s.measure): s.value for s in parse(text)}
    for holdout in ruled.holdout_seeds:
        assert math.isfinite(values[holdout, DIRECT_ARM, Measure.DONOR_MAX])
    assert not _lines(caplog.messages)
