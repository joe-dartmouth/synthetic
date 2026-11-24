"""The added-rows command over a Diabetes 130 store: the coherence rules' counts are not blank.

The toy cohort of `tests/_diabetes_fixtures.py` is scored, stored by `rows` and read back by
`contradictions --rows`, with no API call and no read of `data/`.
"""

from __future__ import annotations

import dataclasses
import io
import logging
from typing import Final

from bko.evaluate.__main__ import main
from bko.evaluate.added_contradictions import ADDED_COLUMNS
from bko.evaluate.perturb import RULES
from tests.test_evaluate_diabetes import TOY_RUN, _context
from tests.test_evaluate_shared import FORMAT


TOY: Final = dataclasses.replace(
    TOY_RUN, arms=("none", "smote"), candidate="smote", comparators=("none",)
)
"""The toy run with `none` and `smote` alone, so it fills with no BKO arm."""

RULE_COUNTS: Final = {
    ("0", "none", "0"): dict.fromkeys(RULES, "0"),
    ("0", "smote", "16"): dict.fromkeys(RULES, "0"),
    ("1", "none", "0"): dict.fromkeys(RULES, "0"),
    ("1", "smote", "16"): {**dict.fromkeys(RULES, "0"), "dose-without-change": "16"},
}
"""Each fill's rows and rule counts: `none` adds no row, and every row SMOTE interpolates at
holdout 1 changes a dose with `change` `No`."""


def test_a_diabetes_store_counts_each_coherence_rule(tmp_path, caplog):
    data = str(tmp_path / "data")
    ctx = dataclasses.replace(_context(toy=TOY), runs_dir=tmp_path)
    folder = tmp_path / "run"
    folder.mkdir()
    caplog.set_level(logging.INFO)
    out = io.StringIO()
    assert main(["score", "toy", "--data-dir", data], context=ctx, out=out) == 0
    (folder / "scores.csv").write_text(out.getvalue(), encoding="utf-8")
    lines = "".join(FORMAT.format(record) + "\n" for record in caplog.records)
    (folder / "scores.log").write_text(lines, encoding="utf-8")
    store = tmp_path / "rows.jsonl"
    argv = ["rows", "toy", "--folder", str(folder), "--rows", str(store), "--data-dir", data]
    assert main(argv, context=ctx, out=io.StringIO()) == 0
    out = io.StringIO()
    argv = ["contradictions", "toy", "--rows", str(store), "--data-dir", data]
    assert main(argv, context=ctx, out=out) == 0
    header, *body = out.getvalue().splitlines()
    assert header == ",".join(ADDED_COLUMNS)
    rules = {
        (holdout, arm, kind): (rows, count)
        for holdout, arm, rows, kind, count in (row.split(",") for row in body)
        if kind in RULES
    }
    assert rules == {
        (holdout, arm, rule): (rows, counts[rule])
        for (holdout, arm, rows), counts in RULE_COUNTS.items()
        for rule in RULES
    }
