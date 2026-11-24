"""The added-rows basis: each kind and rule counted in a fill's added rows, and the command."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import logging
from pathlib import Path

import pandas as pd
import pytest

from bko.datasets.diabetes import CHANGE_CODES, DIABETES_MED_CODES, DRUG_CODES, DRUGS
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.__main__ import main
from bko.evaluate.added_contradictions import ADDED_COLUMNS, ADDED_KINDS, added_counts
from bko.evaluate.experiment import Experiment
from bko.evaluate.perturb import RULES, incoherent, rule_breaks
from bko.gaps.contradictions import KINDS
from bko.gaps.observe import Cell
from tests._contradiction_fixtures import YW1, plant
from tests.test_evaluate_membership import PRIVATE, _context, _flip, _header, _rewrite, _stored


YM0: Cell = ("<55", "male", 0)
"""A cell no planted split holds a record in."""
OF1: Cell = (">=55", "female", 1)
"""Another such cell."""
ASKED: tuple[Cell, ...] = (YW1, YM0, OF1)
"""A request's cells: one every planted split but `empty-cell`'s fills, two none does."""

NONE_RULES: dict[str, int | None] = dict.fromkeys(RULES)


def _diabetes(rows: list[tuple[dict[str, int], int, int]]) -> pd.DataFrame:
    """One row per entry of `rows`: its drugs' codes (every other drug `No`), `diabetesMed`
    and `change`."""
    return pd.DataFrame(
        [
            {
                **dict.fromkeys(DRUGS, DRUG_CODES["No"]),
                **drugs,
                "diabetesMed": medicated,
                "change": changed,
            }
            for drugs, medicated, changed in rows
        ]
    )


NO, YES = DIABETES_MED_CODES["No"], DIABETES_MED_CODES["Yes"]
SAME, CH = CHANGE_CODES["No"], CHANGE_CODES["Ch"]
STEADY, UP = DRUG_CODES["Steady"], DRUG_CODES["Up"]

SIX = _diabetes(
    [
        ({"insulin": STEADY}, NO, SAME),  # drug-without-med
        ({"insulin": STEADY}, NO, CH),  # drug-without-med, change-without-med
        ({}, NO, CH),  # change-without-med
        ({"insulin": UP}, YES, SAME),  # dose-without-change
        ({"insulin": STEADY}, YES, SAME),
        ({}, NO, SAME),
    ]
)


def test_each_rule_counts_its_own_rows():
    breaks = rule_breaks(SIX)
    assert tuple(breaks) == RULES
    assert {rule: int(mask.sum()) for rule, mask in breaks.items()} == {
        "drug-without-med": 2,
        "change-without-med": 2,
        "dose-without-change": 1,
    }
    union = breaks["drug-without-med"] | breaks["change-without-med"]
    union |= breaks["dose-without-change"]
    assert int(union.sum()) == 4
    assert incoherent(SIX) == pytest.approx(4 / 6)


# Each planted split read as one fill's added rows, its expected counts derived by hand from
# `tests/_contradiction_fixtures.py`: the `is_a` pair is `cp` 1 under `exang` 1, `restecg` cycles
# 0, 1, 2 within a cell, so neither nested concept shows in every row of one.
EXPECTED: dict[str, dict[str, int | None]] = {
    # YW1's six rows show `cp` 1 without `exang` 1; all-show: YW1 cp, YM1 cp and exang, OM1 exang.
    "narrow-without-broad": {
        "narrow-without-broad": 6,
        "child-share-above-parent": 1,
        "empty-cell": 2,
        "all-show": 4,
        "code-overlap": 0,
    },
    # YW1's first row shows `cp` 1 alone, so YW1's child share 1 tops its parent's 1/2; YM1's
    # `thal` 7 rows show the parent beside the child. all-show: YW1 cp, OM1 exang.
    "child-share-above-parent": {
        "narrow-without-broad": 1,
        "child-share-above-parent": 1,
        "empty-cell": 2,
        "all-show": 2,
        "code-overlap": 0,
    },
    # YW1 holds no row; YW0's six show `cp` 1 alone. all-show: YM1 both, OM1 exang, YW0 cp.
    "empty-cell": {
        "narrow-without-broad": 6,
        "child-share-above-parent": 1,
        "empty-cell": 3,
        "all-show": 4,
        "code-overlap": 0,
    },
    # Five cells, each showing all four concepts in every row.
    "all-show": {
        "narrow-without-broad": 0,
        "child-share-above-parent": 0,
        "empty-cell": 2,
        "all-show": 20,
        "code-overlap": 0,
    },
    # `ca` cycles 0, 1, 2, 3, 0, 1 in each of five cells; code 2 is the unresolved overlap.
    "code-overlap": {
        "narrow-without-broad": 0,
        "child-share-above-parent": 0,
        "empty-cell": 2,
        "all-show": 0,
        "code-overlap": 5,
    },
}


@pytest.mark.parametrize("kind", list(EXPECTED))
def test_each_planted_kind_is_counted_in_a_fills_added_rows(kind):
    planted = plant(kind)
    found = added_counts(
        planted.frame,
        planted.rows,
        planted.ontologies,
        frame=UCI_FRAME,
        asked=ASKED,
        name="test",
    )
    assert tuple(found) == ADDED_KINDS
    assert found == {**EXPECTED[kind], **NONE_RULES}


def test_an_asked_cell_left_empty_counts_one_empty_cell():
    planted = plant("narrow-without-broad")
    found = added_counts(
        planted.frame,
        planted.rows,
        planted.ontologies,
        frame=UCI_FRAME,
        asked=(YW1, YM0),
        name="test",
    )
    assert found["empty-cell"] == 1


def _with_rules(rows: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """`rows` with `rules`' columns beside them, row by row."""
    return pd.concat([rows, rules.set_axis(rows.index[: len(rules)])], axis=1)


def test_an_arm_that_adds_no_row_counts_zero():
    planted = plant("narrow-without-broad")
    none = planted.frame.iloc[:0]
    found = added_counts(
        none, planted.rows, planted.ontologies, frame=UCI_FRAME, asked=ASKED, name="test"
    )
    assert found == {**dict.fromkeys(KINDS, 0), **NONE_RULES}
    coded = _with_rules(planted.frame, SIX).iloc[:0]
    found = added_counts(
        coded, planted.rows, planted.ontologies, frame=UCI_FRAME, asked=ASKED, name="test"
    )
    assert found == dict.fromkeys(ADDED_KINDS, 0)


def test_coherence_counts_are_blank_off_the_diabetes_columns():
    planted = plant("narrow-without-broad")
    six = planted.frame.iloc[:6]
    off = added_counts(
        six, planted.rows, planted.ontologies, frame=UCI_FRAME, asked=ASKED, name="test"
    )
    assert {rule: off[rule] for rule in RULES} == NONE_RULES
    on = added_counts(
        _with_rules(six, SIX),
        planted.rows,
        planted.ontologies,
        frame=UCI_FRAME,
        asked=ASKED,
        name="test",
    )
    assert {rule: on[rule] for rule in RULES} == {
        "drug-without-med": 2,
        "change-without-med": 2,
        "dose-without-change": 1,
    }
    assert {kind: on[kind] for kind in KINDS} == {kind: off[kind] for kind in KINDS}


# The command.


def _added(tmp_path: Path, *argv: str, experiment: Experiment = PRIVATE) -> tuple[int, str]:
    out = io.StringIO()
    args = ["contradictions", *argv, "--data-dir", str(tmp_path / "data")]
    code = main(args, context=_context(tmp_path, experiment=experiment), out=out)
    return code, out.getvalue()


@pytest.mark.parametrize("replicas", [(), ("smote",)], ids=["gap", "replica"])
def test_the_command_reads_a_store_and_makes_no_api_call(tmp_path, caplog, replicas):
    experiment = dataclasses.replace(PRIVATE, replica_arms=replicas)
    _, store = _stored(tmp_path, caplog, experiment=experiment)
    caplog.set_level(logging.INFO)
    code, text = _added(tmp_path, "e1", "--rows", str(store), experiment=experiment)
    assert code == 0
    lines = text.splitlines()
    assert lines[0] == ",".join(ADDED_COLUMNS)
    body = [line.split(",") for line in lines[1:]]
    fills = {(holdout, arm) for holdout, arm, *_ in body}
    arms = {*PRIVATE.arms, *(f"{arm}-replica" for arm in replicas)}
    assert fills == {(str(holdout), arm) for holdout in (0, 1) for arm in arms}
    assert len(body) == len(fills) * len(ADDED_KINDS)
    for _holdout, _arm, rows, kind, count in body:
        if kind in RULES:
            assert count == ""
        elif rows == "0":
            assert count == "0"
        else:
            assert int(count) >= 0
    assert {kind for *_, kind, _ in body} == set(ADDED_KINDS)
    logged = caplog.messages
    assert logged[0].startswith("versions commit=")
    sha = hashlib.sha256(text.encode()).hexdigest()
    assert f"added rows={len(body)} sha256={sha} stores=1 holdouts=2" in logged
    assert "added holdout=1 done=2/2" in logged
    assert f"added store={store} run=e1 folder=run" in caplog.text
    assert "privacy store=" not in caplog.text


def test_a_run_without_rows_exits_2(tmp_path, caplog):
    assert _added(tmp_path, "e1") == (2, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert "--rows" in errors[-1]


def test_a_tampered_fill_exits_3(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    _rewrite(store, _flip)
    assert _added(tmp_path, "e1", "--rows", str(store)) == (3, "")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_header("run", []), "is run []'s"),
        (_header("run", {"a": 1}), "is run {'a': 1}'s"),
        (_header("folder", "r\x00n"), "not a folder directly under"),
    ],
    ids=["list-run", "dict-run", "nul-folder"],
)
def test_a_malformed_header_exits_3(tmp_path, caplog, edit, message):
    _, store = _stored(tmp_path, caplog)
    _rewrite(store, edit)
    assert _added(tmp_path, "e1", "--rows", str(store)) == (3, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert message in errors[-1]
