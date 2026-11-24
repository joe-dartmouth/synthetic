"""The added-rows basis: each kind and rule counted in a fill's added rows, and the command."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import logging
import re
from pathlib import Path

import pandas as pd
import pytest

from bko.datasets.diabetes import CHANGE_CODES, DIABETES_MED_CODES, DRUG_CODES, DRUGS
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.__main__ import main
from bko.evaluate.added_contradictions import ADDED_COLUMNS, ADDED_KINDS, added_counts, answers
from bko.evaluate.experiment import PRIOR_SHIFT_ARM, REAL_ARM, Experiment
from bko.evaluate.perturb import RULES, incoherent, rule_breaks
from bko.gaps.contradictions import KINDS, survey
from bko.gaps.observe import Cell
from tests._contradiction_fixtures import ROOT, YW1, Planted, plant, row
from tests.test_evaluate_membership import PRIVATE, _context, _flip, _header, _rewrite, _stored
from tests.test_evaluate_shared import edit_log


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
STEADY, UP, DOWN = DRUG_CODES["Steady"], DRUG_CODES["Up"], DRUG_CODES["Down"]

CODED = _diabetes(
    [
        ({"insulin": STEADY}, NO, SAME),  # drug-without-med
        ({"insulin": STEADY}, NO, CH),  # drug-without-med, change-without-med
        ({}, NO, CH),  # change-without-med
        ({"insulin": UP}, YES, SAME),  # dose-without-change
        ({"metformin": DOWN}, YES, SAME),  # dose-without-change
        ({"insulin": STEADY}, YES, SAME),
        ({}, NO, SAME),
    ]
)


RULE_COUNTS: dict[str, int | None] = {
    "drug-without-med": 2,
    "change-without-med": 2,
    "dose-without-change": 2,
}
"""Each rule's breaks in `CODED`."""


def test_each_rule_counts_its_own_rows():
    breaks = rule_breaks(CODED)
    assert tuple(breaks) == RULES
    assert {rule: int(mask.sum()) for rule, mask in breaks.items()} == RULE_COUNTS
    union = breaks["drug-without-med"] | breaks["change-without-med"]
    union |= breaks["dose-without-change"]
    assert int(union.sum()) == 5
    assert incoherent(CODED) == pytest.approx(5 / 7)


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


def test_a_fill_with_no_rows_counts_every_asked_cell_empty_and_zero_else():
    planted = plant("narrow-without-broad")
    none = planted.frame.iloc[:0]
    found = added_counts(
        none, planted.rows, planted.ontologies, frame=UCI_FRAME, asked=ASKED, name="test"
    )
    assert found == {**dict.fromkeys(KINDS, 0), "empty-cell": len(ASKED), **NONE_RULES}
    coded = _with_rules(planted.frame, CODED).iloc[:0]
    found = added_counts(
        coded, planted.rows, planted.ontologies, frame=UCI_FRAME, asked=ASKED, name="test"
    )
    assert found == {**dict.fromkeys(ADDED_KINDS, 0), "empty-cell": len(ASKED)}


@pytest.mark.parametrize("rows", [0, 6], ids=["no-rows", "rows"])
def test_a_fill_that_answers_no_request_leaves_empty_cell_blank(rows):
    planted = plant("narrow-without-broad")
    found = {
        asked: added_counts(
            planted.frame.iloc[:rows],
            planted.rows,
            planted.ontologies,
            frame=UCI_FRAME,
            asked=asked,
            name="test",
        )
        for asked in (None, ASKED)
    }
    assert found[None] == found[ASKED] | {"empty-cell": None}


@pytest.mark.parametrize(
    ("arm", "answering"),
    [
        ("none", False),
        ("class-weight", False),
        (REAL_ARM, False),
        (PRIOR_SHIFT_ARM, False),
        ("smote", True),
        ("ctgan", True),
        ("bko", True),
        ("bko-direct", True),
    ],
)
def test_only_an_arm_that_adds_rows_for_the_request_answers_it(arm, answering):
    assert answers(arm) is answering


def test_two_overlap_findings_sharing_a_row_count_it_once():
    base = plant("code-overlap")
    planted = Planted(base.kind, base.frame, (*base.rows, row(ROOT, "ca", 0, 2)), base.ontologies)
    findings = survey(
        planted.frame, planted.rows, planted.ontologies, frame=UCI_FRAME, name="test"
    ).findings
    overlaps = [f for f in findings if f.kind == "code-overlap"]
    assert len(overlaps) == 3
    assert all(set(f.codes) == {2} for f in overlaps)
    found = added_counts(
        planted.frame,
        planted.rows,
        planted.ontologies,
        frame=UCI_FRAME,
        asked=ASKED,
        name="test",
    )
    assert found["code-overlap"] == 5


def test_coherence_counts_are_blank_off_the_diabetes_columns():
    planted = plant("narrow-without-broad")
    six = planted.frame.iloc[: len(CODED)]
    off = added_counts(
        six, planted.rows, planted.ontologies, frame=UCI_FRAME, asked=ASKED, name="test"
    )
    assert {rule: off[rule] for rule in RULES} == NONE_RULES
    on = added_counts(
        _with_rules(six, CODED),
        planted.rows,
        planted.ontologies,
        frame=UCI_FRAME,
        asked=ASKED,
        name="test",
    )
    assert {rule: on[rule] for rule in RULES} == RULE_COUNTS
    assert {kind: on[kind] for kind in KINDS} == {kind: off[kind] for kind in KINDS}


# The command.


def _added(tmp_path: Path, *argv: str, experiment: Experiment = PRIVATE) -> tuple[int, str]:
    out = io.StringIO()
    args = ["contradictions", *argv, "--data-dir", str(tmp_path / "data")]
    code = main(args, context=_context(tmp_path, experiment=experiment), out=out)
    return code, out.getvalue()


REPLICATED = dataclasses.replace(PRIVATE, replica_arms=("smote",))
"""`PRIVATE` with SMOTE's replica fill: `none` and `real` answer no request, `smote` the gap
request and `smote-replica` the replica request."""


def _emptied(folder: Path, store: Path, *, holdout: int, arm: str) -> None:
    """Make `arm`'s fill of `holdout` hold no row, as `none`'s does, in `folder`'s log and the
    store, resealing the store's `log_sha256`."""

    def edit(lines: list[dict]) -> list[dict]:
        none = next(x for x in lines[1:] if (x["holdout"], x["arm"]) == (holdout, "none"))
        line = next(x for x in lines[1:] if (x["holdout"], x["arm"]) == (holdout, arm))
        was = f"added={line['added']} sha256={line['sha256']}"
        now = f"added=0 sha256={none['sha256']}"
        edit_log(folder, f"arm holdout={holdout} arm={arm} ", lambda text: text.replace(was, now))
        line.update(added=0, csv=none["csv"], sha256=none["sha256"])
        lines[0]["log_sha256"] = hashlib.sha256((folder / "scores.log").read_bytes()).hexdigest()
        return lines

    _rewrite(store, edit)


def _table(text: str) -> dict[tuple[int, str, str], tuple[int, str]]:
    """The CSV's rows and count by holdout, arm and kind."""
    lines = text.splitlines()
    assert lines[0] == ",".join(ADDED_COLUMNS)
    body = [line.split(",") for line in lines[1:]]
    found = {(int(h), arm, kind): (int(rows), count) for h, arm, rows, kind, count in body}
    assert len(found) == len(body)
    return found


EMPTY_CELLS = {
    (0, "none"): "",
    (0, REAL_ARM): "",
    (0, "smote"): "0",
    (0, "smote-replica"): "0",
    (1, "none"): "",
    (1, REAL_ARM): "",
    (1, "smote"): "2",
    (1, "smote-replica"): "0",
}
"""Each fill's `empty-cell`: blank where it answers no request; holdout 1's `smote` fill holds no
row, so it counts both cells holdout 1's gap request asks for (its `quota` lines)."""


def test_the_command_reads_a_store_and_makes_no_api_call(tmp_path, caplog):
    folder, store = _stored(tmp_path, caplog, experiment=REPLICATED)
    _emptied(folder, store, holdout=1, arm="smote")
    caplog.set_level(logging.INFO)
    code, text = _added(tmp_path, "e1", "--rows", str(store), experiment=REPLICATED)
    assert code == 0
    found = _table(text)
    fills = {(holdout, arm) for holdout, arm, _ in found}
    assert fills == set(EMPTY_CELLS)
    assert len(found) == len(fills) * len(ADDED_KINDS)
    assert {fill: found[*fill, "empty-cell"][1] for fill in fills} == EMPTY_CELLS
    assert found[1, "smote", "all-show"] == (0, "0")
    assert found[0, "smote", "all-show"] == (30, "0")
    assert found[0, "smote-replica", "all-show"] == (130, "6")
    assert all(found[*fill, rule][1] == "" for fill in fills for rule in RULES)
    logged = caplog.messages
    assert logged[0].startswith("versions commit=")
    assert "added holdout=1 arm=smote rows=0 asked=2" in logged
    assert not any(re.match(r"added holdout=\d+ arm=(none|real) ", m) for m in logged)
    sha = hashlib.sha256(text.encode()).hexdigest()
    assert f"added rows={len(found)} sha256={sha} stores=1 holdouts=2" in logged
    assert any(re.fullmatch(r"added holdout=1 done=2/2 seconds=\d+\.\d", m) for m in logged)
    assert f"added store={store} run=e1 folder=run" in caplog.text
    assert "privacy store=" not in caplog.text


NEEDS_ROWS = "contradictions reads a run's added rows from its stores, so it needs --rows"


@pytest.mark.parametrize("argv", [("e1",), ("--runs-dir", "elsewhere")], ids=["run", "runs-dir"])
def test_a_run_or_runs_dir_without_rows_exits_2(tmp_path, caplog, argv):
    assert _added(tmp_path, *argv) == (2, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [NEEDS_ROWS]


def test_rows_without_a_run_reads_the_default_run(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    caplog.set_level(logging.INFO)
    code, text = _added(tmp_path, "--rows", str(store))
    assert code == 0
    assert text.startswith(",".join(ADDED_COLUMNS))
    assert any(m.startswith("added run=e1 stores=1 holdouts=2 ") for m in caplog.messages)


def test_a_tampered_fill_exits_3_naming_its_store(tmp_path, caplog):
    _, store = _stored(tmp_path, caplog)
    _rewrite(store, _flip)
    assert _added(tmp_path, "e1", "--rows", str(store)) == (3, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors[-1].startswith(f"the store {store}: ")
    assert "does not hash to its sha256" in errors[-1]


def test_a_split_line_the_data_does_not_reproduce_exits_3(tmp_path, caplog):
    folder, store = _stored(tmp_path, caplog)
    edit_log(folder, "split seed=1", lambda line: re.sub(r"fingerprint=\w+", "fingerprint=0", line))
    log = hashlib.sha256((folder / "scores.log").read_bytes()).hexdigest()
    _rewrite(store, _header("log_sha256", log))
    assert _added(tmp_path, "e1", "--rows", str(store)) == (3, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors[-1].startswith("holdout 1's split ")
    assert errors[-1].endswith(f"which the store {store} is bound to")


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
