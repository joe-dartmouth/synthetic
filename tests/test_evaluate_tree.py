"""`bko-tree` beside `bko-direct` on the `diabetes-codes` toy cohort: run, scored, traced.

The toy cohort of `tests/_diabetes_fixtures.py` holds two mapped diagnosis codes, 486 and 585; the
MONDO-shaped release here puts their concepts under one invented parent, so a drawn leaf's trace
reads three levels. No test reaches the API or reads `data/`.
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
from typing import Final

import pandas as pd
import pytest

from bko.datasets.diabetes import DIAGNOSES
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.data import Dataset
from bko.evaluate.direct_arms import DIRECT_ARM, NOMAP_ARM, TREE_ARM
from bko.evaluate.experiment import E1, REAL_ARM, Experiment, Removal
from bko.evaluate.holdout_candidates import KEPT, read
from bko.evaluate.named_runs import RUNS, NamedRun
from bko.evaluate.reference import REFERENCE
from bko.evaluate.run_scores import code_share_error, value_shares
from bko.evaluate.runs_diabetes import E9_EMPTY, E9_THIN
from bko.evaluate.runs_direct import E15_EMPTY, E15_THIN
from bko.evaluate.scores import Measure, Scope, Source, parse
from bko.generate.messages import SETTINGS, AnySend
from bko.generate.tree import TOP, UNMAPPED, TreePart
from bko.ontology.icd9_codes import ICD9_CODES
from tests._diabetes_fixtures import diabetes_releases, pin, toy_archive
from tests._ontology_fixtures import RoutingOpener
from tests.test_evaluate_diabetes import PREVALENCE, REMOVED
from tests.test_evaluate_trace import _scored


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

ARCHIVE: Final = toy_archive()
PIN: Final = pin(ARCHIVE, "uci-diabetes-130")

PNEUMONIA: Final = "MONDO:0005249"
KIDNEY: Final = "MONDO:0005300"
"""The concepts of the toy cohort's two mapped codes, 486 and 585."""

_UNDER: Final = {row.concept: row.label for row in ICD9_CODES}
"""Each mapped code's concept with its release name; the toy's two sit under an invented parent."""

MONDO: Final = "\n".join(
    [
        "format-version: 1.2",
        "data-version: releases/2026-01-01",
        "ontology: mondo",
        "",
        *("[Term]", "id: MONDO:9000001", "name: disease", ""),
        *("[Term]", "id: MONDO:9000002", "name: invented parent", "is_a: MONDO:9000001", ""),
        *(
            line
            for concept, label in _UNDER.items()
            for line in (
                "[Term]",
                f"id: {concept}",
                f"name: {label}",
                f"is_a: MONDO:{'9000002' if concept in {PNEUMONIA, KIDNEY} else '9000001'}",
                "",
            )
        ),
    ]
)


TREE_RUN: Final = dataclasses.replace(
    E1,
    holdout_seeds=(0, 1),
    arms=("none", REAL_ARM, DIRECT_ARM, NOMAP_ARM, TREE_ARM),
    candidate=TREE_ARM,
    comparators=(DIRECT_ARM, NOMAP_ARM, "none"),
    replica_arms=(),
    criteria=(),
    removal=Removal(REMOVED, 0.9),
    decontaminate=False,
    cutoff=PREVALENCE,
)
"""E15's arms on two holdouts of the toy cohort, 0.9 of `(6-7, female, 1)` removed."""

PART_FIELDS: Final = {field.name for field in dataclasses.fields(TreePart)}


def _never() -> AnySend:
    msg = "a tree run never reaches the API"
    raise AssertionError(msg)


def _context(experiment: Experiment = TREE_RUN) -> Context:
    pins, served = diabetes_releases()
    mondo = MONDO.encode()
    pins = {**pins, "MONDO": pin(mondo, name="mondo")}
    return Context(
        open_url=RoutingOpener({**served, pins["MONDO"].url: mondo, PIN.url: ARCHIVE}),
        runs={"toy": NamedRun(experiment, SETTINGS, Dataset.DIABETES_CODES)},
        releases=pins,
        send=_never,
        diabetes=PIN,
    )


@pytest.fixture(scope="module")
def scored(tmp_path_factory) -> Path:
    """A finished `toy` folder of `TREE_RUN`, its data root beside it."""
    base = tmp_path_factory.mktemp("tree")
    folder = base / "run"
    argv = ["score", "toy", "--candidates", str(folder / "holdout-candidates.jsonl")]
    _scored(folder, _context(), [*argv, "--data-dir", str(base / "data")])
    return folder


def _trace(folder: Path, *args: str) -> tuple[int, list[dict]]:
    out = io.StringIO()
    data = str(folder.parent / "data")
    argv = ["trace", "toy", "--folder", str(folder), "--data-dir", data, "--holdout", "0"]
    code = main([*argv, *args], context=_context(), out=out)
    return code, [json.loads(line) for line in out.getvalue().splitlines()]


def test_the_tree_arm_logs_tree_true(scored):
    log = (scored / "scores.log").read_text(encoding="utf-8")
    arms = re.findall(rf"arm holdout=\d arm={TREE_ARM} attempt=.*", log)
    assert len(arms) == 2
    assert all('"tree": true' in line for line in arms)
    assert re.search(r"direct holdout=0 fill=gap key=\S+ .* tree=true presence=shares", log)
    direct = re.findall(rf"arm holdout=\d arm={DIRECT_ARM} attempt=.*", log)
    assert direct
    assert not any('"tree"' in line for line in direct)


def test_a_tree_fill_traces_to_its_arm_sha(scored, caplog):
    caplog.set_level(logging.INFO)
    code, found = _trace(scored, "--arm", TREE_ARM)
    assert code == 0
    lines = read((scored / "holdout-candidates.jsonl").read_bytes()).candidates
    kept = [c for c in lines if (c.holdout, c.arm, c.outcome) == (0, TREE_ARM, KEPT)]
    assert len(found) == len(kept) > 0
    (arm,) = [m for m in caplog.messages if m.startswith("trace arm ")]
    assert f" arm={TREE_ARM} " in arm
    assert arm.endswith(" kept_lines=true matches=true")


def test_the_trace_names_each_levels_parts(scored):
    _, found = _trace(scored, "--arm", TREE_ARM)
    mapped = []
    for line in found:
        leaves = [fact for fact in line["concepts"] if "leaf" in fact]
        assert [fact["field"] for fact in leaves] == list(DIAGNOSES)
        for fact in leaves:
            parts = fact["parts"]
            assert all(set(part) == PART_FIELDS for part in parts)
            assert parts[0]["parent"] == TOP
            assert parts[-1]["node"] == fact["leaf"]
            assert [part["parent"] for part in parts[1:]] == [part["node"] for part in parts[:-1]]
            assert all(part["weight"] == 20 for part in parts)
            if fact["leaf"] == UNMAPPED:
                assert (fact["concept"], fact["label"], len(parts)) == (None, None, 1)
            else:
                mapped.append(fact)
        concepts = {fact["concept"] for fact in line["concepts"] if "leaf" not in fact}
        assert concepts.isdisjoint({PNEUMONIA, KIDNEY})
    assert mapped
    for fact in mapped:
        assert fact["concept"] == fact["leaf"]
        assert fact["concept"] in {PNEUMONIA, KIDNEY}
        assert [part["node"] for part in fact["parts"]] == [
            "MONDO:9000001",
            "MONDO:9000002",
            fact["leaf"],
        ]
        assert fact["label"] == _UNDER[fact["leaf"]]


def test_a_tree_fill_whose_tree_param_is_1_exits_3(scored, tmp_path, caplog):
    shutil.copytree(scored.parent, tmp_path / "copy")
    folder = tmp_path / "copy" / "run"
    log = folder / "scores.log"

    def edit(found: re.Match[str]) -> str:
        assert '"tree": true' in found[0]
        return found[0].replace('"tree": true', '"tree": 1')

    pattern = rf"arm holdout=0 arm={TREE_ARM} attempt=.*"
    text, count = re.subn(pattern, edit, log.read_text(encoding="utf-8"))
    assert count == 1
    log.write_text(text, encoding="utf-8")
    assert _trace(folder, "--arm", TREE_ARM) == (3, [])
    assert f"{TREE_ARM}'s arm line draws tree 1, not True" in caplog.text


def _error_rows(text: str) -> dict[tuple[int, str], float]:
    return {
        (score.holdout, score.arm): score.value
        for score in parse(text)
        if score.measure is Measure.CODE_SHARE_ERROR
    }


def test_each_arm_of_a_tree_run_writes_its_code_share_error_nan_without_rows(scored):
    scores = parse((scored / "scores.csv").read_text(encoding="utf-8"))
    errors = [s for s in scores if s.measure is Measure.CODE_SHARE_ERROR]
    assert {(s.source, s.scope) for s in errors} == {(Source.ADDED, Scope.FLAGGED)}
    found = _error_rows((scored / "scores.csv").read_text(encoding="utf-8"))
    assert set(found) == {(holdout, arm) for holdout in (0, 1) for arm in TREE_RUN.arms}
    assert all(math.isnan(found[holdout, "none"]) for holdout in (0, 1))
    assert {found[holdout, REAL_ARM] for holdout in (0, 1)} == {0.0}
    filled = [found[key] for key in found if key[1] not in {"none", REAL_ARM}]
    assert all(0 < error <= 1 for error in filled)


def test_the_code_share_error_is_the_hand_value():
    added = pd.DataFrame({"diag_1": [1, 1, 2, 3], "diag_2": [5, 5, 5, 5]})
    removed = {
        "diag_1": value_shares(pd.Series([1, 2, 2, 2])),
        "diag_2": value_shares(pd.Series([5.0, 5.0, 5.0, 6.0])),
    }
    # diag_1: (|1/2 - 1/4| + |1/4 - 3/4| + |1/4 - 0|) / 2 = 1/2; diag_2: (1/4 + 1/4) / 2 = 1/4.
    assert Fraction(code_share_error(added, removed)) == Fraction(3, 8)
    assert Fraction(code_share_error(added, {"diag_1": removed["diag_1"]})) == Fraction(1, 2)


def test_a_column_the_removal_holds_no_value_of_reads_nan():
    added = pd.DataFrame({"diag_1": [1, 2]})
    empty = {"diag_1": value_shares(pd.Series([], dtype="int64"))}
    assert pd.isna(code_share_error(added, empty))


def test_a_run_without_bko_tree_writes_no_code_share_error_row(tmp_path):
    direct = dataclasses.replace(
        TREE_RUN,
        arms=("none", REAL_ARM, DIRECT_ARM),
        candidate=DIRECT_ARM,
        comparators=("none",),
    )
    out = io.StringIO()
    candidates = str(tmp_path / "holdout-candidates.jsonl")
    argv = ["score", "toy", "--candidates", candidates, "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=_context(direct), out=out) == 0
    assert {score.arm for score in parse(out.getvalue())} >= {REAL_ARM, DIRECT_ARM}
    assert _error_rows(out.getvalue()) == {}


def test_a_tree_folder_rescores_to_its_own_bytes(scored, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    store = tmp_path / "rows.jsonl"
    data = str(scored.parent / "data")
    argv = ["rows", "toy", "--folder", str(scored), "--rows", str(store), "--data-dir", data]
    out = io.StringIO()
    assert main(argv, context=_context(), out=out) == 0
    assert out.getvalue() == (scored / "scores.csv").read_text(encoding="utf-8")
    assert any(m.endswith("matches_folder=true") for m in caplog.messages)


@pytest.mark.parametrize(("tree", "e9"), [(E15_THIN, E9_THIN), (E15_EMPTY, E9_EMPTY)])
def test_e15_is_e9_with_bko_tree_beside_bko_direct_and_bko_nomap(tree, e9):
    assert tree.arms == ("none", "real", "bko-direct", "bko-nomap", "bko-tree")
    assert tree.candidate == "bko-tree"
    assert tree.comparators == ("bko-direct", "bko-nomap", "none")
    assert tree.replica_arms == ()
    changed = {"arms", "candidate", "comparators"}
    for field in dataclasses.fields(tree):
        if field.name not in changed:
            assert getattr(tree, field.name) == getattr(e9, field.name), field.name


@pytest.mark.parametrize("fraction", ["thin", "empty"])
def test_each_e15_run_reads_the_codes_frame(fraction):
    whole = E15_THIN if fraction == "thin" else E15_EMPTY
    names = [f"e15-{fraction}", *(f"e15-{fraction}{letter}" for letter in "abcdefghij")]
    assert {RUNS[name].dataset for name in names} == {Dataset.DIABETES_CODES}
    assert RUNS[f"e15-{fraction}"].experiment == whole
    assert RUNS[f"e15-{fraction}"].settings == RUNS[f"e9-{fraction}"].settings
    assert REFERENCE[f"e15-{fraction}"] == DIRECT_ARM
