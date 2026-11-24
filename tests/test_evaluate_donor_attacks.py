"""The field-group match attack and the donor reuse of `python -m bko.evaluate privacy`."""

from __future__ import annotations

import dataclasses
import io
import logging
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bko.evaluate.reference as reference_module
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME
from bko.evaluate import membership
from bko.evaluate.__main__ import _named, main
from bko.evaluate.data import load
from bko.evaluate.donor_attacks import (
    DONOR_MEASURES,
    MATCH_MEASURES,
    donations,
    donor_counts,
    donor_measures,
    match,
    match_groups,
    match_measures,
    match_scores,
)
from bko.evaluate.errors import IdentityError
from bko.evaluate.experiment import DIRECT_ARM, REAL_ARM, gap_request
from bko.evaluate.membership import GROUPS
from bko.evaluate.prepare import prepared
from bko.evaluate.privacy_rule import readings
from bko.evaluate.run_folder import read_folder
from bko.evaluate.scores import Measure, parse
from bko.evaluate.shared import rebuilt
from bko.evaluate.trace_lines import Fill, select
from bko.gaps.scope import load_scope
from bko.generate.direct import Draw
from bko.ontology.frames import (
    DIABETES_FIELD_ROWS,
    DIABETES_FRAME,
    SYNTHEA_FIELD_ROWS,
    SYNTHEA_FRAME,
)
from tests.test_evaluate_membership import PRIVATE, WOBBLE, _reading, _scored
from tests.test_evaluate_membership import _context as privacy_context
from tests.test_evaluate_privacy import random_records
from tests.test_evaluate_shared import FORMAT
from tests.test_evaluate_trace import _scored as _scored_folder
from tests.test_evaluate_trace_direct import _context as trace_context
from tests.test_evaluate_trace_direct import _trace


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

CELL = ("<55", "female", 1)
"""A cell the planted records may fall in."""

UNITS = tuple((name,) for name in UCI_FRAME.columns if name not in {"age", "sex", "target"})
"""UCI's columns as singletons, but `age`, which places a record's cell, `sex` and the label."""


def _planted(count: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Members, and non-members that share no unit value with them: every coded unit at another
    code and every measured unit out of the members' range."""
    members = random_records(UCI_FRAME, count, seed=seed)
    nonmembers = random_records(UCI_FRAME, count, seed=seed + 1)
    for (name,) in UNITS:
        if name in UCI_FRAME.numeric:
            nonmembers[name] += 10_000
        else:
            first, second = list(UCI_FRAME.codebook[name])[:2]
            members[name], nonmembers[name] = first, second
    return members, nonmembers


def _measures(added: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    return match_measures(added, train=train, test=test, cell=CELL, frame=UCI_FRAME, groups=UNITS)


def test_the_match_score_is_the_most_units_one_added_row_copies():
    records = pd.DataFrame({"a": [1, 5, 9], "b": [1, 6, 9], "c": [1, 7, 9], "d": [3, 0, 0]})
    added = pd.DataFrame({"a": [1, 5, 0], "b": [1, 6, 0], "c": [1, 0, 7], "d": [3, 0, 3]})
    groups = (("a",), ("b", "d"), ("c",))
    found = match_scores(records, added, groups)
    assert found.dtype.kind == "i"
    assert found.tolist() == [3, 2, 0]


def test_a_planted_verbatim_copy_scores_auc_1():
    members, nonmembers = _planted(300, seed=21)
    found = _measures(members.copy(), members, nonmembers)
    assert found[Measure.MATCH_AUC_OUTSIDE] == pytest.approx(1.0, abs=0)
    assert found[Measure.MATCH_TPR01_OUTSIDE] == pytest.approx(1.0, abs=0)
    assert found[Measure.MATCH_AUC_LABEL] == pytest.approx(1.0, abs=0)


def test_disjoint_rows_score_one_half():
    members, nonmembers = _planted(300, seed=31)
    added = nonmembers.copy()
    for (name,) in UNITS:
        added[name] = added[name] + 100_000 if name in UCI_FRAME.numeric else -1
    assert not match_scores(members, added, UNITS).any()
    assert not match_scores(nonmembers, added, UNITS).any()
    found = _measures(added, members, nonmembers)
    assert found[Measure.MATCH_AUC_OUTSIDE] == pytest.approx(0.5, abs=0)
    assert found[Measure.MATCH_AUC_LABEL] == pytest.approx(0.5, abs=0)


def test_partial_copies_read_the_hand_auc():
    members, nonmembers = _planted(2, seed=41)
    for frame in (members, nonmembers):
        frame["age"], frame["sex"], frame["target"] = 70, 1, 1
    groups = (("cp",), ("trestbps",), ("chol",))
    copied = nonmembers.iloc[[0]].copy()
    copied["cp"], copied["chol"] = -1, -1
    added = pd.concat([members.iloc[[0]], copied], ignore_index=True)
    assert match_scores(members, added, groups).tolist() == [3, 1]
    assert match_scores(nonmembers, added, groups).tolist() == [1, 0]
    found = match_measures(
        added, train=members, test=nonmembers, cell=CELL, frame=UCI_FRAME, groups=groups
    )
    # Members [3, 1] against non-members [1, 0]: three wins of four pairs and one tie.
    assert found[Measure.MATCH_AUC_OUTSIDE] == pytest.approx(0.875, abs=0)


def test_an_arm_with_no_rows_reads_nan():
    members, nonmembers = _planted(20, seed=51)
    found = _measures(members.iloc[:0], members, nonmembers)
    assert list(found) == list(MATCH_MEASURES)
    assert all(math.isnan(value) for value in found.values())


@pytest.mark.parametrize(
    ("rows", "frame"),
    [(DIABETES_FIELD_ROWS, DIABETES_FRAME), (SYNTHEA_FIELD_ROWS, SYNTHEA_FRAME), ((), UCI_FRAME)],
)
def test_every_column_but_sex_and_the_label_is_in_exactly_one_unit(rows, frame):
    groups = match_groups(rows, frame=frame)
    named = [name for group in groups for name in group]
    assert sorted(named) == sorted(set(named))
    assert set(named) == set(frame.columns) - {"sex", frame.label}
    firsts = [frame.columns.index(group[0]) for group in groups]
    assert firsts == sorted(firsts)


def test_the_match_measures_mirror_the_distance_attacks_scopes():
    assert [m.value for m in MATCH_MEASURES] == [
        "match-auc-outside",
        "match-tpr01-outside",
        "match-tpr001-outside",
        "match-auc-cell",
        "match-tpr01-cell",
        "match-auc-label",
        "match-tpr01-label",
        "match-tpr001-label",
    ]
    assert np.array_equal(
        match_scores(pd.DataFrame({"a": [np.nan]}), pd.DataFrame({"a": [np.nan]}), (("a",),)),
        [1],
    )


# Donor reuse.

DRAWN = Draw(
    ("<55", "female", 1),
    {"a": 1, "b": 2, "c": 3.5},
    frozenset(),
    frozenset(),
    frozenset(),
    {"a": 7, "b": 7, "c": 8},
)
"""A draw that read `a` and `b` from training record 7 and `c` from record 8."""


def test_donor_reuse_reads_max_and_p95_over_donors():
    found = donor_measures(Counter({10: 1, 11: 1, 12: 1, 13: 2}))
    assert found[Measure.DONOR_MAX] == pytest.approx(2.0, abs=0)
    assert found[Measure.DONOR_P95] == pytest.approx(1.85)


def test_a_perturbed_field_stops_counting_as_a_donation():
    groups = (("a", "b"), ("c",))
    assert donations([{"a": 1, "b": 2, "c": 3.5}], [DRAWN], groups) == Counter({7: 1, 8: 1})
    assert donations([{"a": 1, "b": 2, "c": 4.0}], [DRAWN], groups) == Counter({7: 1})
    assert donations([{"a": 1, "b": 9, "c": 3.5}], [DRAWN], groups) == Counter({8: 1})
    mixed = dataclasses.replace(DRAWN, sources={"a": 7, "b": 9, "c": 8})
    assert donations([dict(DRAWN.record)], [mixed], groups) == Counter({8: 1})
    # A record counts once per row, however many units it donates.
    singles = (("a",), ("b",), ("c",))
    assert donations([dict(DRAWN.record)] * 2, [DRAWN] * 2, singles) == Counter({7: 2, 8: 2})


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> Path:
    """A finished `g2` folder of the trace tests' run, its data root beside it."""
    base = tmp_path_factory.mktemp("donors")
    run = base / "run"
    argv = ["score", "g2", "--candidates", str(run / "holdout-candidates.jsonl")]
    _scored_folder(run, trace_context(), [*argv, "--data-dir", str(base / "data")])
    return run


def _donors_of(folder: Path, data_root: Path, *, sha: str | None = None) -> Counter[int]:
    """`donor_counts` of holdout 0's `bko-direct` fill of `folder`, its rows rebuilt from its
    candidate lines; `sha` replaces the `arm` line's sha256."""
    named = _named("g2", trace_context())
    data = load(named.dataset, data_root, named)
    concepts = {row.concept for row in data.field_rows}
    ontologies, _ = load_scope(
        concepts, data_root, releases=named.releases, open_url=named.open_url
    )
    read = read_folder(folder)
    fill = select(read, holdout=0, arm=DIRECT_ARM)
    if sha is not None:
        fill = Fill(0, DIRECT_ARM, re.sub(r" sha256=\w+", f" sha256={sha}", fill.line))
    frame = data.frame
    parts = split(data.records, seed=0, edges=frame.edges)
    train = prepared(parts, named.experiment, edges=frame.edges).ablated.train
    request = gap_request(train, named.experiment, seed=0, frame=frame)
    lines = read.lines[0, DIRECT_ARM, "gap"]
    stored = rebuilt(lines, dataclasses.replace(request, seed=fill.seed), where="the fill")
    return donor_counts(
        fill,
        stored,
        train,
        request,
        data=data,
        ontologies=ontologies,
        budget=named.experiment.budget,
        groups=match_groups(data.field_rows, frame=frame),
    )


def test_donor_reuse_of_a_redrawn_fill_equals_its_trace_sources(folder):
    data = folder.parent / "data"
    code, traced = _trace(folder, data, "--arm", DIRECT_ARM)
    assert code == 0
    named = _named("g2", trace_context())
    groups = match_groups(load(named.dataset, data, named).field_rows, frame=UCI_FRAME)
    expected: Counter[int] = Counter()
    for line in traced:
        sources = line["sources"]
        expected.update(
            {
                sources[unit[0]]
                for unit in groups
                if unit[0] in sources and len({sources.get(name) for name in unit}) == 1
            }
        )
    found = _donors_of(folder, data)
    assert found
    assert found == expected


def test_an_arm_without_a_draw_reads_nan():
    found = donor_measures(Counter())
    assert list(found) == list(DONOR_MEASURES)
    assert all(math.isnan(value) for value in found.values())


def test_a_changed_redraw_is_refused(folder):
    with pytest.raises(
        IdentityError, match=r"re-drawn at seed \d+ hashes to \w+, not its arm line's"
    ):
        _donors_of(folder, folder.parent / "data", sha="0" * 64)
    assert IdentityError.exit_code == 3


# The wiring.

DONORS = dataclasses.replace(
    PRIVATE, arms=(*PRIVATE.arms, DIRECT_ARM), comparators=(*PRIVATE.comparators, DIRECT_ARM)
)
"""`PRIVATE` with `bko-direct` beside its baselines."""


def _stored_direct(tmp_path: Path, caplog) -> Path:
    """`DONORS`' folder scored on `DISTINCT` with its candidates file, and its store."""
    ctx = privacy_context(tmp_path, experiment=DONORS)
    folder = tmp_path / "run"
    folder.mkdir()
    data = ["--data-dir", str(tmp_path / "data")]
    argv = ["score", "--candidates", str(folder / "holdout-candidates.jsonl"), *data]
    caplog.set_level(logging.INFO)
    caplog.clear()
    out = io.StringIO()
    assert main(argv, context=ctx, out=out) == 0
    (folder / "scores.csv").write_text(out.getvalue(), encoding="utf-8")
    lines = "".join(FORMAT.format(record) + "\n" for record in caplog.records)
    (folder / "scores.log").write_text(lines, encoding="utf-8")
    store = tmp_path / "run.jsonl"
    argv = ["rows", "--folder", str(folder), "--rows", str(store), *data]
    assert main(argv, context=ctx, out=io.StringIO()) == 0
    caplog.clear()
    return store


def _read(tmp_path: Path, store: Path, *, experiment=DONORS) -> tuple[int, str]:
    argv = ["privacy", "--data-dir", str(tmp_path / "data"), "--rows", str(store)]
    out = io.StringIO()
    code = main(argv, context=privacy_context(tmp_path, experiment=experiment), out=out)
    return code, out.getvalue()


@pytest.fixture
def reference(monkeypatch):
    """`e1`, which the test context's experiment stands in for, read against `real`."""
    monkeypatch.setattr(
        reference_module, "REFERENCE", {**reference_module.REFERENCE, "e1": REAL_ARM}
    )


def _rule_lines(messages: list[str]) -> list[str]:
    return [line for line in messages if line.startswith("privacy arm=")]


@pytest.mark.usefixtures("reference")
def test_the_new_measures_join_privacy_output_and_the_old_rows_are_unchanged(
    tmp_path, caplog, monkeypatch
):
    store = _stored_direct(tmp_path, caplog)
    caplog.set_level(logging.INFO)
    code, text = _read(tmp_path, store)
    assert code == 0
    ours = parse(text)
    rules = _rule_lines(caplog.messages)
    assert re.search(
        rf"privacy donor holdout=0 arm={DIRECT_ARM} donors=[1-9]\d* max=[1-9]\d* p95=\S+",
        caplog.text,
    )
    new = {*MATCH_MEASURES, *DONOR_MEASURES}
    for arm in DONORS.arms:
        held = {s.measure for s in ours if s.arm == arm and s.holdout == 0}
        assert new <= held
    values = {(s.holdout, s.arm, s.measure): s.value for s in ours}
    assert math.isfinite(values[0, DIRECT_ARM, Measure.DONOR_MAX])
    assert math.isnan(values[0, "smote", Measure.DONOR_MAX])
    assert math.isfinite(values[0, "smote", match("auc", "outside")])
    assert any(" scope=match-outside measure=match-auc-outside " in line for line in rules)
    original = membership.holdout_scores

    def without(*args, extra=None, **kwargs):
        del extra
        return original(*args, **kwargs)

    monkeypatch.setattr(membership, "holdout_scores", without)
    monkeypatch.setattr(membership, "GROUPS", GROUPS[:4])
    caplog.clear()
    code, before = _read(tmp_path, store)
    assert code == 0
    named = {measure.value for measure in new}
    kept = [line for line in text.splitlines() if line.split(",")[3] not in named]
    assert kept == before.splitlines()
    assert [line for line in rules if " scope=match-" not in line] == _rule_lines(caplog.messages)
    (line,) = [
        line
        for line in rules
        if f"arm=smote scope=outside measure={Measure.MI_AUC_OUTSIDE} " in line
    ]
    # `smote` and `bko-direct` read a defined AUC outside the cell; the match groups add none.
    assert " family=2 " in line


def test_a_leaking_arm_is_flagged_by_its_match_group():
    measures = [match("auc", "outside"), match("tpr01", "outside")]
    arms = {"leak": [0.1 + w / 3 for w in WOBBLE], "null": WOBBLE}
    found = readings(_scored(arms, measures), GROUPS, arms=[*arms, REAL_ARM], reference=REAL_ARM)
    leak, null = (_reading(found, arm, match("auc", "outside")) for arm in ("leak", "null"))
    assert (leak.leaks_more, null.leaks_more) == (True, False)
    assert (leak.family, leak.scope) == (4, "match-outside")
    assert not any(r.measure.value.startswith("mi-") for r in found)
