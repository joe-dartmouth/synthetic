"""Diabetes 130 in the harness: its frame, field rows, data line and a free end-to-end rehearsal.

No test reaches the API or reads `data/`: the archive is the toy cohort `toy_archive` builds, the
releases are invented, the rehearsal answers every call with canned records, and the live send
refuses.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import re
from pathlib import Path
from typing import Final

import pytest

from bko.baselines.request import Method
from bko.datasets.cells import LABEL
from bko.datasets.diabetes import STATE_CODES, load_diabetes
from bko.datasets.errors import ContractError
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.data import Dataset, digest, load
from bko.evaluate.diabetes import (
    DIABETES_FIELD_ROWS,
    DIABETES_FRAME,
    DIABETES_INDICATORS,
    _lab_row,
    check_records,
    indicator,
    indicator_codes,
)
from bko.evaluate.experiment import (
    BKO_ARMS,
    CLLM_ARM,
    CLLM_RAW_ARM,
    DIRECT_ARM,
    E1,
    REAL_ARM,
    SKELETON_ARM,
    Experiment,
    Removal,
)
from bko.evaluate.holdout_candidates import read
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.run import method_arms
from bko.evaluate.scores import Source, parse
from bko.gaps.shares import nested
from bko.generate.messages import SETTINGS, AnySend
from bko.generate.prompt import system
from bko.ontology.icd9 import ICD9_MAPPINGS, LAB_MAPPINGS
from bko.ontology.schema import Codes
from tests._diabetes_fixtures import TOY_CELLS, diabetes_releases, pin, toy_archive
from tests._evaluate_bko_fixtures import candidates_for
from tests._evaluate_fixtures import FAST_METHODS
from tests._ontology_fixtures import RoutingOpener


UCI_WORDS: Final = ("chol", "thalach", "trestbps", "oldpeak", "restecg", "angiography", "angina")
"""UCI Heart Disease's columns and findings, none of which a Diabetes prompt may name."""

ARCHIVE: Final = toy_archive()
PIN: Final = pin(ARCHIVE, "uci-diabetes-130")

REMOVED: Final = ("6-7", "female", 1)

PREVALENCE: Final = round(
    sum(n for (_, _, label), n in TOY_CELLS.items() if label) / sum(TOY_CELLS.values()), 2
)
"""The toy cohort's share of label 1, the toy run's cutoff, as E6 reads its own prevalence."""

ARMS: Final = (
    "none",
    REAL_ARM,
    "class-weight",
    "smote",
    "ctgan",
    "bko",
    DIRECT_ARM,
    SKELETON_ARM,
    CLLM_ARM,
)

TOY_RUN: Final = dataclasses.replace(
    E1,
    holdout_seeds=(0, 1),
    arms=ARMS,
    candidate="bko",
    comparators=("none", "class-weight", "smote", "ctgan"),
    replica_arms=(),
    criteria=(),
    removal=Removal(REMOVED, 0.9),
    decontaminate=False,
    cutoff=PREVALENCE,
)
"""Every kind of arm on two holdouts of the toy cohort, 0.9 of `(6-7, female, 1)` removed."""

DIRECT_ONLY: Final = dataclasses.replace(
    E1,
    holdout_seeds=(0, 1),
    arms=("none", DIRECT_ARM),
    candidate=DIRECT_ARM,
    comparators=("none",),
    replica_arms=(),
    criteria=(),
    removal=Removal(REMOVED, 0.9),
)
"""The direct arm beside `none` on two holdouts of the toy cohort, with the toy run's removal."""


def _unsendable() -> AnySend:
    msg = "a rehearsal never builds the live send"
    raise AssertionError(msg)


def _context(source=PIN, **runs: Experiment) -> Context:
    pins, served = diabetes_releases()
    return Context(
        open_url=RoutingOpener({**served, source.url: ARCHIVE}),
        runs={name: NamedRun(run, SETTINGS, Dataset.DIABETES) for name, run in runs.items()},
        arms=method_arms(FAST_METHODS),
        releases=pins,
        send=_unsendable,
        diabetes=source,
    )


def _score(tmp_path: Path, run: str, experiment: Experiment, *flags: str) -> tuple[str, Path]:
    log = tmp_path / "calls.jsonl"
    argv = ["score", run, *flags, "--candidates", str(candidates_for(log))]
    if "--rehearse" in flags:
        argv += ["--log", str(log)]
    argv += ["--data-dir", str(tmp_path / "data")]
    out = io.StringIO()
    assert main(argv, context=_context(**{run: experiment}), out=out) == 0
    return out.getvalue(), log


@pytest.fixture(scope="module")
def rehearsed(tmp_path_factory) -> tuple[str, Path]:
    """The toy run rehearsed once: its scores and its call log."""
    return _score(tmp_path_factory.mktemp("rehearsal"), "toy-e6", TOY_RUN, "--rehearse")


def _requests(log: Path) -> list[dict]:
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line["kind"] == "request"]


def _uci_words(text: str) -> list[str]:
    return [word for word in UCI_WORDS if word in text.lower()]


def test_the_frames_codebooks_are_the_loaders_states_verbatim():
    for name, codes in STATE_CODES.items():
        assert dict(DIABETES_FRAME.codebook[name]) == {code: state for state, code in codes.items()}
    for name in DIABETES_INDICATORS:
        assert dict(DIABETES_FRAME.codebook[name]) == {0: "absent", 1: "present"}
    assert DIABETES_FRAME.edges == (4, 6, 8)
    assert {bound.unit for bound in DIABETES_FRAME.ranges if bound.field == "age"} == {"10.a"}


def test_the_field_rows_read_every_icd9_and_lab_row():
    assert tuple(DIABETES_INDICATORS.values()) == tuple(row.codes for row in ICD9_MAPPINGS)
    by_concept = {row.concept: row for row in DIABETES_FIELD_ROWS}
    assert len(by_concept) == len(ICD9_MAPPINGS) + len(LAB_MAPPINGS)
    for name, row in zip(DIABETES_INDICATORS, ICD9_MAPPINGS, strict=True):
        assert by_concept[row.concept].when == (Codes(name, frozenset({1})),)
        assert by_concept[row.concept].label == row.label
    for row in LAB_MAPPINGS:
        codes = frozenset(STATE_CODES[row.field][state] for state in row.states)
        assert by_concept[row.concept].when == (Codes(row.field, codes),)
        assert by_concept[row.concept].label == row.label


def test_loading_diabetes_builds_its_data_and_logs_its_line(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        data = load(Dataset.DIABETES, tmp_path, _context())
    records = load_diabetes(tmp_path, indicators=DIABETES_INDICATORS, source=PIN)
    assert data.records.equals(records)
    assert tuple(data.records.columns) == DIABETES_FRAME.columns
    assert (data.frame, data.field_rows, data.name) == (
        DIABETES_FRAME,
        DIABETES_FIELD_ROWS,
        PIN.name,
    )
    line = (
        f"data dataset=diabetes frame=diabetes records={len(records)} digest={digest(records)} "
        "agreement=none"
    )
    assert line in caplog.messages


def test_records_outside_the_frames_codebook_are_refused(tmp_path):
    records = load(Dataset.DIABETES, tmp_path, _context()).records
    check_records(records)
    stray = records.copy()
    stray.loc[stray.index[0], "A1Cresult"] = 9
    with pytest.raises(ContractError, match="A1Cresult"):
        check_records(stray)
    with pytest.raises(ContractError, match="columns"):
        check_records(records.drop(columns=[LABEL]))


def test_the_direct_arm_runs_on_diabetes_gap_fills(tmp_path):
    text, _ = _score(tmp_path, "direct", DIRECT_ONLY)
    assert {score.arm for score in parse(text)} >= {"none", DIRECT_ARM}
    kept = read(candidates_for(tmp_path / "calls.jsonl").read_bytes())
    assert kept.header["frame"] == "diabetes"
    direct = [line for line in kept.candidates if line.arm == DIRECT_ARM]
    assert direct
    assert {line.fill for line in direct} == {"gap"}
    for line in direct:
        assert isinstance(line.candidate, dict)
        assert set(line.candidate) == set(DIABETES_FRAME.columns)


def test_diabetes_rehearses_end_to_end_on_a_tmp_archive(rehearsed):
    text, log = rehearsed
    scores = parse(text)
    assert {score.arm for score in scores} >= set(ARMS)
    assert {score.holdout for score in scores} == {0, 1}
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["frame"] == "diabetes"
    assert lines[-1]["status"] == "complete"
    kept = read(candidates_for(log).read_bytes())
    assert kept.header["frame"] == "diabetes"
    assert {line.arm for line in kept.candidates} == {"bko", DIRECT_ARM, SKELETON_ARM, CLLM_ARM}


EVERY_ARM: Final = (
    *(str(method) for method in Method),
    *BKO_ARMS,
    DIRECT_ARM,
    SKELETON_ARM,
    CLLM_ARM,
    CLLM_RAW_ARM,
    REAL_ARM,
)
"""E6's arms: every method, every Messages arm, the direct arm and `real`."""

EMPTIED: Final = dataclasses.replace(
    TOY_RUN,
    arms=EVERY_ARM,
    comparators=tuple(arm for arm in EVERY_ARM if arm not in {"bko", REAL_ARM}),
    removal=Removal(REMOVED, 1.0),
)
"""The toy run at E6's zero-shot removal: every training row of `(6-7, female, 1)` removed."""


def test_every_e6_arm_completes_at_an_emptied_cell(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="bko"):
        text, log = _score(tmp_path, "toy-e6-emptied", EMPTIED, "--rehearse")
    scores = parse(text)
    for holdout in (0, 1):
        scored = {s.arm for s in scores if s.holdout == holdout and s.source is Source.AUGMENTED}
        assert scored == set(EVERY_ARM)
    shortfalls = [m for m in caplog.messages if m.startswith("shortfall holdout=")]
    for arm in ("smote", "adasyn"):
        mine = [m for m in shortfalls if f" arm={arm} " in m]
        assert len(mine) == 2
        assert all(" attempt=1 cell=6-7/female/1 " in m for m in mine)
    assert not any(m.startswith(("retry ", "refused ")) for m in caplog.messages)
    assert json.loads(log.read_text(encoding="utf-8").splitlines()[-1])["status"] == "complete"


def test_a_diabetes_rehearsal_sends_no_uci_words(rehearsed):
    _, log = rehearsed
    requests = _requests(log)
    assert requests
    for line in requests:
        text = json.dumps(line["params"])
        assert not _uci_words(text)
        assert "decades" in text
        assert "essential_hypertension" in text


def test_the_uci_word_check_finds_uci_words_in_a_uci_prompt():
    assert set(_uci_words(system(UCI_FRAME))) >= {"chol", "thalach", "trestbps", "oldpeak"}
    assert not _uci_words(system(DIABETES_FRAME))


def test_two_rows_whose_labels_share_an_indicator_name_are_refused():
    first, second = ICD9_MAPPINGS[:2]
    twin = dataclasses.replace(second, label=first.label.upper())
    with pytest.raises(ValueError, match=f"indicator {indicator(first)!r} names 2 rows"):
        indicator_codes((first, twin))
    assert indicator_codes(ICD9_MAPPINGS) == DIABETES_INDICATORS


@pytest.mark.parametrize(
    ("edit", "named"),
    [
        (lambda records: records.drop(columns=[LABEL]), f"missing ['{LABEL}']"),
        (lambda records: records.assign(extra=0), "extra ['extra']"),
        (
            lambda records: records[[records.columns[1], records.columns[0], *records.columns[2:]]],
            f"first out of order {DIABETES_FRAME.columns[1]!r}",
        ),
    ],
    ids=["missing", "extra", "misordered"],
)
def test_a_column_refusal_names_the_columns(tmp_path, edit, named):
    records = load(Dataset.DIABETES, tmp_path, _context()).records
    with pytest.raises(ContractError, match=re.escape(named)):
        check_records(edit(records))


def test_a_stray_code_is_named_as_a_plain_number(tmp_path):
    records = load(Dataset.DIABETES, tmp_path, _context()).records
    stray = records.copy()
    stray.loc[stray.index[0], "A1Cresult"] = 9
    with pytest.raises(ContractError, match=re.escape("A1Cresult holds codes [9] ")):
        check_records(stray)


@pytest.mark.parametrize(
    ("edit", "named"),
    [({"field": "glucose"}, "field 'glucose'"), ({"states": frozenset({">9"})}, "state '>9'")],
    ids=["field", "state"],
)
def test_a_lab_row_the_frame_cannot_code_is_refused_naming_it(edit, named):
    row = dataclasses.replace(LAB_MAPPINGS[0], **edit)
    with pytest.raises(ValueError, match=re.escape(f"lab row {row.label!r}: {named}")):
        _lab_row(row)


def test_no_field_row_nests_inside_another():
    assert nested(DIABETES_FIELD_ROWS) == {}
