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
from pathlib import Path
from typing import Final

import pytest

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
    check_records,
)
from bko.evaluate.experiment import (
    CLLM_ARM,
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
from bko.evaluate.scores import parse
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

REMOVED: Final = ("4-7", "female", 1)

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
"""Every kind of arm on two holdouts of the toy cohort, 0.9 of `(4-7, female, 1)` removed."""

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
    assert DIABETES_FRAME.edges == (4, 8)
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
