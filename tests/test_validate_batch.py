"""Validating a batch: the accepted frame, the rejections and the per-check counts."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from bko.baselines.records import INDEX_NAME, number
from bko.datasets.uci import UCI_FRAME
from bko.gaps.errors import GapError
from bko.gaps.observe import observe_records
from bko.ontology.mapping import FIELD_MAPPINGS
from bko.validate.checks import CANDIDATE_INDEX, CHECKS, Rejection, validate
from tests._baselines_fixtures import EVALUATION_COLUMNS, EVALUATION_DTYPES, full_records
from tests._validate_fixtures import BASE, PLANTED, planted


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

LOGGER = "bko.validate.checks"


def test_every_fixture_record_passes_unchanged():
    frame = full_records()
    result = validate(frame.to_dict("records"), frame=UCI_FRAME)
    assert result.rejected == ()
    assert result.generated == result.passed == len(frame)
    expected = frame.reset_index(drop=True).rename_axis(CANDIDATE_INDEX)
    pd.testing.assert_frame_equal(result.accepted, expected)


@pytest.mark.parametrize("style", ["int", "float"])
def test_the_accepted_frame_has_the_evaluation_frames_columns_and_dtypes(style):
    candidate = dict(BASE) if style == "int" else {key: float(v) for key, v in BASE.items()}
    accepted = validate([candidate], frame=UCI_FRAME).accepted
    assert list(accepted.columns) == EVALUATION_COLUMNS
    assert accepted.dtypes.astype(str).to_dict() == EVALUATION_DTYPES
    assert accepted.index.name == CANDIDATE_INDEX


def test_an_empty_batch_gives_a_typed_empty_frame_and_zero_counts(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = validate([], frame=UCI_FRAME)
    assert result.generated == result.passed == 0
    assert list(result.accepted.columns) == EVALUATION_COLUMNS
    assert result.accepted.dtypes.astype(str).to_dict() == EVALUATION_DTYPES
    assert result.accepted.index.name == CANDIDATE_INDEX
    assert str(result.accepted.index.dtype) == "int64"
    assert dict(result.rejections) == dict.fromkeys(CHECKS, 0)
    assert "validated no candidates" in caplog.text


def test_accepted_and_rejected_partition_the_input_in_order():
    result = validate([BASE, PLANTED["code:cp"], BASE, "not a record", BASE], frame=UCI_FRAME)
    assert result.accepted.index.tolist() == [0, 2, 4]
    assert result.rejected == (Rejection(1, ("code:cp",)), Rejection(3, ("fields",)))
    assert result.generated == 5
    assert result.passed == 3


def test_every_planted_record_in_one_batch_counts_once_under_its_check():
    result = validate([*PLANTED.values(), BASE, BASE], frame=UCI_FRAME)
    assert dict(result.rejections) == dict.fromkeys(CHECKS, 1)
    assert list(result.rejections) == list(CHECKS)
    assert len(result.rejected) == len(PLANTED)
    assert result.passed == 2


def test_a_candidate_counts_under_every_check_it_fails_and_is_rejected_once():
    result = validate([planted(chol=601, thal=4)], frame=UCI_FRAME)
    [only] = result.rejected
    assert only.failed == ("range:chol", "code:thal")
    assert result.rejections["range:chol"] == result.rejections["code:thal"] == 1
    assert sum(result.rejections.values()) == 2
    assert result.generated - result.passed == 1


def test_validate_reads_a_generator_once():
    assert validate((dict(BASE) for _ in range(3)), frame=UCI_FRAME).passed == 3


def test_an_accepted_value_is_kept_as_given():
    accepted = validate([planted(oldpeak=-0.5, chol=600)], frame=UCI_FRAME).accepted
    assert accepted["oldpeak"].tolist() == [-0.5]
    assert accepted["chol"].tolist() == [600]


@pytest.mark.parametrize(
    "batch",
    [dict(BASE), "abc", b"abc", bytearray(b"abc"), memoryview(b"abc"), pd.DataFrame([dict(BASE)])],
    ids=["record", "str", "bytes", "bytearray", "memoryview", "frame"],
)
def test_a_batch_that_is_one_record_a_string_or_a_frame_is_refused(batch):
    with pytest.raises(TypeError, match="batch of candidates"):
        validate(batch, frame=UCI_FRAME)


def test_validation_logs_one_summary_of_counts(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        validate([BASE, planted(chol=601), planted(chol="x")], frame=UCI_FRAME)
    assert "validated generated=3 passed=1 rejected=2 number:chol=1 range:chol=1" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_a_batch_with_no_rejection_logs_none(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        validate([BASE], frame=UCI_FRAME)
    assert "validated generated=1 passed=1 rejected=0 none" in caplog.text


def test_the_accepted_frame_feeds_observe_records():
    frame = full_records()
    got = observe_records(
        validate(frame.to_dict("records"), frame=UCI_FRAME).accepted, FIELD_MAPPINGS, name="c"
    )
    want = observe_records(frame, FIELD_MAPPINGS, name="c")
    assert got.records == want.records
    assert dict(got.holders) == dict(want.holders)


def test_an_empty_accepted_frame_reaches_observe_records_empty_check():
    accepted = validate([PLANTED["code:cp"]], frame=UCI_FRAME).accepted
    with pytest.raises(GapError, match="holds no record"):
        observe_records(accepted, FIELD_MAPPINGS, name="c")


def test_numbering_the_accepted_frame_gives_a_baselines_added_rows():
    frame = full_records()
    added = number(validate(frame.to_dict("records"), frame=UCI_FRAME).accepted)
    assert added.index.name == INDEX_NAME
    assert added.index.tolist() == list(range(-1, -len(frame) - 1, -1))
    assert added.dtypes.astype(str).to_dict() == EVALUATION_DTYPES
