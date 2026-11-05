"""The `contradictions` census: each planted kind counted in its cell, with no API call."""

from __future__ import annotations

import io
import logging
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest

import bko.evaluate.contradictions as census_module
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.contradictions import HEADER, Framed, census
from bko.evaluate.data import Dataset
from bko.generate.messages import AnySend
from tests._contradiction_fixtures import (
    BASE_ROWS,
    BROAD,
    INNER,
    KINDS,
    NAME,
    NARROW,
    OVERLAP_A,
    OVERLAP_B,
    ontologies,
    plant,
)
from tests._validate_fixtures import planted
from tests.test_generate_direct import frame


PLANTED_ROWS: Final[dict[str, str]] = {
    "narrow-without-broad": (
        f"narrow-without-broad,{NAME},<55/female/1,{NARROW},{BROAD},6,6,0,6,1.000000,0.000000,,"
    ),
    "child-share-above-parent": (
        f"child-share-above-parent,{NAME},<55/female/1,{NARROW},{BROAD},2,2,1,1,1.000000,0.500000,,"
    ),
    "empty-cell": f"empty-cell,{NAME},<55/female/1,,,0,,,,,,,",
    "all-show": f"all-show,{NAME},<55/female/1,{INNER},,6,6,,,,,,",
    "code-overlap": f"code-overlap,{NAME},,{OVERLAP_A},{OVERLAP_B},,,,,,,ca,2",
}
"""The row each kind plants, in the cell it plants it."""

PLANTED_COUNTS: Final[dict[str, dict[str, int]]] = {
    "narrow-without-broad": {
        "narrow-without-broad": 1,
        "child-share-above-parent": 1,
        "empty-cell": 3,
        "all-show": 4,
        "code-overlap": 0,
    },
    "child-share-above-parent": {
        "narrow-without-broad": 1,
        "child-share-above-parent": 1,
        "empty-cell": 3,
        "all-show": 2,
        "code-overlap": 0,
    },
    "empty-cell": {
        "narrow-without-broad": 1,
        "child-share-above-parent": 1,
        "empty-cell": 4,
        "all-show": 4,
        "code-overlap": 0,
    },
    "all-show": {
        "narrow-without-broad": 0,
        "child-share-above-parent": 0,
        "empty-cell": 3,
        "all-show": 20,
        "code-overlap": 0,
    },
    "code-overlap": {
        "narrow-without-broad": 0,
        "child-share-above-parent": 0,
        "empty-cell": 3,
        "all-show": 0,
        "code-overlap": 1,
    },
}
"""Each planted split's count of every kind: the planted kind and those its cells carry too."""


def framed(kind: str) -> Framed:
    found = plant(kind)
    return Framed(NAME, found.frame, found.rows, found.ontologies, UCI_FRAME)


def clean() -> Framed:
    """Every cell of UCI's grid holds six records, half showing the `is_a` pair, half neither."""
    ages = {"<55": 40, ">=55": 63}
    records = [
        planted(
            age=ages[band] + index,
            sex=sex,
            target=label,
            restecg=index % 3,
            thal=3,
            thalach=120 + 5 * index,
            cp=1 if index % 2 else 4,
            exang=index % 2,
        )
        for band in ("<55", ">=55")
        for sex in (0, 1)
        for label in (0, 1)
        for index in range(6)
    ]
    return Framed(NAME, frame(records), BASE_ROWS, ontologies(), UCI_FRAME)


def counted(caplog: pytest.LogCaptureFixture, kinds: tuple[str, ...] = KINDS) -> dict[str, int]:
    found: dict[str, int] = {}
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith("census kind="):
            fields = dict(part.split("=", 1) for part in message.split()[1:])
            assert fields["frame"] == NAME
            found[fields["kind"]] = int(fields["count"])
    assert tuple(found) == kinds
    return found


@pytest.mark.parametrize("kind", KINDS)
def test_each_planted_kind_is_counted_in_its_cell(kind, caplog):
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        text = census([framed(kind)])
    lines = text.splitlines()
    assert lines[0] == HEADER
    assert PLANTED_ROWS[kind] in lines[1:]
    found = counted(caplog)
    assert found == PLANTED_COUNTS[kind]
    for each in KINDS:
        assert sum(line.startswith(f"{each},") for line in lines[1:]) == found[each]


def test_a_clean_frame_prints_the_header_only_and_logs_every_kind_at_zero(caplog):
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        text = census([clean()])
    assert text == f"{HEADER}\n"
    assert counted(caplog) == dict.fromkeys(KINDS, 0)
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        m.startswith(f"census data frame={NAME} records=48 digest=")
        and m.endswith(" basis=train seed=0")
        for m in messages
    )
    assert f"census rows=0 sha256={sha256(text.encode()).hexdigest()}" in messages


def test_two_runs_give_one_sha256(caplog):
    frames = [framed(kind) for kind in KINDS]
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        first, second = census(frames), census([framed(kind) for kind in KINDS])
    assert first == second
    digest = sha256(first.encode()).hexdigest()
    rows = len(first.splitlines()) - 1
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("census rows=")]
    assert lines == [f"census rows={rows} sha256={digest}"] * 2


def _refused() -> AnySend:
    msg = "the census makes no API call"
    raise AssertionError(msg)


def test_the_command_reads_every_dataset_and_never_sends(tmp_path: Path, monkeypatch, caplog):
    read: list[Dataset] = []

    def loaded(dataset: Dataset, data_dir: Path, context: Context) -> Framed:
        assert data_dir == tmp_path.resolve()
        read.append(dataset)
        return framed(KINDS[len(read) - 1])

    monkeypatch.setattr(census_module, "framed", loaded)
    out = io.StringIO()
    context = Context(send=_refused, local_send=lambda _: _refused())
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        code = main(["contradictions", "--data-dir", str(tmp_path)], context=context, out=out)
    assert code == 0
    assert read == list(Dataset)
    assert out.getvalue() == census([framed(kind) for kind in KINDS[: len(Dataset)]])
