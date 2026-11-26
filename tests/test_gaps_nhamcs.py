"""`python -m bko.gaps nhamcs` over a catalog entry whose archive and releases are built here."""

from __future__ import annotations

import dataclasses
import io
import logging
from collections import Counter
from pathlib import Path

import pytest

from bko.datasets.cells import LABEL
from bko.gaps.__main__ import Context, main
from bko.ontology.catalog import CATALOG
from bko.ontology.schema import Codes, FieldMapping
from tests._datasets_fixtures import pin
from tests._gaps_fixtures import ladder_releases
from tests._nhamcs_fixtures import chronic, line, nhamcs_archive
from tests._ontology_fixtures import RoutingOpener


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

HEART_FAILURE = FieldMapping(
    when=(Codes("CHF", frozenset({1})),),
    concept="HP:9000003",
    label="Angina pectoris",
    source="t",
    limits="t",
)
ADMITTED = FieldMapping(
    when=(Codes(LABEL, frozenset({1})),),
    concept="MONDO:9000004",
    label="coronary stenosis",
    source="t",
    limits="t",
)
"""Rows on NHAMCS's fields that name the ladder releases' concepts."""

CELLS = 4 * 2 * 2
"""The grid: four age bands (edges 18, 45 and 65), two sexes and two labels."""

VISITS = (
    *(line(n, AGE="070", SEX="2", **chronic("CHF")) for n in range(1, 21)),
    *(line(n, AGE="070", SEX="1", **chronic()) for n in range(21, 41)),
    line(41, AGE="030", ADMITHOS="1"),
)
"""Heart failure in every male visit of 65 and over that was not admitted, in no such female
visit, and one admission under 45."""


def make_context(**change) -> Context:
    pins, served = ladder_releases()
    archive = nhamcs_archive(*VISITS)
    source = pin(archive, "nhamcs-ed-2022")
    entry = dataclasses.replace(
        CATALOG["nhamcs"], sources=(source,), rows=(HEART_FAILURE, ADMITTED)
    )
    context = Context(
        open_url=RoutingOpener({**served, source.url: archive}),
        releases=pins,
        catalog={"nhamcs": entry},
    )
    return dataclasses.replace(context, **change)


def run(data_dir: Path, context: Context) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["nhamcs", "--data-dir", str(data_dir)], context=context, out=out)
    return code, out.getvalue()


def test_gaps_nhamcs_lists_the_training_splits_gaps(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.gaps"):
        code, out = run(tmp_path, make_context())
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == (
        "detector,source,antecedent,antecedent_label,consequent,consequent_label,"
        "age_band,sex,target,count"
    )
    detectors = Counter(row.split(",")[0] for row in lines[1:])
    assert detectors["count"] == CELLS
    assert (
        "frequency-aware,hpo,HP:9000003,Angina pectoris,HP:9000002,Chest pain,>=65,female,0,"
    ) in lines
    assert "observed dataset=nhamcs records=33 " in caplog.text
    assert any(m.startswith(f"gaps rows={len(lines) - 1} sha256=") for m in caplog.messages)


def test_gaps_nhamcs_logs_its_records_and_its_split_once(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO)
    assert run(tmp_path, make_context())[0] == 0
    found = [
        (record.name, record.getMessage().split(" digest=")[0].split(" fingerprint=")[0])
        for record in caplog.records
        if record.getMessage().startswith(("cohort ", "data ", "split "))
    ]
    assert found == [
        (
            "bko.datasets.nhamcs",
            "cohort dataset=nhamcs rows=41 records=41 admitted=1 chronic_blank=0",
        ),
        ("bko.gaps", "data dataset=nhamcs frame=nhamcs records=41"),
        ("bko.datasets", "split seed=0 train=33 test=8"),
    ]


def test_a_second_run_writes_the_same_bytes_and_fetches_nothing(tmp_path: Path) -> None:
    first = run(tmp_path, make_context())
    again = make_context()
    assert run(tmp_path, again) == first
    opener = again.open_url
    assert isinstance(opener, RoutingOpener)
    assert opener.calls == []


def test_a_pin_mismatch_exits_three(tmp_path: Path) -> None:
    _, served = ladder_releases()
    context = make_context()
    [source] = context.catalog["nhamcs"].sources
    opener = RoutingOpener({**served, source.url: b"x" * source.size})
    assert run(tmp_path, dataclasses.replace(context, open_url=opener)) == (3, "")
