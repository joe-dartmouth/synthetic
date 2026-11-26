"""`python -m bko.gaps acs` over a catalog entry whose archive and releases are built here."""

from __future__ import annotations

import dataclasses
import io
import logging
from collections import Counter
from pathlib import Path

import pytest

from bko.gaps.__main__ import Context, main
from bko.ontology.catalog import CATALOG
from bko.ontology.schema import Codes, FieldMapping
from tests._acs_fixtures import acs_archive
from tests._datasets_fixtures import pin
from tests._gaps_fixtures import ladder_releases
from tests._occupation_fixtures import OCCO_TEXT
from tests._ontology_fixtures import RoutingOpener


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

ARCHIVE = acs_archive()
PIN = pin(ARCHIVE, "acs-ma-2024")
OCCO_PIN = pin(OCCO_TEXT.encode(), "occo")

HEARING = FieldMapping(
    when=(Codes("DEAR", frozenset({1})),),
    concept="HP:9000003",
    label="Angina pectoris",
    source="t",
    limits="t",
)
EXECUTIVE = FieldMapping(
    when=(Codes("OCCP", frozenset({10})),),
    concept="OCCO:11100000",
    label="top executive",
    source="t",
    limits="t",
)
"""Rows on ACS's fields that name the ladder HPO's and the fixture OccO's classes."""

CELLS = 4 * 2 * 2
"""The grid: four age bands (edges 30, 45 and 60), two sexes and two labels."""


def make_context(**change) -> Context:
    pins, served = ladder_releases()
    entry = dataclasses.replace(
        CATALOG["acs"],
        sources=(PIN,),
        rows=(HEARING, EXECUTIVE),
        releases={"HP": pins["HP"], "OCCO": OCCO_PIN},
    )
    context = Context(
        open_url=RoutingOpener({**served, PIN.url: ARCHIVE, OCCO_PIN.url: OCCO_TEXT.encode()}),
        catalog={"acs": entry},
    )
    return dataclasses.replace(context, **change)


def run(data_dir: Path, context: Context) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["acs", "--data-dir", str(data_dir)], context=context, out=out)
    return code, out.getvalue()


def test_gaps_acs_lists_the_gaps_of_its_own_releases(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.INFO):
        code, out = run(tmp_path, make_context())
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == (
        "detector,source,antecedent,antecedent_label,consequent,consequent_label,"
        "age_band,sex,target,count"
    )
    detectors = Counter(line.split(",")[0] for line in lines[1:])
    assert detectors == {"per-subgroup": 56, "count": CELLS}
    assert (
        "per-subgroup,occo,OCCO:11100000,top executive,OCCO:11000000,management occupation,"
        "<30,male,0,"
    ) in lines
    assert {line.split(",")[1] for line in lines[1:] if not line.startswith("count,")} == {
        "hpo",
        "occo",
    }
    assert "release occo data_version=releases/2025-10-06" in caplog.text
    assert "observed dataset=acs records=12 " in caplog.text
    assert any(m.startswith(f"gaps rows={len(lines) - 1} sha256=") for m in caplog.messages)


def test_gaps_acs_logs_its_records_and_its_split_once(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO)
    assert run(tmp_path, make_context())[0] == 0
    found = [
        (record.name, record.getMessage().split(" digest=")[0].split(" fingerprint=")[0])
        for record in caplog.records
        if record.getMessage().startswith(("cohort ", "data ", "split "))
    ]
    assert found == [
        (
            "bko.datasets.acs",
            (
                "cohort dataset=acs rows=12 age_over_16=12 income_over_100=12 hours_over_0=12 "
                "weight_at_least_1=12 records=12 income_over_50000=4"
            ),
        ),
        ("bko.gaps", "data dataset=acs frame=acs records=12"),
        ("bko.datasets", "split seed=0 train=12 test=0"),
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
    tampered = {PIN.url: b"x" * len(ARCHIVE), OCCO_PIN.url: OCCO_TEXT.encode()}
    assert run(tmp_path, make_context(open_url=RoutingOpener({**served, **tampered}))) == (3, "")
