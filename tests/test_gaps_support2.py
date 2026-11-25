"""`python -m bko.gaps support2` over a catalog entry whose archive and releases are built here."""

from __future__ import annotations

import dataclasses
import io
import logging
import urllib.error
from pathlib import Path

import pytest

from bko.datasets.cells import LABEL
from bko.gaps.__main__ import Context, main
from bko.ontology.catalog import CATALOG
from bko.ontology.schema import Codes, FieldMapping
from tests._datasets_fixtures import pin
from tests._gaps_fixtures import ladder_releases
from tests._ontology_fixtures import RoutingOpener
from tests._support2_fixtures import support2_archive


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

ARCHIVE = support2_archive()
PIN = pin(ARCHIVE, "support2")

TACHYCARDIA = FieldMapping(
    when=(Codes("hrt", frozenset({2})),),
    concept="HP:9000003",
    label="Angina pectoris",
    source="t",
    limits="t",
)
DEATH = FieldMapping(
    when=(Codes(LABEL, frozenset({1})),),
    concept="MONDO:9000004",
    label="coronary stenosis",
    source="t",
    limits="t",
)
"""Rows on SUPPORT2's fields that name the ladder releases' concepts."""


def make_context(**change) -> Context:
    pins, served = ladder_releases()
    entry = dataclasses.replace(CATALOG["support2"], sources=(PIN,), rows=(TACHYCARDIA, DEATH))
    context = Context(
        open_url=RoutingOpener({**served, PIN.url: ARCHIVE}),
        releases=pins,
        catalog={"support2": entry},
    )
    return dataclasses.replace(context, **change)


def run(data_dir: Path, context: Context) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["support2", "--data-dir", str(data_dir)], context=context, out=out)
    return code, out.getvalue()


def test_gaps_support2_lists_the_training_splits_gaps(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.gaps"):
        code, out = run(tmp_path, make_context())
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == (
        "detector,source,antecedent,antecedent_label,consequent,consequent_label,"
        "age_band,sex,target,count"
    )
    assert {line.split(",")[0] for line in lines[1:]} <= {
        "per-subgroup",
        "population-level",
        "frequency-aware",
        "count",
    }
    counts = [line for line in lines if line.startswith("count,")]
    assert len(counts) == 16
    assert "count,,,,,,>=80,male,1,0" in counts
    assert any(m.startswith("observed dataset=support2 records=") for m in caplog.messages)
    assert any(m.startswith("gaps rows=") for m in caplog.messages)


def test_gaps_support2_logs_its_records_and_its_split_once(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO)
    assert run(tmp_path, make_context())[0] == 0
    found = [
        (record.name, record.getMessage())
        for record in caplog.records
        if record.getMessage().startswith(("cohort ", "data ", "split "))
    ]
    assert found == [
        (
            "bko.datasets.support2",
            "cohort dataset=support2 rows=12 missing_scoma=0 records=12 hospital_deaths=4",
        ),
        (
            "bko.gaps",
            (
                "data dataset=support2 frame=support2 records=12 digest="
                "b82d2872761d699238d7045ebb7485b05449640f33b5755a8c335c3fcf2177a9 agreement=none"
            ),
        ),
        (
            "bko.datasets",
            (
                "split seed=0 train=12 test=0 "
                "fingerprint=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
        ),
    ]


def test_a_second_run_writes_the_same_bytes_and_fetches_nothing(tmp_path: Path) -> None:
    first = run(tmp_path, make_context())
    again = make_context()
    assert run(tmp_path, again) == first
    opener = again.open_url
    assert isinstance(opener, RoutingOpener)
    assert opener.calls == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("unpinned", "no pinned release for the prefix MONDO"),
        ("unreachable", "down"),
    ],
)
def test_missing_releases_exit_two(tmp_path: Path, caplog, change: str, message: str) -> None:
    pins, _ = ladder_releases()
    if change == "unpinned":
        context = make_context(releases={"HP": pins["HP"]})
    else:
        context = make_context(open_url=RoutingOpener({}, error=urllib.error.URLError("down")))
    with caplog.at_level(logging.ERROR, logger="bko.gaps"):
        assert run(tmp_path, context) == (2, "")
    assert message in caplog.text


def test_a_pin_mismatch_exits_three(tmp_path: Path) -> None:
    _, served = ladder_releases()
    tampered = {PIN.url: b"x" * len(ARCHIVE)}
    assert run(tmp_path, make_context(open_url=RoutingOpener({**served, **tampered}))) == (3, "")


def test_the_parser_offers_each_catalog_name() -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["--help"], context=make_context(), out=io.StringIO())
    assert stopped.value.code == 0
