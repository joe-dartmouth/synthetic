"""`python -m bko.gaps nhanes` over a catalog entry whose component files and releases are built
here."""

from __future__ import annotations

import dataclasses
import io
import logging
from collections import Counter
from pathlib import Path

import pytest

from bko.datasets.cells import LABEL
from bko.datasets.nhanes import COMPONENTS
from bko.gaps.__main__ import Context, main
from bko.ontology.catalog import CATALOG
from bko.ontology.schema import Codes, FieldMapping
from tests._datasets_fixtures import pin
from tests._gaps_fixtures import ladder_releases
from tests._nhanes_fixtures import Respondent, files
from tests._ontology_fixtures import RoutingOpener


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

HIGH_SYSTOLIC = FieldMapping(
    when=(Codes("BPXOSY", frozenset({2})),),
    concept="HP:9000003",
    label="Angina pectoris",
    source="t",
    limits="t",
)
COMPOSITE = FieldMapping(
    when=(Codes(LABEL, frozenset({1})),),
    concept="MONDO:9000004",
    label="coronary stenosis",
    source="t",
    limits="t",
)
"""Rows on NHANES's fields that name the ladder releases' concepts."""

CELLS = 3 * 2 * 2
"""The grid: three age bands (edges 40 and 60), two sexes and two labels."""

PEOPLE = (
    *(Respondent(n, age=65.0, systolic=(140.0,) * 3) for n in range(1, 21)),
    *(Respondent(n, age=65.0, sex=2.0, systolic=(120.0,) * 3) for n in range(21, 41)),
    Respondent(41, age=30.0, answers=(1.0, 2.0, 2.0, 2.0), systolic=(120.0,) * 3),
)
"""A high systolic pressure in every male of 60 and over without the composite, in no such female,
and one composite under 40."""


def make_context(people=PEOPLE, **change) -> Context:
    pins, served = ladder_releases()
    data = files(list(people))
    sources = tuple(pin(data[name], f"nhanes-{name.lower()}") for name in COMPONENTS)
    entry = dataclasses.replace(CATALOG["nhanes"], sources=sources, rows=(HIGH_SYSTOLIC, COMPOSITE))
    served_files = {s.url: data[n] for n, s in zip(COMPONENTS, sources, strict=True)}
    context = Context(
        open_url=RoutingOpener({**served, **served_files}),
        releases=pins,
        catalog={"nhanes": entry},
    )
    return dataclasses.replace(context, **change)


def run(data_dir: Path, context: Context) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["nhanes", "--data-dir", str(data_dir)], context=context, out=out)
    return code, out.getvalue()


def test_gaps_nhanes_lists_the_training_splits_gaps(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="bko.gaps"):
        code, out = run(tmp_path, make_context())
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == (
        "detector,source,antecedent,antecedent_label,consequent,consequent_label,"
        "age_band,sex,target,count"
    )
    detectors = Counter(line.split(",")[0] for line in lines[1:])
    assert detectors["count"] == CELLS
    assert (
        "frequency-aware,hpo,HP:9000003,Angina pectoris,HP:9000002,Chest pain,>=60,female,0,"
    ) in lines
    assert "count,,,,,,>=60,male,0,16" in lines
    assert "observed dataset=nhanes records=33 " in caplog.text
    assert any(m.startswith(f"gaps rows={len(lines) - 1} sha256=") for m in caplog.messages)


def test_gaps_nhanes_logs_its_records_and_its_split_once(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO)
    assert run(tmp_path, make_context())[0] == 0
    found = [
        (record.name, record.getMessage().split(" digest=")[0].split(" fingerprint=")[0])
        for record in caplog.records
        if record.getMessage().startswith(("cohort ", "data ", "split "))
    ]
    assert found == [
        (
            "bko.datasets.nhanes",
            (
                "cohort dataset=nhanes demographics=41 interviewed=41 adults=41 any_yes=1 "
                "all_no=40 left_out=0 records=41 composite=1"
            ),
        ),
        ("bko.gaps", "data dataset=nhanes frame=nhanes records=41"),
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
    entry = context.catalog["nhanes"]
    tampered = {source.url: b"x" * source.size for source in entry.sources}
    opener = RoutingOpener({**served, **tampered})
    assert run(tmp_path, dataclasses.replace(context, open_url=opener)) == (3, "")
