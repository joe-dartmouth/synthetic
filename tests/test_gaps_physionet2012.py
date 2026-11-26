"""`python -m bko.gaps physionet2012` over a catalog entry whose files and releases are built
here."""

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
from tests._gaps_fixtures import ladder_releases
from tests._ontology_fixtures import RoutingOpener
from tests._physionet2012_fixtures import HR_CUTOFF, built, stay


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

TACHYCARDIA = FieldMapping(
    when=(Codes("hr_max", frozenset({2})),),
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
"""Rows on PhysioNet 2012's fields that name the ladder releases' concepts."""

CELLS = 4 * 2 * 2
"""The grid: four age bands (edges 50, 65 and 80), two sexes and two labels."""

FAST = str(HR_CUTOFF + 5)
SLOW = str(HR_CUTOFF - 20)
STAYS = {
    "a": {n: (stay(n, ("01:00", "HR", FAST), age="70", gender="1"), 0) for n in range(1, 21)},
    "b": {n: (stay(n, ("01:00", "HR", SLOW), age="70", gender="0"), 0) for n in range(21, 41)},
    "c": {41: (stay(41, age="40"), 1)},
}
"""A heart rate above the cutoff in every man of 65 to 79 who survived, in no such woman, and one
death under 50."""

RECORDS = sum(len(held) for held in STAYS.values())
"""Every stay is a record: no age or sex is unknown."""

DIED = sum(death for held in STAYS.values() for _, death in held.values())

MEASURED_HR = len(STAYS["a"]) + len(STAYS["b"])
"""The stays holding a heart rate."""

TRAIN, TEST = 33, 8
"""The seed-0 split of the `RECORDS` records, as its `split` line reads."""


def make_context(**change) -> Context:
    pins, served = ladder_releases()
    files = built(STAYS)
    entry = dataclasses.replace(
        CATALOG["physionet2012"],
        sources=tuple(source for source, _ in files),
        rows=(TACHYCARDIA, DEATH),
    )
    context = Context(
        open_url=RoutingOpener({**served, **{source.url: data for source, data in files}}),
        releases=pins,
        catalog={"physionet2012": entry},
    )
    return dataclasses.replace(context, **change)


def run(data_dir: Path, context: Context) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["physionet2012", "--data-dir", str(data_dir)], context=context, out=out)
    return code, out.getvalue()


def test_gaps_physionet2012_lists_the_training_splits_gaps(tmp_path: Path, caplog) -> None:
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
        "frequency-aware,hpo,HP:9000003,Angina pectoris,HP:9000002,Chest pain,65-79,female,0,"
    ) in lines
    assert f"observed dataset=physionet2012 records={TRAIN} " in caplog.text
    assert any(m.startswith(f"gaps rows={len(lines) - 1} sha256=") for m in caplog.messages)


def test_gaps_physionet2012_logs_its_records_and_its_split_once(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO)
    assert run(tmp_path, make_context())[0] == 0
    found = [
        (record.name, record.getMessage().split(" digest=")[0].split(" fingerprint=")[0])
        for record in caplog.records
        if record.getMessage().startswith(("cohort ", "data ", "split "))
    ]
    assert [name for name, _ in found] == ["bko.datasets.physionet2012", "bko.gaps", "bko.datasets"]
    assert found[0][1].startswith(
        f"cohort dataset=physionet2012 stays={RECORDS} stays_a={len(STAYS['a'])} "
        f"stays_b={len(STAYS['b'])} stays_c={len(STAYS['c'])} unknown_age=0 unknown_sex=0 "
        f"left_out=0 records={RECORDS} died={DIED} unnamed=0 missing=0 below_zero=0 "
        f"measured_hr_max={MEASURED_HR} "
    )
    assert TRAIN + TEST == RECORDS
    assert found[1:] == [
        ("bko.gaps", f"data dataset=physionet2012 frame=physionet2012 records={RECORDS}"),
        ("bko.datasets", f"split seed=0 train={TRAIN} test={TEST}"),
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
    [source, *_] = context.catalog["physionet2012"].sources
    files = {s.url: d for s, d in built(STAYS)}
    opener = RoutingOpener({**served, **files, source.url: b"x" * source.size})
    assert run(tmp_path, dataclasses.replace(context, open_url=opener)) == (3, "")
