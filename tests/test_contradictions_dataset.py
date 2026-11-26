"""`contradictions --dataset`: the census of chosen frames; the default census keeps its three."""

from __future__ import annotations

import dataclasses
import io
import logging
from pathlib import Path

import pytest

import bko.evaluate.contradictions as census_module
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.contradictions import CENSUS, CHOICES, Framed, run
from bko.evaluate.data import Dataset
from bko.gaps.contradictions import KINDS
from bko.ontology.catalog import CATALOG
from bko.ontology.schema import Codes, FieldMapping
from tests._datasets_fixtures import pin
from tests._gaps_fixtures import ladder_releases
from tests._ontology_fixtures import RoutingOpener
from tests._support2_fixtures import support2_archive


ARCHIVE = support2_archive()
PIN = pin(ARCHIVE, "support2")
TACHYCARDIA = FieldMapping(
    when=(Codes("hrt", frozenset({2})),),
    concept="HP:9000003",
    label="Angina pectoris",
    source="t",
    limits="t",
)


def _recording(monkeypatch) -> list[tuple[Dataset, ...]]:
    read: list[tuple[Dataset, ...]] = []

    def frames(datasets, data_dir: Path, context: Context) -> list[Framed]:
        read.append(tuple(datasets))
        return []

    monkeypatch.setattr(census_module, "frames", frames)
    return read


def _census(tmp_path: Path, *extra: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["contradictions", "--data-dir", str(tmp_path), *extra], out=out)
    return code, out.getvalue()


def test_the_default_census_is_the_three_frames(tmp_path: Path, monkeypatch) -> None:
    read = _recording(monkeypatch)
    assert CENSUS == (Dataset.UCI, Dataset.SYNTHEA, Dataset.DIABETES)
    assert _census(tmp_path)[0] == 0
    assert read == [CENSUS]


def test_dataset_selects_one_frame(tmp_path: Path, monkeypatch) -> None:
    read = _recording(monkeypatch)
    assert _census(tmp_path, "--dataset", "support2")[0] == 0
    assert _census(tmp_path, "--dataset", "support2", "--dataset", "uci")[0] == 0
    assert read == [(Dataset.SUPPORT2,), (Dataset.SUPPORT2, Dataset.UCI)]


def test_a_repeated_dataset_is_read_once_at_its_first_place(tmp_path: Path, monkeypatch) -> None:
    read = _recording(monkeypatch)
    argv = ("--dataset", "support2", "--dataset", "uci", "--dataset", "support2")
    assert _census(tmp_path, *argv)[0] == 0
    assert read == [(Dataset.SUPPORT2, Dataset.UCI)]


def test_a_dataset_beside_a_run_exits_2_naming_both(tmp_path: Path, monkeypatch, caplog) -> None:
    read = _recording(monkeypatch)
    assert _census(tmp_path, "e1", "--dataset", "support2") == (2, "")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [
        (
            "contradictions --dataset reads the census of the named frames, which takes no run, "
            "not 'e1'; a run's added rows are read with --rows"
        )
    ]
    assert read == []


def test_every_catalog_name_is_a_dataset() -> None:
    assert {name: Dataset(name) for name in CATALOG} == {
        "support2": Dataset.SUPPORT2,
        "acs": Dataset.ACS,
    }


def test_the_choices_are_the_census_and_the_catalog() -> None:
    assert CHOICES == (
        Dataset.UCI,
        Dataset.SYNTHEA,
        Dataset.DIABETES,
        Dataset.SUPPORT2,
        Dataset.ACS,
    )
    assert tuple(CATALOG) == ("support2", "acs")


@pytest.mark.parametrize(
    "extra",
    [
        ("--dataset", "nowhere"),
        ("--dataset", "diabetes-codes"),
        ("--dataset", "support2", "--rows", "x.jsonl"),
    ],
    ids=["unknown", "not-a-census-frame", "with-rows"],
)
def test_an_unknown_dataset_exits_2(tmp_path: Path, monkeypatch, extra) -> None:
    read = _recording(monkeypatch)
    with pytest.raises(SystemExit) as stopped:
        _census(tmp_path, *extra)
    assert stopped.value.code == 2
    assert read == []


def test_the_default_output_keeps_its_bytes(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[Dataset, ...]] = []
    monkeypatch.setattr(census_module, "census", lambda framed: "header\n")
    monkeypatch.setattr(
        census_module, "frames", lambda datasets, *_: calls.append(tuple(datasets)) or []
    )
    assert run(tmp_path, Context()) == "header\n"
    assert run(tmp_path, Context(), None) == "header\n"
    assert calls == [CENSUS, CENSUS]


def test_the_support2_census_reads_its_training_split(tmp_path: Path, caplog) -> None:
    pins, served = ladder_releases()
    entry = dataclasses.replace(CATALOG["support2"], sources=(PIN,), rows=(TACHYCARDIA,))
    context = Context(
        open_url=RoutingOpener({**served, PIN.url: ARCHIVE}),
        releases=pins,
        catalog={"support2": entry},
    )
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        text = run(tmp_path, context, (Dataset.SUPPORT2,))
    assert text.startswith("kind,frame,cell,")
    assert any(m.startswith("census data frame=support2 records=") for m in caplog.messages)
    assert sum(m.startswith("census kind=") for m in caplog.messages) == len(KINDS)
