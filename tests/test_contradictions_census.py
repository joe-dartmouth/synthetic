"""The `contradictions` census: each planted kind counted in its cell, with no API call."""

from __future__ import annotations

import dataclasses
import io
import logging
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest

import bko.evaluate.contradictions as census_module
from bko.datasets.sources import Source
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.contradictions import HEADER, Framed, census, frames
from bko.evaluate.data import Data, Dataset, load
from bko.evaluate.errors import EvaluationError
from bko.generate.messages import AnySend
from bko.ontology.schema import FieldMapping
from bko.ontology.terms import Ontology
from tests._contradiction_fixtures import (
    BASE_ROWS,
    BROAD,
    INNER,
    KINDS,
    NAME,
    NARROW,
    OUTER,
    OVERLAP_A,
    OVERLAP_B,
    ROOT,
    ontologies,
    plant,
    row,
)
from tests._datasets_fixtures import pin
from tests._gaps_fixtures import ANGINA, STENOSIS, ladder_releases
from tests._ontology_fixtures import RoutingOpener
from tests._validate_fixtures import planted
from tests.test_evaluate_direct import VARIED
from tests.test_generate_direct import frame


PLANTED_ROWS: Final[dict[str, str]] = {
    "narrow-without-broad": (
        f"narrow-without-broad,{NAME},<55/female/1,{NARROW},{BROAD},6,6,0,6,1.000000,0.000000,,"
    ),
    "child-share-above-parent": (
        f"child-share-above-parent,{NAME},<55/female/1,{NARROW},{BROAD},2,2,1,1,1.000000,0.500000,,"
    ),
    "empty-cell": f"empty-cell,{NAME},<55/female/1,,,0,,,,,,,",
    "all-show": f"all-show,{NAME},<55/female/1,{INNER},,6,6,,,1.000000,,,",
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


def clean_records() -> list[dict[str, object]]:
    """Six records in every cell of UCI's grid, half showing the `is_a` pair, half neither."""
    ages = {"<55": 40, ">=55": 63}
    return [
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


def clean() -> Framed:
    """`clean_records()` read through `BASE_ROWS`."""
    return Framed(NAME, frame(clean_records()), BASE_ROWS, ontologies(), UCI_FRAME)


def messages(caplog: pytest.LogCaptureFixture, start: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(start)]


def counted(caplog: pytest.LogCaptureFixture, kinds: tuple[str, ...] = KINDS) -> dict[str, int]:
    found: dict[str, int] = {}
    for message in messages(caplog, "census kind="):
        fields = dict(part.split("=", 1) for part in message.split()[1:])
        assert fields["frame"] == NAME
        found[fields["kind"]] = int(fields["count"])
    assert tuple(found) == kinds
    return found


def test_the_header_is_the_documented_columns():
    assert HEADER == (
        "kind,frame,cell,concept,parent,records,shown,with_parent,without_parent,"
        "child_share,parent_share,field,codes"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_each_planted_kind_is_counted_in_its_cell(kind, caplog):
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        text = census([framed(kind)])
    lines = text.splitlines()
    assert lines[0] == HEADER
    assert PLANTED_ROWS[kind] in lines[1:]
    found = counted(caplog)
    assert found == PLANTED_COUNTS[kind]
    assert [line.split(",")[0] for line in lines[1:]] == [
        each for each in KINDS for _ in range(found[each])
    ], "the rows come in KINDS order"


CLEAN_DIGEST: Final = "1c9433d00997b6ada7a8691fecb3f9a6fdc0ed1dd774b52978fb08c84667be72"
"""`bko.evaluate.data.digest` of `clean()`'s records."""

BASE_ROWS_DIGEST: Final = "b85233a6740615994945390ac8eb625832492a9a64219e61e9d14b0590b9c2e2"
"""`rows_digest` of `BASE_ROWS`."""


def test_a_clean_frame_prints_the_header_only_and_logs_every_kind_at_zero(caplog):
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        text = census([clean()])
    assert text == f"{HEADER}\n"
    assert counted(caplog) == dict.fromkeys(KINDS, 0)
    assert messages(caplog, "census data ") == [
        f"census data frame={NAME} records=48 digest={CLEAN_DIGEST} basis=train seed=0"
    ]
    assert messages(caplog, "census rows ") == [
        f"census rows frame={NAME} rows=4 digest={BASE_ROWS_DIGEST} pairs=1"
    ]
    assert messages(caplog, "census cells ") == [
        f"census cells frame={NAME} cells=8 examined=8 following=-"
    ]
    assert messages(caplog, "census pair ") == [
        f"census pair frame={NAME} concept={NARROW} parent={BROAD} shown=24 without=0 cells=0"
    ]
    assert messages(caplog, "census rows=") == [
        f"census rows=0 sha256={sha256(text.encode()).hexdigest()}"
    ]


def test_a_rows_digest_moves_with_any_field_of_a_row():
    digest = census_module.rows_digest
    assert digest(BASE_ROWS) == BASE_ROWS_DIGEST
    first = BASE_ROWS[0]
    changed = [
        dataclasses.replace(
            first, when=(dataclasses.replace(first.when[0], codes=frozenset({1})),)
        ),
        dataclasses.replace(first, label="other"),
        dataclasses.replace(first, source="other"),
        dataclasses.replace(first, limits="other"),
    ]
    seen = {digest((each, *BASE_ROWS[1:])) for each in changed}
    assert BASE_ROWS_DIGEST not in seen
    assert len(seen) == len(changed)
    assert digest(tuple(reversed(BASE_ROWS))) != BASE_ROWS_DIGEST


UNEVEN: Final = (
    {"cp": 1, "exang": 0},
    {"cp": 1, "exang": 0},
    {"cp": 4, "exang": 1},
    {"cp": 4, "exang": 0},
    {"cp": 4, "exang": 0},
    {"cp": 4, "exang": 0},
)
"""`<55/female/0`'s records in `uneven()`: two show the narrow concept alone, one the broad."""


def uneven() -> Framed:
    """`clean()` with its first cell's records changed so the pair's shares are 1/3 and 1/6."""
    records = clean_records()
    for index, values in enumerate(UNEVEN):
        records[index] |= values
    return Framed(NAME, frame(records), BASE_ROWS, ontologies(), UCI_FRAME)


def test_a_cell_whose_shares_differ_writes_each_to_six_places(caplog):
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        lines = census([uneven()]).splitlines()[1:]
    cell = "<55/female/0"
    assert lines == [
        f"narrow-without-broad,{NAME},{cell},{NARROW},{BROAD},6,2,0,2,0.333333,0.166667,,",
        f"child-share-above-parent,{NAME},{cell},{NARROW},{BROAD},6,2,0,2,0.333333,0.166667,,",
    ]
    assert messages(caplog, "census pair ") == [
        f"census pair frame={NAME} concept={NARROW} parent={BROAD} shown=23 without=2 cells=1"
    ]


def test_a_concept_following_the_label_is_never_all_show(caplog):
    follows = row("HP:9000031", UCI_FRAME.label, 1)
    every = dataclasses.replace(clean(), rows=(*BASE_ROWS, follows))
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        text = census([every])
    assert text == f"{HEADER}\n"
    assert messages(caplog, "census cells ") == [
        f"census cells frame={NAME} cells=8 examined=8 following=HP:9000031"
    ]


GRID: Final = tuple(
    f"{band}/{sex}/{label}"
    for band in ("<55", ">=55")
    for sex in ("female", "male")
    for label in (0, 1)
)
"""UCI's cells in the grid's order."""


def chained() -> Framed:
    """`clean()` with the root mapped on a value no record holds, and `>=55/male/1`, the last six
    records, dropped.

    Five `is_a` pairs: every mapped concept is under the root, which no record shows, and half of
    each non-empty cell's records show the broad and narrow concepts.
    """
    records = [values | {"fbs": 1} for values in clean_records()[:-6]]
    rows = (*BASE_ROWS, row(ROOT, "fbs", 0))
    return Framed(NAME, frame(records), rows, ontologies(), UCI_FRAME)


def test_pairs_cells_and_kinds_come_in_their_orders(caplog):
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        lines = census([chained()]).splitlines()[1:]
    pairs = ((BROAD, ROOT), (NARROW, ROOT), (OUTER, ROOT), (INNER, ROOT))
    assert [tuple(line.split(",")[:5]) for line in lines] == [
        *(
            (kind, NAME, cell, narrow, broad)
            for kind in ("narrow-without-broad", "child-share-above-parent")
            for cell in GRID[:-1]
            for narrow, broad in pairs
        ),
        ("empty-cell", NAME, GRID[-1], "", ""),
    ]
    assert messages(caplog, "census pair ") == [
        f"census pair frame={NAME} concept={BROAD} parent={ROOT} shown=21 without=21 cells=7",
        f"census pair frame={NAME} concept={NARROW} parent={ROOT} shown=21 without=21 cells=7",
        f"census pair frame={NAME} concept={NARROW} parent={BROAD} shown=21 without=0 cells=0",
        f"census pair frame={NAME} concept={OUTER} parent={ROOT} shown=28 without=28 cells=7",
        f"census pair frame={NAME} concept={INNER} parent={ROOT} shown=14 without=14 cells=7",
    ]
    assert messages(caplog, "census cells ") == [
        f"census cells frame={NAME} cells=8 examined=7 following=-"
    ]


def test_frames_are_written_in_order_each_under_its_own_name(caplog):
    first = dataclasses.replace(framed("code-overlap"), name="first")
    second = dataclasses.replace(framed("empty-cell"), name="second")
    alone = [census([each]).count("\n") - 1 for each in (first, second)]
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        text = census([first, second])
    names = [line.split(",")[1] for line in text.splitlines()[1:]]
    assert names == ["first"] * alone[0] + ["second"] * alone[1]
    frames_logged = [
        m.split("frame=")[1].split()[0]
        for m in (r.getMessage() for r in caplog.records)
        if m.startswith("census ") and "frame=" in m
    ]
    assert frames_logged == sorted(frames_logged, key=["first", "second"].index)
    assert frames_logged.count("first") == frames_logged.count("second")


def test_two_runs_give_one_sha256(caplog):
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        first, second = census([framed(k) for k in KINDS]), census([framed(k) for k in KINDS])
    assert first == second
    digest = sha256(first.encode()).hexdigest()
    rows = len(first.splitlines()) - 1
    assert messages(caplog, "census rows=") == [f"census rows={rows} sha256={digest}"] * 2


def _refused() -> AnySend:
    msg = "the census makes no API call"
    raise AssertionError(msg)


def test_the_command_reads_every_dataset_and_never_sends(tmp_path: Path, monkeypatch, caplog):
    read: list[tuple[Dataset, ...]] = []

    def loaded(datasets: tuple[Dataset, ...], data_dir: Path, context: Context) -> list[Framed]:
        assert data_dir == tmp_path.resolve()
        read.append(datasets)
        return [framed(kind) for kind in KINDS[: len(datasets)]]

    monkeypatch.setattr(census_module, "frames", loaded)
    out = io.StringIO()
    context = Context(send=_refused, local_send=lambda _: _refused())
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        code = main(["contradictions", "--data-dir", str(tmp_path)], context=context, out=out)
    assert code == 0
    assert read == [tuple(Dataset)]
    assert out.getvalue() == census([framed(kind) for kind in KINDS[: len(Dataset)]])
    logged = [r.getMessage() for r in caplog.records]
    assert logged[0].startswith("versions commit=")
    assert logged[1] == f"census data_dir={tmp_path.resolve()}"


def uci_context(releases: Mapping[str, Source] | None = None) -> Context:
    """A context whose UCI archive and two releases (`HP`, `MONDO`) a routing opener serves."""
    pins, served = ladder_releases()
    uci = pin(VARIED)
    return Context(
        open_url=RoutingOpener({**served, uci.url: VARIED}),
        uci=uci,
        releases=pins if releases is None else releases,
        field_rows=(ANGINA, STENOSIS),
        send=_refused,
    )


def test_frames_reads_the_training_split_at_seed_0_and_each_named_release(tmp_path: Path):
    context = uci_context()
    (found,) = frames((Dataset.UCI,), tmp_path, context)
    data = load(Dataset.UCI, tmp_path, context)
    assert found.name == UCI_FRAME.name
    assert found.frame == UCI_FRAME
    assert found.rows == (ANGINA, STENOSIS)
    assert found.train.equals(split(data.records, seed=0, edges=UCI_FRAME.edges).train)
    assert not found.train.equals(split(data.records, seed=1, edges=UCI_FRAME.edges).train)
    assert sorted(found.ontologies) == ["HP", "MONDO"]
    assert {o.name for o in found.ontologies.values()} == {"hpo", "mondo"}


def test_frames_refuses_a_prefix_with_no_pinned_release_naming_the_frame(tmp_path: Path):
    pins, _ = ladder_releases()
    context = uci_context({"HP": pins["HP"]})
    with pytest.raises(
        EvaluationError, match=r"^the uci frame's rows name no pinned release for the prefix MONDO$"
    ):
        frames((Dataset.UCI,), tmp_path, context)


def test_frames_loads_every_release_once_and_gives_each_frame_its_own(tmp_path, monkeypatch):
    def data(rows: tuple[FieldMapping, ...]) -> Data:
        return Data(clean().train, UCI_FRAME, rows, NAME, None)

    given = {
        Dataset.UCI: data((ANGINA,)),
        Dataset.SYNTHEA: data((STENOSIS,)),
        Dataset.DIABETES: data((ANGINA, STENOSIS)),
    }
    calls: list[list[str]] = []
    released = {
        "HP": ontologies()["HP"],
        "MONDO": dataclasses.replace(ontologies()["HP"], name="mondo"),
    }

    def load_releases(sources, data_dir, *, open_url) -> dict[str, Ontology]:
        calls.append(sorted(sources))
        return {prefix: released[prefix] for prefix in sources}

    monkeypatch.setattr(census_module, "load", lambda dataset, *_: given[dataset])
    monkeypatch.setattr(census_module, "load_releases", load_releases)
    pins, _ = ladder_releases()
    found = frames(tuple(Dataset), tmp_path, Context(releases=pins))
    assert calls == [["HP", "MONDO"]]
    assert [dict(each.ontologies) for each in found] == [
        {"HP": released["HP"]},
        {"MONDO": released["MONDO"]},
        released,
    ]
