"""`bko.evaluate.likelihood`: G7's free gate over a run's candidates file, with no API call."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re

import pandas as pd
import pytest

from bko.datasets.cells import cell_keys
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.data import Data
from bko.evaluate.errors import EvaluationError
from bko.evaluate.holdout_candidates import read
from bko.evaluate.likelihood import CSV_COLUMNS, _scored, likelihood
from bko.ontology.mapping import FIELD_MAPPINGS
from tests._evaluate_fixtures import GRADED, SMALL


DATA = Data(GRADED, UCI_FRAME, FIELD_MAPPINGS, "uci", None)
"""The graded records with UCI's field rows, so their concepts score."""


def header() -> dict[str, object]:
    return {
        "data": DATA.digest,
        "frame": "uci",
        "kind": "header",
        "releases": {},
        "run": "e1",
        "schema": 2,
    }


def candidates(*, same: bool = False) -> bytes:
    """Per holdout of `SMALL`, each training cell's own rows as `bko`'s replica records.

    With `same`, every record of a cell is its first, so the cell's halves are identical.
    """
    lines = [header()]
    for seed in SMALL.holdout_seeds:
        train = split(GRADED, seed=seed, edges=UCI_FRAME.edges).train
        keys = cell_keys(train, edges=UCI_FRAME.edges)
        groups = train.groupby([keys[c] for c in keys.columns])
        for call, (cell, members) in enumerate(groups, start=1):
            rows = members.loc[:, list(UCI_FRAME.columns)].to_dict("records")
            if same:
                rows = [rows[0]] * len(rows)
            for position, row in enumerate(rows):
                lines.append(
                    {
                        "kind": "candidate",
                        "holdout": seed,
                        "arm": "bko",
                        "fill": "replica",
                        "attempt": 1,
                        "returned": True,
                        "call": call,
                        "cell": "/".join(map(str, cell)),
                        "position": position,
                        "candidate": {
                            k: (float(v) if k in UCI_FRAME.fractional else int(v))
                            for k, v in row.items()
                        },
                        "outcome": "kept" if position % 3 else "dropped-quota",
                        "failed": [],
                    }
                )
    return "".join(json.dumps(line) + "\n" for line in lines).encode()


def run(data: bytes, caplog) -> tuple[bool, str]:
    caplog.set_level(logging.INFO, logger="bko.evaluate.likelihood")
    sink = io.StringIO()
    separated = likelihood(SMALL, DATA, read(data), sink)
    return separated, sink.getvalue()


def test_the_command_writes_a_row_per_holdout_and_cell_and_logs_the_gate(caplog):
    separated, text = run(candidates(), caplog)
    table = pd.read_csv(io.StringIO(text))
    assert tuple(table.columns) == CSV_COLUMNS
    assert set(table["holdout"]) == set(SMALL.holdout_seeds)
    assert (table["top"] == table["records"] // 2).all()
    assert (table["minimum"] <= table["median"]).all()
    assert (table["median"] <= table["maximum"]).all()
    measures = [m for m in caplog.messages if m.startswith("likelihood measure=")]
    assert [m.split()[1] for m in measures] == [
        "measure=quality",
        "measure=column-shapes",
        "measure=column-pair-trends",
    ]
    assert all(" arm=bko " in m and m.endswith(" holdouts=2") for m in measures)
    assert f"gate separated={separated}" in caplog.messages
    splits = [m for m in caplog.messages if m.startswith("split seed=")]
    assert [m.split()[1] for m in splits] == ["seed=0", "seed=1"]
    concepts = [m for m in caplog.messages if m.startswith("likelihood concept=")]
    assert {m.split()[1] for m in concepts} == {f"concept={r.concept}" for r in FIELD_MAPPINGS}


def test_the_same_input_twice_gives_the_same_csv(caplog):
    first = run(candidates(), caplog)[1]
    second = run(candidates(), caplog)[1]
    assert hashlib.sha256(first.encode()).hexdigest() == hashlib.sha256(second.encode()).hexdigest()


def test_identical_halves_contrast_exactly_zero(caplog):
    separated, _ = run(candidates(same=True), caplog)
    measures = [m for m in caplog.messages if m.startswith("likelihood measure=")]
    assert all(re.search(r" mean=0\.000000 low=0\.000000 high=0\.000000 ", m) for m in measures)
    assert not separated


def test_ties_at_the_median_take_the_sha256_order():
    found = read(candidates(same=True))
    lines = [c for c in found.candidates if c.holdout == 0]
    train = split(GRADED, seed=0, edges=UCI_FRAME.edges).train
    by_cell = _scored(0, lines, train, DATA)
    for scored in by_cell.values():
        assert len({s.score for s in scored}) == 1
        ordered = sorted(scored, key=lambda s: (-s.score, s.tie))
        assert [s.tie for s in ordered] == sorted(s.tie for s in scored)
    line = lines[0]
    assert any(
        s.tie == hashlib.sha256(f"0:{line.call}:{line.position}".encode()).hexdigest()
        for scored in by_cell.values()
        for s in scored
    )


def test_a_holdout_with_no_scored_record_is_refused(caplog):
    data = candidates()
    kept = [line for line in data.decode().splitlines() if '"holdout": 1' not in line]
    with pytest.raises(EvaluationError, match="holdout 1 holds no bko replica record"):
        run(("\n".join(kept) + "\n").encode(), caplog)


def test_a_schema_1_file_scores_alike(caplog):
    schema_2 = candidates()
    head, *rest = schema_2.decode().splitlines()
    from bko.datasets.sources import UCI_HEART_DISEASE

    old = {k: v for k, v in json.loads(head).items() if k not in {"frame", "data"}}
    old |= {"schema": 1, "uci": UCI_HEART_DISEASE.sha256}
    schema_1 = ("\n".join([json.dumps(old), *rest]) + "\n").encode()
    assert run(schema_1, caplog)[1] == run(schema_2, caplog)[1]
