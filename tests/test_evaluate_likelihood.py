"""`bko.evaluate.likelihood`: G7's free gate over a run's candidates file, with no API call."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import math
import re

import pandas as pd
import pytest

import bko.evaluate.likelihood as likelihood_module
import bko.evaluate.ranking as ranking_module
from bko.datasets.cells import cell_keys
from bko.datasets.sources import UCI_HEART_DISEASE
from bko.datasets.split import split
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.__main__ import Context, main
from bko.evaluate.data import Data, Dataset, load
from bko.evaluate.errors import EvaluationError
from bko.evaluate.holdout_candidates import read
from bko.evaluate.likelihood import CSV_COLUMNS, likelihood
from bko.evaluate.named_runs import NamedRun
from bko.ontology.mapping import FIELD_MAPPINGS
from tests._baselines_fixtures import cleveland_text
from tests._datasets_fixtures import FakeOpener, archive, pin
from tests._diabetes_fixtures import pin as diabetes_pin
from tests._diabetes_fixtures import toy_archive
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


def candidate(seed: int, call: int, cell: str, position: int, row, **fields) -> dict[str, object]:
    """One `bko` replica candidate line holding `row`."""
    values = {k: (float(v) if k in UCI_FRAME.fractional else int(v)) for k, v in row.items()}
    return {
        "kind": "candidate",
        "holdout": seed,
        "arm": "bko",
        "fill": "replica",
        "attempt": 1,
        "returned": True,
        "call": call,
        "cell": cell,
        "position": position,
        "candidate": values,
        "outcome": "kept" if position % 3 else "dropped-quota",
        "failed": [],
    } | fields


def cells(seed: int, records=GRADED):
    """Each training cell of holdout `seed`: its key and its rows, in order."""
    train = split(records, seed=seed, edges=UCI_FRAME.edges).train
    keys = cell_keys(train, edges=UCI_FRAME.edges)
    for cell, members in train.groupby([keys[c] for c in keys.columns]):
        yield "/".join(map(str, cell)), members.loc[:, list(UCI_FRAME.columns)].to_dict("records")


def encode(lines, head=None) -> bytes:
    return "".join(json.dumps(line) + "\n" for line in [head or header(), *lines]).encode()


def candidates(*, same: bool = False, seeds=SMALL.holdout_seeds, records=GRADED, bad=False):
    """Per holdout, each training cell's own rows as `bko`'s replica records.

    With `same`, every record of a cell is its first, so the cell's halves are identical. With
    `bad`, each cell also holds its rows again with `age` moved past any real one and every other
    column reversed, so they share neither the training shapes nor its pairs.
    """
    lines = []
    for seed in seeds:
        for call, (cell, rows) in enumerate(cells(seed, records), start=1):
            kept = [rows[0]] * len(rows) if same else rows
            flipped = [
                {k: (r["age"] + 500 if k == "age" else rows[-1 - i][k]) for k in r}
                | {"sex": r["sex"], "target": r["target"]}
                for i, r in enumerate(rows)
            ]
            written = kept + flipped if bad else kept
            lines += [candidate(seed, call, cell, p, row) for p, row in enumerate(written)]
    return encode(lines)


def run(data: bytes, caplog) -> tuple[bool, str]:
    caplog.set_level(logging.INFO, logger="bko.evaluate.likelihood")
    sink = io.StringIO()
    separated = likelihood(SMALL, DATA, [read(data)], sink)
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


def test_each_holdouts_cells_share_one_stratum_pass(caplog, monkeypatch):
    """Every cell of a holdout is scored through one `Scorers`, so its keys are cut once."""
    keyed: list[int] = []
    scored: list[object] = []
    keys = ranking_module.stratum_keys
    build = ranking_module.Scorers.__call__

    def keys_spy(*args, **kwargs):
        keyed.append(len(args[0]))
        return keys(*args, **kwargs)

    def build_spy(self, cell):
        scored.append(cell)
        return build(self, cell)

    monkeypatch.setattr(ranking_module, "stratum_keys", keys_spy)
    monkeypatch.setattr(ranking_module.Scorers, "__call__", build_spy)
    run(candidates(), caplog)
    assert len(keyed) == len(SMALL.holdout_seeds)
    assert len(scored) > 2 * len(keyed)


def test_a_holdout_with_no_scored_record_is_refused(caplog):
    data = candidates()
    kept = [line for line in data.decode().splitlines() if '"holdout": 1' not in line]
    row = GRADED.iloc[0][list(UCI_FRAME.columns)].to_dict()
    kept.append(json.dumps(candidate(1, 999, "nowhere", 0, row, fill="gap")))
    with pytest.raises(EvaluationError, match="holdout 1 holds no bko replica record"):
        run(("\n".join(kept) + "\n").encode(), caplog)


def test_a_schema_1_file_scores_alike(caplog):
    schema_2 = candidates()
    head, *rest = schema_2.decode().splitlines()
    old = {k: v for k, v in json.loads(head).items() if k not in {"frame", "data"}}
    old |= {"schema": 1, "uci": UCI_HEART_DISEASE.sha256}
    schema_1 = ("\n".join([json.dumps(old), *rest]) + "\n").encode()
    assert run(schema_1, caplog)[1] == run(schema_2, caplog)[1]


class Ranked:
    """A stand-in score: 0 for a record with a real `age`, -1 for one moved past every age."""

    def __init__(self, shares, rows, frame) -> None:
        pass

    def __call__(self, record) -> float:
        return 0.0 if record["age"] < 200 else -1.0


class Tied:
    """A stand-in score: every record ties, so the sha256 order alone splits each cell."""

    def __init__(self, shares, rows, frame) -> None:
        pass

    def __call__(self, record) -> float:
        return 0.0


def bounds(caplog, side: str) -> list[float]:
    """Each `likelihood measure=` line's `low=` or `high=` value."""
    return [
        float(found.group(1))
        for m in caplog.messages
        if m.startswith("likelihood measure=") and (found := re.search(rf" {side}=(\S+)", m))
    ]


def test_a_score_that_ranks_real_records_first_separates(caplog, monkeypatch):
    monkeypatch.setattr(ranking_module, "Likelihood", Ranked)
    many = dataclasses.replace(SMALL, holdout_seeds=tuple(range(6)))
    caplog.set_level(logging.INFO, logger="bko.evaluate.likelihood")
    data = candidates(seeds=many.holdout_seeds, bad=True)
    assert likelihood(many, DATA, [read(data)], io.StringIO())
    lows = bounds(caplog, "low")
    assert len(lows) == 3
    assert all(low > 0 for low in lows)
    assert "gate separated=True" in caplog.messages


def test_reversed_ranks_do_not_separate(caplog, monkeypatch):
    class Reversed(Ranked):
        def __call__(self, record) -> float:
            return -super().__call__(record)

    monkeypatch.setattr(ranking_module, "Likelihood", Reversed)
    many = dataclasses.replace(SMALL, holdout_seeds=tuple(range(6)))
    caplog.set_level(logging.INFO, logger="bko.evaluate.likelihood")
    assert not likelihood(
        many, DATA, [read(candidates(seeds=many.holdout_seeds, bad=True))], io.StringIO()
    )
    highs = bounds(caplog, "high")
    assert len(highs) == 3
    assert all(high < 0 for high in highs)


def test_tied_scores_split_in_sha256_order_and_an_odd_cells_middle_record_sits_out(monkeypatch):
    monkeypatch.setattr(ranking_module, "Likelihood", Tied)
    seen: list[pd.DataFrame] = []

    def capture(train, synthetic, *, frame):
        seen.append(synthetic.reset_index(drop=True))
        return dict.fromkeys(likelihood_module.MEASURES, 0.0)

    monkeypatch.setattr(likelihood_module, "fidelity", capture)
    one = dataclasses.replace(SMALL, holdout_seeds=(0, 1))
    lines = []
    expected_top: list[int] = []
    expected_bottom: list[int] = []
    for seed in one.holdout_seeds:
        for call, (cell, rows) in enumerate(cells(seed), start=1):
            picked = rows[:3]
            # Distinct `chol` per record names it in the halves the gate scores.
            marked = [r | {"chol": 1000 * seed + 10 * call + p} for p, r in enumerate(picked)]
            lines += [candidate(seed, call, cell, p, r) for p, r in enumerate(marked)]
            order = sorted(
                range(len(marked)),
                key=lambda p: hashlib.sha256(f"{seed}:{call}:{p}".encode()).hexdigest(),
            )
            half = len(marked) // 2
            expected_top += [marked[p]["chol"] for p in order[:half]]
            expected_bottom += [marked[p]["chol"] for p in order[len(order) - half :]]
    likelihood(one, DATA, [read(encode(lines))], io.StringIO())
    tops = [chol for frame in seen[0::2] for chol in frame["chol"]]
    bottoms = [chol for frame in seen[1::2] for chol in frame["chol"]]
    assert sorted(tops) == sorted(expected_top)
    assert sorted(bottoms) == sorted(expected_bottom)
    assert set(tops).isdisjoint(bottoms)


def test_an_undrawable_record_reads_minus_infinity_in_its_cells_row(monkeypatch):
    class Undrawable(Tied):
        def __call__(self, record) -> float:
            return -math.inf if record["chol"] == 999 else 0.0

    monkeypatch.setattr(ranking_module, "Likelihood", Undrawable)
    lines = []
    for seed in SMALL.holdout_seeds:
        for call, (cell, rows) in enumerate(cells(seed), start=1):
            rows = [rows[0] | {"chol": 999}, *rows[1:]] if len(rows) > 3 else rows
            lines += [candidate(seed, call, cell, p, r) for p, r in enumerate(rows)]
    sink = io.StringIO()
    likelihood(SMALL, DATA, [read(encode(lines))], sink)
    table = pd.read_csv(io.StringIO(sink.getvalue()), dtype={"minimum": str})
    marked = table[table["records"] > 3]
    assert (marked["minimum"] == "-inf").all()
    assert (marked["at_minimum"] == (1 / marked["records"]).round(6)).all()
    assert (table[table["records"] <= 3]["minimum"] != "-inf").all()


def test_other_arms_fills_and_outcomes_stay_out_of_the_pool(caplog):
    base = candidates()
    noise = [
        candidate(0, 999, "nowhere", 0, GRADED.iloc[0][list(UCI_FRAME.columns)].to_dict(), **f)
        for f in (
            {"arm": "cllm"},
            {"fill": "gap"},
            {"outcome": "rejected", "failed": ["fields"]},
            {"outcome": "dropped-range"},
        )
    ]
    head, *rest = base.decode().splitlines()
    mixed = encode([json.loads(line) for line in rest] + noise, json.loads(head))
    assert run(mixed, caplog)[1] == run(base, caplog)[1]
    pools = [m for m in caplog.messages if m.startswith("likelihood pool ")]
    assert pools[0].endswith(f"lines={len(rest) + 4} scored={len(rest)} outside=0")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"cell": "nowhere"}, "no training cell 'nowhere'"),
        ({"candidate": None}, "has no record"),
        ({"drop": "chol"}, "has no record"),
        ({"chol": None}, "not a finite number"),
        ({"chol": "high"}, "not a finite number"),
    ],
    ids=["unknown-cell", "no-record", "missing-column", "null-value", "text-value"],
)
def test_a_line_that_cannot_be_scored_is_refused(caplog, change, message):
    head, first, *rest = candidates().decode().splitlines()
    line = json.loads(first)
    if "cell" in change:
        line["cell"] = change["cell"]
    elif "candidate" in change:
        line["candidate"] = None
    elif "drop" in change:
        del line["candidate"][change["drop"]]
    else:
        line["candidate"]["chol"] = change["chol"]
    with pytest.raises(EvaluationError, match=re.escape(message)):
        run(encode([line, *map(json.loads, rest)], json.loads(head)), caplog)


def test_a_holdout_of_single_record_cells_is_refused(caplog):
    lines = [
        candidate(seed, call, cell, 0, rows[0])
        for seed in SMALL.holdout_seeds
        for call, (cell, rows) in enumerate(cells(seed), start=1)
    ]
    with pytest.raises(EvaluationError, match="no cell with two bko replica records"):
        run(encode(lines), caplog)


def cli(tmp_path, body: bytes, run_name: str = "e1") -> tuple[int, str]:
    path = tmp_path / "holdout-candidates.jsonl"
    path.write_bytes(body)
    data = archive(cleveland_text(GRADED))
    context = Context(open_url=FakeOpener(data), uci=pin(data), experiment=SMALL)
    out = io.StringIO()
    argv = ["likelihood", run_name, "--candidates", str(path), "--data-dir", str(tmp_path)]
    return main(argv, context=context, out=out), out.getvalue()


def cli_records(tmp_path):
    data = archive(cleveland_text(GRADED))
    context = Context(open_url=FakeOpener(data), uci=pin(data), experiment=SMALL)
    return load(Dataset.UCI, tmp_path / "data", context)


def test_the_cli_scores_a_file_of_the_runs_data_and_logs_its_provenance(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    loaded = cli_records(tmp_path)
    lines = [
        candidate(seed, call, cell, p, row)
        for seed in SMALL.holdout_seeds
        for call, (cell, rows) in enumerate(cells(seed, loaded.records), start=1)
        for p, row in enumerate(rows)
    ]
    body = encode(lines, header() | {"data": loaded.digest})
    code, text = cli(tmp_path, body)
    assert code == 0
    assert text.startswith(",".join(CSV_COLUMNS) + "\n")
    assert any(m.startswith("experiment run=e1 ") for m in caplog.messages)
    assert any(m.startswith("versions commit=") for m in caplog.messages)
    digest = hashlib.sha256(body).hexdigest()
    assert any(m.endswith(f"schema=2 sha256={digest}") for m in caplog.messages)


@pytest.mark.parametrize(
    ("head", "run_name", "message"),
    [
        ({"data": "0" * 64}, "e1", "data digest '" + "0" * 64 + "' is not the run's"),
        ({"run": "g2"}, "e1", "is run 'g2''s, not 'e1''s"),
    ],
    ids=["other-data", "other-run"],
)
def test_the_cli_refuses_a_file_of_other_data_or_another_run(
    tmp_path, caplog, head, run_name, message
):
    code, text = cli(tmp_path, encode([], header() | head), run_name)
    assert code == 2
    assert text == ""
    assert any(message in m for m in caplog.messages)


def test_the_cli_refuses_a_schema_1_file_for_a_run_of_other_data(tmp_path, caplog):
    old = {k: v for k, v in header().items() if k not in {"frame", "data"}}
    old |= {"schema": 1, "uci": UCI_HEART_DISEASE.sha256, "run": "e3"}
    code, _ = cli(tmp_path, encode([], old), "e3")
    assert code == 2
    assert any("schema-1 candidates file holds UCI Heart Disease" in m for m in caplog.messages)


def sliced(data: bytes, *parts: tuple[str, tuple[int, ...]]) -> list[bytes]:
    """`data`'s lines in one file per part: its header under the part's run, its holdouts' lines."""
    head, *rest = (json.loads(line) for line in data.decode().splitlines())
    return [
        encode([line for line in rest if line["holdout"] in held], head | {"run": name})
        for name, held in parts
    ]


def gate(caplog) -> list[str]:
    """The run's `likelihood measure=`, `likelihood concept=` and `gate` lines."""
    return [
        m
        for m in caplog.messages
        if m.startswith(("likelihood measure=", "likelihood concept=", "gate "))
    ]


def test_the_gate_reads_slices_as_one_run(caplog):
    caplog.set_level(logging.INFO, logger="bko.evaluate.likelihood")
    whole = candidates()
    one = io.StringIO()
    separated = likelihood(SMALL, DATA, [read(whole)], one)
    expected = gate(caplog)
    caplog.clear()
    files = sliced(whole, ("e1-b", (1,)), ("e1-a", (0,)))
    two = io.StringIO()
    assert likelihood(SMALL, DATA, [read(f) for f in files], two) is separated
    assert two.getvalue() == one.getvalue()
    assert gate(caplog) == expected
    pools = [m for m in caplog.messages if m.startswith("likelihood pool ")]
    lines = sum(len(f.decode().splitlines()) - 1 for f in files)
    assert pools == [f"likelihood pool arm=bko fill=replica lines={lines} scored={lines} outside=0"]


@pytest.mark.parametrize(
    ("parts", "message"),
    [
        pytest.param((("e1-a", (0, 1)), ("e1-b", (1,))), "holdout 1 is in both", id="overlap"),
        pytest.param((("e1-a", (0,)),), "leave the run's holdouts [1] out", id="missing"),
    ],
)
def test_overlapping_or_missing_slice_holdouts_are_refused(caplog, parts, message):
    files = sliced(candidates(), *parts)
    with pytest.raises(EvaluationError, match=re.escape(message)):
        likelihood(SMALL, DATA, [read(f) for f in files], io.StringIO())


def test_slice_headers_must_agree_but_run(caplog):
    low, high = sliced(candidates(), ("e1-a", (0,)), ("e1-b", (1,)))
    head, *rest = (json.loads(line) for line in high.decode().splitlines())
    other = encode(rest, head | {"releases": {"MONDO": "0" * 64}})
    with pytest.raises(EvaluationError, match="header differs from the first file's in releases"):
        likelihood(SMALL, DATA, [read(low), read(other)], io.StringIO())


def frame_candidates(data: Data, experiment) -> bytes:
    """Per holdout, each training cell's own rows of `data` as `bko`'s replica records."""
    frame = data.frame
    lines = []
    for seed in experiment.holdout_seeds:
        train = split(data.records, seed=seed, edges=frame.edges).train
        keys = cell_keys(train, edges=frame.edges)
        groups = train.groupby([keys[c] for c in keys.columns])
        for call, (cell, members) in enumerate(groups, start=1):
            rows = members.loc[:, list(frame.columns)].to_dict("records")
            for position, row in enumerate(rows):
                values = {
                    k: (float(v) if k in frame.fractional else int(v)) for k, v in row.items()
                }
                key = "/".join(map(str, cell))
                lines.append(candidate(seed, call, key, position, row) | {"candidate": values})
    head = header() | {"data": data.digest, "frame": frame.name, "run": "e6"}
    return encode(lines, head)


def test_the_gate_scores_a_non_uci_frames_candidates(tmp_path, caplog):
    archived = toy_archive()
    source = diabetes_pin(archived, "uci-diabetes-130")
    context = Context(open_url=FakeOpener(archived), diabetes=source)
    data = load(Dataset.DIABETES, tmp_path, context)
    assert data.frame.name != UCI_FRAME.name
    caplog.set_level(logging.INFO, logger="bko.evaluate.likelihood")
    sink = io.StringIO()
    likelihood(SMALL, data, [read(frame_candidates(data, SMALL))], sink)
    table = pd.read_csv(io.StringIO(sink.getvalue()))
    assert set(table["holdout"]) == set(SMALL.holdout_seeds)
    assert (table["top"] == table["records"] // 2).all()
    concepts = [m for m in caplog.messages if m.startswith("likelihood concept=")]
    assert {m.split()[1] for m in concepts} == {f"concept={r.concept}" for r in data.field_rows}
    assert any(m.startswith("gate separated=") for m in caplog.messages)


SLICES = {
    "e1-a": NamedRun(dataclasses.replace(SMALL, spend_ceiling=12.5)),
    "e1-b": NamedRun(SMALL),
}
"""Runs registered as slices of `SMALL`: its configuration but for the spend ceiling, over holdouts
among its own (an `Experiment` holds at least two, so each names both here)."""


def _sliced_paths(tmp_path, names: tuple[str, str]) -> list[str]:
    """`--candidates` flags for slices of the whole run's candidates named `names`."""
    loaded = cli_records(tmp_path)
    lines = [
        candidate(seed, call, cell, p, row)
        for seed in SMALL.holdout_seeds
        for call, (cell, rows) in enumerate(cells(seed, loaded.records), start=1)
        for p, row in enumerate(rows)
    ]
    whole = encode(lines, header() | {"data": loaded.digest})
    paths = []
    for i, body in enumerate(sliced(whole, (names[0], (0,)), (names[1], (1,)))):
        path = tmp_path / f"slice-{i}.jsonl"
        path.write_bytes(body)
        paths += ["--candidates", str(path)]
    return paths


@pytest.mark.parametrize(
    ("runs", "names", "message"),
    [
        (SLICES, ("e1-a", "e9-b"), "run 'e9-b''s, which is not a slice of 'e1'"),
        (
            {**SLICES, "e1-b": NamedRun(dataclasses.replace(SMALL, budget=9))},
            ("e1-a", "e1-b"),
            "run 'e1-b''s, which is not a slice of 'e1'",
        ),
        (
            {**SLICES, "e1-b": NamedRun(dataclasses.replace(SMALL, holdout_seeds=(1, 7)))},
            ("e1-a", "e1-b"),
            "run 'e1-b''s, which is not a slice of 'e1'",
        ),
    ],
    ids=["unknown", "other-budget", "other-holdouts"],
)
def test_the_cli_refuses_a_file_of_a_run_that_is_no_slice_of_the_named_run(
    tmp_path, caplog, runs, names, message
):
    paths = _sliced_paths(tmp_path, names)
    data = archive(cleveland_text(GRADED))
    context = Context(open_url=FakeOpener(data), uci=pin(data), experiment=SMALL, runs=runs)
    out = io.StringIO()
    with caplog.at_level(logging.ERROR):
        code = main(
            ["likelihood", "e1", *paths, "--data-dir", str(tmp_path)], context=context, out=out
        )
    assert (code, out.getvalue()) == (2, "")
    assert message in caplog.text


def test_the_cli_reads_slices_of_other_run_names(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    loaded = cli_records(tmp_path)
    lines = [
        candidate(seed, call, cell, p, row)
        for seed in SMALL.holdout_seeds
        for call, (cell, rows) in enumerate(cells(seed, loaded.records), start=1)
        for p, row in enumerate(rows)
    ]
    whole = encode(lines, header() | {"data": loaded.digest})
    code, expected = cli(tmp_path, whole)
    assert code == 0
    paths = []
    for i, body in enumerate(sliced(whole, ("e1-a", (0,)), ("e1-b", (1,)))):
        path = tmp_path / f"slice-{i}.jsonl"
        path.write_bytes(body)
        paths += ["--candidates", str(path)]
    data = archive(cleveland_text(GRADED))
    context = Context(open_url=FakeOpener(data), uci=pin(data), experiment=SMALL, runs=SLICES)
    out = io.StringIO()
    argv = ["likelihood", "e1", *paths, "--data-dir", str(tmp_path)]
    assert main(argv, context=context, out=out) == 0
    assert out.getvalue() == expected
    found = [m for m in caplog.messages if m.startswith("candidates path=")]
    assert [m.split()[1] for m in found[-2:]] == [
        f"path={tmp_path / f'slice-{i}.jsonl'}" for i in (0, 1)
    ]
