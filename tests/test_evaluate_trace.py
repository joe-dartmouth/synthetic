"""`python -m bko.evaluate trace`: each record's stored or re-derivable facts, with no API call."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import re
import shutil
import zipfile
from collections.abc import Hashable
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pytest

from bko.baselines.request import Request
from bko.bkb.fusion import source_component
from bko.datasets.split import split
from bko.datasets.uci import MEMBER, UCI_FRAME, complete_records, parse_cleveland
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.experiment import DIRECT_ARM, Experiment, gap_request, replica_request
from bko.evaluate.holdout_candidates import KEPT, Candidate, read
from bko.evaluate.prepare import prepared
from bko.evaluate.shared import rebuilt
from bko.gaps.observe import by_concept, cell_key, observe_records, record_cells
from bko.gaps.shares import GROUP_WEIGHT, shares
from bko.generate.messages import AnySend
from bko.generate.prompt import sections
from bko.ontology.fragments import INDIVIDUAL
from tests._evaluate_bko_fixtures import NARROW
from tests._gaps_fixtures import ANGINA, STENOSIS
from tests._generate_fixtures import ScriptedSend, reply
from tests._validate_fixtures import planted
from tests.test_evaluate_direct import SMALL_G2, VARIED, g2_context
from tests.test_evaluate_shared import FORMAT


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

MIXED: Final = dataclasses.replace(
    NARROW, arms=("none", "bko", DIRECT_ARM), comparators=("none",), replica_arms=()
)
"""Two holdouts of `VARIED`: one baseline, the Messages arm `bko` and the direct arm."""

ROWS: Final = (ANGINA, STENOSIS)

LABELS: Final = {"HP:9000003": "Angina pectoris", "MONDO:9000004": "coronary stenosis"}
"""Each mapped concept's name in the ladder releases."""


def distinct() -> ScriptedSend:
    """Both holdouts' `bko` replies, every record of a call distinct and inside training ranges."""

    def records(target: int, count: int) -> list[object]:
        return [planted(age=41 + i % 5, sex=0, target=target, chol=250 + i) for i in range(count)]

    holdout = (reply(records(0, 12)), reply(records(1, 16)))
    return ScriptedSend(*holdout, *holdout)


def never() -> AnySend:
    """A send no trace may build."""
    msg = "trace reached the API"
    raise AssertionError(msg)


class _Kept(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _scored(folder: Path, ctx: Context, argv: list[str]) -> Path:
    """Score `argv` into `folder` as a run of record: `scores.csv` and `scores.log` beside it."""
    folder.mkdir(parents=True)
    kept, root = _Kept(), logging.getLogger()
    level = root.level
    root.addHandler(kept)
    root.setLevel(logging.INFO)
    try:
        out = io.StringIO()
        assert main(argv, context=ctx, out=out) == 0
    finally:
        root.removeHandler(kept)
        root.setLevel(level)
    (folder / "scores.csv").write_text(out.getvalue(), encoding="utf-8")
    log = "".join(FORMAT.format(record) + "\n" for record in kept.records)
    (folder / "scores.log").write_text(log, encoding="utf-8")
    return folder


def context(send: AnySend | None = None) -> Context:
    """`VARIED`'s world under `MIXED`, `g2` registered; with no `send`, any call fails."""
    ctx, _ = g2_context()
    return dataclasses.replace(ctx, experiment=MIXED, send=never if send is None else lambda: send)


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    """A data root and two finished folders: `mixed`, scored over a call log, and `direct`."""
    base = tmp_path_factory.mktemp("trace")
    data = ["--data-dir", str(base / "data")]
    send = distinct()
    mixed = base / "mixed"
    argv = ["score", "--log", str(mixed / "calls.jsonl")]
    argv += ["--candidates", str(mixed / "holdout-candidates.jsonl"), *data]
    _scored(mixed, context(send), argv)
    send.assert_consumed()
    direct = base / "direct"
    argv = ["score", "g2", "--candidates", str(direct / "holdout-candidates.jsonl"), *data]
    _scored(direct, context(), argv)
    return base


def trace(root: Path, folder: str, *args: str, run: str = "e1") -> tuple[int, list[dict]]:
    """Run `trace` on `root / folder` with a send that fails; its exit code and its lines."""
    out = io.StringIO()
    argv = ["trace", run, "--folder", str(root / folder), "--data-dir", str(root / "data"), *args]
    code = main(argv, context=context(), out=out)
    return code, [json.loads(line) for line in out.getvalue().splitlines()]


def train_of(experiment: Experiment, holdout: int) -> pd.DataFrame:
    """`VARIED`'s training rows on `holdout`, as the harness prepares them."""
    with zipfile.ZipFile(io.BytesIO(VARIED)) as bundle:
        records = complete_records(parse_cleveland(bundle.read(MEMBER).decode("ascii")))
    parts = split(records, seed=holdout, edges=UCI_FRAME.edges)
    return prepared(parts, experiment, edges=UCI_FRAME.edges).ablated.train


def lines_of(root: Path, folder: str, holdout: int, arm: str, fill: str) -> list[Candidate]:
    found = read((root / folder / "holdout-candidates.jsonl").read_bytes())
    return [c for c in found.candidates if (c.holdout, c.arm, c.fill) == (holdout, arm, fill)]


def kept_in_order(lines: list[Candidate], request: Request) -> list[dict[Hashable, Any]]:
    """The fill's rows as `shared.rebuilt` orders them, as dicts."""
    return rebuilt(lines, request, where="test").to_dict("records")


def test_a_direct_trace_holds_each_records_concepts_shares_and_sources(root):
    code, found = trace(root, "mixed", "--holdout", "0", "--arm", DIRECT_ARM)
    assert code == 0
    train = train_of(MIXED, 0)
    cells = record_cells(train, ROWS, name="t", edges=UCI_FRAME.edges)
    concepts = by_concept(ROWS)
    lines = lines_of(root, "mixed", 0, DIRECT_ARM, "gap")
    assert len(found) == sum(line.outcome == KEPT for line in lines) > 0
    for index, record in enumerate(found):
        assert (record["holdout"], record["arm"], record["fill"]) == (0, DIRECT_ARM, "gap")
        assert record["record_id"] == index
        assert record["call"] is None
        assert record["untraced"] is None
        values = record["record"]
        shown = {c for c, rows in concepts.items() if any(row.applies(values) for row in rows)}
        assert {c["concept"] for c in record["concepts"] if c["present"]} == shown
        cell = next(c for c in cells.each if cell_key(c) == record["cell"])
        members = train.loc[train["target"].eq(cell[2])]
        expected = shares(cells.each[cell], observe_records(members, ROWS, name="t"))
        assert {c["concept"]: c["share"] for c in record["concepts"]} == expected
        for concept in record["concepts"]:
            assert concept["group_weight"] == GROUP_WEIGHT
            assert concept["cell_records"] == cells.each[cell].observed.records
            assert concept["label"] == LABELS[concept["concept"]]
            for snode in concept["snodes"]:
                assert snode["head"] == [concept["concept"], INDIVIDUAL]
                assert snode["tail"][0][0] == source_component(concept["concept"])
        assert set(record["sources"]) == set(UCI_FRAME.columns) - {"sex", "target"}
        for field, source in record["sources"].items():
            assert train.loc[source, field] == values[field]
        assert set(record["fallback"]) <= set(record["sources"])
        assert set(record["closed"]) >= {c["concept"] for c in record["concepts"] if c["present"]}
        assert record["request"]["cell"] == record["cell"]
        assert record["request"]["seed"] == 0


def test_a_direct_traces_outcome_and_order_are_the_candidates_lines(root):
    _, found = trace(root, "mixed", "--holdout", "1", "--arm", DIRECT_ARM)
    train = train_of(MIXED, 1)
    lines = lines_of(root, "mixed", 1, DIRECT_ARM, "gap")
    request = gap_request(train, MIXED, seed=1, frame=UCI_FRAME)
    expected = list(kept_in_order(lines, request))
    assert len({json.dumps(r, sort_keys=True) for r in expected}) > 1, "the rows are distinct"
    assert [r["record"] for r in found] == expected
    assert {(r["outcome"], tuple(r["failed"])) for r in found} == {(KEPT, ())}


def test_a_messages_trace_names_the_call_and_its_prompts_gap_statements(root):
    code, found = trace(root, "mixed", "--holdout", "0", "--arm", "bko")
    assert code == 0
    calls = {
        line["call"]: line
        for line in map(json.loads, (root / "mixed" / "calls.jsonl").read_text().splitlines())
        if line["kind"] == "request"
    }
    lines = lines_of(root, "mixed", 0, "bko", "gap")
    request = gap_request(train_of(MIXED, 0), MIXED, seed=0, frame=UCI_FRAME)
    expected = list(kept_in_order(lines, request))
    assert len({json.dumps(r, sort_keys=True) for r in expected}) == len(expected) > 1
    assert [r["record"] for r in found] == expected
    for record in found:
        user = calls[record["call"]]["params"]["messages"][0]["content"]
        assert record["cell"] == calls[record["call"]]["cell"]
        read_back = sections(user)
        assert record["request"]["sections"]["findings"] == list(read_back.findings)
        assert record["request"]["sections"]["opening"] == read_back.opening
        assert record["request"]["guidance"] == "summary"
        assert (record["outcome"], record["untraced"]) == (KEPT, None)
        assert record["sources"] == {}
        assert record["concepts"] == []


def test_a_baseline_trace_names_its_method_and_seed_only(root):
    code, found = trace(root, "mixed", "--holdout", "1", "--arm", "none")
    assert code == 0
    assert len(found) == 1
    (line,) = found
    assert line["request"] == {"method": "none", "seed": 1}
    assert (line["record"], line["record_id"], line["call"], line["sources"]) == (
        None,
        None,
        None,
        {},
    )


def test_a_folder_without_a_call_log_reports_untraced(root, tmp_path, caplog):
    shutil.copytree(root / "mixed", tmp_path / "copy")
    (tmp_path / "copy" / "calls.jsonl").unlink()
    caplog.set_level(logging.INFO)
    out = io.StringIO()
    argv = ["trace", "--folder", str(tmp_path / "copy"), "--data-dir", str(root / "data")]
    code = main([*argv, "--holdout", "0", "--arm", "bko"], context=context(), out=out)
    assert code == 0
    found = [json.loads(line) for line in out.getvalue().splitlines()]
    assert found
    assert {r["untraced"] for r in found} == {"no calls.jsonl"}
    assert all(r["request"] is None and isinstance(r["call"], int) for r in found)
    warned = [r for r in caplog.records if r.getMessage().startswith("trace partial ")]
    assert [r.levelno for r in warned] == [logging.WARNING]
    assert warned[0].getMessage() == "trace partial holdout=0 arm=bko missing=calls.jsonl"


def test_a_direct_only_folder_traces_its_replica_fill(root, caplog):
    caplog.set_level(logging.INFO)
    code, found = trace(
        root, "direct", "--holdout", "1", "--arm", f"{DIRECT_ARM}-replica", run="g2"
    )
    assert code == 0
    train = train_of(SMALL_G2, 1)
    request = replica_request(train, SMALL_G2, seed=1, frame=UCI_FRAME)
    lines = lines_of(root, "direct", 1, DIRECT_ARM, "replica")
    assert [r["record"] for r in found] == list(kept_in_order(lines, request))
    assert {r["fill"] for r in found} == {"replica"}
    matched = [m for m in caplog.messages if m.startswith("trace arm ")]
    assert len(matched) == 1
    assert re.fullmatch(
        rf"trace arm holdout=1 arm={DIRECT_ARM}-replica arm_sha256=(\w+) rebuilt_sha256=\1"
        r" matches=true",
        matched[0],
    )
    assert any(m.startswith(f"trace rows={len(found)} sha256=") for m in caplog.messages)
    assert any(
        m == f"trace folder={root / 'direct'} run=g2 holdout=1 arm={DIRECT_ARM}-replica"
        f" fill=replica records={len(found)}"
        for m in caplog.messages
    )


def test_one_record_is_traced_alone(root):
    _, every = trace(root, "mixed", "--holdout", "0", "--arm", DIRECT_ARM)
    code, found = trace(root, "mixed", "--holdout", "0", "--arm", DIRECT_ARM, "--record", "2")
    assert code == 0
    assert found == [every[2]]


def test_an_edited_arm_sha_is_refused(root, tmp_path, caplog):
    shutil.copytree(root / "mixed", tmp_path / "copy")
    log = tmp_path / "copy" / "scores.log"
    text = log.read_text(encoding="utf-8")
    pattern = rf"(arm holdout=0 arm={DIRECT_ARM} attempt=\d+ seed=\d+ added=\d+ sha256=)\w"
    edited, count = re.subn(pattern, r"\g<1>0", text)
    assert count == 1
    log.write_text(edited, encoding="utf-8")
    out = io.StringIO()
    argv = ["trace", "--folder", str(tmp_path / "copy"), "--data-dir", str(root / "data")]
    code = main([*argv, "--holdout", "0", "--arm", DIRECT_ARM], context=context(), out=out)
    assert (code, out.getvalue()) == (3, "")
    assert "re-drawn" in caplog.text


@pytest.mark.parametrize(
    "selection",
    [
        ("--holdout", "0", "--arm", "smote"),
        ("--holdout", "7", "--arm", DIRECT_ARM),
        ("--holdout", "0", "--arm", DIRECT_ARM, "--record", "999"),
        ("--holdout", "0", "--arm", DIRECT_ARM, "--record", "-1"),
    ],
    ids=["arm", "holdout", "record", "negative-record"],
)
def test_a_bad_selection_exits_2(root, selection):
    assert trace(root, "mixed", *selection) == (2, [])


def test_trace_makes_no_api_call_and_fetches_nothing(root):
    ctx = context()
    opener = ctx.open_url
    for arm in ("bko", DIRECT_ARM, "none"):
        out = io.StringIO()
        argv = ["trace", "--folder", str(root / "mixed"), "--data-dir", str(root / "data")]
        assert main([*argv, "--holdout", "0", "--arm", arm], context=ctx, out=out) == 0
    assert opener.calls == []  # ty: ignore[unresolved-attribute]
