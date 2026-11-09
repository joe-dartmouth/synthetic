"""`python -m bko.evaluate trace`: each record's stored or re-derivable facts, with no API call."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import re
import shutil
import zipfile
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pytest

from bko.baselines.request import Request
from bko.bkb.fusion import source_component
from bko.datasets.cells import cell_keys
from bko.datasets.split import split
from bko.datasets.uci import MEMBER, UCI_FRAME, complete_records, parse_cleveland
from bko.evaluate import direct as direct_arm
from bko.evaluate import redraw as redraw_module
from bko.evaluate.__main__ import main
from bko.evaluate.call_prompts import NO_CALL, NO_RUN, Prompts, asked
from bko.evaluate.context import Context
from bko.evaluate.data import Data, Dataset, load
from bko.evaluate.errors import IdentityError
from bko.evaluate.experiment import DIRECT_ARM, Experiment, gap_request, replica_request
from bko.evaluate.holdout_candidates import KEPT, Candidate, read
from bko.evaluate.prepare import prepared
from bko.evaluate.redraw import redraw
from bko.evaluate.run_folder import read_folder
from bko.evaluate.shared import rebuilt
from bko.evaluate.trace_lines import Trace, ordered, select, trace_messages
from bko.gaps.observe import by_concept, cell_key, dataset_fragment, observe_records, record_cells
from bko.gaps.scope import load_scope
from bko.gaps.shares import GROUP_WEIGHT, shares
from bko.generate.errors import SamplingError
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

SOURCE: Final = "uci-heart-disease"
"""UCI Heart Disease's source name, which each fragment's name starts with."""

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
    assert warned[0].getMessage() == (
        f"trace partial holdout=0 arm=bko untraced={len(found)} of {len(found)}"
        " missing=calls.jsonl reasons=no calls.jsonl"
    )
    done = next(m for m in caplog.messages if m.startswith("trace rows="))
    assert done.endswith(f" arm=bko record=all untraced={len(found)}")


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
        r" kept_lines=true matches=true",
        matched[0],
    )
    started = [m for m in caplog.messages if m.startswith("trace start ")]
    assert started == [
        (
            f"trace start folder={root / 'direct'} run=g2 holdout=1 arm={DIRECT_ARM}-replica"
            " fill=replica seed=1"
        )
    ]
    assert caplog.messages.index(started[0]) < caplog.messages.index(matched[0])
    assert re.fullmatch(
        rf"trace rows={len(found)} sha256=\w+ folder={re.escape(str(root / 'direct'))} holdout=1"
        rf" arm={DIRECT_ARM}-replica record=all untraced=0",
        next(m for m in caplog.messages if m.startswith("trace rows=")),
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


def refusing_first(real: Callable[..., Any]) -> Callable[..., Any]:
    """`real`, refusing the first attempt of each fill so the harness retries at a derived seed."""
    tags: set[object] = set()

    def first_refused(*args: Any, **kwargs: Any) -> Any:
        if kwargs["tag"] not in tags:
            tags.add(kwargs["tag"])
            msg = "the first attempt of each fill refuses"
            raise SamplingError(msg)
        return real(*args, **kwargs)

    return first_refused


@pytest.fixture(scope="module")
def retried(tmp_path_factory) -> Path:
    """A `g2` folder whose every direct fill returned on its second attempt, at a derived seed."""
    base = tmp_path_factory.mktemp("retried")
    folder = base / "retried"
    argv = ["score", "g2", "--candidates", str(folder / "holdout-candidates.jsonl")]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(direct_arm, "sample", refusing_first(direct_arm.sample))
        _scored(folder, context(), [*argv, "--data-dir", str(base / "data")])
    return base


def arm_seed(folder: Path, holdout: int, arm: str) -> tuple[int, int]:
    """The `arm` line's attempt and seed of `arm` on `holdout`."""
    text = (folder / "scores.log").read_text(encoding="utf-8")
    pattern = rf"arm holdout={holdout} arm={re.escape(arm)} attempt=(\d+) seed=(\d+) "
    (found,) = re.findall(pattern, text)
    return int(found[0]), int(found[1])


@pytest.mark.parametrize("arm", [DIRECT_ARM, f"{DIRECT_ARM}-replica"])
def test_a_retried_fill_is_re_drawn_at_its_arm_lines_seed(retried, arm):
    attempt, seed = arm_seed(retried / "retried", 1, arm)
    assert (attempt, seed != 1) == (2, True), "the fill returned on a retry at a derived seed"
    code, found = trace(retried, "retried", "--holdout", "1", "--arm", arm, run="g2")
    assert code == 0
    fill = "gap" if arm == DIRECT_ARM else "replica"
    asked = gap_request if fill == "gap" else replica_request
    request = asked(train_of(SMALL_G2, 1), SMALL_G2, seed=1, frame=UCI_FRAME)
    lines = lines_of(retried, "retried", 1, DIRECT_ARM, fill)
    assert [r["record"] for r in found] == list(kept_in_order(lines, request))
    assert {(r["request"]["seed"], r["request"]["attempt"]) for r in found} == {(seed, attempt)}


def _copied(root: Path, tmp_path: Path, folder: str = "mixed") -> Path:
    shutil.copytree(root / folder, tmp_path / "copy")
    return tmp_path / "copy"


def _run(root: Path, folder: Path, *args: str) -> tuple[int, list[dict]]:
    out = io.StringIO()
    argv = ["trace", "--folder", str(folder), "--data-dir", str(root / "data"), *args]
    code = main(argv, context=context(), out=out)
    return code, [json.loads(line) for line in out.getvalue().splitlines()]


def _edit_jsonl(path: Path, edit: Callable[[dict], dict | None]) -> None:
    """Rewrite each JSON line of `path` through `edit`; a None drops the line."""
    lines = [edit(json.loads(raw)) for raw in path.read_text(encoding="utf-8").splitlines()]
    text = "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines if line is not None)
    path.write_text(text, encoding="utf-8")


def test_an_edited_kept_line_is_refused_though_the_arm_sha_holds(root, tmp_path, caplog):
    folder = _copied(root, tmp_path)
    edited = []

    def nudge(line: dict) -> dict:
        wanted = (line.get("holdout"), line.get("arm"), line.get("outcome")) == (
            0,
            DIRECT_ARM,
            KEPT,
        )
        if wanted and not edited:
            edited.append(line["position"])
            line["candidate"]["chol"] += 1
        return line

    _edit_jsonl(folder / "holdout-candidates.jsonl", nudge)
    assert edited
    assert _run(root, folder, "--holdout", "0", "--arm", DIRECT_ARM) == (3, [])
    assert "re-drawn records are not its kept lines" in caplog.text
    matched = [m for m in caplog.messages if m.startswith("trace arm ")]
    assert len(matched) == 1
    assert matched[0].endswith(" kept_lines=false matches=false")


def test_an_edited_arm_sha_logs_both_shas_before_the_refusal(root, tmp_path, caplog):
    folder = _copied(root, tmp_path)
    log = folder / "scores.log"
    pattern = rf"(arm holdout=0 arm={DIRECT_ARM} attempt=\d+ seed=\d+ added=\d+ sha256=)\w+"
    edited, count = re.subn(pattern, r"\g<1>" + "0" * 64, log.read_text(encoding="utf-8"))
    assert count == 1
    log.write_text(edited, encoding="utf-8")
    assert _run(root, folder, "--holdout", "0", "--arm", DIRECT_ARM) == (3, [])
    (matched,) = [r for r in caplog.records if r.getMessage().startswith("trace arm ")]
    assert matched.levelno == logging.WARNING
    assert re.fullmatch(
        rf"trace arm holdout=0 arm={DIRECT_ARM} arm_sha256={'0' * 64} rebuilt_sha256=(?!0{{64}})\w+"
        r" kept_lines=true matches=false",
        matched.getMessage(),
    )


def test_a_split_line_the_data_does_not_reproduce_is_refused(root, tmp_path, caplog):
    folder = _copied(root, tmp_path)
    log = folder / "scores.log"
    edited, count = re.subn(
        r"(split seed=0 train=\d+ test=\d+ fingerprint=)\w", r"\g<1>z", log.read_text()
    )
    assert count == 1
    log.write_text(edited, encoding="utf-8")
    for arm in (DIRECT_ARM, "bko", "none"):
        assert _run(root, folder, "--holdout", "0", "--arm", arm) == (3, [])
    assert "holdout 0's split is not" in caplog.text


def test_a_direct_record_names_its_request_fallback_and_s_nodes(root):
    _, found = trace(root, "mixed", "--holdout", "0", "--arm", DIRECT_ARM)
    train = train_of(MIXED, 0)
    request = gap_request(train, MIXED, seed=0, frame=UCI_FRAME)
    attempt, seed = arm_seed(root / "mixed", 0, DIRECT_ARM)
    cells = record_cells(train, ROWS, name=SOURCE, edges=UCI_FRAME.edges)
    keys = cell_keys(train, edges=UCI_FRAME.edges)
    assert any(record["fallback"] for record in found), "the fixture falls back"
    for record in found:
        cell = next(c for c in request.quota if cell_key(c) == record["cell"])
        assert record["request"] == {
            "attempt": attempt,
            "cell": record["cell"],
            "quota": request.quota[cell],
            "rows": request.rows,
            "seed": seed,
        }
        outside = {f for f, source in record["sources"].items() if tuple(keys.loc[source]) != cell}
        assert record["fallback"] == sorted(outside)
        members = train.loc[train["target"].eq(cell[2])]
        group = observe_records(members, ROWS, name=f"{SOURCE}/target={cell[2]}")
        observed = cells.each[cell].observed
        names = [c["concept"] for c in record["concepts"]]
        assert names == sorted(names) == sorted(group.holders)
        for concept in record["concepts"]:
            name = concept["concept"]
            assert (concept["cell_holders"], concept["group_holders"]) == (
                observed.holders.get(name, 0),
                group.holders[name],
            )
            assert concept["group_records"] == group.records
            expected = [
                {
                    "head": [name, INDIVIDUAL],
                    "tail": [[source_component(name), fragment.name]],
                    "prob": snode.prob,
                }
                for fragment in (dataset_fragment(observed), dataset_fragment(group))
                for snode in fragment.snodes
                if snode.head[0] == name
            ]
            assert concept["snodes"] == expected
            assert len(expected) == (name in observed.held) + (name in group.held)


def test_a_messages_record_names_its_attempt_and_guidance(root):
    _, found = trace(root, "mixed", "--holdout", "1", "--arm", "bko")
    assert found
    for record in found:
        assert set(record["request"]) == {"attempt", "cell", "guidance", "sections"}
        assert (record["request"]["attempt"], record["request"]["guidance"]) == (1, "summary")


def _holdout_calls(root: Path, holdout: int) -> list[int]:
    lines = lines_of(root, "mixed", holdout, "bko", "gap")
    return sorted({line.call for line in lines if line.outcome == KEPT and line.call is not None})


def test_a_call_the_log_lacks_leaves_its_records_untraced(root, tmp_path, caplog):
    folder = _copied(root, tmp_path)
    first, second = _holdout_calls(root, 0)
    _edit_jsonl(
        folder / "calls.jsonl",
        lambda line: None if (line["kind"], line.get("call")) == ("request", first) else line,
    )
    code, found = _run(root, folder, "--holdout", "0", "--arm", "bko")
    assert code == 0
    gone = [r for r in found if r["call"] == first]
    assert gone
    assert {(r["request"] is None, r["untraced"]) for r in gone} == {
        (True, f"call {first} is not in calls.jsonl")
    }
    assert {r["untraced"] for r in found if r["call"] == second} == {None}
    warned = [m for m in caplog.messages if m.startswith("trace partial ")]
    assert warned == [
        (
            f"trace partial holdout=0 arm=bko untraced={len(gone)} of {len(found)}"
            f" missing={first} reasons=call {first} is not in calls.jsonl"
        )
    ]


def test_block_content_is_read_and_a_prompt_no_level_writes_is_untraced(root, tmp_path, caplog):
    folder = _copied(root, tmp_path)
    first, second = _holdout_calls(root, 0)
    other = _holdout_calls(root, 1)

    def edit(line: dict) -> dict:
        if line["kind"] != "request":
            return line
        content = line["params"]["messages"][0]["content"]
        if line["call"] == first:
            half = len(content) // 2
            blocks = [
                {"type": "text", "text": content[:half]},
                {"type": "text", "text": content[half:]},
            ]
            line["params"]["messages"][0]["content"] = blocks
        elif line["call"] == second:
            line["params"]["messages"][0]["content"] = f"Hello.\n\n{content}"
        elif line["call"] in other:
            line["params"] = {}
        return line

    _, before = trace(root, "mixed", "--holdout", "0", "--arm", "bko")
    _edit_jsonl(folder / "calls.jsonl", edit)
    code, found = _run(root, folder, "--holdout", "0", "--arm", "bko")
    assert code == 0
    assert [r for r in found if r["call"] == first] == [r for r in before if r["call"] == first]
    broken = {r["untraced"] for r in found if r["call"] == second}
    reason = (
        f"call {second}'s prompt: the prompt does not open with the cell's opening and its bounds"
    )
    assert broken == {reason}
    warned = [m for m in caplog.messages if m.startswith("trace partial ")]
    assert len(warned) == 1
    assert warned[0].endswith(f" missing=none reasons={reason}")


@pytest.mark.parametrize(
    ("edit", "line"),
    [
        (lambda text: text + "not json\n", "last"),
        (lambda text: text + "[1]\n", "last"),
        (lambda text: text.replace('"messages": [{', '"messages": [], "x": [{', 1), "first"),
    ],
    ids=["not-json", "not-an-object", "no-message"],
)
def test_a_malformed_call_log_exits_2_naming_its_line(root, tmp_path, caplog, edit, line):
    folder = _copied(root, tmp_path)
    log = folder / "calls.jsonl"
    text = log.read_text(encoding="utf-8")
    log.write_text(edit(text), encoding="utf-8")
    lines = log.read_text(encoding="utf-8").splitlines()
    number = (
        len(lines)
        if line == "last"
        else 1 + next(k for k, raw in enumerate(lines) if '"messages": [], "x"' in raw)
    )
    assert _run(root, folder, "--holdout", "0", "--arm", "bko") == (2, [])
    assert f"{log} line {number} is not a call log line" in caplog.text


def test_a_call_log_with_no_run_line_leaves_every_record_untraced(root, tmp_path):
    folder = _copied(root, tmp_path)
    _edit_jsonl(folder / "calls.jsonl", lambda line: None if line["kind"] == "run" else line)
    code, found = _run(root, folder, "--holdout", "0", "--arm", "bko")
    assert code == 0
    assert found
    assert {(r["request"] is None, r["untraced"]) for r in found} == {(True, NO_RUN)}


def test_a_record_out_of_range_is_refused_before_any_re_draw(root, caplog):
    caplog.set_level(logging.INFO)
    assert trace(root, "mixed", "--holdout", "0", "--arm", DIRECT_ARM, "--record", "999") == (2, [])
    assert "record 999 is not one of the fill's" in caplog.text
    assert not [m for m in caplog.messages if m.startswith(("trace arm ", "direct "))]


def test_a_bad_selection_names_what_the_folder_holds(root, caplog):
    assert trace(root, "mixed", "--holdout", "7", "--arm", DIRECT_ARM)[0] == 2
    assert f"holdout 7 is not one of {root / 'mixed'}'s [0, 1]" in caplog.text
    assert trace(root, "mixed", "--holdout", "0", "--arm", "smote")[0] == 2
    arms = ["bko", DIRECT_ARM, "none"]
    assert f"{root / 'mixed'} logs no arm 'smote' on holdout 0; its arms are {arms}" in caplog.text


def test_a_trace_refuses_a_value_json_cannot_hold():
    with pytest.raises(ValueError, match="Out of range float values"):
        Trace(0, "bko", "gap", 0, record={"chol": float("nan")}).line()


def world(root: Path) -> tuple[Data, dict]:
    """`VARIED`'s data and the ladder releases, read from the module's data root."""
    ctx = context()
    data = load(Dataset.UCI, root / "data", ctx)
    concepts = {row.concept for row in data.field_rows}
    found, _ = load_scope(concepts, root / "data", releases=ctx.releases, open_url=ctx.open_url)
    return data, dict(found)


def test_a_rejected_draw_is_left_out_of_the_kept_draws(root, monkeypatch):
    data, ontologies = world(root)
    train = train_of(MIXED, 0)
    request = gap_request(train, MIXED, seed=0, frame=UCI_FRAME)
    real = redraw_module.sample

    def with_a_reject(*args: Any, **kwargs: Any) -> dict:
        drawn = dict(real(*args, **kwargs))
        cell = next(iter(drawn))
        first, *rest = drawn[cell]
        bad = dataclasses.replace(first, record={**first.record, "age": 999})
        drawn[cell] = (first, bad, *rest)
        return drawn

    monkeypatch.setattr(redraw_module, "sample", with_a_reject)
    tag = (0, "gap")
    found = redraw(train, request, data=data, ontologies=ontologies, budget=MIXED.budget, tag=tag)
    plain = real(
        train,
        request.quota,
        rows=data.field_rows,
        ontologies=ontologies,
        name=data.name,
        seed=request.seed,
        frame=data.frame,
        tag=tag,
    )
    expected = [dict(draw.record) for draws in plain.values() for draw in draws]
    assert [dict(draw.record) for draw in found.kept] == expected
    assert len(found.rows) == len(expected)
    with pytest.raises(TypeError):
        found.kept[0].sources["age"] = 0  # ty: ignore[invalid-assignment]


def _in_cell(cell: tuple, chol: int) -> dict[str, object]:
    """A valid record of `cell`, `chol` telling records of one cell apart."""
    band, sex, label = cell
    return planted(age=45 if band.startswith("<") else 60, sex=SEX[sex], target=label, chol=chol)


SEX: Final = {"female": 0, "male": 1}


def test_ordered_places_rows_by_the_requests_cells_then_their_generation():
    request = gap_request(train_of(MIXED, 0), MIXED, seed=0, frame=UCI_FRAME)
    cells = list(request.quota)
    assert len(cells) > 1
    candidates = [_in_cell(cell, 250 + k) for cell in reversed(cells) for k in range(2)]
    placed = [candidates[k] for k in ordered(candidates, request, where="test")]
    assert placed == [_in_cell(cell, 250 + k) for cell in cells for k in range(2)]


def test_ordered_refuses_a_row_that_fails_validation_or_falls_outside_the_request():
    request = gap_request(train_of(MIXED, 0), MIXED, seed=0, frame=UCI_FRAME)
    first, *others = list(request.quota)
    valid = _in_cell(first, 250)
    with pytest.raises(IdentityError, match="1 kept candidates no longer pass validation"):
        ordered([valid, {**valid, "sex": 7}], request, where="test")
    narrow = dataclasses.replace(request, quota={first: 1})
    with pytest.raises(IdentityError, match="test: a kept row falls in no cell its request asks"):
        ordered([valid, _in_cell(others[0], 251)], narrow, where="test")


def test_a_line_with_no_call_is_untraced_though_the_log_holds_call_0(root):
    line = next(c for c in lines_of(root, "mixed", 0, "bko", "gap") if c.outcome == KEPT)
    requests = [
        json.loads(raw)
        for raw in (root / "mixed" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    user = next(r for r in requests if r.get("call") == line.call)
    text = user["params"]["messages"][0]["content"]
    assert line.call is not None
    prompts = Prompts({0: text, line.call: text}, "summary")
    assert asked(line, prompts, {})[1] is None
    no_call = dataclasses.replace(line, seen=dataclasses.replace(line.seen, call=None))
    assert asked(no_call, prompts, {}) == (None, NO_CALL)


def test_a_messages_record_names_its_returned_attempt(root):
    fill = select(read_folder(root / "mixed"), holdout=0, arm="bko")
    lines = [dataclasses.replace(c, attempt=2) for c in lines_of(root, "mixed", 0, "bko", "gap")]
    request = gap_request(train_of(MIXED, 0), MIXED, seed=0, frame=UCI_FRAME)
    found = trace_messages(fill, lines, request, calls=root / "mixed" / "calls.jsonl")
    assert found
    assert {None if t.request is None else t.request["attempt"] for t in found} == {2}
