"""`trace` of a perturbation arm's fill: its rows from its lines, the direct draw underneath."""

from __future__ import annotations

import dataclasses
import io
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from bko.datasets.uci import UCI_FRAME
from bko.evaluate import direct as direct_arm
from bko.evaluate import perturb, trace_perturb
from bko.evaluate import redraw as redraw_module
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.errors import IdentityError
from bko.evaluate.experiment import DIRECT_ARM, PERTURBED_ARM, PRAM_ARM
from bko.evaluate.holdout_candidates import (
    CHANGED,
    KEPT,
    PERTURBED,
    REFUSED,
    REJECTED,
    UNCHANGED,
    Candidate,
    Seen,
    read,
)
from bko.evaluate.redraw import Redrawn
from bko.evaluate.run_folder import read_folder
from bko.generate.errors import SamplingError
from tests.test_evaluate_perturb import Rewriter, perturbed_folder
from tests.test_evaluate_pram import FIELDS, _prammed
from tests.test_evaluate_trace import arm_seed, never


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def _trace(folder: Path, ctx: Context, arm: str, *args: str) -> tuple[int, list[dict]]:
    """Run `trace g2` on `folder`'s holdout 0 with a send that fails; its exit and its lines."""
    out = io.StringIO()
    argv = ["trace", "g2", "--folder", str(folder), "--holdout", "0", "--arm", arm]
    argv += ["--data-dir", str(folder.parent / "data"), *args]
    code = main(argv, context=dataclasses.replace(ctx, send=never), out=out)
    return code, [json.loads(line) for line in out.getvalue().splitlines()]


def _kept(folder: Path, arm: str) -> list[Candidate]:
    """`arm`'s kept lines on holdout 0, in file order."""
    found = read((folder / "holdout-candidates.jsonl").read_bytes()).candidates
    return [c for c in found if (c.holdout, c.arm, c.outcome) == (0, arm, KEPT)]


def _edit(path: Path, edit: Callable[[dict], dict]) -> None:
    """Rewrite each JSON line of `path` through `edit`."""
    lines = [edit(json.loads(raw)) for raw in path.read_text(encoding="utf-8").splitlines()]
    path.write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines), "utf-8")


def _schema_3(line: dict) -> dict:
    """`line` as a schema-3 file holds it: no `perturb` key, the header at schema 3."""
    line.pop("perturb", None)
    return {**line, "schema": 3} if line["kind"] == "header" else line


def _arm_line(caplog: pytest.LogCaptureFixture, arm: str) -> str:
    (line,) = [m for m in caplog.messages if m.startswith(f"trace arm holdout=0 arm={arm} ")]
    return line


@pytest.fixture
def rewritten(tmp_path, caplog) -> tuple[Path, Context]:
    """A `PERTURBED` folder whose first call's second row is refused."""
    return perturbed_folder(tmp_path, caplog, Rewriter(invalid=frozenset({1})))


def test_a_perturbed_fill_traces_to_its_arm_sha(rewritten, caplog):
    folder, ctx = rewritten
    code, found = _trace(folder, ctx, PERTURBED_ARM)
    assert code == 0
    assert len(found) == len(_kept(folder, PERTURBED_ARM)) > 0
    assert _arm_line(caplog, PERTURBED_ARM).endswith(" unperturbed=true matches=true")
    assert [line["record_id"] for line in found] == list(range(len(found)))


def test_each_rewrite_line_carries_its_lines_result(rewritten):
    folder, ctx = rewritten
    _, found = _trace(folder, ctx, PERTURBED_ARM)
    stored = Counter(
        (json.dumps(line.candidate, sort_keys=True), line.perturb)
        for line in _kept(folder, PERTURBED_ARM)
    )
    traced = Counter(
        (json.dumps(line["record"], sort_keys=True), line["perturb"]["result"]) for line in found
    )
    assert traced == stored
    assert {PERTURBED, f"{REFUSED}keys"} <= {result for _, result in traced}


def test_the_fields_differ_from_the_draw_exactly(rewritten):
    folder, ctx = rewritten
    _, drawn = _trace(folder, ctx, DIRECT_ARM)
    _, found = _trace(folder, ctx, PERTURBED_ARM)
    assert len(drawn) == len(found)
    for direct, perturbed in zip(drawn, found, strict=True):
        record, draw = perturbed["record"], direct["record"]
        differ = {name for name in record if record[name] != draw[name]}
        assert len(perturbed["perturb"]["fields"]) == len(differ)
        assert set(perturbed["perturb"]["fields"]) == differ
        assert perturbed["perturb"]["drawn"] == {name: draw[name] for name in differ}
        assert set(differ) <= set(FIELDS)
        assert (perturbed["sources"], perturbed["concepts"]) == (
            direct["sources"],
            direct["concepts"],
        )
    assert any(line["perturb"]["fields"] for line in found)


def test_a_pram_fill_traces_to_its_arm_sha(tmp_path, caplog):
    _, ctx = _prammed(tmp_path, caplog)
    folder = tmp_path / "run"
    code, found = _trace(folder, ctx, PRAM_ARM)
    assert code == 0
    assert len(found) == len(_kept(folder, PRAM_ARM)) > 0
    assert _arm_line(caplog, PRAM_ARM).endswith(" unperturbed=true matches=true")
    assert {line["perturb"]["result"] for line in found} == {CHANGED, UNCHANGED}
    for line in found:
        assert (line["perturb"]["result"] == CHANGED) == bool(line["perturb"]["fields"])


def test_pram_fields_sum_to_the_shapes_counts(tmp_path, caplog):
    _, ctx = _prammed(tmp_path, caplog)
    folder = tmp_path / "run"
    log = (folder / "scores.log").read_text(encoding="utf-8")
    shapes = re.search(rf"perturb shapes holdout=0 arm={PRAM_ARM} .*", log)
    assert shapes is not None
    rows = int(shapes[0].split(" rows_changed=")[1].split(" ")[0])
    values = int(shapes[0].split(" values_changed=")[1])
    _, found = _trace(folder, ctx, PRAM_ARM)
    assert sum(len(line["perturb"]["fields"]) for line in found) == values
    assert sum(bool(line["perturb"]["fields"]) for line in found) == rows


def test_a_schema_3_rewrite_line_reads_kept_when_it_names_no_call(rewritten):
    folder, ctx = rewritten
    stored = {
        json.dumps(line.candidate, sort_keys=True): line.call
        for line in _kept(folder, PERTURBED_ARM)
    }
    _edit(folder / "holdout-candidates.jsonl", _schema_3)
    code, found = _trace(folder, ctx, PERTURBED_ARM)
    assert code == 0
    for line in found:
        call = stored[json.dumps(line["record"], sort_keys=True)]
        changed = bool(line["perturb"]["fields"])
        expected = KEPT if call is None else (PERTURBED if changed else UNCHANGED)
        assert line["perturb"]["result"] == expected
    assert {line["perturb"]["result"] for line in found} == {KEPT, PERTURBED, UNCHANGED}


def test_a_perturbation_fill_whose_lines_miss_the_arm_sha_exits_3(rewritten, caplog):
    folder, ctx = rewritten
    edited: list[int] = []

    def nudge(line: dict) -> dict:
        wanted = (line.get("holdout"), line.get("arm"), line.get("outcome")) == (
            0,
            PERTURBED_ARM,
            KEPT,
        )
        if wanted and not edited:
            edited.append(line["position"])
            line["candidate"]["chol"] += 1
        return line

    _edit(folder / "holdout-candidates.jsonl", nudge)
    assert edited
    assert _trace(folder, ctx, PERTURBED_ARM) == (3, [])
    assert "not its arm line's" in caplog.text
    assert _arm_line(caplog, PERTURBED_ARM).endswith(" matches=false")


def test_a_perturbation_fill_whose_unperturbed_columns_differ_exits_3(
    rewritten, caplog, monkeypatch
):
    folder, ctx = rewritten
    logged = read_folder(folder).arms[0, PERTURBED_ARM]
    sha = re.search(r" sha256=(\w+)", logged)
    assert sha is not None
    edited: list[int] = []

    def nudge(line: dict) -> dict:
        wanted = (line.get("holdout"), line.get("arm"), line.get("outcome")) == (
            0,
            PERTURBED_ARM,
            KEPT,
        )
        if wanted and not edited:
            edited.append(line["position"])
            line["candidate"]["age"] += 1
        return line

    _edit(folder / "holdout-candidates.jsonl", nudge)
    monkeypatch.setattr(trace_perturb, "digest", lambda _: sha[1])
    assert _trace(folder, ctx, PERTURBED_ARM) == (3, [])
    assert "columns no perturbation changes differ from the re-draw: age" in caplog.text
    assert _arm_line(caplog, PERTURBED_ARM).endswith(" unperturbed=false matches=false")


def test_a_bko_direct_line_holds_no_perturb_key(rewritten):
    folder, ctx = rewritten
    code, found = _trace(folder, ctx, DIRECT_ARM)
    assert code == 0
    assert found
    assert all("perturb" not in line for line in found)


def test_a_schema_3_pram_line_reads_changed_or_unchanged_from_its_draw(tmp_path, caplog):
    _, ctx = _prammed(tmp_path, caplog)
    folder = tmp_path / "run"
    _edit(folder / "holdout-candidates.jsonl", _schema_3)
    code, found = _trace(folder, ctx, PRAM_ARM)
    assert code == 0
    assert {line["perturb"]["result"] for line in found} == {CHANGED, UNCHANGED}
    for line in found:
        assert (line["perturb"]["result"] == CHANGED) == bool(line["perturb"]["fields"])


def test_a_returned_attempt_with_a_line_per_draw_missing_exits_3(tmp_path, caplog, monkeypatch):
    _, ctx = _prammed(tmp_path, caplog)
    real = trace_perturb.redraw

    def one_short(*args: Any, **kwargs: Any) -> Redrawn:
        found = real(*args, **kwargs)
        return dataclasses.replace(found, drawn=found.drawn[:-1])

    monkeypatch.setattr(trace_perturb, "redraw", one_short)
    assert _trace(tmp_path / "run", ctx, PRAM_ARM) == (3, [])
    assert "lines in the returned attempt, not one per draw of" in caplog.text
    assert _arm_line(caplog, PRAM_ARM).endswith(" unperturbed=false matches=false")


def _refusing_each_first(real: Callable[..., Any]) -> Callable[..., Any]:
    """`real`, refusing every other call of each fill tag: each arm's first attempt of a fill.

    `bko-direct` and `bko-direct-pram` draw a fill under one tag, each retrying once.
    """
    calls: Counter[object] = Counter()

    def refusing(*args: Any, **kwargs: Any) -> Any:
        calls[kwargs["tag"]] += 1
        if calls[kwargs["tag"]] % 2:
            msg = "each arm's first attempt of each fill refuses"
            raise SamplingError(msg)
        return real(*args, **kwargs)

    return refusing


def test_a_retried_pram_fill_is_traced_at_its_arm_lines_seed_and_attempt(tmp_path, caplog):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(direct_arm, "sample", _refusing_each_first(direct_arm.sample))
        _, ctx = _prammed(tmp_path, caplog)
    folder = tmp_path / "run"
    attempt, seed = arm_seed(folder, 0, PRAM_ARM)
    assert (attempt, seed != 0) == (2, True), "the fill returned on a retry at a derived seed"
    code, found = _trace(folder, ctx, PRAM_ARM)
    assert code == 0
    assert found
    assert {(line["request"]["seed"], line["request"]["attempt"]) for line in found} == {
        (seed, attempt)
    }


def _with_a_reject(real: Callable[..., Any]) -> Callable[..., Any]:
    """`real`, its first cell's draws given a second draw whose `age` fails validation."""

    def rejecting(*args: Any, **kwargs: Any) -> Any:
        drawn = dict(real(*args, **kwargs))
        cell = next(iter(drawn))
        first, *rest = drawn[cell]
        bad = dataclasses.replace(first, record={**first.record, "age": 999})
        drawn[cell] = (first, bad, *rest)
        return drawn

    return rejecting


def test_a_rejected_line_is_left_out_of_a_perturbation_trace(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(redraw_module, "sample", _with_a_reject(redraw_module.sample))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(direct_arm, "sample", _with_a_reject(direct_arm.sample))
        _, ctx = _prammed(tmp_path, caplog)
    folder = tmp_path / "run"
    found = read((folder / "holdout-candidates.jsonl").read_bytes()).candidates
    mine = [c for c in found if (c.holdout, c.arm) == (0, PRAM_ARM)]
    assert any(c.outcome != KEPT for c in mine)
    code, traced = _trace(folder, ctx, PRAM_ARM)
    assert code == 0
    assert len(traced) == len(_kept(folder, PRAM_ARM)) > 0
    assert all(line["record"]["age"] != 999 for line in traced)


def test_a_line_holding_no_record_is_refused():
    seen = Seen(None, "cell", 0, [1, 2], None, REJECTED, ("fields",))
    line = Candidate(0, PRAM_ARM, "gap", 1, True, seen)
    with pytest.raises(IdentityError, match=f"holdout 0's {PRAM_ARM} line 0 holds no record"):
        trace_perturb._record(line)


def test_unmapped_fields_that_are_not_the_frames_pinned_exit_2(tmp_path, caplog, monkeypatch):
    _, ctx = _prammed(tmp_path, caplog)
    monkeypatch.setattr(perturb, "PINNED", {UCI_FRAME.name: ("chol",)})
    assert _trace(tmp_path / "run", ctx, PRAM_ARM) == (2, [])
    assert "are not its pinned ['chol']" in caplog.text


@pytest.mark.parametrize("arm", [PRAM_ARM, PERTURBED_ARM])
def test_a_perturbation_record_out_of_range_exits_2(tmp_path, caplog, arm):
    if arm == PRAM_ARM:
        _, ctx = _prammed(tmp_path, caplog)
        folder = tmp_path / "run"
    else:
        folder, ctx = perturbed_folder(tmp_path, caplog, Rewriter())
    caplog.clear()
    assert _trace(folder, ctx, arm, "--record", "999") == (2, [])
    assert "record 999 is not one of the fill's" in caplog.text
    assert not any(m.startswith("trace arm ") for m in caplog.messages)
