"""`trace` of every direct arm: `bko-logic` re-drawn with its own presence, beside `bko-direct`."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

import bko.generate.logic as logic_module
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.experiment import DIRECT_ARM, LOGIC_ARM
from bko.evaluate.holdout_candidates import KEPT, read
from bko.evaluate.named_runs import NamedRun
from tests.test_evaluate_direct import g2_context
from tests.test_evaluate_logic_only import BOTH
from tests.test_evaluate_trace import _scored, never


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def _context() -> Context:
    """`BOTH` registered as `g2`, with a send that fails."""
    ctx, _ = g2_context()
    return dataclasses.replace(ctx, runs={"g2": NamedRun(BOTH)}, send=never)


@pytest.fixture(scope="module")
def scored(tmp_path_factory) -> Path:
    """A finished `g2` folder of `BOTH`, its data root beside it."""
    base = tmp_path_factory.mktemp("logic")
    folder = base / "run"
    argv = ["score", "g2", "--candidates", str(folder / "holdout-candidates.jsonl")]
    _scored(folder, _context(), [*argv, "--data-dir", str(base / "data")])
    return folder


def _trace(folder: Path, data: Path, *args: str) -> tuple[int, list[dict]]:
    out = io.StringIO()
    argv = ["trace", "g2", "--folder", str(folder), "--data-dir", str(data), "--holdout", "0"]
    code = main([*argv, *args], context=_context(), out=out)
    return code, [json.loads(line) for line in out.getvalue().splitlines()]


def _copied(scored: Path, tmp_path: Path) -> Path:
    shutil.copytree(scored, tmp_path / "copy")
    return tmp_path / "copy"


def test_a_logic_fill_traces_to_its_arm_sha(scored, caplog):
    caplog.set_level(logging.INFO)
    code, found = _trace(scored, scored.parent / "data", "--arm", LOGIC_ARM)
    assert code == 0
    lines = read((scored / "holdout-candidates.jsonl").read_bytes()).candidates
    kept = [c for c in lines if (c.holdout, c.arm, c.outcome) == (0, LOGIC_ARM, KEPT)]
    assert len(found) == len(kept) > 0
    (arm,) = [m for m in caplog.messages if m.startswith("trace arm ")]
    assert f" arm={LOGIC_ARM} " in arm
    assert arm.endswith(" kept_lines=true matches=true")
    assert all(line["sources"] for line in found)


def test_a_logic_trace_names_its_closed_draw_not_shares(scored):
    _, found = _trace(scored, scored.parent / "data", "--arm", LOGIC_ARM)
    assert any(line["closed"] for line in found)
    for line in found:
        assert [each["concept"] for each in line["concepts"]] == line["closed"]
        assert all("share" not in each and "snodes" not in each for each in line["concepts"])
    _, direct = _trace(scored, scored.parent / "data", "--arm", DIRECT_ARM)
    assert all("share" in each for line in direct for each in line["concepts"])


def test_a_direct_fill_whose_presence_param_differs_exits_3(scored, tmp_path, caplog):
    folder = _copied(scored, tmp_path)
    log = folder / "scores.log"
    text = log.read_text(encoding="utf-8")
    assert '"presence": "logic"' in text
    log.write_text(text.replace('"presence": "logic"', '"presence": "shares"'), encoding="utf-8")
    assert _trace(folder, scored.parent / "data", "--arm", LOGIC_ARM) == (3, [])
    assert f"{LOGIC_ARM}'s arm line draws presence 'shares', not 'logic'" in caplog.text


def test_record_n_reaches_a_logic_fills_later_records(scored):
    data = scored.parent / "data"
    _, every = _trace(scored, data, "--arm", LOGIC_ARM)
    code, found = _trace(scored, data, "--arm", LOGIC_ARM, "--record", "1")
    assert code == 0
    assert found == [every[1]]


def test_a_logic_trace_whose_components_cannot_be_enumerated_exits_2(scored, caplog, monkeypatch):
    monkeypatch.setattr(logic_module, "MAX_COMPONENT", 0)
    assert _trace(scored, scored.parent / "data", "--arm", LOGIC_ARM) == (2, [])
    assert f"{LOGIC_ARM} cannot draw dataset uci-heart-disease on frame " in caplog.text
    assert not any(m.startswith("trace arm ") for m in caplog.messages)


def _edit_arm_line(folder: Path, edit: Callable[[str], str]) -> None:
    """Rewrite holdout 0's `bko-logic` `arm` line of `folder`'s `scores.log` through `edit`."""
    log = folder / "scores.log"
    pattern = rf"arm holdout=0 arm={LOGIC_ARM} attempt=.*"
    text, count = re.subn(pattern, lambda found: edit(found[0]), log.read_text(encoding="utf-8"))
    assert count == 1
    log.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("params", ["not json", "[]"])
def test_a_direct_fill_whose_params_are_no_json_object_exits_3(scored, tmp_path, caplog, params):
    folder = _copied(scored, tmp_path)
    _edit_arm_line(folder, lambda line: re.sub(r" params=.*", f" params={params}", line))
    assert _trace(folder, scored.parent / "data", "--arm", LOGIC_ARM) == (3, [])
    assert f"{LOGIC_ARM}'s arm line params are not " in caplog.text
    assert not any(m.startswith("trace arm ") for m in caplog.messages)


def test_a_logic_fill_whose_arm_sha_differs_exits_3(scored, tmp_path, caplog):
    folder = _copied(scored, tmp_path)
    _edit_arm_line(folder, lambda line: re.sub(r" sha256=\w+ ", f" sha256={'0' * 64} ", line))
    assert _trace(folder, scored.parent / "data", "--arm", LOGIC_ARM) == (3, [])
    (arm,) = [m for m in caplog.messages if m.startswith("trace arm ")]
    assert arm.endswith(" kept_lines=true matches=false")
    assert "re-drawn at seed" in caplog.text


def test_a_logic_record_out_of_range_exits_2(scored, caplog):
    caplog.set_level(logging.INFO)
    code = _trace(scored, scored.parent / "data", "--arm", LOGIC_ARM, "--record", "999")
    assert code == (2, [])
    assert "record 999 is not one of the fill's" in caplog.text
    assert not any(m.startswith("trace arm ") for m in caplog.messages)


def test_a_logic_records_closed_facts_name_drawn_and_added_concepts(scored):
    _, found = _trace(scored, scored.parent / "data", "--arm", LOGIC_ARM)
    facts = [each for line in found for each in line["concepts"]]
    assert {each["drawn"] for each in facts} == {True, False}
    assert all(isinstance(each["label"], str) and each["label"] for each in facts)
