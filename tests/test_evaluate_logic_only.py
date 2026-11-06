"""`bko-logic` beside `bko-direct`: both direct arms in one free run, each under its own name."""

from __future__ import annotations

import dataclasses
import io
import logging

import pandas as pd
import pytest

import bko.generate.logic as logic_module
from bko.evaluate.__main__ import main
from bko.evaluate.experiment import DIRECT_ARM, DIRECT_ARMS, LOGIC_ARM, MESSAGES_ARMS
from bko.evaluate.holdout_candidates import read
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.run_folder import BKO
from bko.evaluate.scoring import _arms
from bko.generate.presence import Presence
from bko.ontology.schema import Codes
from tests._gaps_fixtures import ANGINA
from tests.test_evaluate_direct import SMALL_G2, g2_context


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

BOTH = dataclasses.replace(
    SMALL_G2,
    arms=("none", DIRECT_ARM, LOGIC_ARM),
    candidate=LOGIC_ARM,
    comparators=("none", DIRECT_ARM),
    replica_arms=(),
)
"""Two holdouts, one baseline and both direct arms, `bko-logic` the candidate."""


def test_the_direct_arms_are_bko_direct_on_shares_and_bko_logic_on_logic():
    assert dict(DIRECT_ARMS) == {DIRECT_ARM: Presence.SHARES, LOGIC_ARM: Presence.LOGIC}


def test_arms_lists_both_direct_arms():
    ctx, _ = g2_context()
    guided, direct = _arms(NamedRun(BOTH), ctx)
    assert (guided, direct) == ([], [DIRECT_ARM, LOGIC_ARM])
    stood_in = dataclasses.replace(ctx, arms={**ctx.arms, DIRECT_ARM: ctx.arms["none"]})
    assert _arms(NamedRun(BOTH), stood_in) == ([], [LOGIC_ARM])


def test_the_run_folder_reads_both_direct_arms_as_bko_arms():
    assert {DIRECT_ARM, LOGIC_ARM} <= BKO
    assert frozenset({*MESSAGES_ARMS, *DIRECT_ARMS}) == BKO


def test_both_direct_arms_run_in_one_run_with_distinct_lines(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    ctx, send = g2_context()
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(BOTH)})
    candidates = tmp_path / "holdout-candidates.jsonl"
    out = io.StringIO()
    argv = ["score", "g2", "--candidates", str(candidates), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=out) == 0
    assert send.calls == []
    scores = pd.read_csv(io.StringIO(out.getvalue()))
    gap = scores[scores["source"] == "candidates"]
    assert {DIRECT_ARM, LOGIC_ARM} <= set(gap["arm"])
    found = read(candidates.read_bytes())
    assert {line.arm for line in found.candidates} == {DIRECT_ARM, LOGIC_ARM}
    assert {line.call for line in found.candidates} == {None}
    for name in (DIRECT_ARM, LOGIC_ARM):
        assert any(m.startswith(f"direct arm={name} holdout=0 ") for m in caplog.messages)
        assert any(m.startswith(f"validity arm={name} ") for m in caplog.messages)
    attempts = [m for m in caplog.messages if m.startswith("direct arm=")]
    assert all(
        ("fallback_draws=" in m) == m.startswith(f"direct arm={LOGIC_ARM} ") for m in attempts
    )
    inner = [m for m in caplog.messages if m.startswith("direct holdout=")]
    assert {m.rsplit(" ", 1)[1] for m in inner} == {"presence=shares", "presence=logic"}
    logged = [m for m in caplog.messages if m.startswith("arm ")]
    params = {name: [m for m in logged if f"arm={name} " in m] for name in (DIRECT_ARM, LOGIC_ARM)}
    assert all(params.values()), logged[:4]
    assert all('"presence": "logic"' in m for m in params[LOGIC_ARM])
    assert not any('"presence"' in m for m in params[DIRECT_ARM])


def test_a_run_with_only_bko_logic_needs_a_candidates_file(tmp_path, caplog):
    ctx, _ = g2_context()
    only = dataclasses.replace(BOTH, arms=("none", LOGIC_ARM), comparators=("none",))
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(only)})
    argv = ["score", "g2", "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert f"--candidates is required: the arms ['{LOGIC_ARM}']" in caplog.text


def test_a_logic_component_too_large_stops_the_run_before_any_arm_fits(
    tmp_path, caplog, monkeypatch
):
    monkeypatch.setattr(logic_module, "MAX_COMPONENT", 0)
    caplog.set_level(logging.INFO)
    ctx, _ = g2_context()
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(BOTH)})
    candidates = tmp_path / "holdout-candidates.jsonl"
    argv = ["score", "g2", "--candidates", str(candidates), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert f"{LOGIC_ARM} cannot draw dataset uci-heart-disease on frame " in caplog.text
    assert "a component of 1 concepts" in caplog.text
    assert not any(m.startswith(("arm ", "direct ")) for m in caplog.messages)


def test_field_rows_nested_in_a_chain_refuse_the_direct_arms_before_any_download(tmp_path, caplog):
    ctx, _ = g2_context()
    chain = tuple(
        dataclasses.replace(ANGINA, concept=f"HP:900000{index}", when=(Codes("cp", codes),))
        for index, codes in enumerate((frozenset({1}), frozenset({1, 2}), frozenset({1, 2, 3})))
    )
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(BOTH)}, field_rows=chain)
    candidates = tmp_path / "holdout-candidates.jsonl"
    argv = ["score", "g2", "--candidates", str(candidates), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert (
        f"the direct arms {[DIRECT_ARM, LOGIC_ARM]} cannot draw dataset uci's rows: "
        "HP:9000000 lies inside 2 concepts" in caplog.text
    )
    assert not (tmp_path / "data").exists()
