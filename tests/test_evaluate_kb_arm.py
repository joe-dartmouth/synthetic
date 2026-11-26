"""`bko-kb`, the knowledge-only direct arm: its spec, its fill, its attribution and its refusal.

ADR `research/matched-knowledge-controls`, Decision 9, fixes the arm.
"""

from __future__ import annotations

import dataclasses
import io
import logging

import pytest

from bko.baselines.request import Request
from bko.evaluate.__main__ import main
from bko.evaluate.attribution import Attribution, classify
from bko.evaluate.direct import DirectArm
from bko.evaluate.direct_arms import DIRECT_ARMS, KB_ARM
from bko.evaluate.named_runs import NamedRun
from bko.generate.draw_spec import DrawSpec
from bko.generate.presence import Presence
from tests._kb_fixtures import FRAME, ONTOLOGIES, QUOTA, ROWS, kb_train
from tests.test_evaluate_attribution import MAPPED, RECORD, _direct
from tests.test_evaluate_direct import g2_context
from tests.test_evaluate_logic_only import BOTH


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def test_the_kb_arm_draws_at_its_spec():
    assert DIRECT_ARMS[KB_ARM] == DrawSpec(presence=Presence.LOGIC, donors=False)


def test_a_kb_field_is_knowledge_and_its_age_method_seed():
    found = classify(_direct(KB_ARM, sources={}), mapped=MAPPED)
    assert found["cp"] is found["thal"] is Attribution.KNOWLEDGE
    assert found["age"] is Attribution.SEED
    assert {found[f] for f in RECORD if f not in MAPPED} == {Attribution.SEED}
    assert Attribution.DONOR not in found.values()


def test_the_kb_arm_fills_a_request_from_k_alone(caplog):
    caplog.set_level(logging.INFO)
    train = kb_train([25, *range(30, 80, 3), 80], shift=1)
    arm = DirectArm(
        ROWS, ONTOLOGIES, name=KB_ARM, budget=2, sink=None, dataset="toy", spec=DIRECT_ARMS[KB_ARM]
    )
    baseline = arm(train, Request(dict(QUOTA), seed=7, frame=FRAME))
    assert len(baseline.added) == sum(QUOTA.values())
    assert baseline.params == {
        "generator": "direct",
        "seed": 7,
        "presence": "logic",
        "donors": False,
    }
    inner = [m for m in caplog.messages if m.startswith("direct holdout=")]
    assert len(inner) == len(QUOTA)
    assert all(m.endswith(" donors=false presence=logic") for m in inner)
    (attempt,) = [m for m in caplog.messages if m.startswith(f"direct arm={KB_ARM} ")]
    assert attempt.endswith(" fallback_draws=0")


def test_a_frame_with_a_free_field_stops_the_run_before_any_arm_fits(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    ctx, _ = g2_context()
    run = dataclasses.replace(BOTH, arms=("none", KB_ARM), candidate=KB_ARM, comparators=("none",))
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(run)})
    candidates = tmp_path / "holdout-candidates.jsonl"
    argv = ["score", "g2", "--candidates", str(candidates), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert f"{KB_ARM} cannot draw dataset uci-heart-disease on frame " in caplog.text
    assert "is a free field" in caplog.text
    assert not any(m.startswith(("arm ", "direct ")) for m in caplog.messages)
