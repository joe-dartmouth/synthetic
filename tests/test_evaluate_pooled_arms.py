"""The pooled direct arms beside `bko-direct`: their specs, attribution, run and trace.

ADR `research/matched-knowledge-controls` fixes each arm's draw.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import re
from pathlib import Path

import pytest

from bko.evaluate.__main__ import main
from bko.evaluate.attribution import Attribution, classify
from bko.evaluate.context import Context
from bko.evaluate.direct_arms import (
    DIRECT_ARM,
    DIRECT_ARMS,
    DONOR_ARM,
    POOL_ALL_ARM,
    POOL_RANDOM_ARM,
)
from bko.evaluate.holdout_candidates import KEPT, read
from bko.evaluate.named_runs import NamedRun
from bko.generate.draw_spec import DrawSpec, Pool
from bko.generate.presence import Presence
from tests.test_evaluate_attribution import MAPPED, _direct
from tests.test_evaluate_direct import g2_context
from tests.test_evaluate_logic_only import BOTH
from tests.test_evaluate_trace import _scored, never


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

POOLED_ARMS = (DONOR_ARM, POOL_RANDOM_ARM, POOL_ALL_ARM)

POOLED_RUN = dataclasses.replace(
    BOTH,
    arms=("none", DIRECT_ARM, *POOLED_ARMS),
    candidate=POOL_RANDOM_ARM,
    comparators=("none", DIRECT_ARM),
)
"""Two holdouts, one baseline, `bko-direct` and the three pooled arms."""


def _context() -> Context:
    """`POOLED_RUN` registered as `g2`, with a send that fails."""
    ctx, _ = g2_context()
    return dataclasses.replace(ctx, runs={"g2": NamedRun(POOLED_RUN)}, send=never)


@pytest.fixture(scope="module")
def scored(tmp_path_factory) -> Path:
    """A finished `g2` folder of `POOLED_RUN`, its data root beside it."""
    base = tmp_path_factory.mktemp("pooled")
    folder = base / "run"
    argv = ["score", "g2", "--candidates", str(folder / "holdout-candidates.jsonl")]
    _scored(folder, _context(), [*argv, "--data-dir", str(base / "data")])
    return folder


def _run(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    code = main(argv, context=_context(), out=out)
    return code, out.getvalue()


def test_the_pooled_arms_draw_at_their_specs():
    assert DIRECT_ARMS[DONOR_ARM] == DrawSpec(presence=Presence.POOLED)
    assert DIRECT_ARMS[POOL_RANDOM_ARM] == DrawSpec(presence=Presence.POOLED, pool=Pool.RANDOM)
    assert DIRECT_ARMS[POOL_ALL_ARM] == DrawSpec(presence=Presence.POOLED, pool=Pool.ALL)


@pytest.mark.parametrize("arm", POOLED_ARMS)
def test_a_pooled_field_is_attributed_to_its_donor(arm):
    found = classify(_direct(arm), mapped=MAPPED)
    assert found["cp"] is found["thal"] is found["age"] is Attribution.DONOR
    assert Attribution.KNOWLEDGE not in found.values()
    assert classify(_direct(f"{arm}-replica"), mapped=MAPPED)["cp"] is Attribution.DONOR


@pytest.mark.parametrize("arm", POOLED_ARMS)
def test_each_pooled_arm_scores_and_traces_end_to_end(scored, tmp_path, arm, caplog):
    log = (scored / "scores.log").read_text(encoding="utf-8")
    params = re.findall(rf"arm holdout=\d+ arm={arm} attempt=.*", log)
    assert params
    assert all('"presence": "pooled"' in line for line in params)
    pool = DIRECT_ARMS[arm].pool
    assert all((f'"pool": "{pool.value}"' in line) is (pool is not Pool.MAPPED) for line in params)

    caplog.set_level(logging.INFO)
    data = scored.parent / "data"
    argv = ["trace", "g2", "--folder", str(scored), "--data-dir", str(data), "--holdout", "0"]
    code, out = _run([*argv, "--arm", arm])
    assert code == 0
    found = [json.loads(line) for line in out.splitlines()]
    lines = read((scored / "holdout-candidates.jsonl").read_bytes()).candidates
    kept = [c for c in lines if (c.holdout, c.arm, c.outcome) == (0, arm, KEPT)]
    assert len(found) == len(kept) > 0
    assert all(line["concepts"] == [] for line in found)
    (traced,) = [m for m in caplog.messages if m.startswith("trace arm ")]
    assert traced.endswith(" kept_lines=true matches=true")

    path = tmp_path / "trace.jsonl"
    path.write_text(out, encoding="utf-8")
    code, csv = _run(["attribution", "g2", "--trace", str(path)])
    assert code == 0
    counts = {row.split(",")[3]: int(row.split(",")[5]) for row in csv.splitlines()[1:]}
    assert counts[Attribution.KNOWLEDGE] == 0
    assert counts[Attribution.DONOR] > 0


def test_pool_random_logs_its_fields_once_per_draw(scored):
    log = (scored / "scores.log").read_text(encoding="utf-8")
    drawn = re.findall(r"pool-random holdout=(\d+) fill=(\w+) fields=(\S+)", log)
    attempts = re.findall(rf"direct arm={POOL_RANDOM_ARM} holdout=(\d+) fill=(\w+) ", log)
    assert [(h, f) for h, f, _ in drawn] == attempts
    assert {h for h, _, _ in drawn} == {"0", "1"}
    assert all(
        fields
        in {
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        }
        for _, _, fields in drawn
    )
