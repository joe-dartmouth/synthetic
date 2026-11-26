"""`bko-capped` beside `bko-direct`: the draw under a donor floor and a reuse cap, run, traced."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest

from bko.datasets.split import split
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.data import load
from bko.evaluate.direct import DirectArm
from bko.evaluate.direct_arms import CAPPED_ARM, DIRECT_ARM, DIRECT_ARMS
from bko.evaluate.donor_attacks import donor_counts, match_groups
from bko.evaluate.experiment import gap_request
from bko.evaluate.holdout_candidates import KEPT, read
from bko.evaluate.named_runs import RUNS, NamedRun
from bko.evaluate.prepare import prepared
from bko.evaluate.reference import REFERENCE
from bko.evaluate.run_context import named as named_run
from bko.evaluate.run_folder import read_folder
from bko.evaluate.runs_diabetes import E9_EMPTY, E9_THIN
from bko.evaluate.runs_direct import E14_EMPTY, E14_THIN
from bko.evaluate.shared import rebuilt
from bko.evaluate.trace_lines import select
from bko.gaps.scope import load_scope
from bko.generate.draw_spec import DrawSpec
from tests._gaps_fixtures import ANGINA, STENOSIS
from tests.test_evaluate_direct import g2_context, varied_arm
from tests.test_evaluate_logic_only import BOTH
from tests.test_evaluate_trace import _scored, never


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

CAPPED_RUN = dataclasses.replace(
    BOTH,
    arms=("none", DIRECT_ARM, CAPPED_ARM),
    candidate=CAPPED_ARM,
    comparators=("none", DIRECT_ARM),
)
"""Two holdouts, one baseline, `bko-direct` and `bko-capped`, `bko-capped` the candidate."""


def _context() -> Context:
    """`CAPPED_RUN` registered as `g2`, with a send that fails."""
    ctx, _ = g2_context()
    return dataclasses.replace(ctx, runs={"g2": NamedRun(CAPPED_RUN)}, send=never)


@pytest.fixture(scope="module")
def scored(tmp_path_factory) -> Path:
    """A finished `g2` folder of `CAPPED_RUN`, its data root beside it."""
    base = tmp_path_factory.mktemp("capped")
    folder = base / "run"
    argv = ["score", "g2", "--candidates", str(folder / "holdout-candidates.jsonl")]
    _scored(folder, _context(), [*argv, "--data-dir", str(base / "data")])
    return folder


def _trace(folder: Path, data: Path, *args: str) -> tuple[int, list[dict]]:
    out = io.StringIO()
    argv = ["trace", "g2", "--folder", str(folder), "--data-dir", str(data), "--holdout", "0"]
    code = main([*argv, *args], context=_context(), out=out)
    return code, [json.loads(line) for line in out.getvalue().splitlines()]


def test_the_capped_arm_draws_at_a_floor_of_2_and_a_cap_of_1():
    assert DIRECT_ARMS[CAPPED_ARM] == DrawSpec(min_donors=2, reuse_cap=1)


@pytest.mark.parametrize("seed", [0, 3])
def test_the_capped_arm_logs_only_its_nondefault_fields(seed):
    train, request, _ = varied_arm(seed)
    arm = DirectArm(
        (ANGINA, STENOSIS),
        {},
        name=CAPPED_ARM,
        budget=3,
        sink=None,
        dataset="uci-test",
        spec=DIRECT_ARMS[CAPPED_ARM],
    )
    params = arm(train, request).params
    assert params == {"generator": "direct", "min_donors": 2, "reuse_cap": 1, "seed": seed}


def test_a_capped_fill_traces_to_its_arm_sha(scored, caplog):
    caplog.set_level(logging.INFO)
    code, found = _trace(scored, scored.parent / "data", "--arm", CAPPED_ARM)
    assert code == 0
    lines = read((scored / "holdout-candidates.jsonl").read_bytes()).candidates
    kept = [c for c in lines if (c.holdout, c.arm, c.outcome) == (0, CAPPED_ARM, KEPT)]
    assert len(found) == len(kept) > 0
    (arm,) = [m for m in caplog.messages if m.startswith("trace arm ")]
    assert f" arm={CAPPED_ARM} " in arm
    assert arm.endswith(" kept_lines=true matches=true")


def test_a_capped_fill_whose_cap_param_is_a_float_exits_3(scored, tmp_path, caplog):
    shutil.copytree(scored, tmp_path / "copy")
    folder = tmp_path / "copy"
    log = folder / "scores.log"

    def edit(found: re.Match[str]) -> str:
        assert '"reuse_cap": 1, ' in found[0]
        return found[0].replace('"reuse_cap": 1, ', '"reuse_cap": 1.0, ')

    pattern = rf"arm holdout=0 arm={CAPPED_ARM} attempt=.*"
    text, count = re.subn(pattern, edit, log.read_text(encoding="utf-8"))
    assert count == 1
    log.write_text(text, encoding="utf-8")
    assert _trace(folder, scored.parent / "data", "--arm", CAPPED_ARM) == (3, [])
    assert f"{CAPPED_ARM}'s arm line draws reuse_cap 1.0, not 1" in caplog.text


def _unit_donors(folder: Path, data_root: Path, arm: str) -> list[Counter[int]]:
    """Per match unit, how many rows of holdout 0's `arm` gap fill each record donates it to."""
    named = named_run("g2", _context())
    data = load(named.dataset, data_root, named)
    concepts = {row.concept for row in data.field_rows}
    ontologies, _ = load_scope(
        concepts, data_root, releases=named.releases, open_url=named.open_url
    )
    stored = read_folder(folder)
    fill = select(stored, holdout=0, arm=arm)
    frame = data.frame
    parts = split(data.records, seed=0, edges=frame.edges)
    train = prepared(parts, named.experiment, edges=frame.edges).ablated.train
    request = gap_request(train, named.experiment, seed=0, frame=frame)
    lines = list(stored.lines[0, arm, "gap"])
    rows = rebuilt(lines, dataclasses.replace(request, seed=fill.seed), where="the fill")
    return [
        donor_counts(
            fill,
            rows,
            train,
            request,
            lines=lines,
            data=data,
            ontologies=ontologies,
            budget=named.experiment.budget,
            groups=(unit,),
        )
        for unit in match_groups(data.field_rows, frame=frame)
    ]


def test_donor_reuse_of_a_capped_fill_is_at_most_the_cap_per_unit(scored):
    data = scored.parent / "data"
    capped = _unit_donors(scored, data, CAPPED_ARM)
    assert any(capped)
    assert max(max(counts.values(), default=0) for counts in capped) == 1
    direct = _unit_donors(scored, data, DIRECT_ARM)
    assert max(max(counts.values(), default=0) for counts in direct) > 1


@pytest.mark.parametrize(("capped", "e9"), [(E14_THIN, E9_THIN), (E14_EMPTY, E9_EMPTY)])
def test_e14_is_e9_with_the_capped_arm_against_bko_direct(capped, e9):
    assert capped.arms == ("none", "real", "bko-direct", "bko-capped")
    assert capped.candidate == "bko-capped"
    assert capped.comparators == ("bko-direct", "none")
    assert capped.replica_arms == ()
    changed = {"arms", "candidate", "comparators", "replica_arms"}
    for field in dataclasses.fields(capped):
        if field.name not in changed:
            assert getattr(capped, field.name) == getattr(e9, field.name), field.name
    assert capped.removal is not None


@pytest.mark.parametrize("fraction", ["thin", "empty"])
def test_e14_runs_are_registered_with_bko_direct_as_privacy_reference(fraction):
    assert RUNS[f"e14-{fraction}"].experiment == (E14_THIN if fraction == "thin" else E14_EMPTY)
    assert REFERENCE[f"e14-{fraction}"] == DIRECT_ARM
