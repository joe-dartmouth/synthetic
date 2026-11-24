"""`bko-nomap` beside `bko-direct`: the direct draw without the mapping table, run and traced."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.direct import DirectArm
from bko.evaluate.experiment import DIRECT_ARM, DIRECT_ARMS, NOMAP_ARM
from bko.evaluate.holdout_candidates import KEPT, read
from bko.evaluate.named_runs import NamedRun
from bko.generate.draw_spec import DrawSpec
from bko.ontology.schema import Codes
from tests._gaps_fixtures import ANGINA, STENOSIS
from tests.test_evaluate_direct import g2_context, varied_arm
from tests.test_evaluate_logic_only import BOTH
from tests.test_evaluate_trace import _scored, never


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

NOMAP_RUN = dataclasses.replace(
    BOTH,
    arms=("none", DIRECT_ARM, NOMAP_ARM),
    candidate=NOMAP_ARM,
    comparators=("none", DIRECT_ARM),
)
"""Two holdouts, one baseline, `bko-direct` and `bko-nomap`, `bko-nomap` the candidate."""


def _context() -> Context:
    """`NOMAP_RUN` registered as `g2`, with a send that fails."""
    ctx, _ = g2_context()
    return dataclasses.replace(ctx, runs={"g2": NamedRun(NOMAP_RUN)}, send=never)


@pytest.fixture(scope="module")
def scored(tmp_path_factory) -> Path:
    """A finished `g2` folder of `NOMAP_RUN`, its data root beside it."""
    base = tmp_path_factory.mktemp("nomap")
    folder = base / "run"
    argv = ["score", "g2", "--candidates", str(folder / "holdout-candidates.jsonl")]
    _scored(folder, _context(), [*argv, "--data-dir", str(base / "data")])
    return folder


def _trace(folder: Path, data: Path, *args: str) -> tuple[int, list[dict]]:
    out = io.StringIO()
    argv = ["trace", "g2", "--folder", str(folder), "--data-dir", str(data), "--holdout", "0"]
    code = main([*argv, *args], context=_context(), out=out)
    return code, [json.loads(line) for line in out.getvalue().splitlines()]


def test_the_nomap_arm_draws_at_mapped_false():
    assert DIRECT_ARMS[NOMAP_ARM] == DrawSpec(mapped=False)


@pytest.mark.parametrize("seed", [0, 3])
def test_the_nomap_arm_logs_only_its_non_default_spec_fields(seed):
    train, request, _ = varied_arm(seed)
    arm = DirectArm(
        (ANGINA, STENOSIS),
        {},
        name=NOMAP_ARM,
        budget=3,
        sink=None,
        dataset="uci-test",
        spec=DIRECT_ARMS[NOMAP_ARM],
    )
    params = arm(train, request).params
    assert params == {"generator": "direct", "mapped": False, "seed": seed}


def test_a_nomap_arm_runs_beside_bko_direct_in_one_candidates_file(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    ctx = _context()
    candidates = tmp_path / "holdout-candidates.jsonl"
    out = io.StringIO()
    argv = ["score", "g2", "--candidates", str(candidates), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=out) == 0
    scores = pd.read_csv(io.StringIO(out.getvalue()))
    assert {DIRECT_ARM, NOMAP_ARM} <= set(scores[scores["source"] == "candidates"]["arm"])
    found = read(candidates.read_bytes())
    assert {line.arm for line in found.candidates} == {DIRECT_ARM, NOMAP_ARM}
    assert {line.call for line in found.candidates} == {None}
    inner = [m for m in caplog.messages if m.startswith("direct holdout=")]
    assert {m.endswith(" mapped=false presence=shares") for m in inner} == {True, False}
    assert all(m.endswith(" presence=shares") for m in inner)
    logged = [m for m in caplog.messages if m.startswith("arm ")]
    nomap = [m for m in logged if f"arm={NOMAP_ARM} " in m]
    direct = [m for m in logged if f"arm={DIRECT_ARM} " in m]
    assert nomap
    assert direct
    assert all('"mapped": false' in m for m in nomap)
    assert not any('"mapped"' in m for m in direct)


def test_a_nomap_arm_line_ends_with_fallback_draws(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    candidates = tmp_path / "holdout-candidates.jsonl"
    argv = ["score", "g2", "--candidates", str(candidates), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=_context(), out=io.StringIO()) == 0
    attempts = [m for m in caplog.messages if m.startswith("direct arm=")]
    assert any(m.startswith(f"direct arm={NOMAP_ARM} ") for m in attempts)
    assert all(
        (re.search(r" fallback_draws=\d+$", m) is not None)
        == m.startswith(f"direct arm={NOMAP_ARM} ")
        for m in attempts
    )


def test_a_nomap_fill_traces_to_its_arm_sha(scored, caplog):
    caplog.set_level(logging.INFO)
    code, found = _trace(scored, scored.parent / "data", "--arm", NOMAP_ARM)
    assert code == 0
    lines = read((scored / "holdout-candidates.jsonl").read_bytes()).candidates
    kept = [c for c in lines if (c.holdout, c.arm, c.outcome) == (0, NOMAP_ARM, KEPT)]
    assert len(found) == len(kept) > 0
    (arm,) = [m for m in caplog.messages if m.startswith("trace arm ")]
    assert f" arm={NOMAP_ARM} " in arm
    assert arm.endswith(" kept_lines=true matches=true")
    for line in found:
        assert line["concepts"] == []
        assert line["closed"] == []
        assert line["sources"]


def test_a_nomap_fill_whose_mapped_param_is_removed_exits_3(scored, tmp_path, caplog):
    shutil.copytree(scored, tmp_path / "copy")
    folder = tmp_path / "copy"
    log = folder / "scores.log"
    pattern = rf"arm holdout=0 arm={NOMAP_ARM} attempt=.*"

    def edit(found: re.Match[str]) -> str:
        assert '"mapped": false, ' in found[0]
        return found[0].replace('"mapped": false, ', "")

    text, count = re.subn(pattern, edit, log.read_text(encoding="utf-8"))
    assert count == 1
    log.write_text(text, encoding="utf-8")
    assert _trace(folder, scored.parent / "data", "--arm", NOMAP_ARM) == (3, [])
    assert f"{NOMAP_ARM}'s arm line draws mapped True, not False" in caplog.text
    assert not any(m.startswith("trace arm ") for m in caplog.messages)


def _chained_context(arms: tuple[str, ...], tmp_path: Path) -> tuple[Context, list[str]]:
    """`g2` over `arms` on field rows nested in a chain, and its `score` argv."""
    ctx, _ = g2_context()
    chain = tuple(
        dataclasses.replace(ANGINA, concept=f"HP:900000{index}", when=(Codes("cp", codes),))
        for index, codes in enumerate((frozenset({1}), frozenset({1, 2}), frozenset({1, 2, 3})))
    )
    run = dataclasses.replace(NOMAP_RUN, arms=arms, candidate=arms[-1], comparators=("none",))
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(run)}, field_rows=chain, send=never)
    candidates = tmp_path / "holdout-candidates.jsonl"
    return ctx, ["score", "g2", "--candidates", str(candidates), "--data-dir", str(tmp_path / "d")]


def test_field_rows_nested_in_a_chain_do_not_refuse_bko_nomap(tmp_path, caplog):
    ctx, argv = _chained_context(("none", NOMAP_ARM), tmp_path)
    main(argv, context=ctx, out=io.StringIO())
    assert "cannot draw dataset" not in caplog.text
    assert "lies inside" not in caplog.text


def test_field_rows_nested_in_a_chain_still_refuse_bko_direct(tmp_path, caplog):
    ctx, argv = _chained_context(("none", DIRECT_ARM), tmp_path)
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert f"the direct arms {[DIRECT_ARM]} cannot draw dataset uci's rows" in caplog.text
