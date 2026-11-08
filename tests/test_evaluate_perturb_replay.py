"""A paid run with the perturbed arm replays byte for byte from its call log."""

from __future__ import annotations

import dataclasses
import io
import json

import pytest

from bko.evaluate.__main__ import main
from bko.generate.messages import Call, Reply
from bko.generate.rewrite import COUNTS_HEADER
from tests.test_evaluate_perturb import Rewriter, perturbed_folder


def _no_send(call: Call) -> Reply:
    msg = f"a replay sent a call: {call.user[:80]!r}"
    raise AssertionError(msg)


@pytest.mark.parametrize(
    "rewriter",
    [lambda: Rewriter(invalid=frozenset({1})), lambda: Rewriter(skew=True)],
    ids=["invalid", "below"],
)
def test_a_replay_rewrites_the_scores_and_candidates_bytes(tmp_path, caplog, rewriter):
    """A holdout whose reading falls below the unperturbed rows replays byte for byte too."""
    folder, ctx = perturbed_folder(tmp_path, caplog, rewriter())
    again = tmp_path / "replay"
    again.mkdir()
    ctx = dataclasses.replace(ctx, send=lambda: _no_send)
    argv = ["score", "g2", "--replay", str(folder / "calls.jsonl")]
    argv += ["--log", str(again / "calls.jsonl")]
    argv += ["--candidates", str(again / "holdout-candidates.jsonl")]
    out = io.StringIO()
    assert main([*argv, "--data-dir", str(tmp_path / "data")], context=ctx, out=out) == 0
    assert out.getvalue() == (folder / "scores.csv").read_text(encoding="utf-8")
    candidates = (again / "holdout-candidates.jsonl").read_bytes()
    assert candidates == (folder / "holdout-candidates.jsonl").read_bytes()
    logged = (folder / "calls.jsonl").read_text(encoding="utf-8")
    assert json.dumps(COUNTS_HEADER)[1:-1] in logged
    assert any(m.startswith("perturb call=1 ") for m in caplog.messages)
