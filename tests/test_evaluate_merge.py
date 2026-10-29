"""`python -m bko.evaluate merge`: holdout slices' scores join into the bytes one run writes."""

from __future__ import annotations

import dataclasses
import io
import logging

from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.experiment import NamedRun
from bko.evaluate.run import ARMS, run
from bko.evaluate.scores import render
from tests._frame_fixtures import TOY, toy_records
from tests.test_evaluate_frame import TOY_EXPERIMENT


WHOLE = dataclasses.replace(TOY_EXPERIMENT, holdout_seeds=(0, 1, 2, 3))
CONTEXT = Context(runs={"toy": NamedRun(WHOLE)})


def _scored(seeds: tuple[int, ...]) -> str:
    experiment = dataclasses.replace(TOY_EXPERIMENT, holdout_seeds=seeds)
    arms = {name: ARMS[name] for name in experiment.arms}
    return render(run(toy_records(), experiment, arms, frame=TOY))


def _merged(tmp_path, texts: list[str]) -> tuple[int, str]:
    paths = []
    for i, text in enumerate(texts):
        path = tmp_path / f"scores-{i}.csv"
        path.write_text(text, encoding="utf-8")
        paths.append(str(path))
    out = io.StringIO()
    code = main(["merge", "toy", *paths], context=CONTEXT, out=out)
    return code, out.getvalue()


def test_merging_two_slices_equals_one_run(tmp_path, caplog):
    whole = _scored((0, 1, 2, 3))
    low, high = _scored((0, 1)), _scored((2, 3))
    with caplog.at_level(logging.INFO):
        code, merged = _merged(tmp_path, [high, low])
    assert code == 0
    assert merged == whole
    assert any(m.startswith("merge files=2 holdouts=4 rows=") for m in caplog.messages)


def test_merge_refuses_overlapping_or_missing_holdouts_or_other_headers(tmp_path):
    low, high = _scored((0, 1)), _scored((2, 3))
    overlap = _scored((1, 2, 3))
    cases = [
        ([low, overlap], 2),
        ([low], 2),
        ([low, low], 2),
        ([low, high.replace("holdout,arm", "holdout,team", 1)], 3),
        ([low, high.replace("holdout,arm,", "arm,holdout,", 1)], 3),
    ]
    for texts, code in cases:
        got, out = _merged(tmp_path, texts)
        assert (got, out) == (code, "")
    extra = _scored((4, 5))
    assert _merged(tmp_path, [low, high, extra]) == (2, "")
