"""`python -m bko.evaluate merge`: holdout slices' scores join into the bytes one run writes."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import logging

import pytest

from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.experiment import NamedRun
from bko.evaluate.run import ARMS, run
from bko.evaluate.scores import parse, render
from tests._frame_fixtures import TOY, toy_records
from tests.test_evaluate_frame import TOY_EXPERIMENT


WHOLE = dataclasses.replace(TOY_EXPERIMENT, holdout_seeds=(0, 1, 2, 3))
CONTEXT = Context(runs={"toy": NamedRun(WHOLE)})


def _scored(seeds: tuple[int, ...]) -> str:
    experiment = dataclasses.replace(TOY_EXPERIMENT, holdout_seeds=seeds)
    arms = {name: ARMS[name] for name in experiment.arms}
    return render(run(toy_records(), experiment, arms, frame=TOY))


LOW, HIGH, OVERLAP, EXTRA = _scored((0, 1)), _scored((2, 3)), _scored((1, 2, 3)), _scored((4, 5))
"""Slices of the toy run's holdouts, scored once for every test here."""


def _merged(tmp_path, texts: list[str], *, context: Context = CONTEXT) -> tuple[int, str]:
    paths = []
    for i, text in enumerate(texts):
        path = tmp_path / f"scores-{i}.csv"
        path.write_text(text, encoding="utf-8")
        paths.append(str(path))
    out = io.StringIO()
    code = main(["merge", "toy", *paths], context=context, out=out)
    return code, out.getvalue()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_merging_two_slices_equals_one_run(tmp_path, caplog):
    whole = _scored((0, 1, 2, 3))
    with caplog.at_level(logging.INFO):
        code, merged = _merged(tmp_path, [HIGH, LOW])
    assert code == 0
    assert merged == whole
    lines = [m for m in caplog.messages if m.startswith("merge ")]
    rows = [len(parse(text)) for text in (HIGH, LOW, whole)]
    assert lines == [
        f"merge file={tmp_path / 'scores-0.csv'} holdouts=2-3 rows={rows[0]} sha256={_sha(HIGH)}",
        f"merge file={tmp_path / 'scores-1.csv'} holdouts=0-1 rows={rows[1]} sha256={_sha(LOW)}",
        f"merge files=2 holdouts=4 rows={rows[2]} sha256={_sha(whole)}",
    ]


def test_rows_follow_the_runs_seed_order_and_each_files_order_within_a_holdout(tmp_path):
    context = Context(
        runs={"toy": NamedRun(dataclasses.replace(WHOLE, holdout_seeds=(2, 3, 0, 1)))}
    )
    reversed_low = "".join(
        [LOW.splitlines(keepends=True)[0], *LOW.splitlines(keepends=True)[:0:-1]]
    )
    code, merged = _merged(tmp_path, [reversed_low, HIGH], context=context)
    assert code == 0
    found = parse(merged)
    assert [s.holdout for s in found] == sorted((s.holdout for s in found), key=(2, 3, 0, 1).index)
    expected = render(s for s in parse(reversed_low) if s.holdout == 0)
    assert render(s for s in found if s.holdout == 0) == expected
    assert expected != render(s for s in parse(LOW) if s.holdout == 0)


@pytest.mark.parametrize(
    ("texts", "code", "fragment"),
    [
        pytest.param([LOW, OVERLAP], 2, "is in both", id="overlap"),
        pytest.param([LOW], 2, "missing [2, 3]", id="missing-holdouts"),
        pytest.param([LOW, LOW], 2, "is in both", id="same-file-twice"),
        pytest.param([LOW, HIGH, EXTRA], 2, "not the run's [4, 5]", id="extra-holdouts"),
        pytest.param(
            [LOW, HIGH.replace("holdout,arm", "holdout,team", 1)], 3, "columns", id="renamed"
        ),
        pytest.param(
            [LOW, HIGH.replace("holdout,arm,", "arm,holdout,", 1)], 3, "columns", id="reordered"
        ),
    ],
)
def test_merge_refuses_files_one_run_could_not_write(tmp_path, caplog, texts, code, fragment):
    with caplog.at_level(logging.INFO):
        assert _merged(tmp_path, texts) == (code, "")
    assert fragment in caplog.text


def test_merge_refuses_slices_whose_holdouts_hold_other_scores(tmp_path, caplog):
    lines = HIGH.splitlines(keepends=True)
    fewer = "".join(line for line in lines if ",flagged-fingerprint," not in line)
    assert fewer != HIGH
    with caplog.at_level(logging.INFO):
        assert _merged(tmp_path, [LOW, fewer]) == (2, "")
    assert "holdout 2 of" in caplog.text
    assert "other scores than holdout 0 of" in caplog.text


def test_merge_of_a_missing_file_exits_two(tmp_path):
    out = io.StringIO()
    missing = str(tmp_path / "absent.csv")
    path = tmp_path / "low.csv"
    path.write_text(LOW, encoding="utf-8")
    assert main(["merge", "toy", str(path), missing], context=CONTEXT, out=out) == 2
    assert out.getvalue() == ""
