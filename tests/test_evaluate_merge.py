"""`python -m bko.evaluate merge`: holdout slices' scores join into the bytes one run writes."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import logging
from fractions import Fraction
from typing import Final

import pytest

from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.experiment import Experiment, Removal
from bko.evaluate.named_runs import NamedRun
from bko.evaluate.provenance import experiment_line
from bko.evaluate.run import ARMS, run
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source, parse, render
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


DATA_LINE = f"data dataset=toy frame=toy records=150 digest={'a' * 64} agreement=none"
VERSIONS_LINE = 'versions commit=0123abc dirty=false {"python": "3.14.6"}'


def scores_log(
    experiment: Experiment,
    *,
    name: str,
    data: str = DATA_LINE,
    versions: str = VERSIONS_LINE,
) -> str:
    """A slice's `scores.log`: its `data`, `experiment` and `versions` lines among others."""
    messages = [
        data,
        experiment_line(name, experiment, ("lr", "hgb"), frame="toy"),
        versions,
        "split seed=0 train=1 test=1 fingerprint=x",
    ]
    return "".join(f"2026-11-02 10:00:00,000 INFO bko.evaluate: {m}\n" for m in messages)


def _slice_logs(texts: list[str]) -> list[str]:
    """Each slice's log as one run's slices write them: own name, seeds and spend ceiling."""
    logs = []
    for i, text in enumerate(texts):
        held = tuple(sorted({int(row.split(",", 1)[0]) for row in text.splitlines()[1:]})) or (0,)
        experiment = dataclasses.replace(
            WHOLE, holdout_seeds=held, spend_ceiling=WHOLE.spend_ceiling + i
        )
        logs.append(scores_log(experiment, name=f"toy-{'ab'[i % 2]}{i}"))
    return logs


def _merged(
    tmp_path,
    texts: list[str],
    *,
    context: Context = CONTEXT,
    logs: list[str | None] | None = None,
) -> tuple[int, str]:
    """Merge `texts`, each in its own folder beside its log; a None log writes none."""
    written = _slice_logs(texts) if logs is None else logs
    paths = []
    for i, (text, log) in enumerate(zip(texts, written, strict=True)):
        folder = tmp_path / f"slice-{i}"
        folder.mkdir()
        path = folder / "scores.csv"
        path.write_text(text, encoding="utf-8")
        if log is not None:
            (folder / "scores.log").write_text(log, encoding="utf-8")
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
    high, low = (tmp_path / f"slice-{i}" / "scores.csv" for i in (0, 1))
    assert lines == [
        f"merge file={high} holdouts=2-3 rows={rows[0]} sha256={_sha(HIGH)}",
        f"merge file={low} holdouts=0-1 rows={rows[1]} sha256={_sha(LOW)}",
        f"merge files=2 holdouts=4 rows={rows[2]} sha256={_sha(whole)}",
    ]


def test_a_shared_split_row_passes_through_a_merge_unchanged(tmp_path):
    def marked(text: str, seeds: tuple[int, ...]) -> str:
        rows = [
            Score(
                seed,
                HELD_OUT_ARM,
                Source.HELD_OUT,
                Measure.SHARED_SPLIT,
                Scope.ALL,
                float(seed + 7),
            )
            for seed in seeds
        ]
        return render([*parse(text), *rows])

    low, high = marked(LOW, (0, 1)), marked(HIGH, (2, 3))
    code, merged = _merged(tmp_path, [high, low])
    assert code == 0
    found = {s.holdout: s.value for s in parse(merged) if s.measure is Measure.SHARED_SPLIT}
    assert found == {0: 7.0, 1: 8.0, 2: 9.0, 3: 10.0}
    assert len(parse(merged)) == len(parse(low)) + len(parse(high))


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


def _without_distances(text: str, *, rows: str | None) -> str:
    """`text` with holdout 3's `smote` distance rows dropped, and its added count `rows` if set."""
    kept = [
        line
        for line in text.splitlines(keepends=True)
        if not line.startswith(("3,smote,added,dcr-p5,", "3,smote,added,dcr-median,"))
    ]
    if rows is not None:
        kept = [
            f"3,smote,added,rows,all,{rows}\n" if line.startswith("3,smote,added,rows,") else line
            for line in kept
        ]
    return "".join(kept)


def test_an_arm_that_added_no_row_on_one_holdout_still_merges(tmp_path):
    emptied = _without_distances(HIGH, rows="0.0")
    assert len(emptied.splitlines()) == len(HIGH.splitlines()) - 2
    code, merged = _merged(tmp_path, [LOW, emptied])
    assert code == 0
    assert merged == LOW + "".join(emptied.splitlines(keepends=True)[1:])


def test_an_arm_that_added_rows_but_lacks_their_distances_is_refused(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        assert _merged(tmp_path, [LOW, _without_distances(HIGH, rows=None)]) == (2, "")
    assert "holdout 3 of" in caplog.text
    assert "other scores than holdout 0 of" in caplog.text


E3_DATA: Final = (
    "data dataset=synthea frame=synthea records=985 "
    "digest=f9c6a23e30036a12a8647903783cac6cd47eac3f58e40d82b09b7d572898b94f"
)
E3_VERSIONS: Final = (
    "versions commit=8a50dc6e8b0392724ed7079c76891fcd1835c654 dirty=false "
    '{"anthropic": "1.13.0", "imbalanced-learn": "0.14.2", "numpy": "2.5.4", '
    '"pandas": "2.3.3", "platform": "macOS-27.0.1-arm64-arm-64bit-Mach-O", "pronto": "2.7.3", '
    '"python": "3.14.6", "scikit-learn": "1.9.1", "scipy": "1.18.1", "sdmetrics": "0.32.0", '
    '"sdv": "1.38.5", "threadpoolctl": "3.7.0", "torch": "2.14.1"}'
)
E3_EXPERIMENT: Final = (
    "experiment run=e3-{slice} frame=synthea holdouts=2 seeds={seeds} "
    "removal=>=65/female/1@0.9 decontaminate=false "
    "arms=none,class-weight,smote,adasyn,ctgan,tvae,bko,bko-population,bko-frequency,real "
    "replica=smote,adasyn,ctgan,tvae,bko,bko-population,bko-frequency fill_to=20 budget=3 "
    "validity=19/20 spend_ceiling=32.50 criteria=none models=lr,hgb"
)
"""`runs/e3-2/scores.log`'s `data`, `versions` and `experiment` lines as recorded, but for the
holdouts and seeds: a line that predates `agreement=`, `training_fraction=`, `threshold=`,
`cutoff=` and `replica_rows=`, each read at its assumed value."""


def _recorded(slice_: str, seeds: str) -> str:
    messages = [E3_DATA, E3_EXPERIMENT.format(slice=slice_, seeds=seeds), E3_VERSIONS]
    return "".join(f"2026-10-28 22:45:36,232 INFO bko.evaluate: {m}\n" for m in messages)


def test_merge_joins_slices_whose_lines_are_in_the_recorded_older_format(tmp_path):
    logs: list[str | None] = [_recorded("a", "0-1"), _recorded("b", "2-3")]
    for key in ("agreement=", "training_fraction=", "threshold=", "cutoff=", "replica_rows="):
        assert key not in str(logs[0])
    code, merged = _merged(tmp_path, [LOW, HIGH], logs=logs)
    assert code == 0
    assert merged == _scored((0, 1, 2, 3))


def test_merge_refuses_a_recorded_format_slice_beside_one_of_another_cutoff(tmp_path, caplog):
    other = _recorded("b", "2-3").replace(" fill_to=20 ", " fill_to=20 cutoff=0.3 ")
    with caplog.at_level(logging.INFO):
        assert _merged(tmp_path, [LOW, HIGH], logs=[_recorded("a", "0-1"), other]) == (2, "")
    assert "cutoff=0.5 vs 0.3" in caplog.text


def test_merge_of_a_missing_file_exits_two(tmp_path):
    out = io.StringIO()
    missing = str(tmp_path / "absent.csv")
    path = tmp_path / "low.csv"
    path.write_text(LOW, encoding="utf-8")
    (tmp_path / "scores.log").write_text(_slice_logs([LOW])[0], encoding="utf-8")
    assert main(["merge", "toy", str(path), missing], context=CONTEXT, out=out) == 2
    assert out.getvalue() == ""


def _refused(tmp_path, caplog, second: str) -> str:
    """Merge LOW and HIGH, HIGH's log `second`; the run must exit 2 with stdout empty."""
    first = _slice_logs([LOW, HIGH])[0]
    with caplog.at_level(logging.INFO):
        assert _merged(tmp_path, [LOW, HIGH], logs=[first, second]) == (2, "")
    return caplog.text


def _high(experiment: Experiment = WHOLE, **lines: str) -> str:
    """HIGH's log, its experiment `experiment` over HIGH's holdouts."""
    held = dataclasses.replace(experiment, holdout_seeds=(2, 3))
    return scores_log(held, name="toy-b", **lines)


def test_merge_refuses_slices_at_another_training_fraction(tmp_path, caplog):
    text = _refused(
        tmp_path, caplog, _high(dataclasses.replace(WHOLE, training_fraction=Fraction(1, 2)))
    )
    assert "training_fraction" in text
    assert str(tmp_path / "slice-1" / "scores.log") in text


def test_merge_refuses_slices_at_another_removal_fraction(tmp_path, caplog):
    removal = Removal(("45-69", "female", 1), 0.5)
    text = _refused(tmp_path, caplog, _high(dataclasses.replace(WHOLE, removal=removal)))
    assert "removal" in text


def test_merge_refuses_slices_at_another_cutoff(tmp_path, caplog):
    text = _refused(tmp_path, caplog, _high(dataclasses.replace(WHOLE, cutoff=0.3)))
    assert "cutoff" in text


def test_merge_refuses_slices_of_another_commit(tmp_path, caplog):
    other = VERSIONS_LINE.replace("0123abc", "4567def")
    text = _refused(tmp_path, caplog, _high(versions=other))
    assert "versions" in text


def test_merge_refuses_slices_of_other_data(tmp_path, caplog):
    text = _refused(tmp_path, caplog, _high(data=DATA_LINE.replace("a" * 64, "b" * 64)))
    assert "data" in text


def test_merge_refuses_a_slice_without_its_scores_log(tmp_path, caplog):
    first = _slice_logs([LOW, HIGH])[0]
    with caplog.at_level(logging.INFO):
        assert _merged(tmp_path, [LOW, HIGH], logs=[first, None]) == (2, "")
    assert f"{tmp_path / 'slice-1' / 'scores.log'}" in caplog.text


def test_merge_reads_an_older_line_at_the_values_it_lacks(tmp_path):
    lines = _high().splitlines(keepends=True)
    older = "".join(
        line.replace(" training_fraction=1", "").replace(" agreement=none", "") for line in lines
    )
    assert older != _high()
    first = _slice_logs([LOW, HIGH])[0]
    code, merged = _merged(tmp_path, [LOW, HIGH], logs=[first, older])
    assert code == 0
    assert merged == _scored((0, 1, 2, 3))
