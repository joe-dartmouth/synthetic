"""A holdout scored on a shared split: excluded records, its marker row, and `beside`'s pairing."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import re
import shutil
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

import bko.evaluate.run as run_module
from bko.datasets.sources import UCI_HEART_DISEASE
from bko.datasets.split import split
from bko.datasets.uci import FRAME_COLUMNS, UCI_FRAME
from bko.evaluate.__main__ import main
from bko.evaluate.beside import Pairing, pairable
from bko.evaluate.context import Context
from bko.evaluate.run import score_holdout
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source, render
from tests._evaluate_bko_fixtures import REPLICAS, context, replica_script, scripted
from tests._evaluate_fixtures import FAST_ARMS, GRADED, SMALL
from tests._generate_fixtures import ScriptedSend


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

PAIR = dataclasses.replace(
    SMALL, arms=("none", "smote"), candidate="smote", comparators=("none",), replica_arms=()
)


def fits_on(monkeypatch: pytest.MonkeyPatch) -> list[list[int]]:
    """Record the test index every classifier in the harness is scored on."""
    seen: list[list[int]] = []
    real = run_module.probabilities

    def spy(training, test, **options):
        seen.append(list(test.index))
        return real(training, test, **options)

    monkeypatch.setattr(run_module, "probabilities", spy)
    return seen


def test_no_exclusion_keeps_todays_bytes(caplog):
    parts = split(GRADED, seed=0, edges=UCI_FRAME.edges)
    today = score_holdout(parts, PAIR, FAST_ARMS, frame=UCI_FRAME)
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        unshared = score_holdout(parts, PAIR, FAST_ARMS, frame=UCI_FRAME, exclude=None)
    assert render(unshared) == render(today)
    assert all(s.measure is not Measure.SHARED_SPLIT for s in unshared)
    assert "shared holdout=" not in caplog.text


def test_an_excluded_record_leaves_the_split_and_the_marker_fingerprints_it(monkeypatch, caplog):
    parts = split(GRADED, seed=0, edges=UCI_FRAME.edges)
    foreign = parts.test.iloc[[3]].loc[:, list(FRAME_COLUMNS)].reset_index(drop=True)
    seen = fits_on(monkeypatch)
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        scores = score_holdout(parts, PAIR, FAST_ARMS, frame=UCI_FRAME, exclude=[foreign])
    gone = parts.test.index[3]
    kept = sorted(int(i) for i in parts.test.index if i != gone)
    assert seen
    assert all(index == kept for index in seen)
    digits = sha256(repr(kept).encode()).hexdigest()[:12]
    held = [s for s in scores if s.source is Source.HELD_OUT]
    assert held[-1] == Score(
        0, HELD_OUT_ARM, Source.HELD_OUT, Measure.SHARED_SPLIT, Scope.ALL, float(int(digits, 16))
    )
    assert (
        f"shared holdout=0 records={len(parts.test)} copied=0 foreign=1 fingerprint={digits}"
        in caplog.text
    )
    unshared = score_holdout(parts, PAIR, FAST_ARMS, frame=UCI_FRAME)
    nothing = score_holdout(parts, PAIR, FAST_ARMS, frame=UCI_FRAME, exclude=[foreign.iloc[:0]])
    assert render([s for s in nothing if s.measure is not Measure.SHARED_SPLIT]) == render(unshared)


def holdout_rows(holdout, *, copies=0.0, marker=None):
    """One holdout of a decontaminating file: held-out rows, an arm's copies and its recall."""
    held = {Measure.HELD_OUT_RATIO: 0.25, Measure.DCR_P5: 0.02, Measure.DCR_MEDIAN: 0.09}
    if marker is not None:
        held[Measure.SHARED_SPLIT] = marker
    return [
        *(Score(holdout, HELD_OUT_ARM, Source.HELD_OUT, m, Scope.ALL, v) for m, v in held.items()),
        Score(holdout, "smote", Source.ADDED, Measure.TEST_COPIES, Scope.ALL, copies),
        Score(holdout, "smote", Source.AUGMENTED, Measure.RECALL, Scope.ALL, 0.5 + holdout / 100),
    ]


def scored(copied, markers):
    """Four holdouts; `copied` holds a copy, and `markers` maps a holdout to its marker value."""
    return [
        row
        for holdout in range(4)
        for row in holdout_rows(
            holdout, copies=float(holdout in copied), marker=markers.get(holdout)
        )
    ]


def test_beside_pairs_copied_holdouts_when_both_files_share_the_split():
    markers = dict.fromkeys(range(4), 123456.0)
    ours, theirs = scored({1}, markers), scored({2}, markers)
    assert pairable(ours, theirs) == Pairing(
        frozenset(range(4)), shared=4, copied=0, mismatched=0, copies_excluded=False
    )
    other = scored({2}, {**markers, 2: 654321.0, 3: 654321.0})
    assert pairable(ours, other) == Pairing(frozenset({0, 1}), shared=4, copied=1, mismatched=1)


def test_a_shared_file_never_pairs_with_an_unshared_one():
    shared = scored(set(), dict.fromkeys(range(4), 123456.0))
    for unshared in (scored(set(), {}), scored({1}, {})):
        assert pairable(shared, unshared).holdouts == frozenset()
        assert pairable(unshared, shared).holdouts == frozenset()
        assert pairable(shared, unshared, allow_copies=True).holdouts == frozenset()


FORMAT = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
"""The line format `configure_logging` gives a run folder's `scores.log`."""


def run_folder(tmp_path: Path, name: str, caplog, *, replicas: bool = False) -> Path:
    """A paid run's folder: `scores.csv`, `scores.log` and `holdout-candidates.jsonl`.

    The run is `NARROW` over `scripted()` replies, or `REPLICAS` over `replica_script()`, whose
    replica rows copy test records on each holdout.
    """
    ctx = rescorer(replicas=replicas, send=replica_script() if replicas else scripted())
    folder = tmp_path / name
    folder.mkdir()
    argv = ["score", "--log", str(folder / "calls.jsonl")]
    argv += ["--candidates", str(folder / "holdout-candidates.jsonl")]
    argv += ["--data-dir", str(tmp_path / "data")]
    out = io.StringIO()
    caplog.set_level(logging.INFO)
    caplog.clear()
    assert main(argv, context=ctx, out=out) == 0
    (folder / "scores.csv").write_text(out.getvalue(), encoding="utf-8")
    lines = "".join(FORMAT.format(record) + "\n" for record in caplog.records)
    (folder / "scores.log").write_text(lines, encoding="utf-8")
    caplog.clear()
    return folder


def rescorer(*, replicas: bool = False, send: ScriptedSend | None = None) -> Context:
    """The run's context; with no `send`, any call fails, so a rescore that called would show."""
    ctx, _ = context(send or ScriptedSend())
    return dataclasses.replace(ctx, experiment=REPLICAS) if replicas else ctx


def shared(tmp_path: Path, folder: Path, *others: Path, replicas: bool = False) -> tuple[int, str]:
    """Run `shared` over `folder` with `others` as `--with`; return its exit code and stdout."""
    argv = ["shared", "--folder", str(folder), "--data-dir", str(tmp_path / "data")]
    for other in others:
        argv += ["--with", str(other)]
    out = io.StringIO()
    code = main(argv, context=rescorer(replicas=replicas), out=out)
    return code, out.getvalue()


def edit_log(folder: Path, prefix: str, edit) -> None:
    """Rewrite the first `scores.log` line whose message starts with `prefix` through `edit`."""
    path = folder / "scores.log"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if f": {prefix}" in line)
    lines[index] = edit(lines[index])
    path.write_text("".join(lines), encoding="utf-8")


def markers(text: str) -> dict[int, float]:
    """Each holdout's `shared-split` value in a scores CSV."""
    frame = pd.read_csv(io.StringIO(text))
    rows = frame[frame["measure"] == "shared-split"]
    return dict(zip(rows["holdout"], rows["value"], strict=True))


def test_rescoring_with_no_with_rewrites_the_runs_scores_bytes(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog, replicas=True)
    code, text = shared(tmp_path, folder, replicas=True)
    assert code == 0
    assert text == (folder / "scores.csv").read_text(encoding="utf-8")
    assert "shared holdout=" not in caplog.text


def test_the_shared_split_matches_a_joint_run(tmp_path, caplog, monkeypatch):
    # The replica run's arms hold the narrow run's rows, so its own split is the joint run's.
    seen: list[tuple[int, ...]] = []
    real = run_module.probabilities

    def spy(training, test, **options):
        index = tuple(int(i) for i in test.index)
        if not seen or seen[-1] != index:
            seen.append(index)
        return real(training, test, **options)

    monkeypatch.setattr(run_module, "probabilities", spy)
    joint = run_folder(tmp_path, "joint", caplog, replicas=True)
    monkeypatch.setattr(run_module, "probabilities", real)
    narrow = run_folder(tmp_path, "narrow", caplog)
    expected = {
        holdout: float(int(sha256(repr(sorted(index)).encode()).hexdigest()[:12], 16))
        for holdout, index in zip(SMALL.holdout_seeds, seen, strict=True)
    }
    code, ours = shared(tmp_path, narrow, joint)
    assert code == 0
    assert markers(ours) == expected
    assert re.search(r"shared holdout=0 records=\d+ copied=0 foreign=[1-9]", caplog.text)
    code, theirs = shared(tmp_path, joint, narrow, replicas=True)
    assert code == 0
    assert markers(theirs) == expected


def test_a_rebuilt_arm_must_equal_its_logged_sha256(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)
    edited = tmp_path / "edited"
    shutil.copytree(folder, edited)
    edit_log(edited, "arm holdout=0 arm=bko ", lambda line: line.replace("sha256=", "sha256=0"))
    code, text = shared(tmp_path, edited)
    assert (code, text) == (3, "")
    assert "arm holdout=0 arm=bko " in caplog.text
    path = folder / "holdout-candidates.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = next(i for i, line in enumerate(lines) if '"outcome": "kept"' in line)
    lines[kept] = lines[kept].replace('"outcome": "kept"', '"outcome": "dropped-quota"')
    path.write_text("".join(lines), encoding="utf-8")
    assert shared(tmp_path, folder) == (3, "")


def test_a_schema_1_file_rescores_like_schema_2(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)
    path = folder / "holdout-candidates.jsonl"
    first, rest = path.read_text(encoding="utf-8").split("\n", 1)
    header = json.loads(first)
    del header["frame"], header["data"]
    header |= {"schema": 1, "uci": UCI_HEART_DISEASE.sha256}
    path.write_text(json.dumps(header, sort_keys=True) + "\n" + rest, encoding="utf-8")
    log = folder / "scores.log"
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    log.write_text("".join(line for line in lines if ": data " not in line), encoding="utf-8")
    code, text = shared(tmp_path, folder)
    assert code == 0
    assert text == (folder / "scores.csv").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("prefix", "old", "new"),
    [
        ("data ", "digest=", "digest=0"),
        ("split seed=1 ", "fingerprint=", "fingerprint=0"),
        ("arm holdout=1 arm=none ", "seed=", "seed=9"),
        ("experiment ", "fill_to=20", "fill_to=21"),
    ],
    ids=["data", "split", "baseline", "experiment"],
)
def test_other_data_or_holdouts_are_refused_with_exit_3(tmp_path, caplog, prefix, old, new):
    folder = run_folder(tmp_path, "run", caplog)
    other = tmp_path / "other"
    shutil.copytree(folder, other)
    edit_log(other, prefix, lambda line: line.replace(old, new, 1))
    assert shared(tmp_path, folder, other) == (3, "")
    assert shared(tmp_path, other) == (3, "")


@pytest.mark.parametrize("missing", ["scores.log", "holdout-candidates.jsonl"])
def test_a_missing_file_exits_2(tmp_path, caplog, missing):
    folder = run_folder(tmp_path, "run", caplog)
    other = tmp_path / "other"
    shutil.copytree(folder, other)
    (other / missing).unlink()
    assert shared(tmp_path, other) == (2, "")
    assert shared(tmp_path, folder, other) == (2, "")
    assert shared(tmp_path, folder, tmp_path / "absent") == (2, "")
