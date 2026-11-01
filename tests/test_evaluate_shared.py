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
from bko.evaluate.experiment import REAL_ARM, Experiment, Removal
from bko.evaluate.run import score_holdout
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source, render
from bko.evaluate.shared import BKO
from tests._evaluate_bko_fixtures import NARROW, REPLICAS, context, replica_script, scripted
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


def run_folder(
    tmp_path: Path,
    name: str,
    caplog,
    *,
    replicas: bool = False,
    experiment: Experiment | None = None,
) -> Path:
    """A paid run's folder: `scores.csv`, `scores.log` and `holdout-candidates.jsonl`.

    The run is `NARROW` over `scripted()` replies, or `REPLICAS` over `replica_script()`, whose
    replica rows copy test records on each holdout, or `experiment` over `scripted()` replies.
    """
    send = replica_script() if replicas else scripted()
    ctx = rescorer(replicas=replicas, experiment=experiment, send=send)
    folder = tmp_path / name
    folder.mkdir()
    argv = ["score", "--data-dir", str(tmp_path / "data")]
    if BKO & set(ctx.experiment.arms):
        argv += ["--log", str(folder / "calls.jsonl")]
        argv += ["--candidates", str(folder / "holdout-candidates.jsonl")]
    out = io.StringIO()
    caplog.set_level(logging.INFO)
    caplog.clear()
    assert main(argv, context=ctx, out=out) == 0
    (folder / "scores.csv").write_text(out.getvalue(), encoding="utf-8")
    lines = "".join(FORMAT.format(record) + "\n" for record in caplog.records)
    (folder / "scores.log").write_text(lines, encoding="utf-8")
    caplog.clear()
    return folder


def rescorer(
    *,
    replicas: bool = False,
    experiment: Experiment | None = None,
    send: ScriptedSend | None = None,
) -> Context:
    """The run's context; with no `send`, any call fails, so a rescore that called would show."""
    ctx, _ = context(send or ScriptedSend())
    chosen = REPLICAS if replicas else experiment
    return ctx if chosen is None else dataclasses.replace(ctx, experiment=chosen)


def shared(
    tmp_path: Path,
    folder: Path,
    *others: Path,
    replicas: bool = False,
    experiment: Experiment | None = None,
) -> tuple[int, str]:
    """Run `shared` over `folder` with `others` as `--with`; return its exit code and stdout."""
    argv = ["shared", "--folder", str(folder), "--data-dir", str(tmp_path / "data")]
    for other in others:
        argv += ["--with", str(other)]
    out = io.StringIO()
    ctx = rescorer(replicas=replicas, experiment=experiment)
    code = main(argv, context=ctx, out=out)
    return code, out.getvalue()


def refused(tmp_path: Path, caplog, folder: Path, *others: Path) -> str:
    """Run `shared` as `shared` does, assert it exits 3 with no output, and return its error."""
    caplog.clear()
    assert shared(tmp_path, folder, *others) == (3, "")
    (error,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    return error


def edit_log(folder: Path, prefix: str, edit) -> None:
    """Rewrite the first `scores.log` line whose message starts with `prefix` through `edit`."""
    path = folder / "scores.log"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if f": {prefix}" in line)
    lines[index] = edit(lines[index])
    path.write_text("".join(lines), encoding="utf-8")


def edit_candidates(folder: Path, edit) -> None:
    """Rewrite each line of the candidates file, parsed, through `edit`, header first."""
    path = folder / "holdout-candidates.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    text = "".join(json.dumps(edit(line), sort_keys=True) + "\n" for line in lines)
    path.write_text(text, encoding="utf-8")


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
    error = refused(tmp_path, caplog, edited)
    assert re.fullmatch(
        r"rescored 'arm holdout=0 arm=bko .* sha256=[0-9a-f]+ .*' differs from the logged "
        r"'arm holdout=0 arm=bko .* sha256=0[0-9a-f]+ .*'",
        error,
    )
    path = folder / "holdout-candidates.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = next(i for i, line in enumerate(lines) if '"outcome": "kept"' in line)
    lines[kept] = lines[kept].replace('"outcome": "kept"', '"outcome": "dropped-quota"')
    path.write_text("".join(lines), encoding="utf-8")
    error = refused(tmp_path, caplog, folder)
    assert re.fullmatch(r"rescored 'arm holdout=0 arm=bko .*' differs from the logged '.*'", error)


def test_a_with_folders_rebuilt_arm_must_equal_its_logged_sha256(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)
    other = tmp_path / "other"
    shutil.copytree(folder, other)
    edit_log(other, "arm holdout=0 arm=bko ", lambda line: line.replace("sha256=", "sha256=0"))
    error = refused(tmp_path, caplog, folder, other)
    assert error.startswith(f"{other}'s rebuilt rows (")
    assert "differ from its line 'arm holdout=0 arm=bko " in error


def test_a_with_folder_missing_a_bko_arm_line_is_refused(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)
    other = tmp_path / "other"
    shutil.copytree(folder, other)
    log = other / "scores.log"
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if ": arm holdout=1 arm=bko " not in line]
    assert len(kept) == len(lines) - 1
    log.write_text("".join(kept), encoding="utf-8")
    error = refused(tmp_path, caplog, folder, other)
    assert error == f"{other / 'scores.log'} logs no arm line for holdout 1's bko"


def test_a_fill_with_no_returned_attempt_is_refused_as_a_shortfall(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)

    def unreturned(line):
        if line["kind"] == "candidate" and (line["holdout"], line["fill"]) == (0, "gap"):
            line["returned"] = False
        return line

    edit_candidates(folder, unreturned)
    error = refused(tmp_path, caplog, folder)
    assert error == (
        "holdout 0's bko gap fill has no returned attempt: a shortfall's partial fill, "
        "whose rows a rescore does not rebuild"
    )


def test_malformed_params_are_refused_with_exit_3(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)
    edit_log(folder, "arm holdout=0 arm=bko ", lambda line: line.replace("params={", "params={,"))
    error = refused(tmp_path, caplog, folder)
    assert re.fullmatch(
        r"the params of 'arm holdout=0 arm=bko .* params=\{,.*' are not a JSON object", error
    )


def test_a_run_with_the_real_arm_rescores_its_bytes(tmp_path, caplog):
    real = dataclasses.replace(
        NARROW,
        arms=("none", REAL_ARM),
        candidate=REAL_ARM,
        comparators=("none",),
        criteria=(),
        removal=Removal(("<55", "female", 1), 0.5),
        decontaminate=False,
    )
    folder = run_folder(tmp_path, "run", caplog, experiment=real)
    code, text = shared(tmp_path, folder, experiment=real)
    assert code == 0
    assert text == (folder / "scores.csv").read_text(encoding="utf-8")


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


def test_other_data_is_refused_with_exit_3(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)
    other = tmp_path / "other"
    shutil.copytree(folder, other)
    ours = json.loads((folder / "holdout-candidates.jsonl").read_text().split("\n", 1)[0])["data"]
    edit_log(other, "data ", lambda line: line.replace("digest=", "digest=0", 1))
    edit_candidates(other, lambda line: line | {"data": "0" + ours} if "data" in line else line)
    expected = f"{other} holds the data digest 0{ours}, not the loaded data's {ours}"
    assert refused(tmp_path, caplog, folder, other) == expected
    assert refused(tmp_path, caplog, other) == expected


@pytest.mark.parametrize(
    ("prefix", "old", "new", "beside", "alone"),
    [
        (
            "split seed=1 ",
            "fingerprint=",
            "fingerprint=0",
            r"\S+'s split lines differ from \S+'s",
            r"holdout 1's split lines \['split seed=1 .*'\] differ from \S+'s 'split seed=1 .*'",
        ),
        (
            "arm holdout=1 arm=none ",
            "seed=",
            "seed=9",
            r"\S+'s baseline arm lines differ from \S+'s: .*",
            r"rescored 'arm holdout=1 arm=none .*' differs from the logged 'arm holdout=1 .*'",
        ),
        (
            "experiment ",
            "fill_to=20",
            "fill_to=21",
            r"the experiment lines of \S+ and \S+ differ: \{'fill_to': \('20', '21'\)\}",
            (
                r"the experiment lines of \S+ and the run 'e1' now differ: "
                r"\{'fill_to': \('21', '20'\)\}"
            ),
        ),
    ],
    ids=["split", "baseline", "experiment"],
)
def test_other_holdouts_are_refused_with_exit_3(tmp_path, caplog, prefix, old, new, beside, alone):
    folder = run_folder(tmp_path, "run", caplog)
    other = tmp_path / "other"
    shutil.copytree(folder, other)
    edit_log(other, prefix, lambda line: line.replace(old, new, 1))
    assert re.fullmatch(beside, refused(tmp_path, caplog, folder, other))
    assert re.fullmatch(alone, refused(tmp_path, caplog, other))


@pytest.mark.parametrize("missing", ["scores.log", "holdout-candidates.jsonl"])
def test_a_missing_file_exits_2(tmp_path, caplog, missing):
    folder = run_folder(tmp_path, "run", caplog)
    other = tmp_path / "other"
    shutil.copytree(folder, other)
    (other / missing).unlink()
    assert shared(tmp_path, other) == (2, "")
    assert shared(tmp_path, folder, other) == (2, "")
    assert shared(tmp_path, folder, tmp_path / "absent") == (2, "")
