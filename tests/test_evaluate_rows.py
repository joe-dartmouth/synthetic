"""`python -m bko.evaluate rows`: every fill of a checked holdout, stored with its sha256."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import math
import re
import shutil
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import TypedDict, Unpack

import numpy as np
import pandas as pd
import pytest

import bko.evaluate.rows as rows_module
from bko.baselines.methods import Runner
from bko.baselines.records import Baseline, number, unit_weights
from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.__main__ import main
from bko.evaluate.data import DIABETES_FRAME, SYNTHEA_FRAME, digest
from bko.evaluate.errors import IdentityError
from bko.evaluate.experiment import REAL_ARM, Experiment, Removal
from bko.evaluate.filling import Filled, arm_line
from bko.evaluate.rows import RowsFile, compare_holdout, held_out
from bko.evaluate.rows_read import RowsLine, bind, frame_of, read_rows
from bko.evaluate.run_folder import fields, read_folder
from bko.evaluate.scores import HELD_OUT_ARM, Measure, Scope, Score, Source
from tests._evaluate_bko_fixtures import NARROW
from tests._evaluate_fixtures import FAST_ARMS
from tests.test_evaluate_shared import edit_log, rescorer, run_folder


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

REAL = dataclasses.replace(
    NARROW,
    arms=("none", REAL_ARM),
    candidate=REAL_ARM,
    comparators=("none",),
    criteria=(),
    removal=Removal(("<55", "female", 1), 0.5),
    decontaminate=False,
)
"""A run whose `real` arm adds the removed rows, as `test_evaluate_shared` scores it."""

BASELINE = dataclasses.replace(
    NARROW,
    arms=("none", "smote"),
    candidate="smote",
    comparators=("none",),
    criteria=(),
    replica_arms=("smote",),
    decontaminate=False,
)
"""A run whose baseline also fills a replica; on `UCI_E1`'s repeated records SMOTE's rows copy
every test record, so it keeps them."""

BASELINE_ARMS = {"none": FAST_ARMS["none"], "smote": FAST_ARMS["smote"]}


class Kind(TypedDict, total=False):
    """`run_folder`'s and `rescorer`'s keywords for one folder kind."""

    replicas: bool
    experiment: Experiment
    arms: Mapping[str, Runner]


KINDS: dict[str, Kind] = {
    "replicas": {"replicas": True},
    "baseline": {"experiment": BASELINE, "arms": BASELINE_ARMS},
    "real": {"experiment": REAL},
}
"""Each folder kind's `run_folder` and `rescorer` keywords."""


def stored(tmp_path: Path, folder: Path, **kind: Unpack[Kind]) -> tuple[int, str]:
    """Run `rows` over `folder` into `rows.jsonl` beside it; return its exit code and stdout."""
    argv = ["rows", "--folder", str(folder), "--rows", str(tmp_path / "rows.jsonl")]
    argv += ["--data-dir", str(tmp_path / "data")]
    out = io.StringIO()
    code = main(argv, context=rescorer(**kind), out=out)
    return code, out.getvalue()


def holdouts(tmp_path: Path) -> set[int]:
    """The holdouts the store holds a fill line for."""
    return {holdout for holdout, _ in read_rows(tmp_path / "rows.jsonl").lines}


def edit_scores(folder: Path, prefix: str, value: str) -> None:
    """Set the value of the scores row starting `prefix`, and log the new file's sha256."""
    path = folder / "scores.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if line.startswith(prefix))
    lines[index] = f"{prefix}{value}\n"
    text = "".join(lines)
    path.write_text(text, encoding="utf-8")
    new = sha256(text.encode()).hexdigest()
    edit_log(folder, "scores rows=", lambda line: re.sub(r"sha256=\w+", f"sha256={new}", line))


def drop_score(folder: Path, prefix: str) -> None:
    """Delete the first scores row starting `prefix`, and log the new file's rows and sha256."""
    path = folder / "scores.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if line.startswith(prefix))
    text = "".join(lines[:index] + lines[index + 1 :])
    path.write_text(text, encoding="utf-8")
    new, rows = sha256(text.encode()).hexdigest(), text.count("\n") - 1
    resealed = f"scores rows={rows} sha256={new}"
    edit_log(
        folder, "scores rows=", lambda line: re.sub(r"scores rows=\d+ sha256=\w+", resealed, line)
    )


@pytest.mark.parametrize("kind", list(KINDS))
def test_rows_stores_every_fill_whose_sha256_is_its_arm_lines(tmp_path, caplog, kind):
    folder = run_folder(tmp_path, "run", caplog, **KINDS[kind])
    code, text = stored(tmp_path, folder, **KINDS[kind])
    assert code == 0
    assert text == (folder / "scores.csv").read_text(encoding="utf-8")
    arms = read_folder(folder).arms
    found = read_rows(tmp_path / "rows.jsonl").lines
    assert set(found) == set(arms)
    for key, line in found.items():
        logged = fields(arms[key])
        assert sha256(line.csv.encode()).hexdigest() == line.sha256 == logged["sha256"]
        assert str(line.added) == logged["added"]
        assert len(frame_of(line, arm_line=arms[key], frame=UCI_FRAME)) == line.added
        message = f"rows holdout={key[0]} arm={key[1]} added={line.added} sha256={line.sha256}"
        assert f"{message} matches_arm=true" in caplog.messages
    assert found[0, "none"].csv.count("\n") == 1
    names = {arm for _, arm in found}
    assert {"replicas": "bko-replica", "baseline": "smote-replica", "real": REAL_ARM}[kind] in names
    assert f"rows holdout=1 stored={len(arms) // 2} differing=0" in caplog.messages
    assert re.search(r"rows folder=\S+ run=e1 holdouts=2 started_utc=\S+ pid=\d+", caplog.text)


def records(frame: FrameSpec, count: int) -> pd.DataFrame:
    """`count` added records in `frame`'s columns: numbers in range, codes from the codebook."""
    rng = np.random.default_rng(0)
    bounds = {constraint.field: constraint for constraint in frame.ranges}
    columns: dict[str, object] = {}
    for name in frame.columns:
        if name in frame.fractional:
            columns[name] = rng.uniform(bounds[name].low, bounds[name].high, count)
        elif name in frame.numeric:
            low, high = int(bounds[name].low), int(bounds[name].high)
            columns[name] = rng.integers(low, high, count, endpoint=True)
        elif name == frame.label:
            columns[name] = rng.choice(np.array(frame.label_codes), count)
        else:
            columns[name] = rng.choice(np.array(list(frame.codebook[name])), count)
    return number(pd.DataFrame(columns).astype(dict(frame.dtypes)))


def filled(added: pd.DataFrame, *, holdout: int = 0, name: str = "ctgan") -> Filled:
    """A returned fill of `added` and the `arm` line it logs."""
    baseline = Baseline(added, unit_weights(added), {"k": 1})
    line = arm_line(baseline, holdout=holdout, name=name, attempt=1, seed=holdout)
    return Filled(name, "gap", 1, baseline, line)


def logged(*fills: Filled) -> dict[tuple[int, str], str]:
    """The folder's `arm` lines that hold exactly `fills`."""
    return {(int(fields(fill.line)["holdout"]), fill.name): fill.line for fill in fills}


HEADER = {"kind": "header", "schema": 1, "run": "e1", "folder": "run", "log_sha256": "0" * 64}


@pytest.mark.parametrize(
    "frame", [UCI_FRAME, SYNTHEA_FRAME, DIABETES_FRAME], ids=["uci", "synthea", "diabetes"]
)
def test_a_stored_fill_reads_back_to_its_digest(tmp_path, frame):
    added = records(frame, 7)
    if frame is UCI_FRAME:
        added.loc[-1, "oldpeak"] = 0.1 + 0.2
        added.loc[-2, "oldpeak"] = 1 / 3
        assert "0.30000000000000004" in added.to_csv()
    fill = filled(added)
    path = tmp_path / "rows.jsonl"
    with RowsFile(path, HEADER) as store:
        assert store.write(0, [fill], arms=logged(fill)) == 1
    line = read_rows(path).lines[0, "ctgan"]
    back = frame_of(line, arm_line=fill.line, frame=frame)
    pd.testing.assert_frame_equal(back, added)
    assert digest(back) == line.sha256 == fields(fill.line)["sha256"]


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_the_writer_refuses_a_non_finite_value(tmp_path, value):
    added = records(UCI_FRAME, 3)
    added.loc[-2, "oldpeak"] = value
    path = tmp_path / "rows.jsonl"
    with RowsFile(path, HEADER) as store:
        first = filled(records(UCI_FRAME, 2), name="smote")
        store.write(0, [first], arms=logged(first))
        fills = [filled(records(UCI_FRAME, 2), holdout=1), filled(added, holdout=1, name="smote")]
        with pytest.raises(IdentityError, match="non-finite"):
            store.write(1, fills, arms=logged(*fills))
    assert set(read_rows(path).lines) == {(0, "smote")}


@pytest.mark.parametrize(
    ("prefix", "old", "new", "kept"),
    [
        ("arm holdout=1 arm=bko ", "sha256=", "sha256=0", {0}),
        ("arm holdout=0 arm=bko ", "sha256=", "sha256=0", set()),
        ("split seed=1 ", "fingerprint=", "fingerprint=0", {0}),
    ],
    ids=["holdout-1", "holdout-0", "split"],
)
def test_a_tampered_holdout_exits_3_and_stores_nothing_for_it(
    tmp_path, caplog, prefix, old, new, kept
):
    folder = run_folder(tmp_path, "run", caplog)
    edit_log(folder, prefix, lambda line: line.replace(old, new, 1))
    assert stored(tmp_path, folder) == (3, "")
    assert holdouts(tmp_path) == kept


def test_a_store_a_tampered_holdout_stopped_fails_bind(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)
    edit_log(folder, "arm holdout=1 arm=bko ", lambda line: line.replace("sha256=", "sha256=0", 1))
    assert stored(tmp_path, folder) == (3, "")
    with pytest.raises(IdentityError, match="every arm line"):
        bind(read_rows(tmp_path / "rows.jsonl"), read_folder(folder), UCI_FRAME)


@pytest.mark.parametrize(
    ("prefix", "result", "kept"),
    [("1,none,held-out,", 0, {0, 1}), ("1,bko,augmented,recall,all,", 3, {0})],
    ids=["held-out", "arm"],
)
def test_rows_admits_a_new_held_out_key_and_refuses_a_new_arm_key(
    tmp_path, caplog, prefix, result, kept
):
    folder = run_folder(tmp_path, "run", caplog)
    drop_score(folder, prefix)
    assert stored(tmp_path, folder)[0] == result
    assert holdouts(tmp_path) == kept


def test_a_moved_score_exits_3(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)
    edit_scores(folder, "1,bko,augmented,recall,all,", "0.123")
    caplog.clear()
    assert stored(tmp_path, folder) == (3, "")
    assert holdouts(tmp_path) == {0}
    (error,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert error.startswith("holdout 1's score (1, 'bko', 'augmented', 'recall', 'all') reads ")


def test_a_moved_score_line_sha256_exits_3(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)
    path = folder / "scores.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert stored(tmp_path, folder) == (3, "")
    assert not (tmp_path / "rows.jsonl").exists()


def test_a_moved_replica_fidelity_row_passes_only_above_50000_rows(tmp_path, caplog, monkeypatch):
    folder = run_folder(tmp_path, "run", caplog, replicas=True)
    edit_scores(folder, "1,bko,replica,quality,all,", "0.123")
    assert stored(tmp_path, folder, replicas=True) == (3, "")
    assert holdouts(tmp_path) == {0}
    (tmp_path / "rows.jsonl").unlink()
    monkeypatch.setattr(rows_module, "SUBSAMPLED_ABOVE", 100)
    caplog.clear()
    code, _ = stored(tmp_path, folder, replicas=True)
    assert code == 0
    assert holdouts(tmp_path) == {0, 1}
    assert any(re.fullmatch(r"rows holdout=1 stored=\d+ differing=1", m) for m in caplog.messages)


@pytest.mark.parametrize(
    ("train", "expected"), [(50_000, False), (50_001, True)], ids=["50000", "50001"]
)
def test_subsampled_reads_the_split_lines_training_rows(train, expected):
    assert rows_module.subsampled(f"split seed=1 train={train} test=10 fingerprint=ab") is expected
    assert rows_module.subsampled(f"split seed=1 rows_train={train} train=10 test=1") is False


@pytest.mark.parametrize("split", [None, "split seed=1 test=10 fingerprint=ab"])
def test_subsampled_refuses_a_split_line_with_no_training_rows(split):
    with pytest.raises(IdentityError, match="names no training rows"):
        rows_module.subsampled(split)


def scores(*extra: Score, value: float = 0.5) -> list[Score]:
    """One holdout's rows: a held-out ratio, a recall of `value`, a replica quality, `extra`."""
    return [
        Score(0, HELD_OUT_ARM, Source.HELD_OUT, Measure.HELD_OUT_RATIO, Scope.ALL, 0.25),
        Score(0, "bko", Source.AUGMENTED, Measure.RECALL, Scope.ALL, value),
        Score(0, "bko", Source.REPLICA, Measure.QUALITY, Scope.ALL, 0.8),
        *extra,
    ]


def compared(ours: list[Score], theirs: list[Score], *, subsampled: bool = False) -> int:
    """`compare_holdout` on holdout 0 with `rows`' admission."""
    return compare_holdout(ours, theirs, holdout=0, admitted=held_out, subsampled=subsampled)


def test_a_nan_row_compares_equal(tmp_path, caplog):
    assert compared(scores(value=math.nan), scores(value=math.nan)) == 0
    with pytest.raises(IdentityError, match=re.escape("reads '', the folder's '0.5'")):
        compared(scores(value=math.nan), scores())
    folder = run_folder(tmp_path, "run", caplog)
    lines = (folder / "scores.csv").read_text(encoding="utf-8").splitlines()
    assert any(line.endswith(",") for line in lines)
    assert stored(tmp_path, folder)[0] == 0


def test_a_new_held_out_key_is_admitted_and_a_new_arm_key_is_not():
    marker = Score(0, HELD_OUT_ARM, Source.HELD_OUT, Measure.SHARED_SPLIT, Scope.ALL, 1.0)
    assert compared(scores(marker), scores()) == 0
    hgb = Score(0, "bko", Source.AUGMENTED, Measure.RECALL_HGB, Scope.ALL, 0.5)
    with pytest.raises(IdentityError, match="the folder holds no score"):
        compared(scores(hgb), scores())
    with pytest.raises(IdentityError, match="the rescore holds no score"):
        compared(scores(), scores(hgb))
    with pytest.raises(IdentityError, match="reads"):
        compared(scores(), scores(value=0.6), subsampled=True)
    quality = Score(0, "bko", Source.REPLICA, Measure.QUALITY, Scope.ALL, 0.7)
    assert compared(scores(), [*scores()[:2], quality], subsampled=True) == 1
    with pytest.raises(IdentityError, match="reads"):
        compared(scores(), [*scores()[:2], quality])


def test_a_rescore_holding_a_key_twice_is_refused():
    with pytest.raises(IdentityError, match="twice"):
        compared(scores(scores()[1]), scores())


def test_an_existing_store_is_refused(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)
    (tmp_path / "rows.jsonl").write_text("kept\n", encoding="utf-8")
    assert stored(tmp_path, folder) == (2, "")
    assert (tmp_path / "rows.jsonl").read_text(encoding="utf-8") == "kept\n"


def test_the_store_is_mode_0600(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)
    assert stored(tmp_path, folder)[0] == 0
    assert (tmp_path / "rows.jsonl").stat().st_mode & 0o777 == 0o600


def test_the_header_binds_the_folder(tmp_path, caplog):
    folder = run_folder(tmp_path, "run", caplog)
    assert stored(tmp_path, folder)[0] == 0
    first = (tmp_path / "rows.jsonl").read_text(encoding="utf-8").split("\n", 1)[0]
    candidates = (folder / "holdout-candidates.jsonl").read_text(encoding="utf-8")
    assert json.loads(first) == {
        "kind": "header",
        "schema": 1,
        "run": "e1",
        "folder": "run",
        "log_sha256": sha256((folder / "scores.log").read_bytes()).hexdigest(),
        "data": json.loads(candidates.split("\n", 1)[0])["data"],
        "frame": "uci",
    }
    store = read_rows(tmp_path / "rows.jsonl")
    bind(store, read_folder(folder), UCI_FRAME)
    with pytest.raises(IdentityError, match="names the frame 'uci'"):
        bind(store, read_folder(folder), SYNTHEA_FRAME)
    other = tmp_path / "other"
    shutil.copytree(folder, other)
    with pytest.raises(IdentityError, match="names the folder 'run'"):
        bind(store, read_folder(other), UCI_FRAME)
    edit_log(folder, "experiment ", lambda line: line + "\n")
    with pytest.raises(IdentityError, match="log_sha256"):
        bind(store, read_folder(folder), UCI_FRAME)


def test_frame_of_refuses_a_flipped_byte(tmp_path):
    fill, line = stored_line(tmp_path, records(UCI_FRAME, 4))
    last = line.csv[-2]
    flipped = dataclasses.replace(line, csv=line.csv[:-2] + ("1" if last != "1" else "2") + "\n")
    with pytest.raises(IdentityError, match="sha256"):
        frame_of(flipped, arm_line=fill.line, frame=UCI_FRAME)
    resealed = dataclasses.replace(flipped, sha256=sha256(flipped.csv.encode()).hexdigest())
    with pytest.raises(IdentityError, match="arm line"):
        frame_of(resealed, arm_line=fill.line, frame=UCI_FRAME)


def stored_line(tmp_path: Path, added: pd.DataFrame) -> tuple[Filled, RowsLine]:
    """`added`'s fill, written to a store and read back as its line."""
    fill = filled(added)
    path = tmp_path / "rows.jsonl"
    with RowsFile(path, HEADER) as store:
        store.write(0, [fill], arms=logged(fill))
    return fill, read_rows(path).lines[0, "ctgan"]


def test_frame_of_refuses_a_csv_that_does_not_read_back(tmp_path):
    added = records(UCI_FRAME, 4)
    added["age"] = [f"0{value}" for value in added["age"]]
    fill, line = stored_line(tmp_path, added)
    with pytest.raises(IdentityError, match="read back"):
        frame_of(line, arm_line=fill.line, frame=UCI_FRAME)


def test_frame_of_refuses_another_frames_columns(tmp_path):
    fill, line = stored_line(tmp_path, records(UCI_FRAME, 4))
    with pytest.raises(IdentityError, match="columns"):
        frame_of(line, arm_line=fill.line, frame=SYNTHEA_FRAME)


def test_frame_of_refuses_a_value_its_dtype_cannot_hold(tmp_path):
    added = records(UCI_FRAME, 4).astype({"age": float})
    added.loc[-1, "age"] = 1.5
    fill, line = stored_line(tmp_path, added)
    with pytest.raises(IdentityError, match="does not read"):
        frame_of(line, arm_line=fill.line, frame=UCI_FRAME)


def test_frame_of_refuses_a_line_whose_arm_line_differs(tmp_path):
    fill, other = filled(records(UCI_FRAME, 4)), filled(records(UCI_FRAME, 5), name="smote")
    path = tmp_path / "rows.jsonl"
    with RowsFile(path, HEADER) as store:
        store.write(0, [fill, other], arms=logged(fill, other))
    found = read_rows(path).lines
    with pytest.raises(IdentityError, match="arm line"):
        frame_of(found[0, "ctgan"], arm_line=other.line, frame=UCI_FRAME)
    wrong = dataclasses.replace(found[0, "ctgan"], added=5)
    with pytest.raises(IdentityError, match="arm line"):
        frame_of(wrong, arm_line=fill.line, frame=UCI_FRAME)


def test_read_rows_refuses_a_duplicate_key(tmp_path):
    path = tmp_path / "rows.jsonl"
    fill = filled(records(UCI_FRAME, 2))
    with RowsFile(path, HEADER) as store:
        store.write(0, [fill], arms=logged(fill))
    first, line = path.read_text(encoding="utf-8").splitlines(keepends=True)
    doubled = tmp_path / "doubled.jsonl"
    doubled.write_text(first + line + line, encoding="utf-8")
    with pytest.raises(IdentityError, match=r"two lines for \(0, 'ctgan'\)"):
        read_rows(doubled)
    headless = tmp_path / "headless.jsonl"
    headless.write_text(line, encoding="utf-8")
    with pytest.raises(IdentityError, match="header"):
        read_rows(headless)


@pytest.mark.parametrize(
    "text",
    ['{"kind":"header","schema":1}\n{"kind":"ro', "[1]\n", '{"kind":"header","schema":1}\n7\n'],
    ids=["torn", "list-header", "number-line"],
)
def test_read_rows_refuses_a_line_that_is_not_a_json_object(tmp_path, text):
    path = tmp_path / "rows.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(IdentityError, match="JSON object"):
        read_rows(path)


def test_the_writer_refuses_a_fill_another_than_the_folders_or_a_repeat(tmp_path):
    fill, other = filled(records(UCI_FRAME, 2)), filled(records(UCI_FRAME, 3))
    path = tmp_path / "rows.jsonl"
    with RowsFile(path, HEADER) as store:
        with pytest.raises(IdentityError, match="folder's arm line"):
            store.write(0, [fill], arms=logged(other))
        with pytest.raises(IdentityError, match="folder's arm line"):
            store.write(0, [fill], arms={})
        store.write(0, [fill], arms=logged(fill))
        with pytest.raises(IdentityError, match="already"):
            store.write(0, [fill], arms=logged(fill))
        again = filled(records(UCI_FRAME, 2), holdout=1)
        with pytest.raises(IdentityError, match="already"):
            store.write(1, [again, again], arms=logged(again))
    assert set(read_rows(path).lines) == {(0, "ctgan")}
