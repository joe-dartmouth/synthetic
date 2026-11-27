"""Every reader of a run folder whose replica filled partially (`Experiment.partial_replica`).

ADR `research/e1-evaluation-protocol` records the rule; `tests/test_evaluate_partial_replica.py`
holds the fill itself.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import re
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from bko.baselines.errors import ShortfallError
from bko.baselines.fills import CURRENT_FILL
from bko.baselines.records import Baseline, number
from bko.baselines.request import Request
from bko.evaluate import reader_commands
from bko.evaluate.__main__ import main
from bko.evaluate.context import Context
from bko.evaluate.rows_read import read_rows
from bko.evaluate.run_folder import fields, read_folder
from bko.evaluate.scores import Measure, Scope, Score, Source, parse, render
from tests._baselines_fixtures import cell_list
from tests._evaluate_bko_fixtures import NARROW, candidates_for, context, score_argv, scripted
from tests._evaluate_fixtures import FAST_ARMS
from tests._generate_fixtures import ScriptedSend
from tests.test_evaluate_compare import UNJUDGED, compared, scores, write
from tests.test_evaluate_merge import HIGH, LOW, _merged, _scored
from tests.test_evaluate_panel import FAST_PANEL, folder_and_store, paneled
from tests.test_evaluate_partial_replica import shortfalls
from tests.test_evaluate_rows import stored
from tests.test_evaluate_shared import edit_log, run_folder, shared


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def short_replica(train: pd.DataFrame, request: Request) -> Baseline:
    """SMOTE's fill; on a replica fill, short of its first cell's rows on every attempt."""
    full = FAST_ARMS["smote"](train, request)
    fill = CURRENT_FILL.get()
    if fill is None or fill.kind != "replica":
        return full
    cell, *_ = request.quota
    keep = [stratum != cell for stratum in cell_list(full.added)]
    missing = {"/".join(map(str, cell)): request.quota[cell]}
    kept = Baseline(number(full.added.loc[keep]), full.weights, {"short": missing})
    msg = f"short of {missing}"
    raise ShortfallError(msg, missing, kept.added, kept)


PARTIAL: Final = dataclasses.replace(
    NARROW,
    arms=("none", "smote"),
    candidate="smote",
    comparators=("none",),
    criteria=(),
    replica_arms=("smote",),
    decontaminate=False,
    partial_replica=True,
)
"""A baseline whose replica fills partially; SMOTE's rows copy test records, so it keeps them."""

ARMS: Final = {"none": FAST_ARMS["none"], "smote": short_replica}

REPLICA_ARM: Final = "smote-replica"


def partial_folder(tmp_path: Path, caplog) -> Path:
    """A finished run folder of `PARTIAL`, each holdout's replica filled partially."""
    return run_folder(tmp_path, "run", caplog, experiment=PARTIAL, arms=ARMS)


def logged_shortfalls(folder: Path) -> list[dict[str, str]]:
    """The `shortfall` lines of `folder`'s `scores.log`, by their fields."""
    messages = [
        line.split(": ", 1)[1]
        for line in (folder / "scores.log").read_text(encoding="utf-8").splitlines()
    ]
    return shortfalls(messages)


def test_merge_accepts_slices_where_only_some_hold_replica_short(tmp_path):
    row = Score(3, "smote", Source.REPLICA, Measure.REPLICA_SHORT, Scope.ALL, 4.0)
    high = render([*parse(HIGH), row])
    code, merged = _merged(tmp_path, [high, LOW])
    assert code == 0
    found = [s for s in parse(merged) if s.measure is Measure.REPLICA_SHORT]
    assert found == [row]
    assert len(parse(merged)) == len(parse(_scored((0, 1, 2, 3)))) + 1


@pytest.mark.parametrize(
    "row",
    [
        Score(3, "none", Source.REPLICA, Measure.REPLICA_SHORT, Scope.ALL, 4.0),
        Score(3, "smote", Source.ADDED, Measure.REPLICA_SHORT, Scope.ALL, 4.0),
        Score(3, "smote", Source.REPLICA, Measure.REPLICA_SHORT, Scope.FLAGGED, 4.0),
    ],
    ids=["not-a-replica-arm", "other-source", "other-scope"],
)
def test_merge_refuses_a_replica_short_key_no_partial_replica_writes(tmp_path, caplog, row):
    high = render([*parse(HIGH), row])
    assert _merged(tmp_path, [high, LOW]) == (2, "")
    (error,) = errors(caplog)
    assert "holds other scores than holdout" in error


def test_compare_prints_each_partial_holdout_and_arm(tmp_path, caplog):
    short = [
        Score(0, "smote", Source.REPLICA, Measure.REPLICA_SHORT, Scope.ALL, 12.0),
        Score(1, "smote", Source.REPLICA, Measure.REPLICA_SHORT, Scope.ALL, 7.0),
    ]
    lines = compared(
        write(tmp_path, "scores.csv", [*scores(panel=False), *short]), UNJUDGED, caplog
    )
    assert [m for m in lines if m.startswith("replica-short ")] == [
        "replica-short holdout=0 arm=smote rows=12",
        "replica-short holdout=1 arm=smote rows=7",
    ]
    plain = compared(write(tmp_path, "plain.csv", scores(panel=False)), UNJUDGED, caplog)
    assert not any(m.startswith("replica-short ") for m in plain)


def test_rows_reproduces_a_partial_replicas_sha256_and_shortfall_lines(tmp_path, caplog):
    folder = partial_folder(tmp_path, caplog)
    drawn = logged_shortfalls(folder)
    assert [(f["holdout"], f["arm"]) for f in drawn] == [
        (str(seed), REPLICA_ARM) for seed in PARTIAL.holdout_seeds
    ]
    caplog.set_level(logging.INFO)
    code, text = stored(tmp_path, folder, experiment=PARTIAL, arms=ARMS)
    assert code == 0
    assert text == (folder / "scores.csv").read_text(encoding="utf-8")
    assert shortfalls(caplog.messages) == drawn
    arms = read_folder(folder).arms
    store = read_rows(tmp_path / "rows.jsonl").lines
    for seed in PARTIAL.holdout_seeds:
        logged = fields(arms[seed, REPLICA_ARM])
        assert '"short": {' in logged["params"]
        assert store[seed, REPLICA_ARM].sha256 == logged["sha256"]
        expected = f"rows holdout={seed} arm={REPLICA_ARM} added={logged['added']}"
        expected += f" sha256={logged['sha256']} matches_arm=true"
        assert expected in caplog.messages


def test_panel_counts_the_replica_in_partial(tmp_path, caplog):
    folder, store = folder_and_store(tmp_path, caplog, PARTIAL, ARMS)
    caplog.set_level(logging.INFO)
    code, text = paneled(tmp_path, folder, store, experiment=PARTIAL, arms=ARMS, models=FAST_PANEL)
    assert code == 0
    lines = [m for m in caplog.messages if m.startswith("panel holdout=")]
    assert all(" partial=1 " in m for m in lines)
    assert len(lines) == len(PARTIAL.holdout_seeds)
    assert not any(m.startswith("shortfall ") for m in caplog.messages)
    short = [s for s in parse(text) if s.measure is Measure.REPLICA_SHORT]
    assert [(s.holdout, s.arm) for s in short] == [
        (seed, "smote") for seed in PARTIAL.holdout_seeds
    ]


def errors(caplog) -> list[str]:
    """The messages logged at ERROR or above."""
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def test_shared_refuses_a_partial_replica_before_the_dataset_is_loaded(
    tmp_path, caplog, monkeypatch
):
    folder = partial_folder(tmp_path, caplog)
    short = json.loads(fields(read_folder(folder).arms[0, REPLICA_ARM])["params"])["short"]
    shown = json.dumps(short, sort_keys=True, separators=(",", ":"))

    def unloaded(*_args, **_kwargs):
        raise AssertionError("shared loaded the dataset before its partial-replica refusal")

    monkeypatch.setattr(reader_commands, "load", unloaded)
    caplog.clear()
    assert shared(tmp_path, folder, experiment=PARTIAL, arms=ARMS) == (3, "")
    (error,) = errors(caplog)
    assert error == (
        f"{folder}'s holdout 0's {REPLICA_ARM} fill is a partial replica, short of {shown},"
        " which a shared rescore does not read; `rows` or `panel` read it"
    )


def test_shared_refuses_a_with_folder_whose_replica_filled_partially(tmp_path, caplog):
    folder = run_folder(tmp_path, "whole", caplog, experiment=PARTIAL, arms=FAST_ARMS)
    other = partial_folder(tmp_path, caplog)
    caplog.clear()
    assert shared(tmp_path, folder, other, experiment=PARTIAL, arms=ARMS) == (3, "")
    (error,) = errors(caplog)
    assert error.startswith(f"{other}'s holdout 0's {REPLICA_ARM} fill is a partial replica")


def test_shared_refuses_malformed_params_on_a_partial_runs_replica_line(tmp_path, caplog):
    folder = partial_folder(tmp_path, caplog)
    edit_log(
        folder,
        f"arm holdout=0 arm={REPLICA_ARM} ",
        lambda line: line.split(" params=")[0] + " params=[1]\n",
    )
    caplog.clear()
    assert shared(tmp_path, folder, experiment=PARTIAL, arms=ARMS) == (3, "")
    (error,) = errors(caplog)
    assert error.startswith("the params of ")
    assert error.endswith(" are not a JSON object")


def curated_short(train: pd.DataFrame, request: Request) -> Baseline:
    """SMOTE's fill; on a replica fill, its params name a `short` as a curated replica's do."""
    full = FAST_ARMS["smote"](train, request)
    fill = CURRENT_FILL.get()
    if fill is None or fill.kind != "replica":
        return full
    cell, *_ = request.quota
    return Baseline(full.added, full.weights, {"short": {"/".join(map(str, cell)): 1}})


CURATED: Final = dataclasses.replace(PARTIAL, partial_replica=False)
"""`PARTIAL` without the field, as B3's run is: a replica naming `short` is not a partial fill."""


def test_a_replica_naming_short_without_the_field_reads_as_before(tmp_path, caplog):
    arms = {"none": FAST_ARMS["none"], "smote": curated_short}
    folder = run_folder(tmp_path, "curated", caplog, experiment=CURATED, arms=arms)
    assert all(
        '"short": {' in line
        for (_, arm), line in read_folder(folder).arms.items()
        if arm == REPLICA_ARM
    )
    written = (folder / "scores.csv").read_text(encoding="utf-8")
    assert not any(s.measure is Measure.REPLICA_SHORT for s in parse(written))
    assert shared(tmp_path, folder, experiment=CURATED, arms=arms) == (0, written)


def short_gap(train: pd.DataFrame, request: Request) -> Baseline:
    """SMOTE's fill; on a gap fill, short of its first cell's rows on every attempt."""
    full = FAST_ARMS["smote"](train, request)
    fill = CURRENT_FILL.get()
    if fill is None or fill.kind != "gap":
        return full
    cell, *_ = request.quota
    keep = [stratum != cell for stratum in cell_list(full.added)]
    missing = {"/".join(map(str, cell)): request.quota[cell]}
    kept = Baseline(number(full.added.loc[keep]), full.weights, {"short": missing})
    msg = f"short of {missing}"
    raise ShortfallError(msg, missing, kept.added, kept)


def test_shared_reads_a_partial_baseline_gap_fill_under_the_field(tmp_path, caplog):
    arms = {"none": FAST_ARMS["none"], "smote": short_gap}
    folder = run_folder(tmp_path, "gap", caplog, experiment=PARTIAL, arms=arms)
    assert '"short": {' in read_folder(folder).arms[0, "smote"]
    written = (folder / "scores.csv").read_text(encoding="utf-8")
    assert shared(tmp_path, folder, experiment=PARTIAL, arms=arms) == (0, written)


def test_a_line_without_the_field_is_refused_under_a_run_that_sets_it(tmp_path, caplog):
    folder = run_folder(tmp_path, "older", caplog, experiment=CURATED, arms=FAST_ARMS)
    assert "partial_replica" not in read_folder(folder).experiment
    caplog.clear()
    assert shared(tmp_path, folder, experiment=PARTIAL, arms=FAST_ARMS) == (3, "")
    (error,) = errors(caplog)
    assert error.endswith("differ: {'partial_replica': ('false', 'true')}")


def test_trace_writes_a_partial_baseline_replicas_shortfall(tmp_path, caplog):
    folder = partial_folder(tmp_path, caplog)
    line = read_folder(folder).arms[0, REPLICA_ARM]
    logged = json.loads(fields(line)["params"])
    found_seed = re.search(r" seed=(\d+) ", line)
    assert found_seed is not None
    seed = int(found_seed[1])
    argv = ["trace", "--folder", str(folder), "--holdout", "0", "--arm", REPLICA_ARM]
    argv += ["--data-dir", str(tmp_path / "data")]
    calls: list[int] = []

    def drawing(train: pd.DataFrame, request: Request) -> Baseline:
        calls.append(request.seed)
        return short_replica(train, request)

    ctx, _ = context(ScriptedSend(), arms={"none": FAST_ARMS["none"], "smote": drawing})
    ctx = dataclasses.replace(ctx, experiment=PARTIAL)
    out = io.StringIO()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert main(argv, context=ctx, out=out) == 0
    (found,) = [json.loads(each) for each in out.getvalue().splitlines()]
    assert found["request"] == {"method": "smote", "seed": seed, "short": logged["short"]}
    assert calls == []
    short = json.dumps(logged["short"], sort_keys=True, separators=(",", ":"))
    assert f"trace replica-short holdout=0 arm={REPLICA_ARM} short={short}" in caplog.messages


GUIDED: Final = dataclasses.replace(
    NARROW,
    arms=("none", "bko", "smote"),
    comparators=("none",),
    criteria=(),
    replica_arms=("smote",),
    decontaminate=False,
    partial_replica=True,
)
"""A guided arm beside the partial replica: `--replay` and `--rehearse` each need a BKO arm."""


def guided_context(send: ScriptedSend) -> Context:
    """`GUIDED` over `send`, SMOTE's replica short on every attempt."""
    ctx, _ = context(send, arms=ARMS)
    return dataclasses.replace(ctx, experiment=GUIDED)


def logged(argv: list[str], ctx: Context, caplog) -> list[dict[str, str]]:
    """`score` over `argv`, exit 0; its `shortfall` lines' fields."""
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert main(argv, context=ctx, out=io.StringIO()) == 0
    return shortfalls(caplog.messages)


def test_replay_and_rehearse_log_the_shortfall_lines(tmp_path, caplog):
    log = tmp_path / "calls.jsonl"
    live = logged(score_argv(log, tmp_path), guided_context(scripted()), caplog)
    assert [(f["holdout"], f["arm"]) for f in live] == [
        (str(seed), REPLICA_ARM) for seed in GUIDED.holdout_seeds
    ]
    assert candidates_for(log).exists()
    replay = tmp_path / "replay.jsonl"
    argv = score_argv(replay, tmp_path, "--replay", str(log))
    assert logged(argv, guided_context(ScriptedSend()), caplog) == live
    rehearsal = tmp_path / "rehearsal.jsonl"
    argv = score_argv(rehearsal, tmp_path, "--rehearse")
    assert logged(argv, guided_context(ScriptedSend()), caplog) == live
