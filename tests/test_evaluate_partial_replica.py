"""`Experiment.partial_replica`: a replica fill short on its last attempt fills partially.

ADR `research/e1-evaluation-protocol` records the rule; `bko.evaluate.filling.fill_arm` holds it.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Final

import pytest

from bko.baselines.errors import ShortfallError
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.experiment import E1
from bko.evaluate.provenance import experiment_line
from bko.evaluate.run import run
from bko.evaluate.run_folder import ASSUMED
from bko.evaluate.runs_published import P1, P1_PARTIAL
from bko.evaluate.scores import Measure, Scope, Source, render
from tests._evaluate_fixtures import FAST_ARMS, GRADED, SMALL
from tests.test_evaluate_run import ONE_RECORD, ONLY_SMOTE, shorting
from tests.test_evaluate_shared import edit_log, refused, run_folder, shared


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

E1_LINE: Final = (
    "experiment run=e1 frame=uci holdouts=100 seeds=0-99 removal=none training_fraction=1"
    " decontaminate=true arms=none,class-weight,smote,adasyn,ctgan,tvae,bko,bko-population,"
    "bko-frequency replica=smote,adasyn,ctgan,tvae,bko,bko-population,bko-frequency threshold=20"
    " fill_to=20 budget=3 validity=19/20 spend_ceiling=25.00"
    " criteria=primary,secondary,privacy,validity models=lr,hgb"
)
"""E1's `experiment` line, captured at the commit before `partial_replica` existed."""

P1_LINE: Final = (
    "experiment run=p1-id frame=tableshift holdouts=10 seeds=0-9 removal=none training_fraction=1"
    " decontaminate=false arms=none,class-weight,smote,adasyn,ctgan,tvae,gaussian-copula,cart,"
    "bayes-net,tabddpm,label-bootstrap,bko-direct replica=smote,adasyn,ctgan,tvae,gaussian-copula,"
    "cart,bayes-net,tabddpm,label-bootstrap,bko-direct threshold=20 fill_to=20 budget=3"
    " validity=19/20 spend_ceiling=25.00 criteria=none models=lr,hgb"
)
"""P1's `experiment` line under `p1-id`, captured at the same commit."""


def test_the_default_experiment_line_is_unchanged():
    assert experiment_line("e1", E1, ["lr", "hgb"], frame="uci") == E1_LINE
    assert experiment_line("p1-id", P1, ["lr", "hgb"], frame="tableshift") == P1_LINE


REPLICA: Final = dataclasses.replace(ONLY_SMOTE, replica_arms=("smote",))
"""SMOTE on `SMALL`'s holdouts, filling the gap request and a replica."""

PARTIAL: Final = dataclasses.replace(REPLICA, partial_replica=True)
"""`REPLICA` with a replica short on its last attempt filling partially."""

SHORTFALL: Final = re.compile(
    r"shortfall holdout=(?P<holdout>\d+) arm=(?P<arm>\S+) attempt=(?P<attempt>\d+)"
    r" cell=(?P<cell>\S+) short=(?P<short>\d+) quota=(?P<quota>\d+)"
)
"""`bko.evaluate.shortfall.partial_fill`'s line; a `give_up` line does not match it."""

KEYS: Final = frozenset({"holdout", "arm", "attempt", "cell", "short", "quota"})


def shortfalls(messages: list[str]) -> list[dict[str, str]]:
    """Each `shortfall` message's fields, asserting every one is `partial_fill`'s line."""
    lines = [m for m in messages if m.startswith("shortfall ")]
    found = [SHORTFALL.fullmatch(m) for m in lines]
    assert all(each is not None for each in found), lines
    fields = [each.groupdict() for each in found if each is not None]
    assert all(set(each) == KEYS for each in fields)
    return fields


def scored(experiment, runner, caplog):
    """`run` over `GRADED` with `runner` as SMOTE, logging at INFO."""
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        return run(GRADED, experiment, {"smote": runner}, frame=UCI_FRAME, rows=(), source_name="")


def test_a_partial_replica_retries_each_attempt_before_the_last_then_fills_partially(caplog):
    assert PARTIAL.attempts > 1
    calls = []
    scores = scored(PARTIAL, shorting(calls, replica_only=True), caplog)
    assert {score.holdout for score in scores} == set(SMALL.holdout_seeds)
    retries = [m.split()[1:4] for m in caplog.messages if m.startswith("retry ")]
    assert retries == [
        [f"holdout={seed}", "arm=smote-replica", f"attempt={attempt}"]
        for seed in SMALL.holdout_seeds
        for attempt in range(1, PARTIAL.attempts)
    ]
    found = shortfalls(caplog.messages)
    assert [(f["holdout"], f["arm"], f["attempt"]) for f in found] == [
        (str(seed), "smote-replica", str(PARTIAL.attempts)) for seed in SMALL.holdout_seeds
    ]
    assert all(f["short"] == f["quota"] for f in found)
    arms = [m for m in caplog.messages if m.startswith("arm holdout=") and "-replica " in m]
    assert len(arms) == len(SMALL.holdout_seeds)
    assert all(f" attempt={PARTIAL.attempts} " in m and '"short": {' in m for m in arms)


def test_the_default_still_refuses(caplog):
    calls = []
    with pytest.raises(ShortfallError):
        scored(REPLICA, shorting(calls, replica_only=True), caplog)
    assert f"refused holdout=0 arm=smote-replica attempts={REPLICA.attempts}" in caplog.messages
    assert not any(m.startswith("shortfall ") for m in caplog.messages)


def test_a_replica_takes_no_early_partial_on_structural_strata(caplog):
    replica = dataclasses.replace(PARTIAL, threshold=2, fill_to=3)
    with caplog.at_level(logging.INFO, logger="bko.evaluate"):
        run(
            ONE_RECORD,
            replica,
            {"smote": FAST_ARMS["smote"]},
            frame=UCI_FRAME,
            rows=(),
            source_name="",
        )
    retries = [m.split()[1:4] for m in caplog.messages if m.startswith("retry ")]
    assert retries == [
        [f"holdout={seed}", "arm=smote-replica", f"attempt={attempt}"]
        for seed in SMALL.holdout_seeds
        for attempt in range(1, PARTIAL.attempts)
    ]
    replicas = [f for f in shortfalls(caplog.messages) if f["arm"] == "smote-replica"]
    assert replicas
    assert {f["attempt"] for f in replicas} == {str(PARTIAL.attempts)}


def test_a_replica_that_completes_scores_identically_with_the_field_set():
    plain = run(GRADED, REPLICA, FAST_ARMS, frame=UCI_FRAME, rows=(), source_name="")
    partial = run(GRADED, PARTIAL, FAST_ARMS, frame=UCI_FRAME, rows=(), source_name="")
    assert render(partial) == render(plain)


def test_a_partial_replica_writes_a_replica_short_row(caplog):
    calls = []
    scores = scored(PARTIAL, shorting(calls, replica_only=True), caplog)
    short = [s for s in scores if s.measure is Measure.REPLICA_SHORT]
    assert [(s.holdout, s.arm, s.source, s.scope) for s in short] == [
        (seed, "smote", Source.REPLICA, Scope.ALL) for seed in SMALL.holdout_seeds
    ]
    missing = {int(f["holdout"]): float(f["short"]) for f in shortfalls(caplog.messages)}
    assert {s.holdout: s.value for s in short} == missing


def test_a_run_whose_replicas_complete_writes_no_such_row():
    scores = run(GRADED, PARTIAL, FAST_ARMS, frame=UCI_FRAME, rows=(), source_name="")
    assert any(s.source is Source.REPLICA for s in scores)
    assert not any(s.measure is Measure.REPLICA_SHORT for s in scores)


def test_the_experiment_line_names_partial_replica_only_when_set():
    plain = experiment_line("x", REPLICA, ["lr"], frame="uci")
    assert "partial_replica" not in plain
    marked = experiment_line("x", PARTIAL, ["lr"], frame="uci")
    assert " replica=smote partial_replica=true threshold=" in marked
    assert marked.replace("partial_replica=true ", "") == plain
    capped = dataclasses.replace(PARTIAL, replica_rows=40)
    assert " replica=smote replica_rows=40 partial_replica=true threshold=" in experiment_line(
        "x", capped, ["lr"], frame="uci"
    )
    line = experiment_line("p1-id-partial", P1_PARTIAL, ["lr", "hgb"], frame="tableshift")
    assert line.replace("partial_replica=true ", "") == P1_LINE.replace(
        "run=p1-id ", "run=p1-id-partial "
    )
    assert line != P1_LINE.replace("run=p1-id ", "run=p1-id-partial ")


def test_an_older_experiment_line_reads_at_the_assumed_partial_replica(tmp_path, caplog):
    assert ASSUMED["partial_replica"] == "false"
    folder = run_folder(tmp_path, "run", caplog)
    edit_log(
        folder,
        "experiment ",
        lambda line: line.replace(" replica= ", " replica= partial_replica=false ", 1),
    )
    code, text = shared(tmp_path, folder)
    assert code == 0
    assert text == (folder / "scores.csv").read_text(encoding="utf-8")
    edit_log(
        folder,
        "experiment ",
        lambda line: line.replace("partial_replica=false ", "partial_replica=true ", 1),
    )
    error = refused(tmp_path, caplog, folder)
    assert error.endswith("differ: {'partial_replica': ('true', 'false')}")


def test_a_partial_replica_without_replica_arms_is_refused():
    with pytest.raises(ValueError, match="partial_replica needs a replica arm"):
        dataclasses.replace(ONLY_SMOTE, partial_replica=True)


@pytest.mark.parametrize("value", [1, "true", None])
def test_a_partial_replica_that_is_not_a_bool_is_refused(value):
    with pytest.raises(TypeError, match=r"^partial_replica \S+ must be a bool$"):
        dataclasses.replace(REPLICA, partial_replica=value)
