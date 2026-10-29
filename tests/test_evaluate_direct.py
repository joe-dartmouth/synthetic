"""The direct arm: `score g2` fills every holdout from the fused BKB, free, with no call log."""

from __future__ import annotations

import dataclasses
import io
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from bko.baselines.errors import BaselineError
from bko.baselines.request import Request, Strata
from bko.evaluate.__main__ import Context, main
from bko.evaluate.direct import DirectArm
from bko.evaluate.experiment import (
    ADDS_ROWS,
    CRITERION_NAMES,
    DIRECT_ARM,
    E1,
    G2,
    METHOD_ARMS,
    RUNS,
    NamedRun,
)
from bko.evaluate.fills import CURRENT_FILL, Fill
from bko.evaluate.holdout_candidates import KEPT, SCHEMA, read
from bko.evaluate.scores import Source
from bko.ontology.mapping import FIELD_MAPPINGS
from tests._datasets_fixtures import archive, cleveland_line, pin
from tests._evaluate_bko_fixtures import NARROW, candidates_for, score_argv, scripted
from tests._evaluate_fixtures import FAST_ARMS
from tests._gaps_fixtures import ANGINA, STENOSIS, ladder_releases, parse_ladders
from tests._generate_fixtures import SIZES, ScriptedSend
from tests._ontology_fixtures import RoutingOpener
from tests._validate_fixtures import planted
from tests.test_generate_direct import frame


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

SMALL_G2: Final = dataclasses.replace(
    G2,
    holdout_seeds=(0, 1),
    arms=("none", DIRECT_ARM),
    comparators=("none",),
    replica_arms=(DIRECT_ARM,),
)
"""G2 on two holdouts with one baseline: the direct arm fills the gap and a replica."""


VARIED: Final = archive(
    "\n".join(
        cleveland_line(
            age=age + index % 7,
            sex=sex,
            num=num,
            cp="4.0" if (age, sex) == (40, 0) else f"{1 + index % 4}.0",
            trestbps=f"{110 + 3 * index}.0",
            chol=f"{200 + 7 * index}.0",
            thalach=f"{120 + 5 * index}.0",
        )
        for (age, sex, num), size in {**SIZES, (40, 0, 1): 5}.items()
        for index in range(size)
    )
    + "\n"
)
"""`UCI_E1`'s cells with ages, pressures, cholesterol and heart rates spread within each, so a
drawn record, assembled from several training records, copies no test record."""


def g2_context() -> tuple[Context, ScriptedSend]:
    send = ScriptedSend()
    pins, served = ladder_releases()
    uci = pin(VARIED)
    ctx = Context(
        open_url=RoutingOpener({**served, uci.url: VARIED}),
        uci=uci,
        experiment=NARROW,
        runs={"g2": NamedRun(SMALL_G2)},
        arms={"none": FAST_ARMS["none"]},
        releases=pins,
        field_rows=(ANGINA, STENOSIS),
        send=lambda: send,
    )
    return ctx, send


def scored(tmp_path: Path, *extra: str) -> tuple[int, str, Path]:
    ctx, send = g2_context()
    candidates = tmp_path / "holdout-candidates.jsonl"
    out = io.StringIO()
    argv = ["score", "g2", "--candidates", str(candidates), "--data-dir", str(tmp_path / "data")]
    code = main([*argv, *extra], context=ctx, out=out)
    assert send.calls == []
    return code, out.getvalue(), candidates


def test_g2_is_e1_with_the_direct_arm_in_place_of_the_bko_arms_and_no_criterion():
    assert RUNS["g2"] == NamedRun(G2)
    assert G2.arms == (*METHOD_ARMS, DIRECT_ARM)
    assert (G2.candidate, G2.comparators, G2.criteria) == (DIRECT_ARM, METHOD_ARMS, ())
    assert G2.replica_arms == (*ADDS_ROWS, DIRECT_ARM)
    assert E1.replica_arms[: len(ADDS_ROWS)] == ADDS_ROWS
    assert (G2.holdout_seeds, G2.budget, G2.attempts) == (E1.holdout_seeds, E1.budget, E1.attempts)


def test_score_g2_writes_candidate_counts_and_a_candidates_file_with_null_calls(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    code, out, path = scored(tmp_path)
    assert code == 0
    scores = pd.read_csv(io.StringIO(out))
    mine = scores[scores["arm"] == DIRECT_ARM]
    sources = set(mine["source"])
    assert {Source.CANDIDATES.value, Source.REPLICA_CANDIDATES.value} <= sources
    assert {Source.ADDED.value, Source.AUGMENTED.value, Source.REPLICA.value} <= sources
    assert mine["measure"].str.endswith("-hgb").any()
    generated = mine[(mine["source"] == "candidates") & (mine["measure"] == "generated")]
    passed = mine[(mine["source"] == "candidates") & (mine["measure"] == "passed")]
    assert sorted(generated["holdout"]) == [0, 1]
    assert generated["value"].tolist() == passed["value"].tolist() == [28.0, 28.0]
    found = read(path.read_bytes())
    assert found.header["run"] == "g2"
    assert (found.header["schema"], found.header["frame"]) == (SCHEMA, "uci")
    assert "settings" not in found.header
    assert {line.call for line in found.candidates} == {None}
    assert {line.arm for line in found.candidates} == {DIRECT_ARM}
    assert {line.fill for line in found.candidates} == {"gap", "replica"}
    gap = [line for line in found.candidates if line.fill == "gap"]
    assert len(gap) == 56
    assert {line.outcome for line in gap} == {KEPT}
    assert f"validity arm={DIRECT_ARM} generated=" in caplog.text


def test_score_g2_writes_the_same_scores_and_candidates_on_a_second_run(tmp_path):
    first = scored(tmp_path / "a")
    second = scored(tmp_path / "b")
    assert first[1] == second[1]
    assert first[2].read_bytes() == second[2].read_bytes()


def test_compare_g2_reads_the_direct_arm_beside_another_runs_bko(tmp_path, caplog):
    code, out, _ = scored(tmp_path)
    assert code == 0
    ours = tmp_path / "scores.csv"
    ours.write_text(out, encoding="utf-8")
    theirs = tmp_path / "e1-scores.csv"
    theirs.write_text(out.replace(f",{DIRECT_ARM},", ",bko,"), encoding="utf-8")
    ctx, _ = g2_context()
    caplog.set_level(logging.INFO)
    argv = ["compare", "g2", str(ours), "--beside", str(theirs), "--beside-arm", "bko"]
    assert main(argv, context=ctx, out=io.StringIO()) == 0
    assert any(m.startswith("beside scores_sha256=") for m in caplog.messages)


def test_score_g2_refuses_a_call_log_before_any_download(tmp_path, caplog):
    code, out, path = scored(tmp_path, "--log", str(tmp_path / "calls.jsonl"))
    assert (code, out) == (2, "")
    assert "--log needs a BKO arm that calls the Messages API" in caplog.text
    assert not path.exists()
    assert not (tmp_path / "data").exists()


def test_score_g2_needs_a_candidates_file(tmp_path, caplog):
    ctx, _ = g2_context()
    argv = ["score", "g2", "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert "--candidates is required: the arms ['bko-direct']" in caplog.text


def test_a_draw_no_record_matches_is_a_refusal_the_harness_may_retry(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    train = frame(
        [planted(age=40, sex=0, target=1, cp=1, exang=0) for _ in range(5)]
        + [planted(age=40, sex=0, target=1, cp=4, exang=0) for _ in range(5)]
    )

    arm = DirectArm(FIELD_MAPPINGS, {}, name=DIRECT_ARM, budget=3, sink=None, dataset="uci-test")
    request = Request(quota={("<55", "female", 1): 100}, strata=Strata.CELL, seed=0)
    token = CURRENT_FILL.set(Fill(0, "gap"))
    try:
        with pytest.raises(BaselineError, match=r"cannot draw cp\+exang"):
            arm(train, request)
    finally:
        CURRENT_FILL.reset(token)
    assert any(
        m.startswith(f"direct arm={DIRECT_ARM} holdout=0 fill=gap seed=0 outcome=refused asked=")
        for m in caplog.messages
    )
    assert "direct holdout=0 fill=gap key=<55/female/1 rows=100" in caplog.text
    caplog.clear()
    with pytest.raises(BaselineError):
        arm(train, request)
    assert any(
        m.startswith(f"direct arm={DIRECT_ARM} holdout=-1 fill=none seed=0 outcome=refused")
        for m in caplog.messages
    )


def test_the_arm_fills_its_request_from_the_seed_and_names_its_generator(tmp_path):
    ontologies = parse_ladders(tmp_path)
    train = frame(
        [planted(age=40 + i % 10, sex=0, target=1, cp=1 + i % 4) for i in range(12)]
        + [planted(age=60, sex=1, target=0) for _ in range(5)]
    )
    request = Request(quota={("<55", "female", 1): 8}, strata=Strata.CELL, seed=4)

    def fill() -> tuple[pd.DataFrame, Mapping[str, object]]:
        arm = DirectArm(
            (ANGINA, STENOSIS), ontologies, name=DIRECT_ARM, budget=3, sink=None, dataset="t"
        )
        baseline = arm(train, request)
        return baseline.added, baseline.params

    added, params = fill()
    assert len(added) == 8
    assert params == {"generator": "direct", "seed": 4}
    assert fill()[0].equals(added)


def test_a_run_with_bko_arms_and_the_direct_arm_shares_one_candidates_file(tmp_path):
    ctx, _ = g2_context()
    send = scripted()
    mixed = dataclasses.replace(
        NARROW, arms=("none", "bko", DIRECT_ARM), comparators=("none",), replica_arms=()
    )
    ctx = dataclasses.replace(ctx, experiment=mixed, send=lambda: send)
    log = tmp_path / "calls.jsonl"
    argv = score_argv(log, tmp_path)
    out = io.StringIO()
    assert main(argv, context=ctx, out=out) == 0
    send.assert_consumed()
    found = read(candidates_for(log).read_bytes())
    calls = {(line.arm, line.call is None) for line in found.candidates}
    assert calls == {("bko", False), (DIRECT_ARM, True)}
    scores = pd.read_csv(io.StringIO(out.getvalue()))
    assert {"bko", DIRECT_ARM} <= set(scores[scores["source"] == "candidates"]["arm"])


def test_compare_on_a_run_with_no_criteria_logs_e1s_criteria_unjudged(tmp_path, caplog):
    code, out, _ = scored(tmp_path)
    assert code == 0
    path = tmp_path / "scores.csv"
    path.write_text(out, encoding="utf-8")
    ctx, _ = g2_context()
    caplog.set_level(logging.INFO)
    table = io.StringIO()
    assert main(["compare", "g2", str(path)], context=ctx, out=table) == 0
    assert table.getvalue().count("\n") == 1, "the table holds its header alone"
    unjudged = [m for m in caplog.messages if m.startswith("unjudged ")]
    criteria = {m.split()[1] for m in unjudged}
    assert criteria == {f"criterion={name}" for name in CRITERION_NAMES if name != "validity"}
    assert any(
        m.startswith("unjudged criterion=primary scope=all arm=bko-direct against=none ")
        and " measure=recall mean=" in m
        and " refuting_high=" in m
        and " holdouts=2 differing=" in m
        for m in unjudged
    )
    assert not any("verdict=" in m for m in caplog.messages if "criterion=" in m)


def test_compare_on_a_judged_run_logs_nothing_unjudged(tmp_path, caplog):
    code, out, _ = scored(tmp_path)
    assert code == 0
    path = tmp_path / "scores.csv"
    path.write_text(out, encoding="utf-8")
    ctx, _ = g2_context()
    judged = dataclasses.replace(SMALL_G2, criteria=("primary",))
    ctx = dataclasses.replace(ctx, runs={"g2": NamedRun(judged)})
    caplog.set_level(logging.INFO)
    assert main(["compare", "g2", str(path)], context=ctx, out=io.StringIO()) == 0
    assert not any(m.startswith("unjudged ") for m in caplog.messages)
