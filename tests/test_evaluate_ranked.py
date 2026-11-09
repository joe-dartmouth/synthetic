"""The ranking arm `bko-curated`: it asks twice each quota and keeps each call cell's best rows.

Every call goes to a `ScriptedSend` or a replay; no test reaches the API. Most tests replace the
scorer by a table of scores by `thalach`, so each fixes the ranking it reads; one runs the real one.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Callable, Iterator, Mapping
from functools import partial
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

import bko.evaluate.guided as guided_module
import bko.evaluate.ranking as ranking_module
from bko.baselines.fills import CURRENT_FILL, Fill
from bko.baselines.records import Drop
from bko.baselines.request import Request, Strata
from bko.datasets.frames import FrameSpec
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.attempts import Attempt, Ranked
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import CURATED_ARM, E1, MESSAGES_ARMS, Curation, MessagesArm
from bko.evaluate.guided import GuidedArm, Knowledge
from bko.evaluate.holdout_candidates import CandidateFile, Origin, Seen
from bko.evaluate.ranking import Known, Scorers, record, tie
from bko.evaluate.scores import Measure, Source
from bko.evaluate.waves import spendable
from bko.gaps.definitions import Definition
from bko.gaps.observe import Cell, cell_key
from bko.generate.flight import Flight, Limits
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, AnySend
from bko.generate.replay import Replay
from bko.ontology.mapping import FIELD_MAPPINGS, RELEASES
from tests._evaluate_fixtures import GRADED_DATA
from tests._generate_fixtures import ScriptedSend, reply
from tests.test_evaluate_cllm import (
    YOUNG_SICK,
    YOUNG_WELL,
    asked_of,
    cells_of,
    sick_woman,
    train,
    well_woman,
)


CURATED: Final = "bko-curated"
HOLDOUT: Final = 0
KIND: Final = "replica"
QUOTA: Final = 3
REQUEST: Final = Request(quota={YOUNG_SICK: QUOTA}, strata=Strata.CELL, seed=0, frame=UCI_FRAME)
REAL_CHANGES: Final = (
    {"cp": 4},
    {},
    {"chol": 300, "trestbps": 150},
    {"chol": 300},
    {"cp": 4, "chol": 300},
    {"trestbps": 150},
)

pytestmark = pytest.mark.usefixtures("fill")


@pytest.fixture
def fill() -> Iterator[None]:
    """Each test runs inside holdout 0's replica fill, whose holdout the tie order reads."""
    token = CURRENT_FILL.set(Fill(HOLDOUT, KIND))
    yield
    CURRENT_FILL.reset(token)


def scored(monkeypatch: pytest.MonkeyPatch, table: Mapping[int, float]) -> None:
    """Score each record by its `thalach` from `table`, whatever its cell."""

    class Table:
        def __init__(self, train: pd.DataFrame, data: Known, frame: FrameSpec) -> None:
            pass

        def __call__(self, cell: Cell) -> Callable[[Mapping[str, float]], float]:
            return lambda record: table[int(record["thalach"])]

    monkeypatch.setattr(guided_module, "Scorers", Table)


def ranked(
    directory: Path, send: AnySend, *, candidates: CandidateFile | None = None
) -> tuple[GuidedArm, CallLog]:
    spec = MESSAGES_ARMS[CURATED]
    directory.mkdir(parents=True, exist_ok=True)
    log = CallLog(directory / "calls.jsonl", SETTINGS, frame="uci")
    flight = Flight(send, log, Limits(1, None), spendable=partial(spendable, log, 50.0))
    built = GuidedArm(
        CURATED,
        spec.definition,
        Knowledge("uci", {}, (), ()),
        log=log,
        budget=spec.budget or 0,
        flight=flight,
        candidates=candidates,
        curation=spec.curation,
        over_ask=spec.over_ask,
    )
    return built, log


def women(*thalach: int) -> list[object]:
    return [sick_woman(44, value) for value in thalach]


def thalach_of(seen: Seen) -> float:
    candidate = seen.candidate
    assert isinstance(candidate, dict)
    return float(candidate["thalach"])


def seen_of(built: GuidedArm) -> list[Seen]:
    return [seen for _, _, seen in built._seen]


def ranked_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The `ranked` lines the arm logged."""
    messages = [logged.getMessage() for logged in caplog.records]
    return [message for message in messages if message.startswith("ranked ")]


def ranked_prefix(cell: Cell = YOUNG_SICK, *, holdout: int = HOLDOUT, seed: int = 0) -> str:
    """A `ranked` line's fields before its counts, for `cell` on `holdout` at `seed`."""
    return f"ranked holdout={holdout} fill={KIND} arm={CURATED} seed={seed} cell={cell_key(cell)}"


RANKED: Final = ranked_prefix(seed=REQUEST.seed)


def sha_order(count: int, *, call: int = 1) -> list[int]:
    """Positions 0 to `count - 1` of `call` in `sha256(f"{holdout}:{call}:{position}")` order."""
    return sorted(
        range(count),
        key=lambda at: hashlib.sha256(f"{HOLDOUT}:{call}:{at}".encode()).hexdigest(),
    )


def test_a_ranked_arm_asks_over_ask_times_the_quota_and_keeps_the_best(tmp_path, monkeypatch):
    spec = MESSAGES_ARMS[CURATED]
    assert spec == MessagesArm(
        Definition.SUBGROUP, curation=Curation.LIKELIHOOD, over_ask=2, budget=E1.budget * 2
    )
    scores = {101: -1.0, 102: -5.0, 103: -2.0, 104: -6.0, 105: -3.0, 106: -4.0}
    scored(monkeypatch, scores)
    send = ScriptedSend(reply(women(*scores)))
    built, log = ranked(tmp_path, send)
    with log:
        baseline = built(train(), REQUEST)
    send.assert_consumed()
    assert [asked_of(call) for call in send.calls] == [spec.over_ask * QUOTA]
    assert baseline.added["thalach"].tolist() == [101, 103, 105]
    assert baseline.params["curation"] == "likelihood"
    assert baseline.params["over_ask"] == 2
    assert [(s.outcome, s.score) for s in seen_of(built)] == [
        ("kept" if value in {101, 103, 105} else "dropped-likelihood", {"likelihood": score})
        for value, score in scores.items()
    ]
    measures = built.drain(HOLDOUT, Source.REPLICA_CANDIDATES)
    assert measures[Measure.DROPPED_LIKELIHOOD] == QUOTA
    assert measures[Measure.DROPPED_QUOTA] == 0
    assert Measure.DROPPED_CURATION not in measures


def test_ties_break_in_sha256_order(tmp_path, monkeypatch):
    thalach = (101, 102, 103, 104, 105, 106)
    scored(monkeypatch, dict.fromkeys(thalach, -2.0))
    built, log = ranked(tmp_path, ScriptedSend(reply(women(*thalach))))
    with log:
        baseline = built(train(), REQUEST)
    best = sorted(sha_order(len(thalach))[:QUOTA])
    assert best != list(range(QUOTA)), "the control: sha256 order is not (call, position) order"
    assert baseline.added["thalach"].tolist() == [thalach[at] for at in best]
    kept = [s.position for s in seen_of(built) if s.outcome == "kept"]
    assert kept == best


def test_minus_infinity_ranks_last(tmp_path, monkeypatch):
    thalach = (101, 102, 103, 104, 105, 106)
    first = sha_order(len(thalach))[0]
    scores = dict.fromkeys(thalach, -2.0) | {thalach[first]: -math.inf}
    scored(monkeypatch, scores)
    built, log = ranked(tmp_path, ScriptedSend(reply(women(*thalach))))
    with log:
        baseline = built(train(), REQUEST)
    best = sorted([at for at in sha_order(len(thalach)) if at != first][:QUOTA])
    assert baseline.added["thalach"].tolist() == [thalach[at] for at in best]
    undrawable = seen_of(built)[first]
    assert (undrawable.outcome, undrawable.score) == ("dropped-likelihood", {"likelihood": None})


def test_a_ranked_attempt_logs_one_line_per_ranked_cell(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=guided_module.__name__)
    scores = {101: -1.0, 102: -5.0, 103: -2.0, 104: -6.0, 105: -3.0, 106: -4.0}
    scored(monkeypatch, scores)
    built, log = ranked(tmp_path, ScriptedSend(reply(women(*scores))))
    with log:
        built(train(), REQUEST)
    assert ranked_lines(caplog) == [f"{RANKED} rows=6 kept=3 tied=0 minus_infinity=0"]


def test_a_tie_at_the_cut_is_counted(tmp_path, monkeypatch, caplog):
    """The last kept row's -2.0 is shared by two dropped rows: three rows tie at the cut."""
    caplog.set_level(logging.INFO, logger=guided_module.__name__)
    scores = {101: -1.0, 102: -1.5, 103: -2.0, 104: -2.0, 105: -2.0, 106: -4.0}
    scored(monkeypatch, scores)
    built, log = ranked(tmp_path, ScriptedSend(reply(women(*scores))))
    with log:
        built(train(), REQUEST)
    assert ranked_lines(caplog) == [f"{RANKED} rows=6 kept=3 tied=3 minus_infinity=0"]


def test_minus_infinity_rows_are_counted(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=guided_module.__name__)
    thalach = (101, 102, 103, 104, 105, 106)
    first = sha_order(len(thalach))[0]
    scored(monkeypatch, dict.fromkeys(thalach, -2.0) | {thalach[first]: -math.inf})
    built, log = ranked(tmp_path, ScriptedSend(reply(women(*thalach))))
    with log:
        built(train(), REQUEST)
    assert ranked_lines(caplog) == [f"{RANKED} rows=6 kept=3 tied=5 minus_infinity=1"]


def test_an_unranked_attempt_logs_no_ranked_line(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=guided_module.__name__)
    send = ScriptedSend(reply(women(101, 102, 103)))
    log = CallLog(tmp_path / "calls.jsonl", SETTINGS, frame="uci")
    built = GuidedArm(
        "bko",
        None,
        Knowledge("uci", {}, (), ()),
        log=log,
        budget=E1.budget,
        flight=Flight(send, log, Limits(1, None), spendable=partial(spendable, log, 50.0)),
    )
    with log:
        baseline = built(train(), REQUEST)
    assert len(baseline.added) == QUOTA
    assert any(logged.getMessage().startswith("guided ") for logged in caplog.records)
    assert ranked_lines(caplog) == []


def test_a_ranked_line_names_its_fills_holdout_and_its_requests_seed(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=guided_module.__name__)
    scores = {101: -1.0, 102: -5.0, 103: -2.0, 104: -6.0, 105: -3.0, 106: -4.0}
    scored(monkeypatch, scores)
    built, log = ranked(tmp_path, ScriptedSend(reply(women(*scores))))
    request = Request(quota={YOUNG_SICK: QUOTA}, strata=Strata.CELL, seed=5, frame=UCI_FRAME)
    token = CURRENT_FILL.set(Fill(7, KIND))
    try:
        with log:
            built(train(), request)
    finally:
        CURRENT_FILL.reset(token)
    prefix = ranked_prefix(holdout=7, seed=5)
    assert ranked_lines(caplog) == [f"{prefix} rows=6 kept=3 tied=0 minus_infinity=0"]


def test_a_ranked_attempt_logs_its_cells_in_quota_order(tmp_path, monkeypatch, caplog):
    """Each ranked cell logs one line, the quota's cells in its order: the well cell first."""
    caplog.set_level(logging.INFO, logger=guided_module.__name__)
    sick = (101, 102, 103, 104)
    well = (181, 182, 183, 184)
    scored(monkeypatch, dict.fromkeys(sick + well, -1.0) | {101: -2.0, 181: -3.0})

    class ByLabel(ScriptedSend):
        def __call__(self, call):
            self.calls.append(call)
            if call.label == 1:
                return reply(women(*sick))
            return reply([{**well_woman(), "thalach": value} for value in well])

    built, log = ranked(tmp_path, ByLabel())
    request = Request(
        quota={YOUNG_SICK: 2, YOUNG_WELL: 2}, strata=Strata.CELL, seed=0, frame=UCI_FRAME
    )
    with log:
        built(train(), request)
    assert list(request.quota) == [YOUNG_WELL, YOUNG_SICK]
    assert ranked_lines(caplog) == [
        f"{ranked_prefix(YOUNG_WELL)} rows=4 kept=2 tied=3 minus_infinity=0",
        f"{ranked_prefix(YOUNG_SICK)} rows=4 kept=2 tied=3 minus_infinity=0",
    ]


def test_ranking_is_per_call_cell(tmp_path, monkeypatch):
    """Every well row outscores every sick row, and each cell still keeps its own best two."""
    sick = (101, 102, 103, 104)
    well = (181, 182, 183, 184)
    scores = {101: -13.0, 102: -11.0, 103: -12.0, 104: -10.0}
    scores |= {181: -1.0, 182: -3.0, 183: -2.0, 184: -4.0}
    scored(monkeypatch, scores)

    class ByLabel(ScriptedSend):
        def __call__(self, call):
            self.calls.append(call)
            if call.label == 1:
                return reply(women(*sick))
            return reply([{**well_woman(), "thalach": value} for value in well])

    send = ByLabel()
    built, log = ranked(tmp_path, send)
    request = Request(
        quota={YOUNG_SICK: 2, YOUNG_WELL: 2}, strata=Strata.CELL, seed=0, frame=UCI_FRAME
    )
    with log:
        baseline = built(train(), request)
    assert sorted(asked_of(call) for call in send.calls) == [4, 4]
    # The quota holds its cells in grid order: the well cell first.
    assert baseline.added["thalach"].tolist() == [181, 183, 102, 104]


def test_a_short_cell_keeps_everything():
    attempt = Attempt(
        train(),
        REQUEST,
        2 * QUOTA * 6,
        rank=lambda seen: [-thalach_of(item) for item in seen],
        over_ask=2,
        holdout=HOLDOUT,
    )
    attempt.add(1, YOUNG_SICK, 2, women(102, 101))
    rows = attempt.rows()
    assert rows["thalach"].tolist() == [102, 101]
    assert [(s.outcome, s.score) for s in attempt.seen] == [
        ("kept", {"likelihood": -102.0}),
        ("kept", {"likelihood": -101.0}),
    ]
    assert attempt.tally.dropped[Drop.LIKELIHOOD] == 0
    assert attempt.ranking == [Ranked(YOUNG_SICK, rows=2, kept=2, tied=0, minus_infinity=0)]


def test_a_cell_holding_its_quota_keeps_everything_with_no_tie():
    """Every row ties, but none is dropped, so none ties at the cut."""
    attempt = Attempt(
        train(), REQUEST, 2 * QUOTA * 6, rank=lambda seen: [-1.0] * len(seen), over_ask=2
    )
    attempt.add(1, YOUNG_SICK, 3, women(101, 102, 103))
    assert attempt.rows()["thalach"].tolist() == [101, 102, 103]
    assert attempt.ranking == [Ranked(YOUNG_SICK, rows=3, kept=3, tied=0, minus_infinity=0)]


def test_a_cut_at_minus_infinity_counts_no_tie():
    """The second kept row and the dropped one both read minus infinity: no finite tie."""
    request = Request(quota={YOUNG_SICK: 2}, strata=Strata.CELL, seed=0, frame=UCI_FRAME)
    table = {101: -math.inf, 102: -math.inf, 103: -2.0}
    attempt = Attempt(
        train(),
        request,
        24,
        rank=lambda seen: [table[int(thalach_of(item))] for item in seen],
        over_ask=2,
        holdout=HOLDOUT,
    )
    attempt.add(1, YOUNG_SICK, 3, women(*table))
    attempt.rows()
    assert attempt.ranking == [Ranked(YOUNG_SICK, rows=3, kept=2, tied=0, minus_infinity=2)]


def test_a_row_outside_its_calls_cell_is_dropped_for_its_cell():
    """An older sick woman from the young cell's call does not count toward the older cell."""
    old_sick = (">=55", "female", 1)
    request = Request(
        quota={YOUNG_SICK: 1, old_sick: 1}, strata=Strata.CELL, seed=0, frame=UCI_FRAME
    )
    attempt = Attempt(
        train(), request, 12, rank=lambda seen: [0.0] * len(seen), over_ask=2, holdout=HOLDOUT
    )
    attempt.add(1, YOUNG_SICK, 2, [sick_woman(44, 101), sick_woman(63, 102)])
    assert [s.outcome for s in attempt.seen] == ["kept", "dropped-cell"]
    assert attempt.short() == {YOUNG_SICK: 1, old_sick: 2}


def test_a_ranked_arm_replays_byte_for_byte(tmp_path, monkeypatch):
    scores = {101: -1.0, 102: -5.0, 103: -math.inf, 104: -6.0, 105: -3.0, 106: -4.0}
    scored(monkeypatch, scores)
    origin = Origin("e1", GRADED_DATA, RELEASES)
    files = []
    added = []
    drains = []
    live_log: Path | None = None
    for name in ("live", "replay"):
        path = tmp_path / f"{name}.jsonl"
        send: AnySend = (
            ScriptedSend(reply(women(*scores))) if live_log is None else Replay(live_log)
        )
        with CandidateFile(path, origin, SETTINGS) as kept:
            built, log = ranked(tmp_path / name, send, candidates=kept)
            with log:
                added.append(built(train(), REQUEST).added)
            drains.append(built.drain(HOLDOUT, Source.REPLICA_CANDIDATES))
        live_log = log.path
        files.append(path.read_bytes())
    assert added[0].equals(added[1])
    assert drains[0]
    assert drains[0] == drains[1]
    assert files[0] == files[1]
    assert b'"score": {"likelihood": null}' in files[0]
    assert b'"outcome": "dropped-likelihood"' in files[0]


def test_a_ranked_arm_over_an_emptied_cell_refuses_before_any_call(tmp_path):
    """The real scorer reads every quota cell's training rows before the first call is sent."""
    rows = train()
    emptied = rows.loc[[key != YOUNG_SICK for key in cells_of(rows)]]
    send = ScriptedSend(reply(women(101, 102, 103, 104, 105, 106)))
    built, log = ranked(tmp_path, send)
    with log, pytest.raises(EvaluationError, match="hold no record of cell"):
        built(emptied, REQUEST)
    assert send.calls == []


def test_a_ranked_arm_keeps_each_call_cells_best_by_the_real_likelihood(tmp_path):
    """No scorer is replaced: the kept rows are the young sick cell's top three by `Likelihood`."""
    knowledge = Knowledge("uci", {}, (), FIELD_MAPPINGS)
    candidates: list[object] = [
        {**sick_woman(44, 101 + at), **change} for at, change in enumerate(REAL_CHANGES)
    ]
    send = ScriptedSend(reply(candidates))
    log = CallLog(tmp_path / "calls.jsonl", SETTINGS, frame="uci")
    built = GuidedArm(
        CURATED,
        None,
        knowledge,
        log=log,
        budget=E1.budget * 2,
        flight=Flight(send, log, Limits(1, None), spendable=partial(spendable, log, 50.0)),
        curation=Curation.LIKELIHOOD,
        over_ask=2,
    )
    with log:
        baseline = built(train(), REQUEST)
    send.assert_consumed()
    likelihood = Scorers(train(), knowledge, UCI_FRAME)(YOUNG_SICK)
    scores = [likelihood(record(item, UCI_FRAME.columns)) for item in candidates]
    order = sorted(range(len(candidates)), key=lambda at: (-scores[at], tie(HOLDOUT, 1, at)))
    best = sorted(order[:QUOTA])
    assert len(set(scores)) > QUOTA, "the control: the real scores tell the candidates apart"
    assert best != list(range(QUOTA)), "the control: the best are not the first asked"
    assert baseline.added["thalach"].tolist() == [101 + at for at in best]
    assert [s.score for s in seen_of(built)] == [{"likelihood": score} for score in scores]


def test_ranking_scores_are_bit_equal_to_the_pinned_scores():
    """The young sick cell's real scores, pinned to the bit as a per-cell build reads them."""
    knowledge = Knowledge("uci", {}, (), FIELD_MAPPINGS)
    candidates = [{**sick_woman(44, 101 + at), **change} for at, change in enumerate(REAL_CHANGES)]
    likelihood = Scorers(train(), knowledge, UCI_FRAME)(YOUNG_SICK)
    scores = [likelihood(record(item, UCI_FRAME.columns)) for item in candidates]
    assert [float(score).hex() for score in scores] == [
        "-0x1.d2bac8afcef3ap+2",
        "-0x1.7f9354de89457p+2",
        "-0x1.2876775bba37dp+2",
        "-0x1.5404e61d21be9p+2",
        "-0x1.a72c59ee676cdp+2",
        "-0x1.5404e61d21beap+2",
    ]


def test_a_holdouts_label_groups_are_observed_once(monkeypatch):
    """Two cells of one label share one `Scorers`: one stratum pass, one observation per label."""
    observed: list[str] = []
    keyed: list[int] = []
    observe = ranking_module.observe_records
    keys = ranking_module.stratum_keys

    def observe_spy(*args, **kwargs):
        observed.append(kwargs["name"])
        return observe(*args, **kwargs)

    def keys_spy(*args, **kwargs):
        keyed.append(len(args[0]))
        return keys(*args, **kwargs)

    monkeypatch.setattr(ranking_module, "observe_records", observe_spy)
    monkeypatch.setattr(ranking_module, "stratum_keys", keys_spy)
    scorers = Scorers(train(), Knowledge("uci", {}, (), FIELD_MAPPINGS), UCI_FRAME)
    old_sick = (">=55", "female", 1)
    for cell in (YOUNG_SICK, old_sick, YOUNG_WELL, YOUNG_SICK):
        scorers(cell)
    assert keyed == [len(train())]
    assert observed == [f"uci/{UCI_FRAME.label}=1", f"uci/{UCI_FRAME.label}=0"]


def test_a_second_cell_of_a_label_scores_bit_equal_to_its_own_build():
    """A cell scored after another of its label reads the label group the first one observed.

    The pins are the old per-cell `scorer`'s scores of the young sick male cell, which observed
    its label group afresh.
    """
    knowledge = Knowledge("uci", {}, (), FIELD_MAPPINGS)
    candidates = [{**sick_woman(44, 101 + at), **change} for at, change in enumerate(REAL_CHANGES)]
    scorers = Scorers(train(), knowledge, UCI_FRAME)
    scorers(YOUNG_SICK)
    likelihood = scorers(("<55", "male", 1))
    scores = [likelihood(record(item, UCI_FRAME.columns)) for item in candidates]
    assert [float(score).hex() for score in scores] == [
        "-0x1.ef86334b5ec7fp+2",
        "-0x1.9c5ebf7a1919cp+2",
        "-0x1.212611b9562ecp+2",
        "-0x1.682dd325d81ccp+2",
        "-0x1.bb5546f71dcafp+2",
        "-0x1.5556fe0d972bcp+2",
    ]


def test_a_ranked_arm_builds_its_cells_scorers_from_one_stratum_pass(tmp_path, monkeypatch):
    keyed: list[int] = []
    keys = ranking_module.stratum_keys

    def keys_spy(*args, **kwargs):
        keyed.append(len(args[0]))
        return keys(*args, **kwargs)

    monkeypatch.setattr(ranking_module, "stratum_keys", keys_spy)
    spec = MESSAGES_ARMS[CURATED]
    log = CallLog(tmp_path / "calls.jsonl", SETTINGS, frame="uci")
    built = GuidedArm(
        CURATED,
        spec.definition,
        Knowledge("uci", {}, (), FIELD_MAPPINGS),
        log=log,
        budget=spec.budget or 0,
        flight=Flight(
            ScriptedSend(), log, Limits(1, None), spendable=partial(spendable, log, 50.0)
        ),
        curation=spec.curation,
        over_ask=spec.over_ask,
    )
    request = Request(
        quota={YOUNG_SICK: 1, YOUNG_WELL: 1, (">=55", "female", 1): 1},
        strata=Strata.CELL,
        seed=0,
        frame=UCI_FRAME,
    )
    built._rank(train(), request)
    assert keyed == [len(train())]


def test_the_scorer_refuses_a_cell_its_training_rows_lack():
    rows = train()
    emptied = rows.loc[[key != YOUNG_SICK for key in cells_of(rows)]]
    knowledge = Knowledge("uci", {}, (), FIELD_MAPPINGS)
    with pytest.raises(EvaluationError, match="training rows hold no record of cell"):
        Scorers(emptied, knowledge, UCI_FRAME)(YOUNG_SICK)


def test_record_reads_every_column_as_a_float_and_refuses_anything_else():
    columns = UCI_FRAME.columns
    woman = sick_woman(44, 101)
    read = record(woman, columns)
    assert list(read) == list(columns)
    assert all(type(value) is float for value in read.values())
    assert read == {column: float(woman[column]) for column in columns}
    partial_record = {"age": 44}
    for candidate in (None, [1, 2], partial_record):
        with pytest.raises(EvaluationError, match="a record of every column"):
            record(candidate, columns)


@pytest.mark.parametrize(("over_ask", "curation"), [(2, None), (1, Curation.LIKELIHOOD), (0, None)])
def test_a_guided_arm_asks_over_ask_with_likelihood_curation_only(tmp_path, over_ask, curation):
    log = CallLog(tmp_path / "calls.jsonl", SETTINGS, frame="uci")
    flight = Flight(ScriptedSend(), log, Limits(1, None), spendable=partial(spendable, log, 50.0))
    with pytest.raises(
        EvaluationError, match=rf"^{CURATED}: over_ask {over_ask} .*above 1 exactly"
    ):
        GuidedArm(
            CURATED,
            None,
            Knowledge("uci", {}, (), ()),
            log=log,
            budget=1,
            flight=flight,
            curation=curation,
            over_ask=over_ask,
        )


@pytest.mark.parametrize(("over_ask", "ranks"), [(2, False), (0, True), (0, False)])
def test_an_attempt_asks_over_ask_with_a_rank_only(over_ask, ranks):
    with pytest.raises(ValueError, match="above 1 exactly when the arm ranks"):
        Attempt(
            train(),
            REQUEST,
            12,
            rank=(lambda seen: [0.0] * len(seen)) if ranks else None,
            over_ask=over_ask,
        )


@pytest.mark.parametrize(
    ("over_ask", "ranks", "reason"),
    [
        (1, False, None),
        (2, True, None),
        (2, False, "is above 1 but the arm does not rank"),
        (1, True, "is 1 but the arm ranks, so it has nothing to drop"),
        (0, False, "is below 1"),
        (0, True, "is below 1"),
    ],
)
def test_every_reader_of_over_ask_states_one_rule(tmp_path, over_ask, ranks, reason):
    """`MessagesArm`, `GuidedArm` and `Attempt` accept and refuse the same `over_ask`, alike."""
    rule = re.escape(
        f"over_ask {over_ask} {reason}; over_ask is 1 or more, and above 1 exactly when the arm "
        "ranks"
    )
    curation = Curation.LIKELIHOOD if ranks else None
    log = CallLog(tmp_path / "calls.jsonl", SETTINGS, frame="uci")
    flight = Flight(ScriptedSend(), log, Limits(1, None), spendable=partial(spendable, log, 50.0))
    readers: list[tuple[type[Exception], Callable[[], object]]] = [
        (ValueError, lambda: MessagesArm(None, curation=curation, over_ask=over_ask)),
        (
            EvaluationError,
            lambda: GuidedArm(
                CURATED,
                None,
                Knowledge("uci", {}, (), ()),
                log=log,
                budget=1,
                flight=flight,
                curation=curation,
                over_ask=over_ask,
            ),
        ),
        (
            ValueError,
            lambda: Attempt(
                train(),
                REQUEST,
                12,
                rank=(lambda seen: [0.0] * len(seen)) if ranks else None,
                over_ask=over_ask,
            ),
        ),
    ]
    for error, build in readers:
        if reason is None:
            build()
        else:
            with pytest.raises(error, match=rule):
                build()


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_rank_scoring_nan_or_plus_infinity_is_refused_and_ranks_again(bad):
    attempt = Attempt(
        train(),
        REQUEST,
        2 * QUOTA * 6,
        rank=lambda seen: [bad if thalach_of(item) == 102 else -1.0 for item in seen],
        over_ask=2,
        holdout=HOLDOUT,
    )
    attempt.add(1, YOUNG_SICK, 4, women(101, 102, 103, 104))
    for _ in range(2):
        with pytest.raises(ValueError, match=r"<55/female/1"):
            attempt.rows()
    assert attempt.tally.dropped[Drop.LIKELIHOOD] == 0


def test_the_curated_arm_never_shows_training_records():
    assert MESSAGES_ARMS[CURATED_ARM].shows is False
