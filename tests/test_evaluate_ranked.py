"""The ranking arm `bko-curated`: it asks twice each quota and keeps each call cell's best rows.

Every call goes to a `ScriptedSend` or a replay; no test reaches the API. The scorer is replaced by
a table of scores by `thalach`, so each test fixes the ranking it reads.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator, Mapping
from functools import partial
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

import bko.evaluate.guided as guided_module
from bko.baselines.fills import CURRENT_FILL, Fill
from bko.baselines.records import Drop
from bko.baselines.request import Request, Strata
from bko.datasets.frames import FrameSpec
from bko.evaluate.attempts import Attempt
from bko.evaluate.experiment import E1, MESSAGES_ARMS, Curation, MessagesArm
from bko.evaluate.guided import GuidedArm, Knowledge
from bko.evaluate.holdout_candidates import CandidateFile, Origin, Seen
from bko.evaluate.ranking import Known
from bko.evaluate.scores import Measure, Source
from bko.evaluate.waves import spendable
from bko.gaps.definitions import Definition
from bko.gaps.observe import Cell
from bko.generate.flight import Flight, Limits
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, AnySend
from bko.generate.replay import Replay
from bko.ontology.mapping import RELEASES
from tests._evaluate_fixtures import GRADED_DATA
from tests._generate_fixtures import ScriptedSend, reply
from tests.test_evaluate_cllm import YOUNG_SICK, YOUNG_WELL, asked_of, sick_woman, train, well_woman


CURATED: Final = "bko-curated"
HOLDOUT: Final = 0
QUOTA: Final = 3
REQUEST: Final = Request(quota={YOUNG_SICK: QUOTA}, strata=Strata.CELL, seed=0)

pytestmark = pytest.mark.usefixtures("fill")


@pytest.fixture
def fill() -> Iterator[None]:
    """Each test runs inside holdout 0's replica fill, whose holdout the tie order reads."""
    token = CURRENT_FILL.set(Fill(HOLDOUT, "replica"))
    yield
    CURRENT_FILL.reset(token)


def scored(monkeypatch: pytest.MonkeyPatch, table: Mapping[int, float]) -> None:
    """Score each record by its `thalach` from `table`, whatever its cell."""

    def scorer(
        train: pd.DataFrame, data: Known, frame: FrameSpec, cell: Cell
    ) -> Callable[[Mapping[str, float]], float]:
        return lambda record: table[int(record["thalach"])]

    monkeypatch.setattr(guided_module, "scorer", scorer)


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
    request = Request(quota={YOUNG_SICK: 2, YOUNG_WELL: 2}, strata=Strata.CELL, seed=0)
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


def test_a_row_outside_its_calls_cell_is_dropped_for_its_cell():
    """An older sick woman from the young cell's call does not count toward the older cell."""
    old_sick = (">=55", "female", 1)
    request = Request(quota={YOUNG_SICK: 1, old_sick: 1}, strata=Strata.CELL, seed=0)
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
            built.drain(HOLDOUT, Source.REPLICA_CANDIDATES)
        live_log = log.path
        files.append(path.read_bytes())
    assert added[0].equals(added[1])
    assert files[0] == files[1]
    assert b'"score": {"likelihood": null}' in files[0]
    assert b'"outcome": "dropped-likelihood"' in files[0]
