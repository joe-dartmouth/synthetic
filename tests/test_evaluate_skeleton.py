"""The prescribing arm `bko-skeleton`: lines drawn once per attempt, re-asked under waves."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Final

import pytest

from bko.baselines.errors import BaselineError
from bko.baselines.request import Request, Strata
from bko.datasets.uci import UCI_FRAME
from bko.evaluate import skeleton as prescribing
from bko.evaluate.attempts import COMMON
from bko.evaluate.data import Data
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import (
    ADDS_ROWS,
    E1,
    MESSAGES_ARMS,
    METHOD_ARMS,
    RUNS,
    SKELETON_ARM,
)
from bko.evaluate.guided import GuidedArm, spendable
from bko.evaluate.rehearse import CannedSend
from bko.evaluate.scores import Measure, Source
from bko.evaluate.skeleton import Queues, prescriber
from bko.evaluate.validity import validity
from bko.gaps.definitions import Definition
from bko.generate.direct import label_only, sample
from bko.generate.flight import Flight, Limits
from bko.generate.guidance import Guidance, intervals
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Call, Level, Reply, Settings
from bko.generate.prompt import render
from bko.generate.skeleton import prescribable, skeleton
from bko.ontology.mapping import FIELD_MAPPINGS
from bko.ontology.schema import shown
from tests._evaluate_bko_fixtures import UCI_E1
from tests._evaluate_fixtures import GRADED, SMALL
from tests._fusion_fixtures import training_split, young
from tests._generate_fixtures import ScriptedSend, reply
from tests._guided_fixtures import REQUEST, YOUNG_SICK, YOUNG_WELL, guided, knowledge, train
from tests.test_evaluate_validity import FULL, counts


LINE: Final = re.compile(r"^Record (\d+): (.*)$", re.MULTILINE)


def _lines(call: Call) -> list[str]:
    """Each record line's text, in order."""
    return [found[2] for found in LINE.finditer(call.user)]


class FollowSend:
    """A `Send` answering each record as its line asks, cp 1 for angina and 2 for none.

    Attributes:
        calls: Every call made, in order.
        flip: Per call number, from 1, the record positions answered against their line.
    """

    def __init__(self, flip: Mapping[int, frozenset[int]] | None = None) -> None:
        """Answer against the line at each position `flip` names."""
        self.calls: list[Call] = []
        self.flip = dict(flip or {})

    def __call__(self, call: Call) -> Reply:
        """Record `call` and answer one record per line."""
        self.calls.append(call)
        flipped = self.flip.get(len(self.calls), frozenset())
        records: list[object] = []
        for position, text in enumerate(_lines(call)):
            angina = (text == "Angina pectoris") != (position in flipped)
            records.append(young(call.label, cp=1 if angina else 2))
        return reply(records)


BOTH: Final = Request(quota={YOUNG_WELL: 3, YOUNG_SICK: 3}, strata=Strata.CELL, seed=3)
"""Three rows of each young-women cell: two chains, so one wave of two calls. On `UCI_E1`'s
training split, seed 3 draws lines that differ within each cell."""


def _arm(
    directory: Path, send: FollowSend, *, in_flight: int = 8, budget: int = 3
) -> tuple[GuidedArm, CallLog]:
    log = CallLog(directory / "calls.jsonl", SETTINGS, frame="uci")
    limits = Limits(in_flight, None)
    flight = Flight(send, log, limits, spendable=partial(spendable, log, 50.0), ceiling=50.0)
    known = knowledge(directory / "data")
    arm = GuidedArm(
        SKELETON_ARM,
        Definition.SUBGROUP,
        known,
        log=log,
        budget=budget,
        flight=flight,
        prescribe=prescriber(known),
    )
    return arm, log


def test_a_top_up_re_asks_the_unconsumed_lines_under_waves(tmp_path):
    send = FollowSend({1: frozenset({2}), 2: frozenset({0, 1})})
    arm, log = _arm(tmp_path, send)
    with log:
        baseline = arm(training_split(tmp_path / "train", UCI_E1), BOTH)
    assert len(baseline.added) == 6
    first, second, third, fourth = send.calls
    assert [call.label for call in send.calls] == [0, 1, 0, 1]
    angina, none = "Angina pectoris", "none of these findings"
    assert (_lines(first), _lines(second)) == ([angina, angina, none], [none, angina, angina])
    assert _lines(third) == [none]
    assert _lines(fourth) == [none, angina]
    found = arm.drain(0, Source.CANDIDATES)
    assert found[Measure.DROPPED_CONCEPTS] == 3
    assert found[Measure.DROPPED_QUOTA] == 0


def test_lines_are_drawn_once_per_request_and_equal_bko_directs_draw(tmp_path, monkeypatch):
    drawn: list[int] = []

    def counted(*args, **kwargs):
        drawn.append(kwargs["seed"])
        return sample(*args, **kwargs)

    monkeypatch.setattr(prescribing, "sample", counted)
    send = FollowSend({1: frozenset({0})})
    arm, log = _arm(tmp_path, send, in_flight=1)
    rows = train(tmp_path / "train")
    with log:
        arm(rows, REQUEST)
    assert drawn == [REQUEST.seed]
    assert len(send.calls) == 2
    known = knowledge(tmp_path / "data")
    direct = sample(
        rows,
        REQUEST.quota,
        rows=known.field_rows,
        ontologies=known.ontologies,
        name=known.name,
        seed=REQUEST.seed,
        frame=REQUEST.frame,
    )
    following = label_only(known.field_rows)
    expected = {
        cell: tuple(d.concepts - following for d in draws) for cell, draws in direct.items()
    }
    assert prescriber(known)(rows, REQUEST) == expected
    asked = [
        frozenset({"HP:9000003"}) if text == "Angina pectoris" else frozenset()
        for text in _lines(send.calls[0])
    ]
    assert tuple(asked) == expected[YOUNG_SICK]


def test_existing_arms_keep_their_attempt_line_and_measures(tmp_path):
    arm, log = guided(tmp_path, ScriptedSend(reply([young(1, chol=234)] * 3)), budget=1)
    attempt_line = (
        "asked=3 generated=3 passed=3 dropped_range=3 dropped_cell=0 dropped_quota=0 kept=0"
    )
    with log, pytest.raises(BaselineError, match=re.escape(attempt_line)):
        arm(train(tmp_path / "train"), REQUEST)
    found = arm.drain(0, Source.CANDIDATES)
    assert Measure.DROPPED_CONCEPTS not in found
    assert arm.totals.reasons == COMMON


def test_a_prescribing_arm_reports_dropped_concepts(tmp_path, caplog):
    caplog.set_level("INFO", logger="bko.evaluate.guided")
    arm, log = _arm(tmp_path, FollowSend({1: frozenset({0})}))
    with log:
        arm(train(tmp_path / "train"), REQUEST)
    found = arm.drain(0, Source.CANDIDATES)
    assert list(found) == [
        Measure.GENERATED,
        Measure.PASSED,
        Measure.DROPPED_RANGE,
        Measure.DROPPED_CELL,
        Measure.DROPPED_CONCEPTS,
        Measure.DROPPED_QUOTA,
    ]
    assert found[Measure.DROPPED_CONCEPTS] == 1
    assert any(
        "dropped_cell=0 dropped_concepts=1 dropped_quota=0 kept=3" in r.message
        for r in caplog.records
    )


def test_validity_sums_dropped_concepts_only_for_arms_that_hold_it():
    experiment = dataclasses.replace(SMALL, replica_arms=())
    plain = validity(counts("bko", FULL), experiment, "bko")
    assert "concepts" not in plain.line()
    assert Measure.DROPPED_CONCEPTS not in plain.counts
    held = counts(SKELETON_ARM, {**FULL, Measure.DROPPED_CONCEPTS: 2.0})
    found = validity([*counts("bko", FULL), *held], experiment, SKELETON_ARM)
    assert found.counts[Measure.DROPPED_CONCEPTS] == 4
    assert "dropped_quota=0 dropped_concepts=4 rate=0.9500" in found.line()
    with pytest.raises(EvaluationError, match="dropped-concepts"):
        validity(held[:-1], experiment, SKELETON_ARM)


def test_queues_hand_back_lines_at_the_end_and_refuse_running_short():
    one, two, three = frozenset({"a"}), frozenset(), frozenset({"b"})
    queues = Queues({YOUNG_SICK: (one, two, three)})
    assert queues.take(YOUNG_SICK, 2) == (one, two)
    queues.give_back(YOUNG_SICK, [one])
    assert queues.take(YOUNG_SICK, 2) == (three, one)
    with pytest.raises(EvaluationError, match="0 lines for 1 records"):
        queues.take(YOUNG_SICK, 1)


SICK_GUIDANCE: Final = Guidance(
    cell=YOUNG_SICK,
    sex=0,
    label=1,
    records=4,
    interval=intervals(YOUNG_SICK, GRADED, frame=UCI_FRAME),
)


def test_canned_records_show_their_prescribed_concepts():
    data = Data(GRADED, UCI_FRAME, FIELD_MAPPINGS, "uci")
    send = CannedSend(data, SMALL, SETTINGS)
    lines = (
        frozenset({"HP:0001681", "HP:0100749", "HP:0003124"}),
        frozenset(),
        frozenset({"HP:0012250", "HP:0003115", "HP:0001712"}),
        frozenset({"HP:0001681", "HP:0004421", "HP:0003074", "HP:0001717", "HP:0500018"}),
        frozenset({"HP:0100749", "HP:0003115"}),
    )
    prescribed = skeleton(lines, FIELD_MAPPINGS)
    system, user = render(SICK_GUIDANCE, len(lines), Level.SUMMARY, skeleton=prescribed)
    call = Call(SETTINGS, 0, 1, system, user, frame=UCI_FRAME)
    answer = send(call).text
    assert answer is not None
    records = json.loads(answer)["records"]
    rows = prescribable(FIELD_MAPPINGS)
    assert [shown(record, rows) for record in records] == list(lines)


def test_g5_is_e1_plus_the_skeleton_arm_with_literal_settings():
    run = RUNS["g5"]
    assert run.experiment == dataclasses.replace(
        E1,
        arms=(*METHOD_ARMS, "bko", "bko-skeleton"),
        candidate="bko-skeleton",
        comparators=METHOD_ARMS,
        replica_arms=(*ADDS_ROWS, "bko", "bko-skeleton"),
        spend_ceiling=25.0,
        criteria=(),
    )
    assert run.settings == Settings(
        model="claude-haiku-5-5",
        effort="medium",
        max_tokens=16000,
        per_call=20,
        guidance=Level.SUMMARY,
    )
    assert run.settings == SETTINGS
    assert MESSAGES_ARMS[SKELETON_ARM] is Definition.SUBGROUP
