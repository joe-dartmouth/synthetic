"""The prescribing arm `bko-skeleton`: lines drawn once per attempt, re-asked under waves."""

from __future__ import annotations

import dataclasses
import io
import json
import re
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Final

import pytest

from bko.baselines.errors import BaselineError
from bko.baselines.records import Drop
from bko.baselines.request import Request, Strata
from bko.datasets.uci import UCI_FRAME
from bko.evaluate import skeleton as prescribing
from bko.evaluate.__main__ import main
from bko.evaluate.attempts import COMMON, Attempt
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
from bko.evaluate.guided import GuidedArm, _committer, spendable
from bko.evaluate.rehearse import CannedSend
from bko.evaluate.scores import Measure, Source
from bko.evaluate.skeleton import Queues, prescriber
from bko.evaluate.validity import validity
from bko.gaps.definitions import Definition
from bko.generate.calls import Batch
from bko.generate.decode import Reason, Undecoded
from bko.generate.direct import label_only, sample
from bko.generate.flight import Flight, Limits, Planned
from bko.generate.guidance import Guidance, intervals
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Call, Level, Reply, Settings
from bko.generate.prompt import render
from bko.generate.skeleton import prescribable, skeleton
from bko.ontology.mapping import FIELD_MAPPINGS
from bko.ontology.schema import Codes, FieldMapping, shown
from tests._evaluate_bko_fixtures import NARROW, UCI_E1, context, score_argv
from tests._evaluate_fixtures import GRADED, SMALL
from tests._fusion_fixtures import training_split, young
from tests._gaps_fixtures import ANGINA, STENOSIS
from tests._generate_fixtures import ScriptedSend, reply
from tests._guided_fixtures import REQUEST, YOUNG_SICK, YOUNG_WELL, guided, knowledge, train
from tests._validate_fixtures import planted
from tests.test_evaluate_validity import FULL, counts


LINE: Final = re.compile(r"^Record (\d+): (.*)$", re.MULTILINE)


def _lines(call: Call) -> list[str]:
    """Each record line's text, in order."""
    return [found[2] for found in LINE.finditer(call.user)]


ASK: Final = re.compile(r"Write exactly \d+ records .*?, age (\d+) to (\d+) years")

type Positions = Mapping[int, frozenset[int]]
"""Per call number, from 1, a set of record positions."""


class FollowSend:
    """A `Send` answering each record as its line asks, cp 1 for angina and 2 for none.

    Each record is aged 40, or 63 when the call's band leaves 40 out.

    Attributes:
        calls: Every call made, in order.
        flip: Per call, the positions answered against their line.
        spoil: Per call, the positions answered with a cholesterol outside the training range.
        older: Per call, the positions aged 63, the older band of the call's sex and label.
        chest: Whether every record has cp 3, which shows `CHEST` and no line ever names.
    """

    def __init__(
        self,
        flip: Positions | None = None,
        *,
        spoil: Positions | None = None,
        older: Positions | None = None,
        chest: bool = False,
    ) -> None:
        """Answer each record as its line asks, but at the positions each mapping names."""
        self.calls: list[Call] = []
        self.flip = dict(flip or {})
        self.spoil = dict(spoil or {})
        self.older = dict(older or {})
        self.chest = chest

    def __call__(self, call: Call) -> Reply:
        """Record `call` and answer one record per line."""
        self.calls.append(call)
        number = len(self.calls)
        found = ASK.search(call.user)
        assert found is not None, call.user
        band = 40 if int(found[1]) <= 40 <= int(found[2]) else 63
        records: list[object] = []
        for position, text in enumerate(_lines(call)):
            angina = (text == "Angina pectoris") != (position in self.flip.get(number, ()))
            changes: dict[str, object] = {"cp": 3 if self.chest else 1 if angina else 2}
            if position in self.spoil.get(number, ()):
                changes["chol"] = 234
            age = 63 if position in self.older.get(number, ()) else band
            records.append(planted(age=age, sex=call.sex, target=call.label, **changes))
        return reply(records)


BOTH: Final = Request(quota={YOUNG_WELL: 3, YOUNG_SICK: 3}, strata=Strata.CELL, seed=3)
"""Three rows of each young-women cell: two chains, so one wave of two calls. On `UCI_E1`'s
training split, seed 3 draws lines that differ within each cell."""


def _arm(
    directory: Path,
    send: FollowSend,
    *,
    in_flight: int = 8,
    budget: int = 3,
    settings: Settings = SETTINGS,
) -> tuple[GuidedArm, CallLog]:
    log = CallLog(directory / "calls.jsonl", settings, frame="uci")
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
    send = FollowSend({1: frozenset({2}), 2: frozenset({0})})
    arm, log = _arm(tmp_path, send)
    with log:
        baseline = arm(training_split(tmp_path / "train", UCI_E1), BOTH)
    assert len(baseline.added) == 6
    first, second, third, fourth = send.calls
    assert [call.label for call in send.calls] == [0, 1, 0, 1]
    angina, none = "Angina pectoris", "none of these findings"
    assert (_lines(first), _lines(second)) == ([angina, angina, none], [none, angina, angina])
    assert _lines(third) == [none]
    assert _lines(fourth) == [none]
    found = arm.drain(0, Source.CANDIDATES)
    assert found[Measure.DROPPED_CONCEPTS] == 2
    assert found[Measure.DROPPED_QUOTA] == 0


OLD_SICK: Final = (">=55", "female", 1)

SIBLINGS: Final = Request(quota={YOUNG_SICK: 4, OLD_SICK: 1}, strata=Strata.CELL, seed=6)
"""Two cells of one sex and label, so one wave each. On `UCI_E1`'s training split, seed 6 draws
the young cell's lines angina, angina, none, angina and the older cell's angina."""

ANGINA_LINE: Final = "Angina pectoris"
NONE_LINE: Final = "none of these findings"


def test_a_record_kept_in_a_sibling_cell_is_dropped_and_its_line_re_asked(tmp_path):
    send = FollowSend(older={1: frozenset({1})})
    arm, log = _arm(tmp_path, send)
    with log:
        baseline = arm(training_split(tmp_path / "train", UCI_E1), SIBLINGS)
    assert len(baseline.added) == 5
    a, n = ANGINA_LINE, NONE_LINE
    assert [_lines(call) for call in send.calls] == [[a, a, n, a], [a], [a]]
    assert ["age 40 to" in call.user for call in send.calls] == [True, False, True]
    found = arm.drain(0, Source.CANDIDATES)
    assert found[Measure.DROPPED_CELL] == 1
    assert found[Measure.DROPPED_CONCEPTS] == 0


PER_TWO: Final = dataclasses.replace(SETTINGS, per_call=2)
"""Two records per call, so one cell's ask spans more than one call of a wave."""


def test_a_wave_assigns_lines_in_plan_order_and_re_queues_them_in_call_order(tmp_path):
    send = FollowSend(spoil={1: frozenset({0, 1}), 2: frozenset({0})})
    arm, log = _arm(tmp_path, send, settings=PER_TWO)
    with log:
        arm(training_split(tmp_path / "train", UCI_E1), SIBLINGS)
    a, n = ANGINA_LINE, NONE_LINE
    # The first wave's two calls take the young cell's lines in plan order; the three spoiled
    # records' lines return in call order, then line order, and the top-up asks them so.
    assert [_lines(call) for call in send.calls] == [[a, a], [n, a], [a], [a, a], [n]]
    assert arm.drain(0, Source.CANDIDATES)[Measure.DROPPED_RANGE] == 3


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


def test_a_call_keeps_reordered_records_and_drops_only_a_skipped_ones_line(tmp_path):
    rows = train(tmp_path / "train")
    angina, none = frozenset({ANGINA.concept}), frozenset()
    lined = skeleton((angina, none, none), (ANGINA, STENOSIS))
    reordered = Attempt(rows, REQUEST, 9, prescribing=True)
    sick, well = young(1, cp=1), young(1, cp=2)
    assert reordered.add(1, YOUNG_SICK, 3, [well, well, sick], skeleton=lined) == {0, 1, 2}
    assert reordered.tally.dropped[Drop.CONCEPTS] == 0
    assert reordered.held() == {YOUNG_SICK: 3}
    skipped = Attempt(rows, REQUEST, 9, prescribing=True)
    short = [well, well, Undecoded(Reason.SHORT)]
    assert skipped.add(1, YOUNG_SICK, 3, short, skeleton=lined) == {1, 2}
    assert skipped.tally.dropped[Drop.CONCEPTS] == 0
    assert skipped.held() == {YOUNG_SICK: 2}


CHEST: Final = FieldMapping(
    when=(Codes("cp", frozenset({3})),),
    concept="HP:9000002",
    label="Chest pain",
    source="uci-heart-disease",
    limits="test",
)
"""A prescribable concept no `UCI_E1` training record shows, so no draw names it."""


def test_a_starved_arm_refuses_every_attempt_and_the_run_exits_2(tmp_path, caplog):
    built, _ = context(FollowSend(chest=True))
    narrow = dataclasses.replace(NARROW, arms=("none", SKELETON_ARM), candidate=SKELETON_ARM)
    built = dataclasses.replace(built, experiment=narrow, field_rows=(ANGINA, STENOSIS, CHEST))
    log = tmp_path / "calls.jsonl"
    caplog.set_level("INFO")
    assert main(score_argv(log, tmp_path), context=built, out=io.StringIO()) == 2
    retries = [m for m in caplog.messages if m.startswith("retry holdout=0 arm=bko-skeleton ")]
    assert [m.split()[3] for m in retries] == [f"attempt={n}" for n in range(1, 5)]
    assert "refused holdout=0 arm=bko-skeleton attempts=5" in caplog.messages
    outcomes = [m.split()[3] for m in caplog.messages if m.startswith("guided ")]
    assert outcomes == ["outcome=refused"] * 5
    end = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert (end["kind"], end["status"], end["retryable"]) == ("end", "failed", False)


def test_a_batch_answering_another_cell_than_its_planned_call_is_refused(tmp_path):
    attempt = Attempt(train(tmp_path / "train"), REQUEST, 9, prescribing=True)
    planned = [Planned(SICK_GUIDANCE, 1, skeleton((frozenset(),), (ANGINA, STENOSIS)))]
    commit = _committer(attempt, planned, Queues({YOUNG_SICK: ()}))
    batch = Batch(1, YOUNG_WELL, 1, (young(0, cp=2),), reply([young(0, cp=2)]), 0.0)
    with pytest.raises(EvaluationError, match="answered cell <55/female/0, but its planned call"):
        commit(batch)


def test_a_tally_refuses_counts_reporting_other_drop_reasons():
    with pytest.raises(ValueError, match="cannot add"):
        Attempt.tally_for(prescribing=False).add(Attempt.tally_for(prescribing=True))
