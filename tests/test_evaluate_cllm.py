"""The CLLM arms: each call shows seeded training records, and `cllm` curates what comes back."""

from __future__ import annotations

import json
import logging
from functools import partial
from pathlib import Path

import pandas as pd
import pytest

from bko.baselines.errors import BaselineError
from bko.baselines.request import Request, Strata
from bko.datasets.cells import cell_keys
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.cllm import SHOW, Shower
from bko.evaluate.experiment import CLLM_ARM, CLLM_RAW_ARM, MESSAGES_ARMS
from bko.evaluate.guided import SEQUENTIAL, GuidedArm, Knowledge, guided_arms
from bko.evaluate.holdout_candidates import Origin
from bko.evaluate.named_runs import RUNS
from bko.evaluate.scores import Measure, Source
from bko.evaluate.waves import spendable
from bko.generate.flight import Flight, Limits
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, AnySend
from bko.generate.replay import Replay
from bko.ontology.mapping import RELEASES
from tests._evaluate_fixtures import GRADED_DATA, SMALL
from tests._generate_fixtures import ScriptedSend, reply
from tests.test_evaluate_curation import separable


YOUNG_SICK = ("<55", "female", 1)
QUOTA = 3


def train() -> pd.DataFrame:
    return separable().loc[:, list(UCI_FRAME.columns)]


def cells_of(rows: pd.DataFrame) -> list[tuple[object, ...]]:
    return list(cell_keys(rows, edges=UCI_FRAME.edges).itertuples(index=False, name=None))


def shown_frame(shown) -> pd.DataFrame:
    return pd.DataFrame(list(shown), columns=list(UCI_FRAME.columns))


def test_the_prompt_shows_the_cells_records_first_then_its_group_at_one_seed():
    rows = train()
    first = Shower(rows, frame=UCI_FRAME, seed=7).take(YOUNG_SICK)
    assert first == Shower(rows, frame=UCI_FRAME, seed=7).take(YOUNG_SICK)
    assert first != Shower(rows, frame=UCI_FRAME, seed=8).take(YOUNG_SICK)
    shown = shown_frame(first)
    assert len(shown) == SHOW
    assert not shown.duplicated().any()
    own = sum(key == YOUNG_SICK for key in cells_of(rows))
    keys = cells_of(shown)
    assert all(key == YOUNG_SICK for key in keys[:own])
    assert all(key[1:] == YOUNG_SICK[1:] for key in keys[own:])
    assert all(tuple(record) == UCI_FRAME.columns for record in first)


def test_each_call_of_an_attempt_draws_on():
    shower = Shower(train(), frame=UCI_FRAME, seed=7)
    assert shower.take(YOUNG_SICK) != shower.take(YOUNG_SICK)


def records() -> list[object]:
    """Four young sick women: typical, one the curator puts at the other label, then typical."""
    rows = train()
    template = rows[[key == YOUNG_SICK for key in cells_of(rows)]].iloc[0]
    found: list[object] = []
    for thalach in (100, 190, 105, 110):
        record: dict[str, int | float] = {
            name: float(template[name]) if name in UCI_FRAME.fractional else int(template[name])
            for name in UCI_FRAME.columns
        }
        record["thalach"] = thalach
        found.append(record)
    return found


def arm(directory: Path, name: str, send: AnySend) -> tuple[GuidedArm, CallLog]:
    spec = MESSAGES_ARMS[name]
    directory.mkdir(parents=True, exist_ok=True)
    log = CallLog(directory / "calls.jsonl", SETTINGS, frame="uci")
    flight = Flight(send, log, Limits(1, None), spendable=partial(spendable, log, 50.0))
    built = GuidedArm(
        name,
        spec.definition,
        Knowledge("uci", {}, (), ()),
        log=log,
        budget=spec.budget or 3,
        flight=flight,
        shows=spec.shows,
        curation=spec.curation,
    )
    return built, log


REQUEST = Request(quota={YOUNG_SICK: QUOTA}, strata=Strata.CELL, seed=0)


def test_cllm_keeps_an_exact_set_at_one_seed(tmp_path):
    built, log = arm(tmp_path, CLLM_ARM, ScriptedSend(reply(records())))
    with log:
        baseline = built(train(), REQUEST)
    assert baseline.added["thalach"].tolist() == [100, 105, 110]
    assert baseline.params["curation"] == "classifier"
    assert baseline.params["shows"] == SHOW
    assert built.drain(0, built_source())[dropped("curation")] == 1


def test_cllm_uncurated_keeps_the_first_quota_rows(tmp_path):
    built, log = arm(tmp_path, CLLM_RAW_ARM, ScriptedSend(reply(records())))
    with log:
        baseline = built(train(), REQUEST)
    assert baseline.added["thalach"].tolist() == [100, 190, 105]
    assert "curation" not in baseline.params
    measures = built.drain(0, built_source())
    assert dropped("curation") not in measures
    assert measures[dropped("quota")] == 1


def test_cllm_replays_byte_for_byte(tmp_path):
    live, log = arm(tmp_path / "live", CLLM_ARM, ScriptedSend(reply(records())))
    with log:
        first = live(train(), REQUEST)
    replay = Replay(log.path)
    again, replayed = arm(tmp_path / "replay", CLLM_ARM, replay)
    with replayed:
        second = again(train(), REQUEST)
    assert first.added.equals(second.added)
    assert replay.remaining == 0
    live_requests, replayed_requests = requests_of(log.path), requests_of(replayed.path)
    assert len(live_requests) == len(replayed_requests) == 1
    assert live_requests == replayed_requests
    assert shown_in(live_requests[0])


def requests_of(path: Path) -> list[dict[str, object]]:
    """Each logged request's params, in call order."""
    return [
        json.loads(line)["params"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["kind"] == "request"
    ]


def user_turn(params: dict[str, object]) -> str:
    messages = params["messages"]
    assert isinstance(messages, list)
    return str(messages[0]["content"])


def shown_in(params: dict[str, object]) -> list[dict[str, object]]:
    """The JSON records a logged request's user turn lists."""
    return [json.loads(line) for line in user_turn(params).splitlines() if line.startswith("{")]


def test_guided_arms_builds_every_messages_arm_of_every_run(tmp_path):
    names = sorted(
        {arm for run in RUNS.values() for arm in run.experiment.arms} & set(MESSAGES_ARMS)
    )
    assert {CLLM_ARM, CLLM_RAW_ARM} <= set(names)
    with guided_arms(
        names,
        Knowledge("uci", {}, (), ()),
        SMALL,
        log=tmp_path / "calls.jsonl",
        candidates=tmp_path / "holdout-candidates.jsonl",
        origin=Origin("e1", GRADED_DATA, RELEASES),
        send=ScriptedSend,
        run="e1",
        limits=SEQUENTIAL,
    ) as run:
        assert sorted(run.arms) == names
        assert all(isinstance(built, GuidedArm) for built in run.arms.values())


def built_source() -> Source:
    return Source.CANDIDATES


def dropped(reason: str) -> Measure:
    return Measure(f"dropped-{reason}")


@pytest.mark.parametrize("name", [CLLM_ARM, CLLM_RAW_ARM, "bko"])
def test_a_cllm_arms_logged_request_shows_seeded_training_records_and_bkos_none(tmp_path, name):
    rows = train()
    built, log = arm(tmp_path, name, ScriptedSend(reply(records())))
    with log:
        built(rows, REQUEST)
    [params] = requests_of(log.path)
    listed = shown_in(params)
    if name == "bko":
        assert listed == []
        return
    expected = Shower(rows, frame=UCI_FRAME, seed=REQUEST.seed).take(YOUNG_SICK)
    assert listed == [dict(record) for record in expected]
    assert len(listed) == SHOW


def small_train() -> pd.DataFrame:
    """Five young sick women, three older sick women and 30 sick men: tier 3 is reached."""
    rows = train()
    keys = cells_of(rows)
    own = rows[[key == YOUNG_SICK for key in keys]].head(5)
    group = rows[[key == (">=55", "female", 1) for key in keys]].head(3)
    men = rows[[key[1:] == ("male", 1) for key in keys]].head(30)
    well = rows[[key[2] == 0 for key in keys]].head(10)
    return pd.concat([own, group, men, well])


def test_the_draw_reaches_the_other_sex_last_and_shows_each_value_in_its_type():
    rows = small_train()
    shown = Shower(rows, frame=UCI_FRAME, seed=3).take(YOUNG_SICK)
    keys = cells_of(shown_frame(shown))
    assert keys[:5] == [YOUNG_SICK] * 5
    assert keys[5:8] == [(">=55", "female", 1)] * 3
    assert [key[1:] for key in keys[8:]] == [("male", 1)] * (SHOW - 8)
    for record in shown:
        assert all(
            type(record[name]) is (float if name in UCI_FRAME.fractional else int)
            for name in UCI_FRAME.columns
        )


def test_a_cell_with_fewer_records_than_shown_shows_what_its_label_holds():
    rows = small_train()
    sick = [key[2] == 1 for key in cells_of(rows)]
    shown = Shower(rows[sick].head(6), frame=UCI_FRAME, seed=3).take(YOUNG_SICK)
    assert len(shown) == 6
    empty = Shower(rows[[not s for s in sick]], frame=UCI_FRAME, seed=3).take(YOUNG_SICK)
    assert empty == ()


def wrong_records(count: int) -> list[object]:
    """`count` young sick women the curator puts at the other label."""
    found = records()
    wrong = found[1]
    return [wrong] * count


def shortfall_send(first: list[object], short: int) -> ScriptedSend:
    """`first`, then a reply of `short` rows the curator drops for every round the budget allows.

    Each round asks for the cell's shortfall, so after `first` the arm asks `short` per call
    until the next ask would pass `QUOTA` times its budget.
    """
    budget = MESSAGES_ARMS[CLLM_ARM].budget or 0
    rounds = (QUOTA * budget - QUOTA) // short
    return ScriptedSend(reply(first), *[reply(wrong_records(short))] * rounds)


def test_a_curated_arm_returns_the_rows_it_kept_when_its_budget_runs_out(tmp_path, caplog):
    request = Request(quota={YOUNG_SICK: QUOTA}, strata=Strata.CELL, seed=0)
    send = shortfall_send([*records()[:1], *wrong_records(2)], QUOTA - 1)
    built, log = arm(tmp_path, CLLM_ARM, send)
    caplog.set_level(logging.INFO, logger="bko.evaluate")
    with log:
        baseline = built(train(), request)
    send.assert_consumed()
    assert baseline.added["thalach"].tolist() == [100]
    assert baseline.params["short"] == {"<55/female/1": QUOTA - 1}
    assert any(
        m.startswith("shortfall arm=cllm seed=0 cell=<55/female/1 short=2 ")
        for m in caplog.messages
    )
    assert any(" outcome=partial " in m for m in caplog.messages if m.startswith("guided arm=cllm"))


def test_a_curated_arm_that_keeps_nothing_adds_no_rows(tmp_path):
    request = Request(quota={YOUNG_SICK: QUOTA}, strata=Strata.CELL, seed=0)
    send = shortfall_send(wrong_records(QUOTA), QUOTA)
    built, log = arm(tmp_path, CLLM_ARM, send)
    with log:
        baseline = built(train(), request)
    send.assert_consumed()
    assert baseline.added.empty
    assert list(baseline.added.columns) == list(UCI_FRAME.columns)
    assert baseline.params["short"] == {"<55/female/1": QUOTA}


def test_an_uncurated_arm_still_refuses_past_its_budget(tmp_path):
    request = Request(quota={YOUNG_SICK: QUOTA}, strata=Strata.CELL, seed=0)
    record = records()[0]
    assert isinstance(record, dict)
    out_of_cell: list[object] = [{**record, "age": 63}] * 20
    budget = MESSAGES_ARMS[CLLM_RAW_ARM].budget or 0
    built, log = arm(tmp_path, CLLM_RAW_ARM, ScriptedSend(*[reply(out_of_cell)] * budget))
    with pytest.raises(BaselineError, match="budget"), log:
        built(train(), request)


def test_a_cell_the_curator_rejects_is_given_up_at_its_own_budget(tmp_path, caplog):
    """The rejected cell stops at `budget` times its quota; the other cell still fills."""
    rows = train()
    well_template = rows[[key == ("<55", "female", 0) for key in cells_of(rows)]].iloc[0]
    well: dict[str, int | float] = {
        name: float(well_template[name])
        if name in UCI_FRAME.fractional
        else int(well_template[name])
        for name in UCI_FRAME.columns
    }
    well["thalach"] = 185
    young_well = ("<55", "female", 0)
    request = Request(quota={YOUNG_SICK: 2, young_well: 2}, strata=Strata.CELL, seed=0)
    budget = MESSAGES_ARMS[CLLM_ARM].budget or 0

    class Keyed(ScriptedSend):
        """Answers each call by its cell: the sick cell's records all curated away."""

        def __call__(self, call):
            self.calls.append(call)
            asked = int(call.user.split("Write exactly ", 1)[1].split(" ", 1)[0])
            rows_out: list[object] = wrong_records(asked) if call.label == 1 else [well] * asked
            return reply(rows_out)

    send = Keyed()
    built, log = arm(tmp_path, CLLM_ARM, send)
    caplog.set_level(logging.INFO, logger="bko.evaluate")
    with log:
        baseline = built(rows, request)
    sick_asked = sum(
        int(call.user.split("Write exactly ", 1)[1].split(" ", 1)[0])
        for call in send.calls
        if call.label == 1
    )
    assert sick_asked == budget * 2
    assert baseline.params["short"] == {"<55/female/1": 2}
    assert len(baseline.added) == 2
    assert any(
        m.startswith("shortfall arm=cllm seed=0 cell=<55/female/1 short=2 asked=18 limit=18")
        for m in caplog.messages
    )
