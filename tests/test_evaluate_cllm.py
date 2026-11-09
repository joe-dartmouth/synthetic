"""The CLLM arms: each call shows seeded training records, and `cllm` curates what comes back."""

from __future__ import annotations

import json
import logging
from functools import partial
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from bko.baselines.errors import BaselineError
from bko.baselines.request import Request, Strata
from bko.datasets.cells import cell_keys
from bko.datasets.uci import UCI_FRAME
from bko.evaluate.attempts import Attempt
from bko.evaluate.cllm import SHOW, Shower
from bko.evaluate.curation import ALEATORIC_AT_MOST, CONFIDENCE_AT_MOST, classifier_scores
from bko.evaluate.experiment import (
    CLLM_ARM,
    CLLM_RAW_ARM,
    MESSAGES_ARMS,
    PAID_ARMS,
    PERTURBED_ARM,
)
from bko.evaluate.guided import SEQUENTIAL, GuidedArm, Knowledge
from bko.evaluate.guided_run import guided_arms
from bko.evaluate.holdout_candidates import Candidate, Origin
from bko.evaluate.named_runs import RUNS
from bko.evaluate.perturbed import PerturbedArm
from bko.evaluate.scores import Measure, Source
from bko.evaluate.waves import asks, spendable
from bko.generate.flight import Flight, Limits
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, AnySend
from bko.generate.replay import Replay
from bko.ontology.mapping import RELEASES
from bko.validate.checks import validate
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


def arm(
    directory: Path, name: str, send: AnySend, *, in_flight: int = 1
) -> tuple[GuidedArm, CallLog]:
    spec = MESSAGES_ARMS[name]
    directory.mkdir(parents=True, exist_ok=True)
    log = CallLog(directory / "calls.jsonl", SETTINGS, frame="uci")
    flight = Flight(send, log, Limits(in_flight, None), spendable=partial(spendable, log, 50.0))
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


REQUEST = Request(quota={YOUNG_SICK: QUOTA}, strata=Strata.CELL, seed=0, frame=UCI_FRAME)


def test_cllm_keeps_an_exact_set_at_one_seed(tmp_path):
    built, log = arm(tmp_path, CLLM_ARM, ScriptedSend(reply(records())))
    with log:
        baseline = built(train(), REQUEST)
    assert baseline.added["thalach"].tolist() == [100, 105, 110]
    assert baseline.params["curation"] == "classifier"
    assert baseline.params["shows"] == SHOW
    assert "over_ask" not in baseline.params
    assert built.drain(0, built_source())[dropped("curation")] == 1


def test_curated_cllm_lines_carry_their_two_scores(tmp_path):
    rows = train()
    built, log = arm(tmp_path, CLLM_ARM, ScriptedSend(reply(records())))
    with log:
        built(rows, REQUEST)
    seen = [item for _, _, item in built._seen if item.outcome != "rejected"]
    assert all(item.score is None for _, _, item in built._seen if item.outcome == "rejected")
    accepted = validate(records(), frame=UCI_FRAME).accepted
    read = classifier_scores(rows, frame=UCI_FRAME)(accepted)
    expected = [
        {"aleatoric": float(aleatoric), "confidence": float(confidence)}
        for confidence, aleatoric in read[["confidence", "aleatoric"]].to_numpy().tolist()
    ]
    assert [item.score for item in seen] == expected
    assert [item.outcome for item in seen] == ["kept", "dropped-curation", "kept", "kept"]
    dropped = seen[1].score
    assert dropped is not None
    confidence, aleatoric = dropped["confidence"], dropped["aleatoric"]
    assert confidence is not None
    assert aleatoric is not None
    assert confidence <= CONFIDENCE_AT_MOST
    assert aleatoric <= ALEATORIC_AT_MOST
    line = Candidate(0, CLLM_ARM, "gap", 1, True, seen[1]).line()
    assert line["score"] == expected[1]
    raw, raw_log = arm(tmp_path / "raw", CLLM_RAW_ARM, ScriptedSend(reply(records())))
    with raw_log:
        raw(rows, REQUEST)
    assert all(item.score is None for _, _, item in raw._seen)
    assert "score" not in Candidate(0, CLLM_RAW_ARM, "gap", 1, True, raw._seen[0][2]).line()


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


def test_guided_arms_builds_every_paid_arm_of_every_run(tmp_path):
    names = sorted(
        {arm for run in RUNS.values() for arm in run.experiment.arms} & PAID_ARMS | {PERTURBED_ARM}
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
        assert isinstance(run.arms[PERTURBED_ARM], PerturbedArm)
        guided = [built for name, built in run.arms.items() if name in MESSAGES_ARMS]
        assert len(guided) == len(names) - 1
        assert all(isinstance(built, GuidedArm) for built in guided)


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
    """`first`, then a reply of `short` rows the curator drops for every call the budget allows.

    Each curating call asks a full call, capped by what is left of the cell's limit, `QUOTA` times
    its budget, so the cell makes one call per `per_call` of its limit, the last asking the
    remainder, which still covers `short`.
    """
    budget = MESSAGES_ARMS[CLLM_ARM].budget or 0
    calls = -(-QUOTA * budget // SETTINGS.per_call)
    assert QUOTA * budget % SETTINGS.per_call >= short
    return ScriptedSend(reply(first), *[reply(wrong_records(short))] * (calls - 1))


def test_a_curated_arm_returns_the_rows_it_kept_when_its_budget_runs_out(tmp_path, caplog):
    request = Request(quota={YOUNG_SICK: QUOTA}, strata=Strata.CELL, seed=0, frame=UCI_FRAME)
    send = shortfall_send([*records()[:1], *wrong_records(2)], QUOTA - 1)
    built, log = arm(tmp_path, CLLM_ARM, send)
    caplog.set_level(logging.INFO, logger="bko.evaluate")
    with log:
        baseline = built(train(), request)
    send.assert_consumed()
    assert baseline.added["thalach"].tolist() == [100]
    assert baseline.params["short"] == {"<55/female/1": QUOTA - 1}
    budget = MESSAGES_ARMS[CLLM_ARM].budget or 0
    cap = QUOTA * budget
    expected = (
        f"shortfall arm=cllm seed=0 cell=<55/female/1 short=2 quota={QUOTA} asked={cap} limit={cap}"
    )
    assert expected in caplog.messages
    assert any(
        " outcome=partial " in m
        for m in caplog.messages
        if m.startswith("guided ") and " arm=cllm " in m
    )


def test_a_curated_arm_that_keeps_nothing_adds_no_rows(tmp_path):
    request = Request(quota={YOUNG_SICK: QUOTA}, strata=Strata.CELL, seed=0, frame=UCI_FRAME)
    send = shortfall_send(wrong_records(QUOTA), QUOTA)
    built, log = arm(tmp_path, CLLM_ARM, send)
    with log:
        baseline = built(train(), request)
    send.assert_consumed()
    assert baseline.added.empty
    assert list(baseline.added.columns) == list(UCI_FRAME.columns)
    assert baseline.params["short"] == {"<55/female/1": QUOTA}


def test_an_uncurated_arm_still_refuses_past_its_budget(tmp_path):
    request = Request(quota={YOUNG_SICK: QUOTA}, strata=Strata.CELL, seed=0, frame=UCI_FRAME)
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
    well = well_woman()
    request = Request(
        quota={YOUNG_SICK: 2, YOUNG_WELL: 2}, strata=Strata.CELL, seed=0, frame=UCI_FRAME
    )
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
    cap = budget * 2
    expected = (
        f"shortfall arm=cllm seed=0 cell=<55/female/1 short=2 quota=2 asked={cap} limit={cap}"
    )
    assert expected in caplog.messages


YOUNG_WELL = ("<55", "female", 0)


def well_woman() -> dict[str, int | float]:
    """A young well woman from the training rows, thalach 185."""
    rows = train()
    template = rows[[key == YOUNG_WELL for key in cells_of(rows)]].iloc[0]
    well: dict[str, int | float] = {
        name: float(template[name]) if name in UCI_FRAME.fractional else int(template[name])
        for name in UCI_FRAME.columns
    }
    well["thalach"] = 185
    return well


def asked_of(call) -> int:
    """The records a call's user turn asks for."""
    return int(call.user.split("Write exactly ", 1)[1].split(" ", 1)[0])


def sick_woman(age: int, thalach: int) -> dict[str, int | float]:
    """A sick woman of `age` from `records()`'s template; thalach 190 the curator drops, 100 not."""
    record = records()[0]
    assert isinstance(record, dict)
    return {**record, "age": age, "thalach": thalach}


class ByCell(ScriptedSend):
    """Answers each call from a per-age-band script: a list of replies, the last one repeated."""

    def __init__(self, script: dict[str, list[list[object]]]) -> None:
        super().__init__()
        self._bands = script
        self._at: dict[str, int] = {}

    def __call__(self, call):
        self.calls.append(call)
        band = "<55" if "age 44 to 54" in call.user or "age 0 to 54" in call.user else ">=55"
        replies = self._bands[band]
        at = self._at.get(band, 0)
        self._at[band] = at + 1
        template = replies[min(at, len(replies) - 1)]
        return reply([template[i % len(template)] for i in range(asked_of(call))])


def test_a_partial_attempts_candidates_are_marked_returned(tmp_path):
    send = shortfall_send([*records()[:1], *wrong_records(2)], QUOTA - 1)
    built, log = arm(tmp_path, CLLM_ARM, send)
    with log:
        built(
            train(), Request(quota={YOUNG_SICK: QUOTA}, strata=Strata.CELL, seed=0, frame=UCI_FRAME)
        )
    assert built._seen
    assert all(returned for _, returned, _ in built._seen)


def test_a_cell_short_by_one_record_asks_a_full_call_within_its_cap(tmp_path, caplog):
    """Quota 3 at budget 22 is a limit of 66: three full calls of 20, then the 6 left of it."""
    budget = MESSAGES_ARMS[CLLM_ARM].budget or 0
    limit = budget * QUOTA
    per_call = SETTINGS.per_call
    first = [records()[0], records()[2], *wrong_records(per_call - 2)]
    send = ByCell({"<55": [first, wrong_records(1)]})
    built, log = arm(tmp_path, CLLM_ARM, send)
    caplog.set_level(logging.INFO, logger="bko.evaluate")
    with log:
        baseline = built(train(), REQUEST)
    full = limit // per_call
    assert [asked_of(call) for call in send.calls] == [per_call] * full + [limit % per_call]
    assert 0 < limit % per_call < per_call
    assert baseline.params["short"] == {"<55/female/1": 1}
    expected = (
        f"shortfall arm=cllm seed=0 cell=<55/female/1 short=1 quota={QUOTA} "
        f"asked={limit} limit={limit}"
    )
    assert expected in caplog.messages


def test_every_in_flight_count_shows_the_same_records(tmp_path):
    """Two chains in one wave, each asked over several rounds: the logged requests match."""
    half: list[object] = [records()[0], records()[1]]
    script = {"<55": [half]}

    class ByLabel(ByCell):
        """The sick cell keeps half its rows; the well cell keeps every one."""

        def __call__(self, call):
            if call.label == 1:
                return super().__call__(call)
            self.calls.append(call)
            return reply([well_woman()] * asked_of(call))

    request = Request(
        quota={YOUNG_SICK: 25, YOUNG_WELL: 30}, strata=Strata.CELL, seed=0, frame=UCI_FRAME
    )
    logged = []
    for in_flight in (1, 4):
        built, log = arm(tmp_path / str(in_flight), CLLM_ARM, ByLabel(script), in_flight=in_flight)
        with log:
            built(train(), request)
        logged.append(requests_of(log.path))
    assert len(logged[0]) > 2
    assert logged[0] == logged[1]
    assert len({json.dumps(shown_in(params)) for params in logged[0]}) == len(logged[0])


def test_two_cells_given_up_in_different_rounds_are_each_logged_once(tmp_path, caplog):
    old_sick = (">=55", "female", 1)
    send = ByCell(
        {
            "<55": [[sick_woman(44, 190)]],
            ">=55": [
                [sick_woman(63, 100), *[sick_woman(63, 190)] * (SETTINGS.per_call - 1)],
                [sick_woman(63, 190)],
            ],
        }
    )
    built, log = arm(tmp_path, CLLM_ARM, send)
    caplog.set_level(logging.INFO, logger="bko.evaluate")
    request = Request(
        quota={YOUNG_SICK: 2, old_sick: 3}, strata=Strata.CELL, seed=0, frame=UCI_FRAME
    )
    with log:
        baseline = built(train(), request)
    lines = [m for m in caplog.messages if m.startswith("shortfall ")]
    assert [m.split()[3] for m in lines] == ["cell=<55/female/1", "cell=>=55/female/1"]
    assert baseline.params["short"] == {"<55/female/1": 2, ">=55/female/1": 2}


@pytest.mark.parametrize(
    ("short", "per_call", "expected"),
    [
        (0, 20, []),
        (1, 20, [1]),
        (19, 20, [19]),
        (20, 20, [20]),
        (21, 20, [20, 1]),
        (45, 20, [20, 20, 5]),
        (60, 20, [20, 20, 20]),
        (7, 3, [3, 3, 1]),
    ],
)
def test_an_arm_that_does_not_curate_asks_its_shortfall_split_as_before_the_cell_limit(
    short, per_call, expected
):
    """The asks every non-curating arm made at B3's run commits: one call per `per_call` of the
    shortfall, the last asking what is left. B3's `cllm` call logs replay at their run commits,
    whose curating arm asked the same split (ADR `research/cllm-arm-design`)."""
    old = [min(per_call, short - start) for start in range(0, short, per_call)]
    assert old == expected
    attempt = cast("Attempt", None)
    assert asks(attempt, ("<55", "female", 0), short, per_call=per_call, budget=None) == expected
