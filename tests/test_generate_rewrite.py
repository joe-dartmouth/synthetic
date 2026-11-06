"""`bko.generate.rewrite`: a rewrite call's prompt, its reply schema and its request."""

from __future__ import annotations

import json

import pytest

from bko.datasets.uci import UCI_FRAME
from bko.evaluate.diabetes import DIABETES_FIELD_ROWS, DIABETES_FRAME
from bko.evaluate.perturb import DIABETES_UNMAPPED
from bko.generate.calls import request
from bko.generate.flight import Flight, Limits, Planned
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Call, Reply
from bko.generate.prompt import field_units
from bko.generate.rewrite import REWRITE, ROW, Rewrite, render_rewrite, rewrite_schema
from bko.generate.skeleton import Skeleton
from tests._datasets_fixtures import FIXED_NOW
from tests._generate_fixtures import GUIDANCE, reply


FIELDS = ("chol", "fbs", "oldpeak", "thal")
"""Fields of each kind: a whole-number measure, two coded ones and a fractional measure."""

ROWS = (
    {"chol": 210, "fbs": 0, "oldpeak": 1.5, "thal": 3},
    {"chol": 260, "fbs": 1, "oldpeak": 0.0, "thal": 7},
    {"chol": 199, "fbs": 0, "oldpeak": 2.3, "thal": 6},
)

BOUNDS = {"chol": (180.0, 300.0), "oldpeak": (0.0, 4.0)}

REWRITTEN = Rewrite(FIELDS, ROWS, BOUNDS)


def _log(tmp_path) -> CallLog:
    return CallLog(tmp_path / "calls.jsonl", SETTINGS, now=lambda: FIXED_NOW, frame="uci")


def test_the_render_shows_every_row_with_its_index():
    system, user = render_rewrite(REWRITTEN, frame=UCI_FRAME)
    assert "rewrite fields of patient records" in system
    found = REWRITE.match(user)
    assert found is not None
    assert (int(found["rows"]), found["fields"]) == (3, "chol, fbs, oldpeak, thal")
    listed = [json.loads(line) for line in user.splitlines() if line.startswith("{")]
    assert listed == [{ROW: index, **row} for index, row in enumerate(ROWS)]
    assert all(next(iter(line)) == ROW for line in listed)
    assert f"chol 180 to 300 {field_units(UCI_FRAME)['chol']}; oldpeak 0 to 4 " in user
    assert "- fbs: " in user
    assert "- thal: " in user
    for name in ("age", "sex", "target", "cp"):
        assert f'"{name}"' not in user


def test_the_schema_pins_codebooks_rows_and_whole_number_ranges():
    schema = json.loads(json.dumps(rewrite_schema(REWRITTEN, frame=UCI_FRAME)))
    items = schema["properties"]["records"]["items"]
    properties = items["properties"]
    assert items["required"] == [ROW, *FIELDS]
    assert items["additionalProperties"] is False
    assert schema["additionalProperties"] is False
    assert properties[ROW] == {"type": "integer", "enum": [0, 1, 2]}
    assert properties["fbs"]["enum"] == sorted(UCI_FRAME.codebook["fbs"])
    assert properties["thal"]["enum"] == sorted(UCI_FRAME.codebook["thal"])
    assert properties["chol"] == {"type": "integer", "enum": list(range(180, 301))}
    assert properties["oldpeak"] == {"type": "number"}
    assert set(properties) == {ROW, *FIELDS}


def test_a_row_that_lacks_or_adds_a_field_is_refused():
    with pytest.raises(ValueError, match="not the fields"):
        Rewrite(FIELDS, ({**ROWS[0], "age": 50},), BOUNDS)
    with pytest.raises(ValueError, match="not the fields"):
        Rewrite(FIELDS, ({"chol": 210},), BOUNDS)


@pytest.mark.parametrize("cell", ["age", "sex"])
def test_a_rewrite_of_the_cells_fields_is_refused(cell):
    with pytest.raises(ValueError, match="the cell's fields"):
        Rewrite((*FIELDS, cell), tuple({**row, cell: 1} for row in ROWS), BOUNDS)


def test_a_rewrite_of_the_label_is_refused():
    rewrite = Rewrite((*FIELDS, "target"), tuple({**row, "target": 1} for row in ROWS), BOUNDS)
    with pytest.raises(ValueError, match="the label target"):
        render_rewrite(rewrite, frame=UCI_FRAME)


def test_the_diabetes_rewrite_sends_the_19_pinned_fields_and_nothing_mapped():
    """Egress pin: a Diabetes rewrite's records hold only `row` and the 19 unmapped fields, never
    a mapped field, `age`, `sex` or the label."""
    first = {name: min(DIABETES_FRAME.codebook.get(name, {0: ""})) for name in DIABETES_UNMAPPED}
    bounds = {
        name: (0.0, 50.0) for name in DIABETES_UNMAPPED if name not in DIABETES_FRAME.categorical
    }
    rewrite = Rewrite(DIABETES_UNMAPPED, (first, first), bounds)
    _, user = render_rewrite(rewrite, frame=DIABETES_FRAME)
    sent = [json.loads(line) for line in user.splitlines() if line.startswith("{")]
    assert [set(row) for row in sent] == [{ROW, *DIABETES_UNMAPPED}] * 2
    mapped = {c.field for row in DIABETES_FIELD_ROWS for c in row.when}
    kept = {"age", "sex", DIABETES_FRAME.label, *mapped} - set(DIABETES_UNMAPPED)
    for name in kept:
        assert f'"{name}"' not in user, name


def test_a_measured_field_with_no_range_is_refused():
    with pytest.raises(ValueError, match="no training range"):
        render_rewrite(Rewrite(FIELDS, ROWS, {"chol": (180.0, 300.0)}), frame=UCI_FRAME)


def test_request_sends_the_rewrite_and_its_schema(tmp_path):
    with _log(tmp_path) as log:
        pending = request(GUIDANCE, 3, log=log, frame=UCI_FRAME, rewrite=REWRITTEN)
    assert (pending.index, pending.asked) == (1, 3)
    params = json.loads(json.dumps(pending.call.params()))
    turns = (params["system"], params["messages"][0]["content"])
    assert turns == render_rewrite(REWRITTEN, frame=UCI_FRAME)
    schema = params["output_config"]["format"]["schema"]
    assert schema == rewrite_schema(REWRITTEN, frame=UCI_FRAME)
    logged = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    assert logged[1]["kind"] == "request"
    assert logged[1]["params"] == params


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"skeleton": Skeleton((), (frozenset(),) * 3)}, "neither a skeleton nor shown"),
        ({"shown": (dict.fromkeys(UCI_FRAME.columns, 1),)}, "neither a skeleton nor shown"),
    ],
    ids=["skeleton", "shown"],
)
def test_request_refuses_a_rewrite_with_a_skeleton_or_shown_records(tmp_path, extra, message):
    with _log(tmp_path) as log:
        with pytest.raises(ValueError, match=message):
            request(GUIDANCE, 3, log=log, frame=UCI_FRAME, rewrite=REWRITTEN, **extra)
        assert log.calls == 0


def test_request_refuses_a_rewrite_of_other_than_the_rows_asked(tmp_path):
    with _log(tmp_path) as log:
        with pytest.raises(ValueError, match="3 rows for 2 records asked"):
            request(GUIDANCE, 2, log=log, frame=UCI_FRAME, rewrite=REWRITTEN)
        assert log.calls == 0


def test_a_planned_rewrite_goes_out_through_a_flight(tmp_path):
    sent: list[Call] = []

    def send(call: Call) -> Reply:
        sent.append(call)
        return reply([{ROW: index, **row} for index, row in enumerate(ROWS)])

    with _log(tmp_path) as log:
        flight = Flight(send, log, Limits(1, None), spendable=lambda: None)
        (batch,) = flight.wave((Planned(GUIDANCE, 3, rewrite=REWRITTEN),), frame=UCI_FRAME)
    assert sent[0].user == render_rewrite(REWRITTEN, frame=UCI_FRAME)[1]
    assert list(batch.candidates) == [{ROW: index, **row} for index, row in enumerate(ROWS)]
