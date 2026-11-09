"""`bko.generate.rewrite`: a rewrite call's prompt, its reply schema and its request."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence

import pytest

from bko.datasets.uci import UCI_FRAME
from bko.evaluate.perturb import DIABETES_UNMAPPED
from bko.generate.calls import request
from bko.generate.flight import Flight, Limits, Planned
from bko.generate.log import CallLog
from bko.generate.messages import SETTINGS, Call, Reply
from bko.generate.prompt import field_units
from bko.generate.rewrite import (
    COUNTS_HEADER,
    OPENING,
    REWRITE,
    ROW,
    Rewrite,
    Share,
    render_rewrite,
    rewrite_schema,
)
from bko.generate.skeleton import Skeleton
from bko.ontology.frames import DIABETES_FIELD_ROWS, DIABETES_FRAME
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


def quotas_for(
    fields: Sequence[str], rows: Sequence[Mapping[str, int | float]]
) -> dict[str, tuple[Share, ...]]:
    """One `Share` per distinct value of each field in `rows`, counted from them."""
    return {
        name: tuple(
            Share(value, value, count)
            for value, count in sorted(Counter(row[name] for row in rows).items())
        )
        for name in fields
    }


QUOTAS = quotas_for(FIELDS, ROWS)

REWRITTEN = Rewrite(FIELDS, ROWS, BOUNDS, QUOTAS)


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


def _counts(user: str) -> dict[str, str]:
    """Each field's counts line in `user`'s quota block, which ends before the records."""
    block = user.split(COUNTS_HEADER + "\n", 1)[1].split("\n\nThe records:", 1)[0]
    lines = [line.removeprefix("- ").split(": ", 1) for line in block.splitlines()]
    return dict(lines)


def test_the_render_states_each_fields_counts_between_the_ranges_and_the_records():
    _, user = render_rewrite(REWRITTEN, frame=UCI_FRAME)
    parts = user.split("\n\n")
    (block,) = [part for part in parts if part.startswith(COUNTS_HEADER)]
    assert parts.index(block) == len(parts) - 2
    assert parts[-3].startswith("Each measured field's range in the training data: ")
    assert parts[-1].startswith("The records:")
    assert _counts(user) == {
        "chol": "199 x1, 210 x1, 260 x1",
        "fbs": "0 x2, 1 x1",
        "oldpeak": "0.0 x1, 1.5 x1, 2.3 x1",
        "thal": "3 x1, 6 x1, 7 x1",
    }
    ranged = Rewrite(
        FIELDS, ROWS, BOUNDS, {**QUOTAS, "chol": (Share(180, 249, 2), Share(250, 300, 1))}
    )
    assert _counts(render_rewrite(ranged, frame=UCI_FRAME)[1])["chol"] == "180-249 x2, 250-300 x1"


def test_a_share_states_its_ends_exactly():
    quotas = {
        **QUOTAS,
        "chol": (Share(180, 1234567, 2), Share(1234568, 1234568, 1)),
        "oldpeak": (Share(0.0, 0.1234567, 2), Share(2.3, 4.0, 1)),
    }
    counts = _counts(render_rewrite(Rewrite(FIELDS, ROWS, BOUNDS, quotas), frame=UCI_FRAME)[1])
    assert counts["chol"] == "180-1234567 x2, 1234568 x1"
    assert counts["oldpeak"] == "0.0-0.1234567 x2, 2.3-4.0 x1"


def test_only_record_lines_start_with_a_brace():
    _, user = render_rewrite(REWRITTEN, frame=UCI_FRAME)
    assert sum(line.startswith("{") for line in user.splitlines()) == len(ROWS)


def test_the_quota_block_names_exactly_the_fields_rewritten():
    _, user = render_rewrite(REWRITTEN, frame=UCI_FRAME)
    assert tuple(_counts(user)) == FIELDS


def test_the_opening_lets_a_field_keep_its_value():
    _, user = render_rewrite(REWRITTEN, frame=UCI_FRAME)
    opening = f"Rewrite these {len(ROWS)} records' fields: {', '.join(FIELDS)}. {OPENING}"
    assert user.split("\n\n", 1)[0] == opening
    assert "a field may keep its value" in OPENING
    assert REWRITE.match(user)


def test_a_quota_for_a_field_not_rewritten_is_refused():
    quotas = {**QUOTAS, "cp": (Share(1, 1, 3),)}
    with pytest.raises(ValueError, match="quotas"):
        Rewrite(FIELDS, ROWS, BOUNDS, quotas)


def test_a_rewritten_field_with_no_quota_is_refused():
    quotas = {name: shares for name, shares in QUOTAS.items() if name != "thal"}
    with pytest.raises(ValueError, match="quotas"):
        Rewrite(FIELDS, ROWS, BOUNDS, quotas)


def test_a_fields_quota_that_does_not_sum_to_the_rows_is_refused():
    quotas = {**QUOTAS, "fbs": (Share(0, 0, 2),)}
    with pytest.raises(ValueError, match="sum to 2, not the 3 rows"):
        Rewrite(FIELDS, ROWS, BOUNDS, quotas)


@pytest.mark.parametrize(
    "shares",
    [(Share(1, 0, 3),), (*QUOTAS["fbs"], Share(0, 0, 0))],
    ids=["low-above-high", "empty"],
)
def test_an_inverted_or_empty_share_is_refused(shares):
    with pytest.raises(ValueError, match="share"):
        Rewrite(FIELDS, ROWS, BOUNDS, {**QUOTAS, "fbs": shares})


@pytest.mark.parametrize(
    "shares",
    [
        (Share(0, 0, 1), Share(1, 1, 1), Share(0, 0, 1)),
        (Share(180, 220, 2), Share(220, 300, 1)),
        (Share(180, 250, 2), Share(200, 300, 1)),
    ],
    ids=["descending", "touching", "overlapping"],
)
def test_shares_that_do_not_strictly_ascend_are_refused(shares):
    with pytest.raises(ValueError, match="ascend"):
        Rewrite(FIELDS, ROWS, BOUNDS, {**QUOTAS, "chol": shares})


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
        Rewrite(FIELDS, ({**ROWS[0], "age": 50},), BOUNDS, quotas_for(FIELDS, ROWS[:1]))
    with pytest.raises(ValueError, match="not the fields"):
        Rewrite(FIELDS, ({"chol": 210},), BOUNDS, quotas_for(FIELDS, ROWS[:1]))


@pytest.mark.parametrize("cell", ["age", "sex"])
def test_a_rewrite_of_the_cells_fields_is_refused(cell):
    rows = tuple({**row, cell: 1} for row in ROWS)
    with pytest.raises(ValueError, match="the cell's fields"):
        Rewrite((*FIELDS, cell), rows, BOUNDS, quotas_for((*FIELDS, cell), rows))


def test_a_rewrite_of_the_label_is_refused():
    rows = tuple({**row, "target": 1} for row in ROWS)
    rewrite = Rewrite((*FIELDS, "target"), rows, BOUNDS, quotas_for((*FIELDS, "target"), rows))
    with pytest.raises(ValueError, match="the label target"):
        render_rewrite(rewrite, frame=UCI_FRAME)


def test_the_diabetes_rewrite_sends_the_19_pinned_fields_and_nothing_mapped():
    """Egress pin: a Diabetes rewrite's records hold only `row` and the 19 unmapped fields, never
    a mapped field, `age`, `sex` or the label."""
    first = {name: min(DIABETES_FRAME.codebook.get(name, {0: ""})) for name in DIABETES_UNMAPPED}
    bounds = {
        name: (0.0, 50.0) for name in DIABETES_UNMAPPED if name not in DIABETES_FRAME.categorical
    }
    quotas = quotas_for(DIABETES_UNMAPPED, (first, first))
    rewrite = Rewrite(DIABETES_UNMAPPED, (first, first), bounds, quotas)
    _, user = render_rewrite(rewrite, frame=DIABETES_FRAME)
    sent = [json.loads(line) for line in user.splitlines() if line.startswith("{")]
    assert [set(row) for row in sent] == [{ROW, *DIABETES_UNMAPPED}] * 2
    mapped = {c.field for row in DIABETES_FIELD_ROWS for c in row.when}
    kept = {"age", "sex", DIABETES_FRAME.label, *mapped} - set(DIABETES_UNMAPPED)
    for name in kept:
        assert f'"{name}"' not in user, name
    assert list(_counts(user)) == list(DIABETES_UNMAPPED)


def test_a_measured_field_with_no_range_is_refused():
    with pytest.raises(ValueError, match="no training range"):
        render_rewrite(Rewrite(FIELDS, ROWS, {"chol": (180.0, 300.0)}, QUOTAS), frame=UCI_FRAME)


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
