"""The candidates file: `render_lines` writes what `read` reads back."""

from __future__ import annotations

import dataclasses
import json
from typing import Final

import pytest

from bko.gaps.definitions import Definition
from bko.generate.candidates import (
    CandidateLine,
    Candidates,
    Header,
    Request,
    read,
    render_header,
    render_lines,
)
from bko.generate.decode import Reason, Undecoded
from bko.generate.errors import CandidatesFormatError
from bko.generate.messages import Level
from tests._fusion_fixtures import scenario_a, young


CELL = ("<55", "female", 1)

HEADER: Final = Header(
    request=Request.COMPLETENESS,
    definition=Definition.SUBGROUP,
    guidance=Level.ONTOLOGY,
    uci="a" * 64,
    fingerprint="b" * 64,
    releases=(("HP", "c" * 64), ("MONDO", "d" * 64)),
    calls=(("<55/female/1", 2),),
)
HEAD: Final = render_header(HEADER)


def test_render_lines_writes_the_bytes_the_command_always_wrote():
    text = render_lines(2, CELL, ({"age": 41}, Undecoded(Reason.SHORT)))
    assert text == (
        '{"call": 2, "candidate": {"age": 41}, "cell": "<55/female/1", "position": 0}\n'
        '{"call": 2, "cell": "<55/female/1", "position": 1, "undecoded": "short"}\n'
    )


def test_read_returns_what_render_lines_wrote():
    data = (HEAD + render_lines(1, CELL, ({"age": 41}, Undecoded(Reason.SHORT)))).encode()
    assert read(data).lines == (
        CandidateLine(1, "<55/female/1", 0, {"age": 41}),
        CandidateLine(1, "<55/female/1", 1, Undecoded(Reason.SHORT)),
    )


def test_read_keeps_a_decoded_null_apart_from_a_placeholder():
    (line,) = read((HEAD + render_lines(1, CELL, (None,))).encode()).lines
    assert line.candidate is None


def test_read_reads_the_tests_oracle():
    lines = read(HEAD.encode() + scenario_a()).lines
    assert len(lines) == 40
    assert lines[0] == CandidateLine(1, "<55/female/0", 0, young(0, cp=4))


def test_a_header_round_trips():
    lines = render_lines(1, CELL, [young(1)] * 2)
    found = read((HEAD + lines).encode())
    assert found.header == HEADER
    assert [line.call for line in found.lines] == [1, 1]


def test_a_headerless_file_is_refused():
    with pytest.raises(CandidatesFormatError, match="line 1 is not a schema 1 header"):
        read(render_lines(1, CELL, [young(1)]).encode())


def test_an_empty_file_is_refused():
    with pytest.raises(CandidatesFormatError, match="no header"):
        read(b"")


def test_a_header_only_file_reads_with_no_lines():
    assert read(HEAD.encode()) == Candidates(HEADER, ())


def test_an_empty_plan_round_trips():
    empty = dataclasses.replace(HEADER, request=Request.E1, calls=())
    assert read(render_header(empty).encode()).header == empty


@pytest.mark.parametrize(
    "edit",
    [
        lambda h: {**h, "schema": 2},
        lambda h: {**h, "request": "other"},
        lambda h: {**h, "extra": 1},
        lambda h: {k: v for k, v in h.items() if k != "calls"},
        lambda h: {**h, "calls": [["<55/female/1", 0]]},
        lambda h: {**h, "uci": "short"},
        lambda h: {**h, "releases": {"HP": "short"}},
        lambda h: {**h, "calls": [["<55/female/1", True]]},
        lambda h: {**h, "kind": "other"},
        lambda h: {**h, "schema": True},
        lambda h: {**h, "fingerprint": "B" * 64},
        lambda h: {**h, "uci": "a" * 63},
        lambda h: {**h, "releases": [["HP", "c" * 64]]},
        lambda h: {**h, "calls": {"<55/female/1": 2}},
        lambda h: {**h, "calls": [["<55/female/1", 2, 0]]},
        lambda h: {**h, "calls": [[7, 2]]},
        lambda h: {**h, "guidance": "loud"},
    ],
    ids=[
        "schema",
        "request",
        "extra key",
        "missing key",
        "empty call",
        "bad sha256",
        "bad release",
        "bool asked",
        "other kind",
        "bool schema",
        "uppercase fingerprint",
        "short uci",
        "releases list",
        "calls object",
        "three-item call",
        "number cell",
        "unknown guidance",
    ],
)
def test_a_malformed_header_is_refused(edit):
    raw = json.loads(HEAD)
    with pytest.raises(CandidatesFormatError, match="header"):
        read((json.dumps(edit(raw)) + "\n").encode())


def test_a_header_after_the_first_line_is_refused():
    with pytest.raises(CandidatesFormatError, match="line 3"):
        read((HEAD + render_lines(1, CELL, [young(1)]) + HEAD).encode())


LINE = '{"call": 1, "candidate": {"age": 41}, "cell": "<55/female/1", "position": 0}'
OTHER_CELL = '{"call": 1, "candidate": {}, "cell": "<55/female/0", "position": 1}'


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"\xff\n", "not UTF-8"),
        (LINE.encode(), "has no newline"),
        (b"{\n", "line 2 is not JSON"),
        (b'{"call": 1, "cell": "<55/female/1", "position": 0}\n', "one of candidate or undecoded"),
        (
            b'{"call": 1, "candidate": 1, "cell": "a", "position": 0, "undecoded": "short"}\n',
            "one of candidate or undecoded",
        ),
        (b'{"call": true, "candidate": 1, "cell": "a", "position": 0}\n', "positive"),
        (b'{"call": 0, "candidate": 1, "cell": "a", "position": 0}\n', "positive"),
        (b'{"call": 1, "candidate": 1, "cell": "a", "position": -1}\n', "non-negative"),
        (b'{"call": 1, "candidate": 1, "cell": "a", "position": false}\n', "non-negative"),
        (
            b'{"call": 1, "candidate": 1, "cell": "a", "extra": 1, "position": 0}\n',
            "one of candidate or undecoded",
        ),
        (b"[]\n", "one of candidate or undecoded"),
        (b'{"call": 1, "candidate": 1, "cell": 7, "position": 0}\n', "cell is not text"),
        (b'{"call": 1, "cell": "a", "position": 0, "undecoded": "odd"}\n', "not a known reason"),
        (f"{LINE}\n{LINE}\n".encode(), "line 3 repeats call 1 position 0"),
        (f"{LINE}\n{OTHER_CELL}\n".encode(), "line 3 puts call 1 in '<55/female/0'"),
    ],
    ids=[
        "utf8",
        "newline",
        "json",
        "no-outcome",
        "both-outcomes",
        "bool-call",
        "zero-call",
        "negative-position",
        "bool-position",
        "extra-key",
        "not-object",
        "cell-type",
        "reason",
        "repeat",
        "two-cells",
    ],
)
def test_a_malformed_file_is_refused(data, message):
    with pytest.raises(CandidatesFormatError, match=message):
        read(data if data.startswith(b"\xff") else HEAD.encode() + data)
