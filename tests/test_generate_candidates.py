"""The candidates file: `render_lines` writes what `read` reads back."""

from __future__ import annotations

import pytest

from bko.generate.candidates import CandidateLine, read, render_lines
from bko.generate.decode import Reason, Undecoded
from bko.generate.errors import CandidatesFormatError
from tests._fusion_fixtures import scenario_a, young


CELL = ("<55", "female", 1)


def test_render_lines_writes_the_bytes_the_command_always_wrote():
    text = render_lines(2, CELL, ({"age": 41}, Undecoded(Reason.SHORT)))
    assert text == (
        '{"call": 2, "candidate": {"age": 41}, "cell": "<55/female/1", "position": 0}\n'
        '{"call": 2, "cell": "<55/female/1", "position": 1, "undecoded": "short"}\n'
    )


def test_read_returns_what_render_lines_wrote():
    data = (render_lines(1, CELL, ({"age": 41}, Undecoded(Reason.SHORT)))).encode()
    assert read(data) == (
        CandidateLine(1, "<55/female/1", 0, {"age": 41}),
        CandidateLine(1, "<55/female/1", 1, Undecoded(Reason.SHORT)),
    )


def test_read_keeps_a_decoded_null_apart_from_a_placeholder():
    (line,) = read(render_lines(1, CELL, (None,)).encode())
    assert line.candidate is None


def test_read_reads_the_tests_oracle():
    lines = read(scenario_a())
    assert len(lines) == 40
    assert lines[0] == CandidateLine(1, "<55/female/0", 0, young(0, cp=4))


def test_an_empty_file_reads_as_no_lines():
    assert read(b"") == ()


LINE = '{"call": 1, "candidate": {"age": 41}, "cell": "<55/female/1", "position": 0}'
OTHER_CELL = '{"call": 1, "candidate": {}, "cell": "<55/female/0", "position": 1}'


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"\xff\n", "not UTF-8"),
        (LINE.encode(), "has no newline"),
        (b"{\n", "line 1 is not JSON"),
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
        (f"{LINE}\n{LINE}\n".encode(), "line 2 repeats call 1 position 0"),
        (f"{LINE}\n{OTHER_CELL}\n".encode(), "line 2 puts call 1 in '<55/female/0'"),
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
        read(data)
