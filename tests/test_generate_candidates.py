"""The candidates file: `render_lines` writes what `read` reads back."""

from __future__ import annotations

from bko.generate.candidates import CandidateLine, read, render_lines
from bko.generate.decode import Reason, Undecoded
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
