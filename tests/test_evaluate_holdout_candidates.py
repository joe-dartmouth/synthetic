"""The per-holdout candidates file: every candidate a BKO arm generates, with its outcome.

Every call goes to a `ScriptedSend`; no test reaches the API.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from bko.evaluate import holdout_candidates
from bko.evaluate.__main__ import Context, main
from bko.evaluate.errors import CandidatesFormatError
from bko.evaluate.guided import Knowledge, guided_arms
from bko.evaluate.holdout_candidates import SCHEMA, Candidate, CandidateFile, Origin, read
from bko.generate.messages import SETTINGS, Reply
from tests._evaluate_bko_fixtures import (
    BAD,
    NARROW,
    PIN,
    REPLICAS,
    candidates_for,
    context,
    replica_script,
    score_argv,
    scripted,
)
from tests._evaluate_fixtures import FAST_ARMS
from tests._fusion_fixtures import young
from tests._generate_fixtures import ScriptedSend, reply
from tests._validate_fixtures import PLANTED, planted


OLD: Final = planted(age=63, sex=0, target=1)
"""A valid candidate of `(">=55", "female", 1)`, a cell E1's gap request here does not ask for."""

KEYS: Final = {
    "kind",
    "holdout",
    "arm",
    "fill",
    "attempt",
    "returned",
    "call",
    "cell",
    "position",
    "outcome",
    "failed",
}
"""The keys every candidate line holds, beside exactly one of `candidate` and `undecoded`."""

HEADER: Final = (
    b'{"kind": "header", "schema": 1, "run": "e1", "uci": "", "releases": {}, "settings": {}}\n'
)
"""A header line with no candidate after it."""

LINE: Final = {
    "kind": "candidate",
    "holdout": 0,
    "arm": "bko",
    "fill": "gap",
    "attempt": 1,
    "returned": True,
    "call": 1,
    "cell": "<55/female/1",
    "position": 0,
    "undecoded": "short",
    "outcome": "rejected",
    "failed": ["fields"],
}
"""A well-formed candidate line: an undecoded record."""

SOURCES: Final = {"candidates": "gap", "replica-candidates": "replica"}
"""Each candidate-count source of the scores, by the fill it counts."""


def _run(ctx: Context, log: Path, tmp_path: Path, *flag: str) -> str:
    out = io.StringIO()
    assert main(score_argv(log, tmp_path, *flag), context=ctx, out=out) == 0
    return out.getvalue()


def _counts(lines: tuple[Candidate, ...]) -> dict[tuple[str, int, str, str], float]:
    """Each (arm, holdout, fill, measure) as the scores count it, read from the lines."""
    found: Counter[tuple[str, int, str, str]] = Counter()
    for line in lines:
        key = (line.arm, line.holdout, line.fill)
        found[*key, "generated"] += 1
        found[*key, "passed"] += line.outcome != "rejected"
        if line.outcome.startswith("dropped-"):
            found[*key, line.outcome] += 1
    return {key: float(count) for key, count in found.items() if count}


def _scored(text: str) -> dict[tuple[str, int, str, str], float]:
    frame = pd.read_csv(io.StringIO(text))
    counted = frame[frame["source"].isin(SOURCES)]
    columns = ("arm", "holdout", "source", "measure", "value")
    return {
        (str(arm), int(holdout), SOURCES[str(source)], str(measure)): float(value)
        for arm, holdout, source, measure, value in counted[list(columns)].to_numpy().tolist()
        if value
    }


def test_a_paid_run_keeps_every_candidate_and_a_replay_rewrites_it(tmp_path, caplog):
    send = replica_script()
    ctx, _ = context(send)
    ctx = dataclasses.replace(ctx, experiment=REPLICAS)
    log = tmp_path / "calls.jsonl"
    caplog.set_level(logging.INFO)
    text = _run(ctx, log, tmp_path)
    send.assert_consumed()
    data = candidates_for(log).read_bytes()
    kept = read(data)
    assert kept.header == {
        "kind": "header",
        "schema": SCHEMA,
        "run": "e1",
        "uci": PIN.sha256,
        "releases": {prefix: source.sha256 for prefix, source in ctx.releases.items()},
        "settings": json.loads(json.dumps(dataclasses.asdict(SETTINGS))),
    }
    assert _counts(kept.candidates) == _scored(text)
    assert {line.fill for line in kept.candidates} == {"gap", "replica"}
    assert {line.arm for line in kept.candidates} == set(REPLICAS.arms[1:])
    rejected = [line for line in kept.candidates if line.outcome == "rejected"]
    assert [(line.arm, line.fill, line.failed) for line in rejected] == [
        ("bko", "replica", ("code:cp",)),
        ("bko", "replica", ("code:cp",)),
    ]
    assert rejected[0].candidate == PLANTED["code:cp"]
    replayed = tmp_path / "replay.jsonl"
    again = _run(
        dataclasses.replace(ctx, send=ScriptedSend), replayed, tmp_path, "--replay", str(log)
    )
    assert again == text
    assert candidates_for(replayed).read_bytes() == data


def _outcome_replies() -> tuple[Reply, ...]:
    """Holdout 0 keeps, drops for cell and quota and rejects; holdout 1 refuses, then returns.

    Holdout 0's well call holds one record of a cell not asked, its sick call one record past the
    quota and one that fails `code:cp`. Holdout 1's first attempt drops every record for range
    over three rounds and is refused; its second leaves one sick record undecoded, then asks it
    again.
    """
    first = (
        reply([*[young(0)] * 12, OLD]),
        reply([*[young(1)] * 16, young(1), PLANTED["code:cp"]]),
    )
    refused = (reply([BAD] * 12), reply([BAD] * 16)) * 3
    second = (reply([young(0)] * 12), reply([young(1)] * 15), reply([young(1)]))
    return (*first, *refused, *second)


def test_each_line_holds_its_keys_sorted_and_one_candidate_or_reason(tmp_path):
    ctx, _ = context(ScriptedSend(*_outcome_replies()))
    log = tmp_path / "calls.jsonl"
    _run(ctx, log, tmp_path)
    header, *lines = candidates_for(log).read_text(encoding="utf-8").splitlines()
    for raw in (header, *lines):
        assert raw == json.dumps(json.loads(raw), sort_keys=True)
    for raw in lines:
        line = json.loads(raw)
        assert line["kind"] == "candidate"
        extra = set(line) - KEYS
        assert extra in ({"candidate"}, {"undecoded"}), line
    assert candidates_for(log).stat().st_mode & 0o777 == 0o600


def test_every_outcome_is_kept_with_its_attempt_and_call(tmp_path):
    ctx, _ = context(ScriptedSend(*_outcome_replies()))
    log = tmp_path / "calls.jsonl"
    text = _run(ctx, log, tmp_path)
    lines = read(candidates_for(log).read_bytes()).candidates
    assert _counts(lines) == _scored(text)
    first, second = NARROW.holdout_seeds
    early = [line for line in lines if line.holdout == first]
    assert Counter(line.outcome for line in early) == {
        "kept": 28,
        "dropped-cell": 1,
        "dropped-quota": 1,
        "rejected": 1,
    }
    assert {(line.attempt, line.returned) for line in early} == {(1, True)}
    assert [(line.call, line.cell, line.position) for line in early[:2]] == [
        (1, "<55/female/0", 0),
        (1, "<55/female/0", 1),
    ]
    assert early[12].outcome == "dropped-cell"
    assert early[12].candidate == OLD
    (rejected,) = [line for line in early if line.outcome == "rejected"]
    assert rejected.failed == ("code:cp",)
    assert (rejected.call, rejected.position) == (2, 17)
    late = [line for line in lines if line.holdout == second]
    refused = [line for line in late if line.attempt == 1]
    assert len(refused) == 84
    assert {(line.returned, line.outcome) for line in refused} == {(False, "dropped-range")}
    returned = [line for line in late if line.attempt == 2]
    assert {line.returned for line in returned} == {True}
    (undecoded,) = [line for line in returned if line.undecoded is not None]
    assert undecoded.undecoded == "short"
    assert undecoded.candidate is None
    assert (undecoded.outcome, undecoded.failed) == ("rejected", ("fields",))
    assert [line.call for line in returned][-1] == 11
    assert {line.fill for line in lines} == {"gap"}
    assert {line.failed for line in lines if line.outcome != "rejected"} == {()}


def test_a_bko_run_without_candidates_exits_two_before_any_download_or_log(tmp_path, caplog):
    send = scripted()
    ctx, opener = context(send)
    log = tmp_path / "calls.jsonl"
    argv = ["score", "--log", str(log), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert "--candidates is required" in caplog.text
    assert not log.exists()
    assert opener.calls == []
    assert send.calls == []


def test_an_existing_candidates_file_exits_two_before_the_call_log(tmp_path):
    send = scripted()
    ctx, opener = context(send)
    log = tmp_path / "calls.jsonl"
    candidates_for(log).write_text("", encoding="utf-8")
    assert main(score_argv(log, tmp_path), context=ctx, out=io.StringIO()) == 2
    assert candidates_for(log).read_text(encoding="utf-8") == ""
    assert not log.exists()
    assert opener.calls == []
    assert send.calls == []


def test_candidates_need_a_bko_arm(tmp_path, caplog):
    ctx, _ = context(scripted(), arms={"none": FAST_ARMS["none"], "bko": FAST_ARMS["bko"]})
    path = tmp_path / "candidates.jsonl"
    argv = ["score", "--candidates", str(path), "--data-dir", str(tmp_path / "data")]
    assert main(argv, context=ctx, out=io.StringIO()) == 2
    assert "--candidates needs a BKO arm" in caplog.text
    assert not path.exists()


@pytest.mark.parametrize("spelling", ["calls.jsonl", "sub/../calls.jsonl"])
def test_candidates_and_the_call_log_must_be_two_files(tmp_path, caplog, spelling):
    ctx, opener = context(scripted())
    (tmp_path / "sub").mkdir()
    log = tmp_path / "calls.jsonl"
    argv = ["score", "--log", str(log), "--candidates", str(tmp_path / spelling)]
    assert main([*argv, "--data-dir", str(tmp_path / "data")], context=ctx, out=io.StringIO()) == 2
    assert "name one file" in caplog.text
    assert not log.exists()
    assert opener.calls == []


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b'{"kind": "candidate"}\n',
        b'{"kind": "header", "schema": 0}\n',
        HEADER + b'{"kind": "candidate", "holdout": 0}\n',
        HEADER.replace(b'"schema": 1', b'"schema": 0'),
        HEADER + HEADER,
        HEADER + json.dumps({**LINE, "holdout": "0"}).encode() + b"\n",
        HEADER + json.dumps({**LINE, "failed": []}).encode() + b"\n",
        HEADER + json.dumps({**LINE, "outcome": "kept"}).encode() + b"\n",
        HEADER.replace(b'"schema": 1', b'"schema": true'),
        HEADER + json.dumps({**LINE, "undecoded": None}).encode() + b"\n",
        HEADER + json.dumps({**LINE, "outcome": "dropped-range", "failed": []}).encode() + b"\n",
        b"not json\n",
    ],
)
def test_read_refuses_a_file_that_is_not_a_candidates_file(data):
    with pytest.raises(CandidatesFormatError):
        read(data)


def test_read_takes_a_header_and_a_well_formed_line():
    kept = read(HEADER + json.dumps(LINE).encode() + b"\n")
    assert kept.header["run"] == "e1"
    (line,) = kept.candidates
    assert (line.undecoded, line.candidate, line.failed) == ("short", None, ("fields",))
    assert line.line() == LINE
    assert read(HEADER).candidates == ()


def test_a_call_log_that_cannot_be_made_leaves_no_candidates_file(tmp_path):
    log = tmp_path / "calls.jsonl"
    log.write_text("", encoding="utf-8")
    candidates = tmp_path / "candidates.jsonl"
    knowledge = Knowledge("x", {}, (), ())
    origin = Origin("e1", PIN, {})
    with (
        pytest.raises(FileExistsError),
        guided_arms(
            ("bko",),
            knowledge,
            NARROW,
            log=log,
            candidates=candidates,
            origin=origin,
            send=ScriptedSend,
        ),
    ):
        pass
    assert not candidates.exists()


def _arms(tmp_path: Path, send) -> tuple[Path, Path]:
    """Enter `guided_arms` over files in `tmp_path` with `send`; return the two paths."""
    log, candidates = tmp_path / "calls.jsonl", tmp_path / "candidates.jsonl"
    with guided_arms(
        ("bko",),
        Knowledge("x", {}, (), ()),
        NARROW,
        log=log,
        candidates=candidates,
        origin=Origin("e1", PIN, {}),
        send=send,
    ):
        pass
    return log, candidates


def test_a_send_that_cannot_be_built_leaves_neither_file(tmp_path):
    def unbuildable():
        raise RuntimeError("no credential")

    with pytest.raises(RuntimeError, match="no credential"):
        _arms(tmp_path, unbuildable)
    assert list(tmp_path.iterdir()) == []


def test_an_existing_candidates_file_is_refused_and_left_untouched(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("theirs\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _arms(tmp_path, ScriptedSend)
    assert candidates.read_text(encoding="utf-8") == "theirs\n"
    assert not (tmp_path / "calls.jsonl").exists()


@pytest.mark.parametrize("fault", ["fdopen", "header"])
def test_a_candidates_file_whose_setup_fails_is_removed(tmp_path, monkeypatch, fault):
    def refuse(*args, **kwargs):
        raise OSError("disk full")

    if fault == "fdopen":
        monkeypatch.setattr(holdout_candidates.os, "fdopen", refuse)
    else:
        monkeypatch.setattr(CandidateFile, "_write", refuse)
    path = tmp_path / "candidates.jsonl"
    with pytest.raises(OSError, match="disk full"):
        CandidateFile(path, Origin("e1", PIN, {}), SETTINGS)
    assert not path.exists()
