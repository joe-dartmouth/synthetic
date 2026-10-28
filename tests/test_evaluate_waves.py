"""`python -m bko.evaluate score`'s output, pinned byte for byte against `tests/golden/waves/`.

The harness runs every BKO arm, with gap and replica fills, on three holdouts, over a send whose
replies are keyed by call number. Its script covers what makes one call's ask depend on another's
reply: a record kept in another age band of its sex and label, out-of-range values, short replies,
quota overflow, an attempt that exhausts its budget, and a replica round whose short cells run
A1, A2, B1 (two cells of one sex and label, then another). `tests/golden/waves/regenerate.py`
rewrites the golden files.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import logging
import random
import re
import sys
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Final

import pytest

from bko.evaluate.__main__ import main
from bko.evaluate.attempts import Attempt
from bko.generate.calls import Batch
from bko.generate.messages import Call, Reply
from tests._evaluate_bko_fixtures import REPLICAS, candidates_for, context, score_argv
from tests._generate_fixtures import reply
from tests._validate_fixtures import PLANTED, planted


GOLDEN: Final = Path(__file__).parent / "golden" / "waves"
FILES: Final = ("scores.csv", "candidates.jsonl", "calls.jsonl")
"""The golden files, in the order `_run` returns their bytes."""

WAVES: Final = dataclasses.replace(REPLICAS, holdout_seeds=(0, 1, 2))
"""Every BKO arm filling its gap and a replica, on three holdouts."""

ASK: Final = re.compile(r"Write exactly (\d+) records .*?, age (\d+) to (\d+) years")

type Records = list[object]
type Edit = Callable[[Call, Records], Records]

VOLATILE: Final = frozenset({"at", "started", "ended", "in_flight", "anthropic"})
"""Call log keys that vary from run to run, with how many calls are in flight, or with the SDK."""

KINDS: Final = ("run", "request", "reply", "error", "end")


def _asked(call: Call) -> tuple[int, int]:
    """The rows `call` asks for, and an age inside its band."""
    found = ASK.search(call.user)
    assert found is not None, call.user
    low, high = int(found[2]), int(found[3])
    return int(found[1]), 40 if low <= 40 <= high else 63


def _records(call: Call, rows: int, **changes: object) -> Records:
    """`rows` valid records of `call`'s cell, with chest pain 2 so none copies a test record."""
    _, age = _asked(call)
    record = planted(age=age, sex=call.sex, target=call.label, cp=2, **changes)
    return [record] * rows


def _overflow(call: Call, records: Records) -> Records:
    return [*records, *_records(call, 2)]


def _short(call: Call, records: Records) -> Records:
    return records[: len(records) * 2 // 3]


def _out_of_range(call: Call, records: Records) -> Records:
    return [*records[:-2], *_records(call, 2, chol=234)]


def _invalid(call: Call, records: Records) -> Records:
    return [PLANTED["code:cp"]] * len(records)


def _other_band(call: Call, records: Records) -> Records:
    """Two records short, and one of them kept in the older band of the cell's sex and label."""
    older = planted(age=63, sex=call.sex, target=call.label, cp=2)
    return [*records[:-2], older]


SCRIPT: Final[Mapping[int, Edit]] = {
    # Holdout 0, `bko`'s gap fill.
    1: _overflow,
    2: _short,
    # `bko-population`'s gap fill.
    4: _out_of_range,
    # `bko-frequency`'s gap fill: its first attempt exhausts its budget on invalid records.
    7: _invalid,
    8: _invalid,
    9: _invalid,
    10: _invalid,
    11: _invalid,
    12: _invalid,
    # `bko`'s replica fill: cells (<55, female, 0), (>=55, female, 0) and (>=55, male, 1) stay
    # short, so its second round runs A1, A2, B1.
    15: _other_band,
    19: _short,
    22: _short,
}
"""Each scripted call's edit of the records its ask would be answered with in full."""


class WaveSend:
    """A fake `Send` answering each call in full, or as `SCRIPT` edits it, by call number.

    Attributes:
        calls: Every call made, in order.
    """

    def __init__(self) -> None:
        """Start with no call made."""
        self.calls: list[Call] = []

    def __call__(self, call: Call) -> Reply:
        """Record `call` and return its reply."""
        self.calls.append(call)
        rows, _ = _asked(call)
        records = _records(call, rows)
        edit = SCRIPT.get(len(self.calls))
        return reply(records if edit is None else edit(call, records))


class LateSend:
    """`WaveSend`, numbering each call at dispatch, whose replies land after a random delay.

    Each call's delay is `random.Random(call).uniform(0, 0.02)` seconds, so replies land out of
    order.

    Attributes:
        calls: Every call made, in dispatch order.
        landed: Each call's number, counted from 1 at dispatch, in the order its reply landed.
    """

    def __init__(self, inner: WaveSend | None = None) -> None:
        """Answer through `inner`, a fresh `WaveSend` when None."""
        self._inner = inner or WaveSend()
        self.calls = self._inner.calls
        self.landed: list[int] = []

    def __call__(self, call: Call) -> Awaitable[Reply]:
        """Number `call` and build its reply now; return them after its delay."""
        answer = self._inner(call)
        number = len(self.calls)
        delay = random.Random(number).uniform(0, 0.02)  # ruff: ignore[suspicious-non-cryptographic-random-usage]

        async def land() -> Reply:
            await asyncio.sleep(delay)
            self.landed.append(number)
            return answer

        return land()


class CrossingSend(WaveSend):
    """`WaveSend` whose call 15 answers with men's records: call 17 of its wave asks men's cell."""

    def __call__(self, call: Call) -> Reply:
        """Record `call`; answer call 15 with records of sex 1, the others as `WaveSend` does."""
        answer = super().__call__(call)
        if len(self.calls) != 15:
            return answer
        rows, age = _asked(call)
        crossed = planted(age=age, sex=1, target=call.label, cp=2)
        return reply([crossed] * rows)


def _sort_key(line: Mapping[str, object]) -> tuple[int, int, str]:
    """Call, then kind, then the line itself, so no tie falls back to file order."""
    kind = str(line["kind"])
    default = -1 if kind == "run" else sys.maxsize
    call = line.get("call", default)
    assert isinstance(call, int)
    return call, KINDS.index(kind), json.dumps(line, sort_keys=True)


def _normalised(log: Path) -> bytes:
    """The call log without its volatile keys, in call order, each call's lines in kind order."""
    lines = [json.loads(text) for text in log.read_text(encoding="utf-8").splitlines()]
    kept = [{key: value for key, value in line.items() if key not in VOLATILE} for line in lines]
    kept.sort(key=_sort_key)
    return "".join(json.dumps(line, sort_keys=True) + "\n" for line in kept).encode()


def _run(
    tmp_path: Path,
    *,
    in_flight: int | None,
    send: Callable[[Call], Reply | Awaitable[Reply]] | None = None,
    replay: Path | None = None,
) -> tuple[bytes, bytes, bytes]:
    """Score `WAVES` in-process; return the scores, the candidates file and the normalised log.

    `in_flight` None passes no `--in-flight`. Every call goes to `send`, a fresh `WaveSend` when
    None, or is answered from the call log `replay`.
    """
    built, _ = context(send or WaveSend())
    built = dataclasses.replace(built, experiment=WAVES)
    log = tmp_path / "calls.jsonl"
    flag = [] if in_flight is None else ["--in-flight", str(in_flight)]
    if replay is not None:
        flag += ["--replay", str(replay)]
    out = io.StringIO()
    code = main(score_argv(log, tmp_path, *flag), context=built, out=out)
    assert code == 0
    return out.getvalue().encode(), candidates_for(log).read_bytes(), _normalised(log)


def _differ(made: tuple[bytes, bytes, bytes]) -> list[str]:
    """The golden files `made` differs from."""
    pairs = zip(FILES, made, strict=True)
    return [name for name, data in pairs if data != (GOLDEN / name).read_bytes()]


def test_the_harness_matches_its_golden(tmp_path: Path) -> None:
    assert _differ(_run(tmp_path, in_flight=1)) == []


@pytest.mark.parametrize("in_flight", [1, 2, 5, 32])
def test_any_in_flight_matches_the_golden(tmp_path: Path, in_flight: int) -> None:
    send = LateSend()
    assert _differ(_run(tmp_path, in_flight=in_flight, send=send)) == []
    if in_flight > 2:
        # Replies landed out of call order, so the golden pins a concurrent run, not a serial one.
        assert send.landed != sorted(send.landed)


ANSWERED: Final = frozenset({"replay", "live_usd"})
"""Call log keys saying where calls were answered from: a replay names its log and pays nothing."""


def _unmarked(log: bytes) -> bytes:
    """A normalised call log without the keys that say where its calls were answered from."""
    lines = (json.loads(line) for line in log.splitlines())
    kept = ({key: value for key, value in line.items() if key not in ANSWERED} for line in lines)
    return "".join(json.dumps(line, sort_keys=True) + "\n" for line in kept).encode()


def test_a_replay_at_32_in_flight_matches_the_golden(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _run(source, in_flight=1)
    replayed = tmp_path / "replayed"
    replayed.mkdir()
    scores, candidates, log = _run(
        replayed, in_flight=32, send=_never, replay=source / "calls.jsonl"
    )
    assert _differ((scores, candidates, (GOLDEN / "calls.jsonl").read_bytes())) == []
    assert _unmarked(log) == _unmarked((GOLDEN / "calls.jsonl").read_bytes())


def _never(call: Call) -> Reply:
    raise AssertionError(call)


def test_a_reply_crossing_sex_inside_a_wave_stops(tmp_path: Path, caplog, monkeypatch) -> None:
    taken: list[int] = []
    take = Attempt.take

    def spied(attempt: Attempt, batch: Batch) -> None:
        taken.append(batch.index)
        take(attempt, batch)

    monkeypatch.setattr(Attempt, "take", spied)
    built, _ = context(CrossingSend())
    built = dataclasses.replace(built, experiment=WAVES)
    log = tmp_path / "calls.jsonl"
    caplog.set_level(logging.ERROR)
    code = main(score_argv(log, tmp_path, "--in-flight", "8"), context=built, out=io.StringIO())
    assert code == 3
    assert "call 15 " in caplog.text
    end = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert end["status"] == "failed"
    # Call 15's batch is the last taken: the wave's later batches are never committed.
    assert taken[-1] == 15
    assert sorted(taken) == list(range(1, 16))


def test_score_holds_each_call_to_the_experiments_idle_limit(tmp_path: Path, caplog) -> None:
    def stalled(call: Call) -> Awaitable[Reply]:
        async def never() -> Reply:
            await asyncio.Event().wait()
            raise AssertionError(call)

        return never()

    built, _ = context(stalled)
    built = dataclasses.replace(built, experiment=dataclasses.replace(WAVES, idle_limit=0.5))
    log = tmp_path / "calls.jsonl"
    caplog.set_level(logging.WARNING)
    code = main(score_argv(log, tmp_path, "--in-flight", "2"), context=built, out=io.StringIO())
    assert code == 2
    end = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert end["stop"] == "idle"
    assert "stop rule=idle call=1 limit_seconds=0.5" in caplog.messages
