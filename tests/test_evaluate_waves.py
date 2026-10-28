"""`python -m bko.evaluate score`'s output, pinned byte for byte against `tests/golden/waves/`.

The harness runs every BKO arm, with gap and replica fills, on three holdouts, over a send whose
replies are keyed by call number. Its script covers what makes one call's ask depend on another's
reply: a record kept in another age band of its sex and label, out-of-range values, short replies,
quota overflow, an attempt that exhausts its budget, and a replica round whose short cells run
A1, A2, B1 (two cells of one sex and label, then another). `tests/golden/waves/regenerate.py`
rewrites the golden files.
"""

from __future__ import annotations

import dataclasses
import io
import json
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

from bko.evaluate.__main__ import main
from bko.generate.messages import Call, Reply
from tests._evaluate_bko_fixtures import REPLICAS, candidates_for, context, score_argv
from tests._generate_fixtures import reply
from tests._validate_fixtures import PLANTED, planted


GOLDEN: Final = Path(__file__).parent / "golden" / "waves"

WAVES: Final = dataclasses.replace(REPLICAS, holdout_seeds=(0, 1, 2))
"""Every BKO arm filling its gap and a replica, on three holdouts."""

ASK: Final = re.compile(r"Write exactly (\d+) records .*?, age (\d+) to (\d+) years")

type Records = list[object]
type Edit = Callable[[Call, Records], Records]

VOLATILE: Final = frozenset({"at", "started", "ended", "in_flight"})
"""Call log keys that vary from run to run, or with how many calls are in flight."""

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
        """Start at call 1."""
        self.calls: list[Call] = []

    def __call__(self, call: Call) -> Reply:
        """Record `call` and return its reply."""
        self.calls.append(call)
        rows, _ = _asked(call)
        records = _records(call, rows)
        edit = SCRIPT.get(len(self.calls))
        return reply(records if edit is None else edit(call, records))


def _sort_key(line: Mapping[str, object]) -> tuple[float, int]:
    kind = str(line["kind"])
    default = -1 if kind == "run" else sys.maxsize
    call = line.get("call", default)
    assert isinstance(call, int)
    return call, KINDS.index(kind)


def _normalised(log: Path) -> bytes:
    """The call log without its volatile keys, in call order, each call's lines in kind order."""
    lines = [json.loads(text) for text in log.read_text(encoding="utf-8").splitlines()]
    kept = [{key: value for key, value in line.items() if key not in VOLATILE} for line in lines]
    kept.sort(key=_sort_key)
    return "".join(json.dumps(line, sort_keys=True) + "\n" for line in kept).encode()


def _run(tmp_path: Path, *, in_flight: int | None) -> tuple[bytes, bytes, bytes]:
    """Score `WAVES` in-process; return the scores, the candidates file and the normalised log.

    `in_flight` 1 or None passes no `--in-flight`.
    """
    send = WaveSend()
    built, _ = context(send)
    built = dataclasses.replace(built, experiment=WAVES)
    log = tmp_path / "calls.jsonl"
    flag = [] if in_flight is None or in_flight == 1 else ["--in-flight", str(in_flight)]
    out = io.StringIO()
    code = main(score_argv(log, tmp_path, *flag), context=built, out=out)
    assert code == 0
    return out.getvalue().encode(), candidates_for(log).read_bytes(), _normalised(log)


def test_the_harness_matches_its_golden(tmp_path: Path) -> None:
    scores, candidates, calls = _run(tmp_path, in_flight=1)
    assert scores == (GOLDEN / "scores.csv").read_bytes()
    assert candidates == (GOLDEN / "candidates.jsonl").read_bytes()
    assert calls == (GOLDEN / "calls.jsonl").read_bytes()
