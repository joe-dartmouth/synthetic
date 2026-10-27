"""`score --rehearse`: answer every BKO call with canned records, so a paid run's rules run free.

Only the send is replaced: every call still goes through the real prompt, call log, validation,
drops, candidates file, spend ceiling, projection and scoring. `CannedSend` answers each call with
exactly the records it asks for, all in the call's own cell (sex and target from the `Call`, age
from the interval its user turn names), all valid, all inside every holdout's training range, and
none equal to a real record, each billed at `e1-3`'s tokens per record asked. Its spend is a wiring
check, not an estimate. ADR `research/e1-evaluation-protocol` says when the tokens per record are
recomputed.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from typing import Final

import pandas as pd

from bko.datasets.split import split
from bko.datasets.uci import CODEBOOK, FRACTION_DIGITS, FRACTIONAL, FRAME_COLUMNS, NUMERIC
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import Experiment
from bko.generate.messages import Call, Reply, Send, Settings, Usage


INPUT_PER_RECORD: Final = 368  # e1-3 Runs row: 34843870 / 94727
"""Billed input tokens per record asked: `e1-3`'s input tokens over its records asked, rounded."""

OUTPUT_PER_RECORD: Final = 197  # 18665554 / 94727
"""Billed output tokens per record asked: `e1-3`'s output tokens over its records asked, rounded."""

PROMPT: Final = re.compile(
    r"Write exactly (?P<rows>\d+) records of patients in this group: "
    r"sex (?P<sex>\d) \([a-z]+\), target (?P<label>\d) \([^)]*\), "
    r"age (?P<low>\d+(?:\.\d+)?) to (?P<high>\d+(?:\.\d+)?) years\."
)
"""The first sentence of every user turn `bko.generate.prompt.render` writes, at every level."""

STEPS: Final = {"age": 11, "trestbps": 7, "chol": 13, "thalach": 11, "oldpeak": 7}
"""Each measured field's stride through its safe values, so successive records differ."""

SKIPS: Final = 10_000
"""Canned records in a row that may equal real ones before the send gives up."""

type Rehearse = Callable[[pd.DataFrame, Experiment, Settings], Send]
"""Build the send a rehearsal answers every call with, from the frame, the run and its settings."""


def safe_ranges(frame: pd.DataFrame, seeds: Sequence[int]) -> dict[str, tuple[float, float]]:
    """Each measured field's range inside the training split of every one of `seeds`.

    Raises:
        EvaluationError: If no value of a field lies inside every split's range.
    """
    trains = [split(frame, seed=seed).train for seed in seeds]
    ranges: dict[str, tuple[float, float]] = {}
    for name in NUMERIC:
        low = max(float(train[name].min()) for train in trains)
        high = min(float(train[name].max()) for train in trains)
        if low > high:
            msg = f"no {name} lies inside every holdout's training range"
            raise EvaluationError(msg)
        ranges[name] = (low, high)
    return ranges


def _grid(low: float, high: float, name: str) -> tuple[int, int, int]:
    """The scaled whole-number bounds of `name`'s values in `low` to `high`, and its scale."""
    scale = 10**FRACTION_DIGITS if name in FRACTIONAL else 1
    first, last = math.ceil(low * scale - 1e-9), math.floor(high * scale + 1e-9)
    if first > last:
        msg = f"no {name} on the archive's precision lies in {low:g} to {high:g}"
        raise EvaluationError(msg)
    return first, last, scale


class CannedSend:
    """A `Send` that answers every call with exactly the records it asks for, in its own cell.

    Attributes:
        calls: The calls answered.
        records: The canned records drawn, those skipped as equal to a real record included.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        experiment: Experiment,
        settings: Settings,
        *,
        per_record: tuple[int, int] = (INPUT_PER_RECORD, OUTPUT_PER_RECORD),
    ) -> None:
        """Read the safe ranges over `experiment`'s holdouts of `frame`.

        Args:
            frame: The records the run splits; no canned record equals one of them.
            experiment: The run; its holdout seeds bound the ranges.
            settings: The run's settings; each reply names their model.
            per_record: The input and output tokens billed per record asked.

        Raises:
            EvaluationError: As `safe_ranges`, or if a field has no value on its precision there.
        """
        ranges = safe_ranges(frame, experiment.holdout_seeds)
        self._grids = {name: _grid(*ranges[name], name) for name in NUMERIC}
        self._real = frozenset(frame.loc[:, list(FRAME_COLUMNS)].itertuples(index=False, name=None))
        self._model = settings.model
        self._input, self._output = per_record
        self.calls = 0
        self.records = 0

    def __call__(self, call: Call) -> Reply:
        """Answer `call` with its records, billed at the tokens per record.

        Raises:
            ValueError: If the user turn does not read as the prompt renders it or names another
                cell than the call's.
            EvaluationError: If no canned record of the cell's ages fits the safe ranges, or
                `SKIPS` in a row equal real records.
        """
        found = PROMPT.match(call.user)
        if found is None:
            msg = f"the user turn does not read as the prompt renders it: {call.user[:200]!r}"
            raise ValueError(msg)
        if (int(found["sex"]), int(found["label"])) != (call.sex, call.label):
            msg = f"the user turn names sex {found['sex']} target {found['label']}, not the call's"
            raise ValueError(msg)
        rows = int(found["rows"])
        first, last, _ = self._grids["age"]
        ages = (
            max(first, math.ceil(float(found["low"]))),
            min(last, math.floor(float(found["high"]))),
        )
        if ages[0] > ages[1]:
            msg = f"no age of {found['low']} to {found['high']} lies inside every training range"
            raise EvaluationError(msg)
        batch = [self._next(call.sex, call.label, ages) for _ in range(rows)]
        self.calls += 1
        text = json.dumps({"records": batch})
        return Reply(
            request_id=f"req_rehearsal_{self.calls}",
            model=self._model,
            stop_reason="end_turn",
            text=text,
            usage=Usage(
                input_tokens=self._input * rows,
                output_tokens=self._output * rows,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
            raw={
                "id": f"msg_rehearsal_{self.calls}",
                "model": self._model,
                "content": [{"type": "text", "text": text}],
            },
            attempts=1,
        )

    def _next(self, sex: int, label: int, ages: tuple[int, int]) -> dict[str, int | float]:
        for _ in range(SKIPS):
            item = self._record(self.records, sex, label, ages)
            self.records += 1
            if tuple(item[name] for name in FRAME_COLUMNS) not in self._real:
                return item
        msg = f"{SKIPS} canned records in a row equal real records"
        raise EvaluationError(msg)

    def _record(
        self, n: int, sex: int, label: int, ages: tuple[int, int]
    ) -> dict[str, int | float]:
        """The n-th canned record: each code cycles at its own pace, each measure strides."""
        values: dict[str, int | float] = {}
        for name in NUMERIC:
            first, last, scale = (*ages, 1) if name == "age" else self._grids[name]
            scaled = first + n * STEPS[name] % (last - first + 1)
            values[name] = scaled / scale if scale > 1 else scaled
        paces = {"cp": 1, "fbs": 1, "restecg": 1, "exang": 2, "slope": 3, "ca": 5, "thal": 7}
        for name, pace in paces.items():
            codes = sorted(CODEBOOK[name])
            values[name] = codes[n // pace % len(codes)]
        values["sex"], values["target"] = sex, label
        return {name: values[name] for name in FRAME_COLUMNS}
