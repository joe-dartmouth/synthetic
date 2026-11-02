"""`score --rehearse`: answer every BKO call with canned records, so a paid run's rules run free.

Only the send is replaced: every call still goes through the real prompt, call log, validation,
drops, candidates file, spend ceiling, projection and scoring. `CannedSend` answers each call with
exactly the records it asks for, all in the call's own cell (sex and target from the `Call`, age
from the interval its user turn names), all valid, all inside every holdout's training range, and
none equal to a real record, each billed at the judged run `e1-4`'s tokens per record asked. A
prescribing call's records each show exactly the concepts its record line names, by the run's field
rows (`bko.generate.skeleton`), so a rehearsal drops none for concepts. Its
records are in the run's frame (`bko.evaluate.data.Data`), so it refuses a call for another frame,
raising `GenerationError` so the call log's `error` line keeps the reason. Its spend is a wiring
check, not an estimate. ADR `research/e1-evaluation-protocol` says when the tokens per record are
recomputed.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from collections.abc import Callable, Sequence
from typing import Final

import pandas as pd

from bko.datasets.frames import FrameSpec
from bko.datasets.split import split
from bko.evaluate.data import Data
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import Experiment
from bko.evaluate.prepare import prepared
from bko.generate.errors import GenerationError
from bko.generate.messages import Call, Reply, Send, Settings, Usage
from bko.generate.prompt import NO_FINDING, field_units
from bko.generate.skeleton import Named, skeleton
from bko.ontology.schema import AtLeast, FieldMapping, shown


INPUT_PER_RECORD: Final = 301  # e1-4 Runs row: 23287576 / 77487
"""Billed input tokens per record asked: `e1-4`'s input tokens over its records asked, rounded."""

OUTPUT_PER_RECORD: Final = 211  # 16362315 / 77487
"""Billed output tokens per record asked: `e1-4`'s output tokens over its records asked, rounded."""

PROMPT: Final = re.compile(
    r"Write exactly (?P<rows>\d+) records of patients in this group: "
    r"sex (?P<sex>\d) \([a-z]+\), target (?P<label>\d) \([^)]*\), "
    r"age (?P<low>\d+(?:\.\d+)?) to (?P<high>\d+(?:\.\d+)?) (?P<unit>[a-z]+(?: [a-z]+)*)\."
)
"""The first sentence of every user turn `bko.generate.prompt.render` writes, at every level; `unit`
is age's unit as the prompt writes it for the call's frame (`bko.generate.prompt.field_units`)."""

RECORD_LINE: Final = re.compile(r"^Record (?P<number>\d+): (?P<labels>.*)$", re.MULTILINE)
"""A prescribing user turn's line for one record: its number from 1, then its concepts' labels."""

STEPS: Final = {"age": 11, "trestbps": 7, "chol": 13, "thalach": 11, "oldpeak": 7}
"""Each of UCI's measured fields' stride through its safe values, so successive records differ."""

PACES: Final = {"cp": 1, "fbs": 1, "restecg": 1, "exang": 2, "slope": 3, "ca": 5, "thal": 7}
"""Each of UCI's coded fields' pace: the records a code holds before the next one comes."""

FALLBACK_STEPS: Final = (7, 11, 13)
"""The strides of a measured field `STEPS` lacks, by its position among the frame's numeric
fields, cycling."""

SKIPS: Final = 10_000
"""Canned records in a row that may equal real ones before the send gives up."""

type Rehearse = Callable[[Data, Experiment, Settings], Send]
"""Build the send a rehearsal answers every call with, from the run's data, the run and its
settings."""


def safe_ranges(
    records: pd.DataFrame, experiment: Experiment, *, frame: FrameSpec
) -> dict[str, tuple[float, float]]:
    """Each of `frame`'s measured fields' range inside every holdout's training rows.

    Each split bands its cells by `frame`'s edges, and its training rows are those the
    experiment's training fraction and removal leave (`bko.evaluate.prepare.prepared`), as the
    harness's are.

    Raises:
        EvaluationError: If no value of a field lies inside every split's range.
    """
    trains = [
        prepared(
            split(records, seed=seed, edges=frame.edges), experiment, edges=frame.edges
        ).ablated.train
        for seed in experiment.holdout_seeds
    ]
    ranges: dict[str, tuple[float, float]] = {}
    for name in frame.numeric:
        low = max(float(train[name].min()) for train in trains)
        high = min(float(train[name].max()) for train in trains)
        if low > high:
            msg = f"no {name} lies inside every holdout's training range"
            raise EvaluationError(msg)
        ranges[name] = (low, high)
    return ranges


def _grid(low: float, high: float, name: str, frame: FrameSpec) -> tuple[int, int, int]:
    """The scaled whole-number bounds of `name`'s values in `low` to `high`, and its scale."""
    scale = 10**frame.fraction_digits if name in frame.fractional else 1
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
        data: Data,
        experiment: Experiment,
        settings: Settings,
        *,
        per_record: tuple[int, int] = (INPUT_PER_RECORD, OUTPUT_PER_RECORD),
    ) -> None:
        """Read the safe ranges over `experiment`'s holdouts of `data`'s records.

        Args:
            data: The run's data; its frame shapes every canned record, and no canned record
                equals one of its records.
            experiment: The run; its holdout seeds bound the ranges.
            settings: The run's settings; each reply names their model.
            per_record: The input and output tokens billed per record asked.

        Raises:
            EvaluationError: As `safe_ranges`, or if a field has no value on its precision there.
        """
        frame = data.frame
        self._frame = frame
        ranges = safe_ranges(data.records, experiment, frame=frame)
        self._grids = {name: _grid(*ranges[name], name, frame) for name in frame.numeric}
        self._steps = {
            name: STEPS.get(name, FALLBACK_STEPS[position % len(FALLBACK_STEPS)])
            for position, name in enumerate(frame.numeric)
        }
        self._paces = {
            name: PACES.get(name, position + 1)
            for position, name in enumerate(frame.categorical)
            if name != "sex"
        }
        self._real = frozenset(
            data.records.loc[:, list(frame.columns)].itertuples(index=False, name=None)
        )
        self._named = {one.label: one for one in skeleton((), data.field_rows, frame=frame).named}
        self._model = settings.model
        self._input, self._output = per_record
        self.calls = 0
        self.records = 0

    def __call__(self, call: Call) -> Reply:
        """Answer `call` with its records, billed at the tokens per record.

        Raises:
            GenerationError: If the call is for another frame than the run's, or its user turn
                does not read as the prompt renders it, gives age in another unit than the
                frame's, or names another cell than the call's.
            EvaluationError: If no canned record of the cell's ages fits the safe ranges, or
                `SKIPS` in a row equal real records.
        """
        if call.frame is not self._frame:
            msg = (
                f"a rehearsal of frame {self._frame.name} answers its own calls only, not a call "
                f"for frame {call.frame.name}"
            )
            raise GenerationError(msg)
        found = PROMPT.match(call.user)
        if found is None:
            msg = f"the user turn does not read as the prompt renders it: {call.user[:200]!r}"
            raise GenerationError(msg)
        unit = field_units(self._frame)["age"]
        if found["unit"] != unit:
            msg = f"the user turn gives age in {found['unit']}, not the frame's {unit}"
            raise GenerationError(msg)
        if (int(found["sex"]), int(found["label"])) != (call.sex, call.label):
            msg = f"the user turn names sex {found['sex']} target {found['label']}, not the call's"
            raise GenerationError(msg)
        rows = int(found["rows"])
        first, last, _ = self._grids["age"]
        ages = (
            max(first, math.ceil(float(found["low"]))),
            min(last, math.floor(float(found["high"]))),
        )
        if ages[0] > ages[1]:
            msg = f"no age of {found['low']} to {found['high']} lies inside every training range"
            raise EvaluationError(msg)
        lines = self._lines(call.user, rows)
        batch = [
            self._next(call.sex, call.label, ages, None if lines is None else lines[position])
            for position in range(rows)
        ]
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

    def _lines(self, user: str, rows: int) -> list[tuple[Named, ...]] | None:
        """Each record line's named concepts, or None for a call that prescribes none.

        Raises:
            GenerationError: If the lines are not numbered 1 to `rows` or name an unknown label.
        """
        found = list(RECORD_LINE.finditer(user))
        if not found:
            return None
        if [int(line["number"]) for line in found] != list(range(1, rows + 1)):
            msg = f"the user turn's record lines are not numbered 1 to {rows}"
            raise GenerationError(msg)
        lines: list[tuple[Named, ...]] = []
        for line in found:
            labels = [] if line["labels"] == NO_FINDING else line["labels"].split("; ")
            if stray := sorted(set(labels) - self._named.keys()):
                msg = f"record {line['number']} names {stray}, which no field row of the run maps"
                raise GenerationError(msg)
            lines.append(tuple(self._named[text] for text in labels))
        return lines

    def _next(
        self,
        sex: int,
        label: int,
        ages: tuple[int, int],
        line: tuple[Named, ...] | None = None,
    ) -> dict[str, int | float]:
        columns = self._frame.columns
        for _ in range(SKIPS):
            item = self._record(self.records, sex, label, ages)
            if line is not None:
                item = self._shaped(item, line)
            self.records += 1
            if tuple(item[name] for name in columns) not in self._real:
                return item
        msg = f"{SKIPS} canned records in a row equal real records"
        raise EvaluationError(msg)

    def _shaped(
        self, record: dict[str, int | float], line: tuple[Named, ...]
    ) -> dict[str, int | float]:
        """`record` with each named concept's fields set so it shows exactly `line`'s concepts.

        Each group of fields the named concepts' rows join is searched apart, the record's own
        values first, so a field changes only when its concepts need it to.

        Raises:
            GenerationError: If no values on the safe ranges and codebooks show `line` exactly.
        """
        out = dict(record)
        wanted = frozenset(one.concept for one in line)
        for fields, rows in _groups(tuple(self._named.values())):
            choices = [self._choices(name, rows, out[name]) for name in fields]
            target = wanted & {row.concept for row in rows}
            for values in itertools.product(*choices):
                trial = {**out, **dict(zip(fields, values, strict=True))}
                if shown(trial, rows) == target:
                    out = trial
                    break
            else:
                msg = f"no canned value of {', '.join(fields)} shows exactly {sorted(target)}"
                raise GenerationError(msg)
        return out

    def _choices(self, name: str, rows: Sequence[FieldMapping], value: float) -> list[int | float]:
        """`value` first, then each value of `name` a row's clause can tell apart.

        The cell's fields, age, sex and the label, keep `value`, so the record stays in its cell.
        """
        frame = self._frame
        if name in {"age", "sex", frame.label}:
            return [value]
        if name in frame.categorical:
            return [value, *(code for code in sorted(frame.codebook[name]) if code != value)]
        first, last, scale = self._grids[name]
        limits = {
            clause.value
            for row in rows
            for clause in row.when
            if isinstance(clause, AtLeast) and clause.field == name
        }
        values = [first, last]
        for limit in sorted(limits):
            at = min(max(math.ceil(limit * scale - 1e-9), first), last)
            values += [at, max(at - 1, first)]
        scaled = [step / scale if scale > 1 else step for step in values]
        return [value, *dict.fromkeys(item for item in scaled if item != value)]

    def _record(
        self, n: int, sex: int, label: int, ages: tuple[int, int]
    ) -> dict[str, int | float]:
        """The n-th canned record: each code cycles at its own pace, each measure strides."""
        frame = self._frame
        values: dict[str, int | float] = {}
        for name in frame.numeric:
            first, last, scale = (*ages, 1) if name == "age" else self._grids[name]
            scaled = first + n * self._steps[name] % (last - first + 1)
            values[name] = scaled / scale if scale > 1 else scaled
        for name, pace in self._paces.items():
            codes = sorted(frame.codebook[name])
            values[name] = codes[n // pace % len(codes)]
        values["sex"], values[frame.label] = sex, label
        return {name: values[name] for name in frame.columns}


def _groups(named: Sequence[Named]) -> list[tuple[tuple[str, ...], tuple[FieldMapping, ...]]]:
    """The fields the named concepts' rows join, each with every row reading one of them.

    Two fields join when one concept's rows read both; each group's fields are sorted by name,
    and its rows follow the concepts' order.
    """
    groups: list[tuple[set[str], list[FieldMapping]]] = []
    for one in named:
        fields = {clause.field for row in one.states for clause in row.when}
        joined = [group for group in groups if group[0] & fields]
        merged = (
            set().union(fields, *(group[0] for group in joined)),
            [row for group in joined for row in group[1]] + list(one.states),
        )
        groups = [group for group in groups if group not in joined] + [merged]
    return [(tuple(sorted(fields)), tuple(rows)) for fields, rows in groups]
