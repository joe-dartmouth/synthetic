"""G7's free gate: does the BKB likelihood separate the judged arm's records by fidelity?

`likelihood` reads a run's per-holdout candidates file, or a sliced run's files as one, and
scores every record of `bko`'s replica fill that counted toward a cell (`kept` or
`dropped-quota`) by its BKB likelihood
(`bko.gaps.likelihood`) at its cell's shares on that holdout's training rows. Each cell's records
split at the cell's median score, ties at the median in the order of
`sha256(f"{holdout}:{call}:{position}")`, and the cells' top halves pooled against their bottom
halves, so both carry the training cell mix. The tops' and bottoms' fidelity against the training
rows (quality, column shapes, column pair trends; `bko.evaluate.fidelity`) are paired over the
holdouts, and the gate is separated when every top-minus-bottom corrected interval
(`bko.evaluate.criteria.corrected_interval`, at the largest held-out ratio as the criteria take it)
lies above zero.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import pandas as pd

from bko.datasets.split import split
from bko.evaluate.criteria import corrected_interval
from bko.evaluate.data import Data, Dataset
from bko.evaluate.errors import EvaluationError
from bko.evaluate.experiment import Experiment
from bko.evaluate.fidelity import MEASURES, fidelity
from bko.evaluate.holdout_candidates import KEPT, Candidate, HoldoutCandidates, read
from bko.evaluate.prepare import held_out_ratio, prepared
from bko.gaps.likelihood import Likelihood
from bko.gaps.observe import Cell, by_concept, cell_key, observe_records, record_cells
from bko.gaps.shares import shares
from bko.ontology.schema import FieldMapping


logger = logging.getLogger(__name__)

ARM = "bko"
"""The arm whose records the gate scores: the judged one."""

FILL = "replica"
"""The fill scored: the replica asks every cell, so every cell has records."""

COUNTED = frozenset({KEPT, "dropped-quota"})
"""The outcomes of a record that passed every check before the quota."""

CSV_COLUMNS = (
    "holdout",
    "cell",
    "records",
    "minimum",
    "median",
    "maximum",
    "at_minimum",
    "at_median",
    "top",
    "bottom",
)
"""The per-cell CSV's columns, in order."""


type Values = dict[str, float]
"""A record's value per column."""


@dataclass(frozen=True, slots=True)
class _Scored:
    """One candidate record with its score and its tie order."""

    record: Values
    score: float
    tie: str


@dataclass(slots=True)
class _Tally:
    """What the gate sums over the holdouts: each measure's contrasts and each side's concepts."""

    rows: tuple[FieldMapping, ...]
    mapped: dict[str, list[FieldMapping]] = field(init=False)
    contrasts: dict[str, list[float]] = field(default_factory=dict)
    present: dict[str, Counter[str]] = field(default_factory=dict)
    totals: Counter[str] = field(default_factory=Counter)
    ratios: list[float] = field(default_factory=list)
    lines: list[tuple[object, ...]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Group the rows by concept once."""
        self.mapped = by_concept(self.rows)

    def add_sides(self, sides: Mapping[str, Sequence[Values]]) -> None:
        """Count each mapped concept's records on each side."""
        for side, records in sides.items():
            self.totals[side] += len(records)
            counts = self.present.setdefault(side, Counter())
            for concept, concept_rows in self.mapped.items():
                counts[concept] += sum(any(row.applies(r) for row in concept_rows) for r in records)


def likelihood(
    experiment: Experiment, data: Data, files: Iterable[HoldoutCandidates], sink: TextIO
) -> bool:
    """Write the per-cell CSV to `sink`, log the paired measures and the gate, and return it.

    `files` are one run's candidates files, a sliced run's one per slice, read one at a time:
    only each file's scored lines are kept. A file's holdouts are those its lines name.

    Raises:
        EvaluationError: If no file is given, two files hold the same holdout, the files leave a
            holdout of the experiment out, a file's header differs from the first file's in any
            key but `run`, a holdout of the experiment holds no scored record or no cell of two,
            a scored line has no record of the frame's finite values, or the file names a cell the
            holdout's training rows do not hold.
    """
    pool: dict[int, list[Candidate]] = {}
    owner: dict[int, int] = {}
    first: Mapping[str, object] | None = None
    lines = 0
    for number, candidates in enumerate(files, start=1):
        first = _agree(first, candidates.header, number)
        lines += len(candidates.candidates)
        for holdout in sorted({line.holdout for line in candidates.candidates}):
            if holdout in owner:
                msg = f"holdout {holdout} is in both file {owner[holdout]} and file {number}"
                raise EvaluationError(msg)
            owner[holdout] = number
        for line in candidates.candidates:
            if (line.arm, line.fill) == (ARM, FILL) and line.outcome in COUNTED:
                pool.setdefault(line.holdout, []).append(line)
    if first is None:
        msg = "the gate needs at least one candidates file"
        raise EvaluationError(msg)
    missing = sorted(set(experiment.holdout_seeds) - owner.keys())
    if missing:
        msg = f"the candidates files leave the run's holdouts {missing} out"
        raise EvaluationError(msg)
    seeds = set(experiment.holdout_seeds)
    logger.info(
        "likelihood pool arm=%s fill=%s lines=%d scored=%d outside=%d",
        ARM,
        FILL,
        lines,
        sum(len(v) for k, v in pool.items() if k in seeds),
        sum(len(v) for k, v in pool.items() if k not in seeds),
    )
    tally = _Tally(data.field_rows)
    for seed in experiment.holdout_seeds:
        _holdout(seed, pool.get(seed, []), experiment, data, tally)
    separated = _log_gate(tally, experiment)
    writer = csv.writer(sink, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    writer.writerows(tally.lines)
    return separated


def candidates_file(
    path: Path, run: str, dataset: Dataset, *, single: bool
) -> tuple[bytes, HoldoutCandidates]:
    """A `--candidates` file's bytes and contents, its header checked against the named run.

    Raises:
        EvaluationError: If `single` and the file is another run's, or the file is schema 1 and
            `dataset` is not UCI Heart Disease.
        CandidatesFormatError: If the file is not a candidates file.
    """
    body = path.read_bytes()
    candidates = read(body)
    header = candidates.header
    if single and header["run"] != run:
        msg = f"the candidates file is run {header['run']!r}'s, not {run!r}'s"
        raise EvaluationError(msg)
    if header["schema"] == 1 and dataset is not Dataset.UCI:
        msg = f"a schema-1 candidates file holds UCI Heart Disease, not run {run!r}'s data"
        raise EvaluationError(msg)
    return body, candidates


def _agree(
    first: Mapping[str, object] | None, header: Mapping[str, object], number: int
) -> Mapping[str, object]:
    """The first file's header, after refusing file `number`'s if it differs but in `run`.

    Raises:
        EvaluationError: If `header` differs from `first` in a key other than `run`.
    """
    if first is None:
        return header
    differ = sorted(
        key for key in (first.keys() | header.keys()) - {"run"} if first.get(key) != header.get(key)
    )
    if differ:
        msg = f"file {number}'s header differs from the first file's in {', '.join(differ)}"
        raise EvaluationError(msg)
    return first


def _holdout(
    seed: int, lines: Sequence[Candidate], experiment: Experiment, data: Data, tally: _Tally
) -> None:
    """Score one holdout's records, split each cell at its median, and add the paired contrasts."""
    frame = data.frame
    parts = split(data.records, seed=seed, edges=frame.edges)
    ready = prepared(parts, experiment, edges=frame.edges)
    tally.ratios.append(held_out_ratio(ready.sized, edges=frame.edges))
    ablated = ready.ablated
    train = ablated.train
    logger.info(
        "split seed=%d train=%d test=%d fingerprint=%s",
        seed,
        len(train),
        len(ablated.test),
        ablated.fingerprint,
    )
    if not lines:
        msg = f"holdout {seed} holds no {ARM} {FILL} record to score"
        raise EvaluationError(msg)
    tops, bottoms, cell_lines = _split(seed, lines, train, data)
    if not tops:
        msg = f"holdout {seed} holds no cell with two {ARM} {FILL} records to split"
        raise EvaluationError(msg)
    tally.lines += cell_lines
    found_top = fidelity(train, _frame(tops, data), frame=frame)
    found_bottom = fidelity(train, _frame(bottoms, data), frame=frame)
    for measure in MEASURES:
        tally.contrasts.setdefault(str(measure), []).append(
            found_top[measure] - found_bottom[measure]
        )
    tally.add_sides({"top": tops, "bottom": bottoms, "training": _values(train)})


def _log_gate(tally: _Tally, experiment: Experiment) -> bool:
    """Log each measure's corrected interval, each concept's shares and the gate; return it."""
    ratio = max(tally.ratios)
    separated = True
    for measure, values in tally.contrasts.items():
        interval = corrected_interval(
            values, ratio=ratio, confidence=experiment.confidence, family=len(MEASURES)
        )
        separated = separated and interval.low > 0
        logger.info(
            "likelihood measure=%s arm=%s mean=%.6f low=%.6f high=%.6f holdouts=%d",
            measure,
            ARM,
            interval.mean,
            interval.low,
            interval.high,
            interval.holdouts,
        )
    for concept in tally.mapped:
        shares_of = (
            tally.present[side][concept] / tally.totals[side]
            for side in ("top", "bottom", "training")
        )
        logger.info("likelihood concept=%s top=%.4f bottom=%.4f training=%.4f", concept, *shares_of)
    logger.info("gate separated=%s", separated)
    return separated


def _split(
    seed: int, lines: Sequence[Candidate], train: pd.DataFrame, data: Data
) -> tuple[list[Values], list[Values], list[tuple[object, ...]]]:
    """Each cell's records split at its median: the tops, the bottoms and the cell's CSV row."""
    tops: list[Values] = []
    bottoms: list[Values] = []
    rows: list[tuple[object, ...]] = []
    by_cell = _scored(seed, lines, train, data)
    for cell in sorted(by_cell):
        scored = sorted(by_cell[cell], key=lambda s: (-s.score, s.tie))
        half = len(scored) // 2
        tops += [s.record for s in scored[:half]]
        bottoms += [s.record for s in scored[len(scored) - half :]]
        values = [s.score for s in scored]
        middle, least = statistics.median(values), min(values)
        rows.append(
            (
                seed,
                cell_key(cell),
                len(values),
                _number(least),
                _number(middle),
                _number(max(values)),
                f"{sum(v == least for v in values) / len(values):.6f}",
                f"{sum(v == middle for v in values) / len(values):.6f}",
                half,
                half,
            )
        )
    return tops, bottoms, rows


def _scored(
    seed: int, lines: Sequence[Candidate], train: pd.DataFrame, data: Data
) -> dict[Cell, list[_Scored]]:
    """Each record scored at its cell's shares on `train`, by cell.

    Raises:
        EvaluationError: If a record names a cell `train` does not hold, or holds no record.
    """
    frame = data.frame
    cells = record_cells(train, data.field_rows, name=data.name, edges=frame.edges)
    by_key = {cell_key(cell): cell for cell in cells.each}
    groups = {
        label: observe_records(
            train.loc[train[frame.label].eq(label)],
            data.field_rows,
            name=f"{data.name}/{frame.label}={label}",
        )
        for label in sorted({int(cell[2]) for cell in cells.each})
    }
    scorers: dict[Cell, Likelihood] = {}
    by_cell: dict[Cell, list[_Scored]] = {}
    for line in lines:
        cell = by_key.get(line.cell)
        if cell is None:
            msg = f"holdout {seed} has no training cell {line.cell!r} (call {line.call})"
            raise EvaluationError(msg)
        record = _record(line, frame.columns)
        if cell not in scorers:
            share = shares(cells.each[cell], groups[int(cell[2])])
            scorers[cell] = Likelihood(share, data.field_rows, frame)
        score = scorers[cell](record)
        tie = hashlib.sha256(f"{seed}:{line.call}:{line.position}".encode()).hexdigest()
        by_cell.setdefault(cell, []).append(_Scored(record, score, tie))
    return by_cell


def _record(line: Candidate, columns: Sequence[str]) -> Values:
    """`line`'s record as the frame's columns' finite values.

    Raises:
        EvaluationError: If the record is missing, lacks a column or holds a non-finite value.
    """
    found = line.candidate
    if not isinstance(found, Mapping) or not set(columns) <= set(found):
        msg = f"holdout {line.holdout} call {line.call} position {line.position} has no record"
        raise EvaluationError(msg)
    values = {column: found[column] for column in columns}
    if not all(
        isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
        for v in values.values()
    ):
        msg = (
            f"holdout {line.holdout} call {line.call} position {line.position} holds a value "
            f"that is not a finite number: {values!r}"
        )
        raise EvaluationError(msg)
    return {column: float(v) for column, v in values.items()}


def _number(value: float) -> str:
    return "-inf" if math.isinf(value) else f"{value:.6f}"


def _frame(records: Sequence[Values], data: Data) -> pd.DataFrame:
    columns = list(data.frame.columns)
    return pd.DataFrame([[r[c] for c in columns] for r in records], columns=columns).astype(
        dict(data.frame.dtypes)
    )


def _values(train: pd.DataFrame) -> list[Values]:
    return [{str(k): float(v) for k, v in r.items()} for r in train.to_dict("records")]
