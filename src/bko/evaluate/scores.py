"""Scores, one per row: what `python -m bko.evaluate score` writes and `compare` reads.

A row names the holdout seed, the arm, the rows the score describes (`Source`), the measure and the
test records it covers (`Scope`). A measure a scope cannot define is NaN, written as an empty
field.
"""

from __future__ import annotations

import enum
import io
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import pandas as pd

from bko.baselines.request import Method
from bko.evaluate.errors import ScoresFormatError


class Source(enum.StrEnum):
    """The rows a score describes."""

    AUGMENTED = "augmented"
    """A classifier fit on the real training rows and the arm's added rows."""
    REPLICA = "replica"
    """The arm's replica of the training split: a classifier fit on it alone, and its fidelity."""
    ADDED = "added"
    """The arm's added rows: how many, and how near they sit to the training rows."""
    HELD_OUT = "held-out"
    """The test rows: how near they sit to the training rows, and the split's held-out ratio."""
    CANDIDATES = "candidates"
    """A BKO arm's gap fill's candidates on the holdout: generated, passed and dropped, refused
    attempts included."""
    REPLICA_CANDIDATES = "replica-candidates"
    """A BKO arm's replica fill's candidates on the holdout, counted as `CANDIDATES` counts the gap
    fill's."""


class Measure(enum.StrEnum):
    """What a score measures."""

    RECALL = "recall"
    PRECISION = "precision"
    AUC_PR = "auc-pr"
    RECALL_HGB = "recall-hgb"
    """`RECALL` of the boosting model (`bko.evaluate.classifier.Model.HGB`), which no criterion
    reads; each model but the regression suffixes its measures with its name."""
    PRECISION_HGB = "precision-hgb"
    AUC_PR_HGB = "auc-pr-hgb"
    QUALITY = "quality"
    COLUMN_SHAPES = "column-shapes"
    COLUMN_PAIR_TRENDS = "column-pair-trends"
    ROWS = "rows"
    DCR_P5 = "dcr-p5"
    DCR_MEDIAN = "dcr-median"
    HELD_OUT_RATIO = "held-out-ratio"
    FLAGGED_FINGERPRINT = "flagged-fingerprint"
    """The holdout's flagged cells, as the first 12 hex digits of the sha256 of their sorted
    list's `repr`, read as an integer: below 2**48, so a float holds it exactly. `compare
    --beside` pairs a holdout only when both files flag the same cells."""
    COPIES_KEPT = "copies-kept"
    """1.0 on a holdout of a run that keeps copies (`Experiment.decontaminate` false), whose arms
    are scored on the whole test split; a decontaminating run writes no such row. `compare
    --beside` lets a copy pair a holdout only when both files hold it there."""
    SHARED_SPLIT = "shared-split"
    """A holdout scored on a split shared with other runs, which drops the test records their
    rows copy: the first 12 hex digits of the sha256 of the sorted kept record ids' `repr`, read
    as an integer (`bko.evaluate.shared_split.marker`). `compare --beside` lets a copy pair a
    holdout when both files hold bitwise-equal rows there."""
    GENERATED = "generated"
    PASSED = "passed"
    DROPPED_RANGE = "dropped-range"
    DROPPED_CELL = "dropped-cell"
    DROPPED_CONCEPTS = "dropped-concepts"
    """A prescribing arm's candidates dropped for showing other concepts than their line; no
    other arm writes it."""
    DROPPED_CURATION = "dropped-curation"
    """A curating arm's candidates its curator dropped; no other arm writes it."""
    DROPPED_QUOTA = "dropped-quota"
    TEST_COPIES = "test-copies"


class Scope(enum.StrEnum):
    """The test records a classifier's score covers."""

    ALL = "all"
    FLAGGED = "flagged-subgroups"
    """The test records of every age-sex subgroup that holds an underrepresented cell."""


@dataclass(frozen=True, slots=True)
class Score:
    """One measured value.

    Attributes:
        holdout: The split seed.
        arm: The arm's name; the held-out rows' scores sit under `HELD_OUT_ARM`.
        source: The rows the value describes.
        measure: What it measures.
        scope: The test records it covers; `Scope.ALL` for a score of rows rather than records.
        value: The value; NaN where the scope cannot define the measure.
    """

    holdout: int
    arm: str
    source: Source
    measure: Measure
    scope: Scope
    value: float


SCORE_COLUMNS: Final = ("holdout", "arm", "source", "measure", "scope", "value")
"""The scores CSV's columns, in order."""

HELD_OUT_ARM: Final = Method.NONE.value
"""The arm the held-out rows' scores sit under: no augmentation, whose rows are the real ones."""


def render(scores: Iterable[Score]) -> str:
    """Write `scores` as the scores CSV, in the order given."""
    rows = [
        (s.holdout, s.arm, s.source.value, s.measure.value, s.scope.value, s.value) for s in scores
    ]
    return pd.DataFrame(rows, columns=list(SCORE_COLUMNS)).to_csv(index=False, lineterminator="\n")


def parse(text: str) -> list[Score]:
    """Read a scores CSV back.

    Raises:
        ScoresFormatError: If the text is not CSV, its columns differ, a row is malformed or names
            an unknown source, measure or scope, a holdout is not an integer, an arm is empty, or
            two rows share a holdout, arm, source, measure
            and scope.
    """
    try:
        frame = pd.read_csv(
            io.StringIO(text),
            dtype={"arm": str},
            keep_default_na=False,
            na_values={"value": [""]},
            float_precision="round_trip",
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        msg = f"the scores are not CSV: {exc}"
        raise ScoresFormatError(msg) from exc
    if tuple(frame.columns) != SCORE_COLUMNS:
        msg = f"the scores' columns are {list(frame.columns)}, not {list(SCORE_COLUMNS)}"
        raise ScoresFormatError(msg)
    if not pd.api.types.is_integer_dtype(frame["holdout"]) or frame["arm"].str.len().eq(0).any():
        msg = "a score row has a holdout that is not an integer or an empty arm"
        raise ScoresFormatError(msg)
    try:
        scores = [
            Score(int(holdout), arm, Source(source), Measure(measure), Scope(scope), float(value))
            for holdout, arm, source, measure, scope, value in frame.itertuples(index=False)
        ]
    except (TypeError, ValueError) as exc:
        msg = f"a score row is malformed: {exc}"
        raise ScoresFormatError(msg) from exc
    keys = [(s.holdout, s.arm, s.source, s.measure, s.scope) for s in scores]
    if len(set(keys)) != len(keys):
        msg = "two score rows share a holdout, arm, source, measure and scope"
        raise ScoresFormatError(msg)
    return scores
