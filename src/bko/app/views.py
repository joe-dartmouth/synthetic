"""The frames the page shows, built from what `bko.app.steps` and `bko.app.runs` return.

Nothing here imports streamlit, so tests read the values the page shows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from bko.app.runs import Run
from bko.app.steps import Analysis, Outcome
from bko.datasets.cells import CELL_COLUMNS, LABEL, LABEL_MEANINGS
from bko.evaluate.experiment import E1
from bko.evaluate.scores import Measure, Scope, Score, Source
from bko.evaluate.validity import reported, validity, verdict
from bko.gaps.definitions import Definition
from bko.gaps.report import STATEMENT_COLUMNS, cell_cells, labeller, statement_cells
from bko.generate.candidates import read
from bko.generate.decode import Undecoded


GAP_COLUMNS: Final = (*STATEMENT_COLUMNS, *CELL_COLUMNS)
PAIR_COLUMNS: Final = ("definition", "status", *GAP_COLUMNS)
SUMMARY_COLUMNS: Final = ("definition", "before", "closed", "still_open", "new")
SCORE_INDEX: Final = ("source", "measure", "scope")
PREDICTION: Final = (Measure.RECALL, Measure.PRECISION, Measure.AUC_PR)
"""The classifier measures the page compares across arms, in the order it shows them."""
ROLES: Final = ("filled", "comparison")
"""The `role` words of `prediction_table`: the arm that filled, and every other arm."""
TOTALS: Final = (
    "status",
    "calls",
    "failed",
    "retried",
    "input_tokens",
    "output_tokens",
    "usd",
    "live_usd",
    "replay",
    "resume",
)
"""The `end` line's fields the page shows."""


def cell_table(analysis: Analysis) -> pd.DataFrame:
    """The training split's cell counts, with a `status` word, so colour never carries it alone."""
    counts = analysis.counts
    words = counts["underrepresented"].map({True: "underrepresented", False: "covered"})
    return counts.assign(status=words)


def balance(analysis: Analysis) -> pd.DataFrame:
    """The training records per label, with each label's meaning in words."""
    totals = analysis.counts.groupby(LABEL, as_index=False)[["count"]].sum()
    return totals.assign(meaning=totals[LABEL].map(LABEL_MEANINGS))


def prediction_column(measure: Measure, scope: Scope) -> str:
    """The `prediction_table` column holding `measure` over `scope`."""
    return f"{measure.value} ({scope.value})"


def prediction_table(scores: Sequence[Score], filled: str) -> pd.DataFrame:
    """Each arm's classifier on one holdout: the real rows plus what the arm adds or reweights.

    `scores` hold one holdout, as `bko.evaluate.run.score_holdout` returns them. One row per arm,
    the arm named `filled` first and the others in `scores`' order, with a `role` word, then one
    column per measure of `PREDICTION` and scope; a score the split cannot define stays NaN. It
    ranks nothing and judges nothing: one split is not E1's verdict.
    """
    values: dict[str, dict[str, float]] = {}
    for s in scores:
        if s.source is Source.AUGMENTED and s.measure in PREDICTION:
            values.setdefault(s.arm, {})[prediction_column(s.measure, s.scope)] = s.value
    arms = [filled] if filled in values else []
    arms += [arm for arm in values if arm != filled]
    columns = [prediction_column(m, scope) for scope in Scope for m in PREDICTION]
    frame = pd.DataFrame.from_dict(values, orient="index").reindex(index=arms, columns=columns)
    roles = [ROLES[0] if arm == filled else ROLES[1] for arm in arms]
    return frame.rename_axis("arm").reset_index().assign(role=roles)[["arm", "role", *columns]]


def gap_table(analysis: Analysis, definition: Definition) -> pd.DataFrame:
    """One row per gap under `definition`: its statement with labels, then its cell."""
    label = labeller(analysis.knowledge.ontologies)
    rows = [
        (*statement_cells(gap.statement, label), *cell_cells(gap.cell))
        for gap in analysis.found[definition]
    ]
    return pd.DataFrame(rows, columns=list(GAP_COLUMNS))


def closure_summary(outcome: Outcome) -> pd.DataFrame:
    """Each definition's gaps before the fill, and how many closed, stayed open or are new."""
    rows = [
        (c.definition.value, len(c.before), len(c.closed), len(c.still_open), len(c.opened))
        for c in outcome.closures
    ]
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def closure_pairs(analysis: Analysis, outcome: Outcome) -> pd.DataFrame:
    """Every gap listed before or after the fill, with whether it closed, stayed open or is new."""
    label = labeller(analysis.knowledge.ontologies)
    rows = []
    for reading in outcome.closures:
        closed, opened = set(reading.closed), set(reading.opened)
        for gap in reading.listed(analysis.cells.each):
            status = "closed" if gap in closed else "new" if gap in opened else "still open"
            cells = (*statement_cells(gap.statement, label), *cell_cells(gap.cell))
            rows.append((reading.definition.value, status, *cells))
    return pd.DataFrame(rows, columns=list(PAIR_COLUMNS))


def score_table(scores: Sequence[Score]) -> pd.DataFrame:
    """One holdout's scores, one row per source, measure and scope and one column per arm."""
    rows = [(s.source.value, s.measure.value, s.scope.value, s.arm, s.value) for s in scores]
    frame = pd.DataFrame(rows, columns=[*SCORE_INDEX, "arm", "value"])
    return frame.pivot(index=list(SCORE_INDEX), columns="arm", values="value")


def totals_table(totals: Mapping[str, object]) -> pd.DataFrame:
    """A call log's `end` line as one row."""
    return pd.DataFrame([{key: totals.get(key) for key in TOTALS}])


def arm_means(scores: Sequence[Score]) -> pd.DataFrame:
    """Each arm's mean of every measure over the holdouts, and how many holdouts there are.

    `Measure.SHARED_SPLIT` is a fingerprint read as a number, not a quantity, so it has no mean.
    """
    kept = [s for s in scores if s.measure is not Measure.SHARED_SPLIT]
    rows = [(s.source.value, s.measure.value, s.scope.value, s.arm, s.value) for s in kept]
    frame = pd.DataFrame(rows, columns=[*SCORE_INDEX, "arm", "value"])
    means = frame.pivot_table(
        index=list(SCORE_INDEX), columns="arm", values="value", aggfunc="mean"
    )
    return means.assign(holdouts=len({s.holdout for s in kept}))


def e1_validity(scores: Sequence[Score]) -> pd.DataFrame:
    """Each BKO arm's candidate counts and rate over the run, with E1's verdict on its candidate.

    Raises:
        EvaluationError: As `bko.evaluate.validity.validity`, for a run missing one of E1's
            holdouts.
    """
    rows = []
    for arm in reported(scores):
        found = validity(scores, E1, arm)
        counts = found.counts
        rows.append(
            {
                "arm": arm,
                "generated": counts[Measure.GENERATED],
                "passed": counts[Measure.PASSED],
                "rate": float(found.rate),
                "verdict": verdict(found, E1).value if arm == E1.candidate else "reported",
            }
        )
    return pd.DataFrame(rows)


def role_table(run: Run) -> pd.DataFrame:
    """Each role the run folder holds, with its size in bytes."""
    rows = [(role, path.stat().st_size) for role, path in run.roles.items()]
    return pd.DataFrame(rows, columns=["role", "bytes"])


def candidates_table(path: Path) -> pd.DataFrame:
    """One row per record line of a candidates file: its call, cell and position, then its fields.

    The header line is read, refused if malformed, and shown as no row.

    Raises:
        CandidatesFormatError: As `bko.generate.candidates.read`.
    """
    rows = []
    for line in read(path.read_bytes()).lines:
        candidate = line.candidate
        if isinstance(candidate, Undecoded):
            item = {"undecoded": candidate.reason.value}
        elif isinstance(candidate, Mapping):
            item = dict(candidate)
        else:
            item = {"undecoded": repr(candidate)}
        rows.append({"call": line.call, "cell": line.cell, "position": line.position, **item})
    return pd.DataFrame(rows)


def _flags(frame: pd.DataFrame, column: str) -> pd.Series:
    """`column` as read, refusing anything but True and False.

    Raises:
        KeyError: If `frame` has no `column`.
        ValueError: If `column` holds a blank or any value but True and False.
    """
    values = frame[column]
    if not pd.api.types.is_bool_dtype(values):
        msg = f"the {column!r} column holds values other than True and False"
        raise ValueError(msg)
    return values


def closure_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Each definition's pairs that closed, stayed open or are new, in `Definition` order.

    Raises:
        KeyError: If a column is missing.
        ValueError: If a reading holds anything but True and False.
    """
    before, after = _flags(frame, "before"), _flags(frame, "after")
    marks = frame.assign(closed=before & ~after, still_open=before & after, new=~before & after)
    counts = marks.groupby("definition")[["closed", "still_open", "new"]].sum()
    order = [definition.value for definition in Definition]
    return counts.reindex(order, fill_value=0).rename_axis("definition").reset_index()


def gap_counts(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """A gap table's pairs per group of `keys`, and how many are listed under each reading.

    Raises:
        KeyError: If a key column is missing.
        ValueError: If a reading holds anything but True and False.
    """
    readings = [column for column in ("before", "after", "truth") if column in frame.columns]
    marks = frame.assign(pairs=1, **{column: _flags(frame, column) for column in readings})
    return marks.groupby(list(keys), sort=False)[["pairs", *readings]].sum().reset_index()


def status_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """A reproduction's rows per status."""
    return frame["status"].value_counts(sort=False).rename_axis("status").reset_index()
