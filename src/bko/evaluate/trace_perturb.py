"""A perturbation arm's fill rebuilt from its candidate lines, each record over its direct draw.

`bko-direct-pram` and `bko-direct-perturbed` change the unmapped fields of a `bko-direct` draw
(`bko.evaluate.perturb.unmapped`), so a re-draw alone gives their unperturbed rows.
`trace_perturbation` rebuilds the fill's rows from its lines (`bko.evaluate.shared.rebuilt`) and
refuses (`IdentityError`, exit 3) unless they hash to the `arm` line's sha256. It then re-draws the
fill at that line's seed (`bko.evaluate.redraw.redraw`), aligns the returned attempt's lines with
the draws in generation order, and refuses unless every other column of every line is its draw's.
Each record carries its draw's facts, as `bko.evaluate.trace_direct` writes them, and a `perturb`
object: the line's result, the fields that differ from the draw and the draw's values for them
(ADR `research/record-explainability`).

A schema-4 line holds its result. An older line's is read from the draw: a PRAM line is
`CHANGED` when a field differs, else `UNCHANGED`; a rewrite line that names its call is
`PERTURBED` when a field differs, else `UNCHANGED`; and a rewrite line that names no call kept its
draw, `KEPT`, for a reason the line does not hold. That `KEPT` is a result only `trace` writes;
it is none of the arm's `PERTURBS`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import pandas as pd

from bko.baselines.request import Request
from bko.evaluate.data import Data, digest
from bko.evaluate.errors import IdentityError
from bko.evaluate.experiment import PRAM_ARM
from bko.evaluate.holdout_candidates import CHANGED, KEPT, PERTURBED, UNCHANGED, Candidate
from bko.evaluate.perturb import unmapped
from bko.evaluate.redraw import Shares, redraw
from bko.evaluate.run_folder import fields
from bko.evaluate.shared import rebuilt
from bko.evaluate.trace_lines import (
    Fill,
    Trace,
    drawn_trace,
    json_value,
    log_arm,
    ordered,
    returned_lines,
)
from bko.generate.direct import Draw
from bko.ontology.terms import Ontology


def trace_perturbation(
    fill: Fill,
    lines: Sequence[Candidate],
    train: pd.DataFrame,
    request: Request,
    *,
    data: Data,
    ontologies: Mapping[str, Ontology],
    budget: int,
) -> list[Trace]:
    """Rebuild a perturbation arm's fill from its lines and explain each kept record.

    Raises:
        IdentityError: If the rebuilt rows' sha256 is not the `arm` line's, a returned line holds
            no record, the returned attempt's lines are not one per draw, or a column no
            perturbation changes differs from its draw's; or as `bko.evaluate.shared.rebuilt`.
        EvaluationError: As `bko.evaluate.perturb.unmapped`.
    """
    where = f"holdout {fill.holdout}'s {fill.arm} fill"
    seeded = dataclasses.replace(request, seed=fill.seed)
    logged, found = fields(fill.line)["sha256"], digest(rebuilt(lines, seeded, where=where))
    attempt, returned = returned_lines(lines)
    # The re-draw comes before the sha256 refusal, so the `trace arm` line names both checks.
    tag = (fill.holdout, fill.kind)
    redrawn = redraw(train, seeded, data=data, ontologies=ontologies, budget=budget, tag=tag)
    records = [_record(line) for line in returned]
    free = unmapped(data.field_rows, frame=data.frame)
    differs = _unperturbed(records, redrawn.drawn, free)
    log_arm(fill, logged=logged, rebuilt=found, check="unperturbed", held=not differs)
    if found != logged:
        msg = f"{where}'s rows rebuilt from its lines hash to {found}, not its arm line's {logged}"
        raise IdentityError(msg)
    if differs:
        msg = f"{where}: {differs}"
        raise IdentityError(msg)
    kept = [
        (line, record, draw)
        for line, record, draw in zip(returned, records, redrawn.drawn, strict=True)
        if line.outcome == KEPT
    ]
    facts = Shares(train, seeded.quota, data=data, ontologies=ontologies)
    return _explained(fill, seeded, attempt, kept, facts)


def _explained(
    fill: Fill,
    request: Request,
    attempt: int,
    kept: Sequence[tuple[Candidate, Mapping[str, object], Draw]],
    facts: Shares,
) -> list[Trace]:
    """Each kept line's `Trace`, its record over its draw's facts with its `perturb`, in row order.

    Raises:
        IdentityError: As `ordered`.
    """
    where = f"holdout {fill.holdout}'s {fill.arm} fill"
    order = ordered([record for _, record, _ in kept], request, where=where)
    traces = []
    for record_id, k in enumerate(order):
        line, record, draw = kept[k]
        traces.append(
            drawn_trace(
                fill,
                request,
                attempt=attempt,
                record_id=record_id,
                line=line,
                record=record,
                draw=draw,
                concepts=facts.concepts(draw),
                perturb=_perturb(fill.name, line, record, draw),
            )
        )
    return traces


def _record(line: Candidate) -> Mapping[str, object]:
    """A line's candidate, which a drawn row's line always holds as a mapping.

    Raises:
        IdentityError: If it holds anything else.
    """
    candidate = line.candidate
    if not isinstance(candidate, Mapping):
        msg = f"holdout {line.holdout}'s {line.arm} line {line.position} holds no record"
        raise IdentityError(msg)
    return {str(name): value for name, value in candidate.items()}


def _changed(record: Mapping[str, object], draw: Draw) -> list[str]:
    """The fields of `draw`, in its order, whose value `record` does not hold."""
    return [name for name in draw.record if record.get(name) != draw.record[name]]


def _unperturbed(
    records: Sequence[Mapping[str, object]], drawn: Sequence[Draw], free: Sequence[str]
) -> str:
    """Why `records` are not `drawn` outside `free`, or the empty string when they are."""
    if len(records) != len(drawn):
        return f"{len(records)} lines in the returned attempt, not one per draw of {len(drawn)}"
    names = {
        name
        for record, draw in zip(records, drawn, strict=True)
        for name in _changed(record, draw)
        if name not in free
    }
    if names:
        return f"columns no perturbation changes differ from the re-draw: {','.join(sorted(names))}"
    return ""


def _perturb(
    arm: str, line: Candidate, record: Mapping[str, object], draw: Draw
) -> dict[str, object]:
    """A line's `perturb` object: its result, the fields its draw differs in, and their values."""
    changed = _changed(record, draw)
    if line.perturb is not None:
        result = line.perturb
    elif arm == PRAM_ARM:
        result = CHANGED if changed else UNCHANGED
    elif line.call is None:
        result = KEPT
    else:
        result = PERTURBED if changed else UNCHANGED
    return {
        "result": result,
        "fields": changed,
        "drawn": {name: json_value(draw.record[name]) for name in changed},
    }
