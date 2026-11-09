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
draw, `KEPT`, for a reason the line does not hold.
"""

from __future__ import annotations

import dataclasses
import logging
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
from bko.evaluate.shared import rebuilt, returned_attempt
from bko.evaluate.trace_lines import Fill, Trace, json_value, ordered
from bko.gaps.observe import cell_key
from bko.generate.direct import Draw
from bko.ontology.terms import Ontology


logger = logging.getLogger("bko.evaluate")


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
        IdentityError: If the rebuilt rows' sha256 is not the `arm` line's, the returned attempt's
            lines are not one per draw, or a column no perturbation changes differs from its
            draw's; or as `bko.evaluate.shared.rebuilt`.
        EvaluationError: As `bko.evaluate.perturb.unmapped`.
    """
    where = f"holdout {fill.holdout}'s {fill.arm} fill"
    seeded = dataclasses.replace(request, seed=fill.seed)
    logged, found = fields(fill.line)["sha256"], digest(rebuilt(lines, seeded, where=where))
    attempt = returned_attempt(lines)
    tag = (fill.holdout, fill.kind)
    redrawn = redraw(train, seeded, data=data, ontologies=ontologies, budget=budget, tag=tag)
    returned = [line for line in lines if line.attempt == attempt]
    free = unmapped(data.field_rows, frame=data.frame)
    differs = _unperturbed(returned, redrawn.drawn, free)
    logger.log(
        logging.INFO if not differs and found == logged else logging.WARNING,
        "trace arm holdout=%d arm=%s arm_sha256=%s rebuilt_sha256=%s kept_lines=%s matches=%s",
        fill.holdout,
        fill.arm,
        logged,
        found,
        str(not differs).lower(),
        str(not differs and found == logged).lower(),
    )
    if found != logged:
        msg = f"{where}'s rows rebuilt from its lines hash to {found}, not its arm line's {logged}"
        raise IdentityError(msg)
    if differs:
        msg = f"{where}: {differs}"
        raise IdentityError(msg)
    pairs = [
        (line, draw)
        for line, draw in zip(returned, redrawn.drawn, strict=True)
        if line.outcome == KEPT
    ]
    facts = Shares(train, seeded.quota, data=data, ontologies=ontologies)
    order = ordered([line.candidate for line, _ in pairs], seeded, where=where)
    return [
        _trace(fill, seeded, attempt, record_id, pairs[k], facts)
        for record_id, k in enumerate(order)
    ]


def _trace(
    fill: Fill,
    request: Request,
    attempt: int,
    record_id: int,
    pair: tuple[Candidate, Draw],
    facts: Shares,
) -> Trace:
    """One kept record's `Trace`: its line's values over its draw's facts and its `perturb`."""
    line, draw = pair
    return Trace(
        fill.holdout,
        fill.arm,
        fill.kind,
        record_id,
        line.cell,
        dict(_record(line)),
        line.outcome,
        line.failed,
        {
            "attempt": attempt,
            "cell": cell_key(draw.cell),
            "quota": request.quota[draw.cell],
            "rows": request.rows,
            "seed": fill.seed,
        },
        facts.concepts(draw),
        dict(draw.sources),
        tuple(sorted(draw.fallback)),
        tuple(sorted(draw.closed)),
        perturb=_perturb(fill.name, line, draw),
    )


def _record(line: Candidate) -> Mapping[str, object]:
    """A line's candidate, which a drawn row's line always holds as a mapping."""
    candidate = line.candidate
    if not isinstance(candidate, Mapping):
        msg = f"holdout {line.holdout}'s {line.arm} line {line.position} holds no record"
        raise IdentityError(msg)
    return {str(name): value for name, value in candidate.items()}


def _unperturbed(lines: Sequence[Candidate], drawn: Sequence[Draw], free: Sequence[str]) -> str:
    """Why `lines` are not `drawn` outside `free`, or the empty string when they are."""
    if len(lines) != len(drawn):
        return f"{len(lines)} lines in the returned attempt, not one per draw of {len(drawn)}"
    names: set[str] = set()
    for line, draw in zip(lines, drawn, strict=True):
        record = _record(line)
        names.update(
            name
            for name in draw.record
            if name not in free and record.get(name) != draw.record[name]
        )
    if names:
        return f"columns no perturbation changes differ from the re-draw: {','.join(sorted(names))}"
    return ""


def _perturb(arm: str, line: Candidate, draw: Draw) -> dict[str, object]:
    """A line's `perturb` object: its result, the fields its draw differs in, and their values."""
    record = _record(line)
    changed = [name for name in draw.record if record[name] != draw.record[name]]
    result = line.perturb
    if result is None:
        moved = CHANGED if arm == PRAM_ARM else PERTURBED
        result = KEPT if arm != PRAM_ARM and line.call is None else moved if changed else UNCHANGED
    return {
        "result": result,
        "fields": changed,
        "drawn": {name: json_value(draw.record[name]) for name in changed},
    }
