"""A direct arm's fill re-drawn at its `arm` line's seed, each kept record with its facts.

`trace_direct` re-draws a fill of an arm of `DIRECT_ARMS` through `bko.evaluate.redraw.redraw`
at that arm's spec, which draws nothing more to name its sources, and refuses (`IdentityError`,
exit 3) unless every spec field the `arm` line logs is its arm's, the re-drawn rows hash to the
line's sha256 and its kept draws are the candidates file's kept lines. Each record names its
request's cell and quota, the training record each field was read from, its fallback fields and
its closure, and its concepts: for a draw at shares each one's share with its parts and S-nodes,
for a logic-only draw each one of its closed draw (`closed_facts`), for a tree draw each tree
field's leaf with its parts at every level (`bko.evaluate.redraw.TreeShares`), for a conditional
`is_a` draw each broad concept's pair parts beside its share, and for a draw without the mapping
table none; labels come from the pinned releases (ADR
`research/record-explainability`).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from functools import partial

import pandas as pd

from bko.baselines.request import Request
from bko.evaluate.call_prompts import Json
from bko.evaluate.data import Data, digest
from bko.evaluate.errors import EvaluationError, IdentityError
from bko.evaluate.experiment import DIRECT_ARMS
from bko.evaluate.holdout_candidates import Candidate
from bko.evaluate.redraw import Shares, TreeShares, redraw
from bko.evaluate.run_folder import fields
from bko.evaluate.trace_lines import Fill, Trace, drawn_trace, kept_lines, log_arm, ordered
from bko.generate.direct import Draw, logic_components, plan_hierarchy
from bko.generate.draw_spec import DEFAULT_SPEC, DrawSpec
from bko.generate.presence import Presence
from bko.ontology.terms import Ontology


def trace_direct(
    fill: Fill,
    lines: Sequence[Candidate],
    train: pd.DataFrame,
    request: Request,
    *,
    data: Data,
    ontologies: Mapping[str, Ontology],
    budget: int,
) -> list[Trace]:
    """Re-draw a direct arm's fill at its `arm` line's seed and explain each kept record.

    The fill is re-drawn at its arm's spec (`DIRECT_ARMS`). A `Presence.SHARES` record's
    concepts are its share facts (`bko.evaluate.redraw.Shares`); a `Presence.LOGIC` record's are
    its closed draw (`closed_facts`), since a logic-only draw reads no share; a tree draw's are
    `TreeShares`'; a conditional `is_a` draw's are its share facts with each broad concept's
    pair parts (`bko.generate.hierarchy.PairPart`); a record drawn without the mapping table, or
    at `Presence.POOLED`, has none.

    Raises:
        IdentityError: If the `arm` line's params are not a JSON object or a spec field it logs
            is not its arm's, the re-drawn rows' sha256 is not the `arm` line's, or its kept
            draws are not the candidates file's kept lines.
        EvaluationError: Before any draw, if a logic-only arm's components cannot be enumerated
            (`bko.generate.direct.logic_components`).
    """
    spec = arm_spec(fill, data, ontologies)
    seeded = dataclasses.replace(request, seed=fill.seed)
    tag = (fill.holdout, fill.kind)
    redrawn = redraw(
        train, seeded, data=data, ontologies=ontologies, budget=budget, tag=tag, spec=spec
    )
    logged, rebuilt = fields(fill.line)["sha256"], digest(redrawn.rows)
    kept = redrawn.kept
    attempt, stored = kept_lines(lines)
    same = [dict(draw.record) for draw in kept] == [line.candidate for line in stored]
    log_arm(fill, logged=logged, rebuilt=rebuilt, check="kept_lines", held=same)
    if rebuilt != logged:
        msg = f"holdout {fill.holdout}'s {fill.arm} fill re-drawn at seed {fill.seed} hashes to"
        msg += f" {rebuilt}, not its arm line's {logged}"
        raise IdentityError(msg)
    if not same:
        msg = f"holdout {fill.holdout}'s {fill.arm} re-drawn records are not its kept lines"
        raise IdentityError(msg)
    concepts: Callable[[Draw], tuple[Json, ...]]
    if not spec.mapped or spec.presence is Presence.POOLED:
        concepts = _no_concepts
    elif spec.tree:
        concepts = TreeShares(train, seeded.quota, data=data, ontologies=ontologies).concepts
    elif spec.presence is Presence.SHARES:
        hierarchy = plan_hierarchy(data.field_rows, ontologies, data.frame, spec)
        concepts = Shares(
            train, seeded.quota, data=data, ontologies=ontologies, hierarchy=hierarchy
        ).concepts
    else:
        concepts = partial(closed_facts, ontologies=ontologies)
    where = f"holdout {fill.holdout}'s {fill.arm} fill"
    order = ordered([draw.record for draw in kept], seeded, where=where)
    return [
        drawn_trace(
            fill,
            seeded,
            attempt=attempt,
            record_id=record_id,
            line=stored[k],
            record=kept[k].record,
            draw=kept[k],
            concepts=concepts(kept[k]),
        )
        for record_id, k in enumerate(order)
    ]


def arm_spec(fill: Fill, data: Data, ontologies: Mapping[str, Ontology]) -> DrawSpec:
    """The spec `fill`'s arm draws at, checked against its `arm` line before any draw.

    Each field of `DrawSpec` the line's params omit reads as its default, since an arm logs only
    the fields off their defaults.

    Raises:
        IdentityError: If the `arm` line's params are not a JSON object, or name another value
            of a spec field, a value of another type included (JSON `0` is not `false`).
        EvaluationError: If a logic-only arm's components cannot be enumerated.
    """
    spec = DIRECT_ARMS[fill.name]
    params = fields(fill.line)["params"]
    try:
        named = json.loads(params)
    except ValueError as exc:
        msg = f"holdout {fill.holdout}'s {fill.arm}'s arm line params are not JSON: {params!r}"
        raise IdentityError(msg) from exc
    if not isinstance(named, dict):
        msg = f"holdout {fill.holdout}'s {fill.arm}'s arm line params are not an object: {params!r}"
        raise IdentityError(msg)
    wanted, defaults = spec.values(), DEFAULT_SPEC.values()
    for name, default in defaults.items():
        logged = named.get(name, default)
        if type(logged) is not type(wanted[name]) or logged != wanted[name]:
            msg = f"holdout {fill.holdout}'s {fill.arm}'s arm line draws {name} {logged!r},"
            msg += f" not {wanted[name]!r}"
            raise IdentityError(msg)
    if spec.presence is Presence.LOGIC:
        try:
            logic_components(data.field_rows, ontologies, data.frame)
        except ValueError as exc:
            msg = f"{fill.name} cannot draw dataset {data.name} on frame {data.frame.name}: {exc}"
            raise EvaluationError(msg) from exc
    return spec


def _no_concepts(draw: Draw) -> tuple[Json, ...]:
    """A draw without the mapping table, or a pooled one, names no concept."""
    del draw
    return ()


def closed_facts(draw: Draw, *, ontologies: Mapping[str, Ontology]) -> tuple[Json, ...]:
    """Each concept of a logic-only draw's closure, sorted: its label and whether it was drawn.

    The label is the term's name in its pinned release, None when the release lacks it; `drawn`
    is false for a concept the closure added under `is_a` or nesting.
    """
    facts = []
    for concept in sorted(draw.closed):
        release = ontologies.get(concept.partition(":")[0])
        term = None if release is None else release.terms.get(concept)
        label = None if term is None else term.name
        facts.append({"concept": concept, "label": label, "drawn": concept in draw.concepts})
    return tuple(facts)
