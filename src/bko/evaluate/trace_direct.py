"""A direct arm's fill re-drawn at its `arm` line's seed, each kept record with its facts.

`trace_direct` re-draws a fill of an arm of `DIRECT_ARMS` through `bko.evaluate.redraw.redraw`
with that arm's presence, which draws nothing more to name its sources, and refuses
(`IdentityError`, exit 3) unless the `arm` line's `presence` is its arm's, the re-drawn rows hash
to the line's sha256 and its kept draws are the candidates file's kept lines. Each record names
its request's cell and quota, the training record each field was read from, its fallback fields
and its closure, and its concepts: for a draw at shares each one's share with its parts and
S-nodes, for a logic-only draw each one of its closed draw (`closed_facts`); labels come from the
pinned releases (ADR `research/record-explainability`).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from functools import partial

import pandas as pd

from bko.baselines.request import Request
from bko.evaluate.call_prompts import Json
from bko.evaluate.data import Data, digest
from bko.evaluate.errors import EvaluationError, IdentityError
from bko.evaluate.experiment import DIRECT_ARMS
from bko.evaluate.holdout_candidates import Candidate
from bko.evaluate.redraw import Shares, redraw
from bko.evaluate.run_folder import fields
from bko.evaluate.trace_lines import Fill, Trace, drawn_trace, kept_lines, log_arm, ordered
from bko.generate.direct import Draw, logic_components
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

    The fill is re-drawn with its arm's presence (`DIRECT_ARMS`). A `Presence.SHARES` record's
    concepts are its share facts (`bko.evaluate.redraw.Shares`); a `Presence.LOGIC` record's are
    its closed draw (`closed_facts`), since a logic-only draw reads no share.

    Raises:
        IdentityError: If the `arm` line's params are not a JSON object or its `presence` is not
            its arm's, the re-drawn rows' sha256 is not the `arm` line's, or its kept draws are
            not the candidates file's kept lines.
        EvaluationError: Before any draw, if a logic-only arm's components cannot be enumerated
            (`bko.generate.direct.logic_components`).
    """
    presence = _presence(fill, data, ontologies)
    seeded = dataclasses.replace(request, seed=fill.seed)
    tag = (fill.holdout, fill.kind)
    redrawn = redraw(
        train, seeded, data=data, ontologies=ontologies, budget=budget, tag=tag, presence=presence
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
    concepts = (
        Shares(train, seeded.quota, data=data, ontologies=ontologies).concepts
        if presence is Presence.SHARES
        else partial(closed_facts, ontologies=ontologies)
    )
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


def _presence(fill: Fill, data: Data, ontologies: Mapping[str, Ontology]) -> Presence:
    """The presence `fill`'s arm draws with, checked against its `arm` line before any draw.

    Raises:
        IdentityError: If the `arm` line's params are not a JSON object, or name another
            presence.
        EvaluationError: If a logic-only arm's components cannot be enumerated.
    """
    presence = DIRECT_ARMS[fill.name]
    params = fields(fill.line)["params"]
    try:
        named = json.loads(params)
    except ValueError as exc:
        msg = f"holdout {fill.holdout}'s {fill.arm}'s arm line params are not JSON: {params!r}"
        raise IdentityError(msg) from exc
    if not isinstance(named, dict):
        msg = f"holdout {fill.holdout}'s {fill.arm}'s arm line params are not an object: {params!r}"
        raise IdentityError(msg)
    logged = named.get("presence", presence.value)
    if logged != presence.value:
        msg = f"holdout {fill.holdout}'s {fill.arm}'s arm line draws presence {logged!r},"
        msg += f" not {presence.value!r}"
        raise IdentityError(msg)
    if presence is Presence.LOGIC:
        try:
            logic_components(data.field_rows, ontologies, data.frame)
        except ValueError as exc:
            msg = f"{fill.name} cannot draw dataset {data.name} on frame {data.frame.name}: {exc}"
            raise EvaluationError(msg) from exc
    return presence


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
