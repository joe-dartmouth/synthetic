"""A direct arm's fill re-drawn at its `arm` line's seed, each kept record with its facts.

`trace_direct` re-draws a `bko-direct` fill through `bko.evaluate.redraw.redraw`, which draws
nothing more to name its sources, and refuses (`IdentityError`, exit 3) unless the re-drawn rows
hash to the `arm` line's sha256 and its kept draws are the candidates file's kept lines. Each
record names its request's cell and quota, each concept's share with its parts and S-nodes, the
training record each field was read from, its fallback fields and its closure; labels come from
the pinned releases (ADR `research/record-explainability`).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence

import pandas as pd

from bko.baselines.request import Request
from bko.evaluate.data import Data, digest
from bko.evaluate.errors import IdentityError
from bko.evaluate.holdout_candidates import Candidate
from bko.evaluate.redraw import Shares, redraw
from bko.evaluate.run_folder import fields
from bko.evaluate.trace_lines import Fill, Trace, kept_lines, ordered
from bko.gaps.observe import cell_key
from bko.ontology.terms import Ontology


logger = logging.getLogger("bko.evaluate")


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
    """Re-draw a `bko-direct` fill at its `arm` line's seed and explain each kept record.

    Raises:
        IdentityError: If the re-drawn rows' sha256 is not the `arm` line's, or its kept draws are
            not the candidates file's kept lines.
    """
    seeded = dataclasses.replace(request, seed=fill.seed)
    tag = (fill.holdout, fill.kind)
    redrawn = redraw(train, seeded, data=data, ontologies=ontologies, budget=budget, tag=tag)
    logged, rebuilt = fields(fill.line)["sha256"], digest(redrawn.rows)
    kept = redrawn.kept
    attempt, stored = kept_lines(lines)
    same = [dict(draw.record) for draw in kept] == [line.candidate for line in stored]
    logger.log(
        logging.INFO if same and rebuilt == logged else logging.WARNING,
        "trace arm holdout=%d arm=%s arm_sha256=%s rebuilt_sha256=%s kept_lines=%s matches=%s",
        fill.holdout,
        fill.arm,
        logged,
        rebuilt,
        str(same).lower(),
        str(same and rebuilt == logged).lower(),
    )
    if rebuilt != logged:
        msg = f"holdout {fill.holdout}'s {fill.arm} fill re-drawn at seed {fill.seed} hashes to"
        msg += f" {rebuilt}, not its arm line's {logged}"
        raise IdentityError(msg)
    if not same:
        msg = f"holdout {fill.holdout}'s {fill.arm} re-drawn records are not its kept lines"
        raise IdentityError(msg)
    facts = Shares(train, seeded.quota, data=data, ontologies=ontologies)
    where = f"holdout {fill.holdout}'s {fill.arm} fill"
    order = ordered([draw.record for draw in kept], seeded, where=where)
    return [
        Trace(
            fill.holdout,
            fill.arm,
            fill.kind,
            record_id,
            stored[k].cell,
            dict(kept[k].record),
            stored[k].outcome,
            stored[k].failed,
            {
                "attempt": attempt,
                "cell": cell_key(kept[k].cell),
                "quota": seeded.quota[kept[k].cell],
                "rows": seeded.rows,
                "seed": fill.seed,
            },
            facts.concepts(kept[k]),
            dict(kept[k].sources),
            tuple(sorted(kept[k].fallback)),
            tuple(sorted(kept[k].closed)),
        )
        for record_id, k in enumerate(order)
    ]
